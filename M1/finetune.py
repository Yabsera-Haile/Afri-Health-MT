from transformers import  AutoProcessor,AutoModelForCausalLM,AutoModelForSeq2SeqLM,AutoModelForSequenceClassification, AutoModelForMultimodalLM, TrainingArguments, Trainer, DataCollatorForSeq2Seq, DataCollatorWithPadding
import torch
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
import math, os, glob, copy, hashlib, json


Head= {
    "causal_lm":      AutoModelForCausalLM,
    "seq2seq":        AutoModelForSeq2SeqLM,
    "classification": AutoModelForSequenceClassification,
    "multimodal":     AutoModelForMultimodalLM,
}

def _dtype(x):
    return getattr(torch, x) if isinstance(x, str) and x != "auto" else x

def load_model_and_processor(cfg):
    training = cfg.get("mode", "train") == "train"

    processor = AutoProcessor.from_pretrained(cfg["model_id"])
    tokenizer = getattr(processor, "tokenizer", processor)      

    default_map = {"": 0} if torch.cuda.is_available() else None
    model = Head[cfg["task"]].from_pretrained(
        cfg["model_id"],
        dtype=_dtype(cfg.get("dtype", "auto")),
        device_map=cfg.get("device_map", default_map),
        **cfg.get("load_kwargs", {}),
    )

    use_cache = cfg.get("use_cache", not training)
    model.config.use_cache = use_cache
    if hasattr(model.config, "text_config"):                    
        model.config.text_config.use_cache = use_cache

    if cfg.get("verbose"):
        print("Linear layer types:", sorted({n.split(".")[-1] for n, m in model.named_modules()
                                             if isinstance(m, torch.nn.Linear)}))

    quantized = cfg.get("load_kwargs", {}).get("quantization_config") is not None
    grad_ckpt = cfg.get("gradient_checkpointing", True)

    if cfg.get("adapter_path"):                                   
        model = PeftModel.from_pretrained(model, cfg["adapter_path"], is_trainable=training)
        if not training and cfg.get("merge_adapter") and not quantized:
            model = model.merge_and_unload()                     

    elif training and cfg.get("use_lora"):                       
        if quantized:
            model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=grad_ckpt)
        elif grad_ckpt:
            model.enable_input_require_grads()
        model = get_peft_model(model, LoraConfig(**cfg["lora_config"]))
        model.print_trainable_parameters()

    frozen_kw = cfg.get("must_stay_frozen", [])
    bad = [n for n, p in model.named_parameters() if p.requires_grad and any(k in n for k in frozen_kw)]
    assert not bad, f"these should be frozen but are trainable: {bad[:3]}"

    if not training:
        model.eval()

    return processor, tokenizer, model

def _ids(id):
    return id["input_ids"] if isinstance(id,dict) else id

def setup_tokenizer(tokenizer, pad_side="right"):
    tokenizer.padding_side = pad_side
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def tokenize(row, tokenizer,max_len=None):
    target = row["completion"][0]["content"]

    prompt_text = tokenizer.apply_chat_template(row["prompt"], tokenize=False,
                                                add_generation_prompt=True, enable_thinking=False)

    full_text = tokenizer.apply_chat_template(row["prompt"] + row["completion"], tokenize=False,
                                              enable_thinking=False)
    end = full_text[full_text.rindex(target) + len(target):]

    prompt = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    answer = tokenizer(target + end, add_special_tokens=False)["input_ids"]

    full = prompt + answer
    labels = [-100] * len(prompt) + answer

    if max_len and len(full) > max_len:
        raise ValueError(f"Row longer than max_len={max_len}: {len(full)} tokens")

    return {"input_ids": full, "labels": labels, "attention_mask": [1] * len(full)}


def get_collator(cfg, tokenizer, model=None):
    if cfg["task"] in ("causal_lm", "multimodal", "seq2seq"):
        return DataCollatorForSeq2Seq(
            tokenizer,
            model=model if cfg["task"] == "seq2seq" else None,   # builds decoder inputs for NLLB
            padding=True, label_pad_token_id=-100, pad_to_multiple_of=8)
    if cfg["task"] == "classification":
        return DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
    raise ValueError(f"No collator for task {cfg['task']}")

_RUN_KEYS = ["learning_rate", "num_train_epochs", "max_steps", "per_device_train_batch_size",
             "gradient_accumulation_steps", "seed", "lr_scheduler_type", "warmup_ratio", "warmup_steps"]


def run_fingerprint(cfg, train_ds, training_config):
    """What identifies a run: model/adapter config, training data (size + first/last rows) and
    the main hyperparameters. Stored as <output_dir>/run_info.json."""
    n = len(train_ds)
    idx = sorted(set(range(min(20, n))) | set(range(max(0, n - 20), n)))
    sample = [train_ds[i].get("input_ids", str(train_ds[i])) for i in idx]
    info = {"model_id": cfg.get("model_id"), "lora_config": cfg.get("lora_config"), "n_train": n,
            "train_sample_sha1": hashlib.sha1(json.dumps(sample, default=str).encode()).hexdigest(),
            "args": {k: training_config.get(k) for k in _RUN_KEYS if k in training_config}}
    return json.loads(json.dumps(info, default=str, sort_keys=True))   # same form as when re-read


def should_resume(output_dir, info, resume="auto"):
    """
    resume="auto": resume only if output_dir has checkpoints from the SAME run (run_info.json
    matches); checkpoints from another or an unknown run raise instead of being silently
    continued (e.g. m1_sent pointed at m1_mix's folder). resume=True/False forces the choice.
    """
    ckpts = glob.glob(os.path.join(output_dir, "checkpoint-*"))
    if resume != "auto":
        return bool(resume) and bool(ckpts)
    if not ckpts:
        return False
    path = os.path.join(output_dir, "run_info.json")
    if not os.path.exists(path):
        raise ValueError(f"{output_dir} has checkpoints but no run_info.json (unknown run). "
                         "Use a new output_dir, or pass resume=True to continue it anyway.")
    with open(path) as f:
        saved = json.load(f)
    if saved.get("fingerprint") != info:
        raise ValueError(f"{output_dir} has checkpoints from a different run "
                         f"(run_info.json differs). Use a new output_dir for this run.")
    return True


def train(cfg, model, tokenizer, train_ds, eval_ds, training_config, smoke_test=False, smoke_steps=50,
          resume="auto"):
    tc = copy.deepcopy(training_config)

    if smoke_test:
        tc["output_dir"] = tc["output_dir"].rstrip("/") + "_smoke"
        tc["max_steps"] = smoke_steps
        tc["save_strategy"] = "no"
        tc["eval_strategy"] = "no"
        tc["logging_steps"] = 5

    info = run_fingerprint(cfg, train_ds, training_config)
    do_resume = False if smoke_test else should_resume(tc["output_dir"], info, resume)
    if not smoke_test:
        os.makedirs(tc["output_dir"], exist_ok=True)
        with open(os.path.join(tc["output_dir"], "run_info.json"), "w") as f:
            json.dump({"fingerprint": info, "training_config": tc,
                       "cfg": {k: v for k, v in cfg.items() if k != "load_kwargs"}},
                      f, indent=2, default=str)

    args = TrainingArguments(**tc)
    trainer = Trainer(model=model, args=args, train_dataset=train_ds,
                      eval_dataset=None if smoke_test else eval_ds,
                      data_collator=get_collator(cfg, tokenizer, model))

    trainer.train(resume_from_checkpoint=do_resume)

    if smoke_test:
        steps_done = trainer.state.global_step
        runtime = next(x["train_runtime"] for x in reversed(trainer.state.log_history) if "train_runtime" in x)
        eff_batch = tc["per_device_train_batch_size"] * tc.get("gradient_accumulation_steps", 1)
        total_steps = math.ceil(len(train_ds) / eff_batch) * tc.get("num_train_epochs", 1)
        sec = runtime / steps_done
        print(f"~{sec:.1f}s/step -> full run ~{sec * total_steps / 3600:.1f} h ({total_steps} steps)")
    else:
        final = os.path.join(tc["output_dir"], "final")
        trainer.save_model(final)
        tokenizer.save_pretrained(final)

    return trainer



    


