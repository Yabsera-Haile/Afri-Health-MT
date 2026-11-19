from transformers import  AutoProcessor,AutoModelForCausalLM,AutoModelForSeq2SeqLM,AutoModelForSequenceClassification, AutoModelForMultimodalLM


Head= {
    "causal_lm":      AutoModelForCausalLM,
    "seq2seq":        AutoModelForSeq2SeqLM,
    "classification": AutoModelForSequenceClassification,
    "multimodal":     AutoModelForMultimodalLM,
}

def load_model_and_processor(cfg):
    processor=AutoProcessor.from_pretrained(cfg['model_id'])
    model = Head[cfg['task']].from_pretrained(
        cfg['model_id'],
        dtype=cfg.get('dtype','auto'),
        device_map=cfg.get('device_map'),
        **cfg.get('load_kwargs',{})
        )

    return processor, model



