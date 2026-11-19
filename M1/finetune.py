from transformers import  AutoProcessor,AutoModelForCausalLM,AutoModelForSeq2SeqLM,AutoModelForSequenceClassification, AutoModelForMultimodalLM


Head= {
    "causal_lm":      AutoModelForCausalLM,
    "seq2seq":        AutoModelForSeq2SeqLM,
    "classification": AutoModelForSequenceClassification,
    "multimodal":     AutoModelForMultimodalLM,
}

def load_model(cfg):
    processor=AutoProcessor.from_pretrained(cfg['model_id'])
    model = Head[cfg['task']].from_pretrained(cfg['model_id'])

    return processor, model



