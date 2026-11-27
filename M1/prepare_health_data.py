from datasets import load_dataset
import numpy as np
import re, unicodedata
import random
from collections import Counter

ZW = re.compile(r"[\u200b\u200c\u200d\u2060\ufeff]")
CTRL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
LANG = {"en": "English", "am": "Amharic", "ha": "Hausa",
        "sw": "Swahili", "yo": "Yoruba", "zu": "Zulu"}
# AFRIDOC-MT revision used to build data/ (data/manifest.json "dataset_sha");
# same value as evaluate.AFRIDOC_REVISION (kept separate: the COMET worker env lacks `datasets`).
AFRIDOC_REVISION = "8ae06a3f73aa9194a0d29b32e6c1037b3f9420fb"

def clean_row(row,languages):
  c_row={}
  for lang in languages:
    text=row[lang] or ""
    text = unicodedata.normalize("NFC", text)
    text=ZW.sub("",text)
    text=CTRL.sub("",text)
    text = re.sub(r"[ \t\u00a0]+", " ", text)      
    text = re.sub(r"\n{3,}", "\n\n", text)  

    c_row[lang]= text.strip()
  
  return c_row


def filter_by_length(data, tok, max_len):
    kept, dropped = [], Counter()
    for ex in data:
        ids = tok.apply_chat_template(ex["prompt"] + ex["completion"],
                                      tokenize=True, enable_thinking=False)
        ids = ids["input_ids"] if isinstance(ids, dict) else ids
        ex["n_tokens"] = len(ids)
        if ex["n_tokens"] <= max_len:
            kept.append(ex)
        else:   # older records have only 'lang' (= target), source is English
            dropped[(ex.get("src", "en"), ex.get("trg") or ex["lang"], ex["level"])] += 1
    print("dropped:", dict(dropped))
    return kept

def check_pair(text1,text2):
  if not text1 or not text2:
    return False
  if text1==text2:
    return False
  
  return True


def token_len(col,tokenizer,lang,batch=512):
  out=[]

  for i in range(0,len(col),batch):
    batch_col=col[i:i+batch]
    out+=[len(x) for x in tokenizer(batch_col,add_special_tokens=False)["input_ids"]]
  
  return np.array(out)
  

def load_mt_dataset(name, config, split, lang1,lang2,tokenizer,is_test=False,revision=AFRIDOC_REVISION):
  # revision pins the HF dataset version (pass None for the latest)
  dataset=load_dataset(name,config,split=split,revision=revision).select_columns([lang1,lang2])
  dataset = dataset.map(lambda r, i: {"row_id": i}, with_indices=True) 
  dataset = dataset.map(clean_row, fn_kwargs={"languages": [lang1, lang2]})
  if not is_test:
    print("data size before ",len(dataset[lang1]),len(dataset[lang2]))
    dataset = dataset.filter(lambda row: check_pair(row[lang1],row[lang2]))
    total_tokens= token_len(dataset[lang1],tokenizer,lang1) + token_len(dataset[lang2],tokenizer,lang2)
    dataset = dataset.add_column("n_tokens", total_tokens.tolist())
    p=np.percentile(total_tokens,[50,95,99])
    print(f"Token size: p50={p[0]:.0f}  p95={p[1]:.0f}  p99={p[2]:.0f}  max={total_tokens.max()}")
    
    print("data size after ", len(dataset[lang1]),len(dataset[lang2]))
  return dataset

def convert_row_to_prompt_translate_psy(row,src,trg,level):
  source_lang,target_lang=LANG[src],LANG[trg]
  source_text,target_text=row[src],row[trg]
  system=(f"You are a professional {source_lang} to {target_lang} translator. "
            f"Your goal is to accurately convey the meaning and nuances of the "
            f"original {source_lang} text while adhering to {target_lang} grammar, "
            f"vocabulary, and cultural sensitivities. Produce only the "
            f"{target_lang} translation, without any additional explanations "
            "or commentary. "
  )
  user=(f"Please translate the following {source_lang} text into "
            f"{target_lang}: {source_text}.\n\nTranslation:"
        )
  return {
        "id": f"{level}-{row['row_id']}-{src}-{trg}", "lang": trg, "src": src, "trg": trg,
        "level": level, "row_id": row["row_id"],
        "prompt": [{"role": "system", "content": system},
                   {"role": "user", "content": user}],
        "completion": [{"role": "assistant", "content": target_text}],
    }


def build_lang_pair_dataset(model,dataset,src,trg,direc,level):
  p_dataset=[]
  if model=="TranslatePsy":
    if direc=="ONE":
      p_dataset.extend(dataset.map(convert_row_to_prompt_translate_psy,fn_kwargs={"src":src,"trg":trg,"level":level},remove_columns=dataset.column_names))
    elif direc=="BOTH":
      p_dataset.extend(dataset.map(convert_row_to_prompt_translate_psy,fn_kwargs={"src":src,"trg":trg,"level":level},remove_columns=dataset.column_names))
      p_dataset.extend(dataset.map(convert_row_to_prompt_translate_psy,fn_kwargs={"src":trg,"trg":src,"level":level},remove_columns=dataset.column_names))
    else:
      raise ValueError(f"DIrection not supported: {direc}")
  else:
    raise ValueError(f"Model not supported: {model}")
  return p_dataset

def build_dataset(model,datasets,src_langs,trg_langs,direc,level,seed=42):
  data=[]
  for i in range(len(src_langs)):
    src,trg,dataset=src_langs[i],trg_langs[i],datasets[i]
    data+=build_lang_pair_dataset(model,dataset,src,trg,direc,level)

  random.seed(seed)
  random.shuffle(data)

  return data
  


