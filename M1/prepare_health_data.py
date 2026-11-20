from datasets import load_dataset
import numpy as np
import re, unicodedata

ZW = re.compile(r"[\u200b\u200c\u200d\u2060\ufeff]")
CTRL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")

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
  

def load_mt_dataset(name, config, split, lang1,lang2,tokenizer):
  dataset=load_dataset(name,config,split=split).select_columns([lang1,lang2])
  print("data size before ",len(dataset[lang1]),len(dataset[lang2]))
  dataset = dataset.map(lambda row: clean_row(row,[lang1,lang2]))
  dataset = dataset.filter(lambda row: check_pair(row[lang1],row[lang2]))
  total_tokens= token_len(dataset[lang1],tokenizer,lang1) + token_len(dataset[lang2],tokenizer,lang2)
  p=np.percentile(total_tokens,[50,95,99])
  print(f"Token sieze: p50={p[0]:.0f}  p95={p[1]:.0f}  p99={p[2]:.0f}  max={total_tokens.max()}")
  
  print("data size after ", len(dataset[lang1]),len(dataset[lang2]))
  return dataset
  


