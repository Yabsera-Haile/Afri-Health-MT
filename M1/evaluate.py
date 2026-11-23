import json, os, re, random, unicodedata
from collections import Counter

import sacrebleu

LANG_NAMES = {"en": "English", "am": "Amharic", "ha": "Hausa",
              "sw": "Swahili", "yo": "Yoruba", "zu": "Zulu"}
NLLB_CODES = {"en": "eng_Latn", "am": "amh_Ethi", "ha": "hau_Latn",
              "sw": "swh_Latn", "yo": "yor_Latn", "zu": "zul_Latn"}
GLOTLID_OK = {"en": {"eng_Latn"}, "am": {"amh_Ethi"}, "ha": {"hau_Latn"},
              "sw": {"swh_Latn", "swa_Latn"}, "yo": {"yor_Latn"}, "zu": {"zul_Latn"}}

# max_new_tokens per (level, target language): Amharic needs ~3x more tokens per word
MAX_NEW = {"sent": {"am": 512, "default": 320},
           "chunk": {"am": 2048, "default": 1536}}

# SSA-COMET checkpoints (Hugging Face, collection "McGill-NLP/ssa-comet").
# QE name is confirmed; check the exact reference-based name in the collection.
SSA_COMET_QE = "McGill-NLP/ssa-comet-qe"
SSA_COMET_REF = "McGill-NLP/ssa-comet-mtl"


# ----------------------------------------------------------------------------- utils
def norm(s):
    """NFC + collapse whitespace (applied to hypotheses and references alike)."""
    return " ".join(unicodedata.normalize("NFC", s or "").split())


def _level_key(level):
    return "sent" if level == "sent" else "chunk"


def max_new_tokens_for(level, lang):
    table = MAX_NEW[_level_key(level)]
    return table.get(lang, table["default"])


# ----------------------------------------------------------------------------- output files
def save_outputs(outputs, path, append=False):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a" if append else "w", encoding="utf-8") as f:
        for o in outputs:
            f.write(json.dumps(o, ensure_ascii=False) + "\n")


def load_outputs(path):
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_records(path):
    """Test records written by the data-prep code (prompt/completion + row_id, trg, level)."""
    return load_outputs(path)


# ----------------------------------------------------------------------------- generation
_THINK = re.compile(r"<think>.*?</think>", re.S)


def _clean_generation(text):
    return _THINK.sub("", text).replace("<think>", "").replace("</think>", "").strip()


def generate_llm(model, tokenizer, records, out_path, batch_size=16, max_new_tokens=None):
    """
    Translate test records with a chat LLM (TranslatePsy). Uses the record's prompt exactly
    as in training: apply_chat_template(..., add_generation_prompt=True, enable_thinking=False).
    Greedy decoding. Appends to out_path after every batch and skips ids already done,
    so it resumes after a disconnect.
    """
    import torch

    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    stop_ids = list({tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|im_end|>")} - {None})

    done = {o["id"] for o in load_outputs(out_path)}
    todo = [r for r in records if r["id"] not in done]
    prompts = {r["id"]: tokenizer.apply_chat_template(r["prompt"], tokenize=False,
                                                      add_generation_prompt=True,
                                                      enable_thinking=False) for r in todo}
    # group by target language and level (same max_new_tokens), longest first
    groups = {}
    for r in todo:
        groups.setdefault((r["trg"], r["level"]), []).append(r)
    print(f"{len(done)} already done, {len(todo)} to translate")

    model.eval()
    for (lang, level), recs in groups.items():
        mnt = max_new_tokens or max_new_tokens_for(level, lang)
        recs.sort(key=lambda r: len(prompts[r["id"]]), reverse=True)
        for i in range(0, len(recs), batch_size):
            batch = recs[i:i + batch_size]
            enc = tokenizer([prompts[r["id"]] for r in batch], return_tensors="pt",
                            padding=True, add_special_tokens=False).to(model.device)
            with torch.no_grad():
                gen = model.generate(**enc, max_new_tokens=mnt, do_sample=False,
                                     eos_token_id=stop_ids, pad_token_id=tokenizer.pad_token_id)
            new = gen[:, enc["input_ids"].shape[1]:]
            out = []
            for r, ids in zip(batch, new):
                ids = ids.tolist()
                n = len(ids)
                while n and ids[n - 1] == tokenizer.pad_token_id:   # trailing pads after stop
                    n -= 1
                ended = any(t in stop_ids for t in ids[:n])
                text = tokenizer.decode(ids[:n], skip_special_tokens=True)
                out.append({"id": r["id"], "row_id": r["row_id"], "trg": lang, "level": level,
                            "hypothesis": _clean_generation(text),
                            "n_gen_tokens": n, "hit_limit": (not ended) and n >= mnt})
            save_outputs(out, out_path, append=True)
            print(f"{lang} {level}: {min(i + batch_size, len(recs))}/{len(recs)}", end="\r")
        print()
    return load_outputs(out_path)


def generate_nllb(model, tokenizer, source_ds, langs, out_path, level="sent",
                  batch_size=32, max_new_tokens=None):
    """NLLB baseline (sentence level). source_ds = raw HF test split with 'en' column."""
    import torch

    tokenizer.src_lang = NLLB_CODES["en"]
    done = {o["id"] for o in load_outputs(out_path)}
    model.eval()
    for lang in langs:
        mnt = max_new_tokens or max_new_tokens_for(level, lang)
        rows = [(i, source_ds[i]["en"]) for i in range(len(source_ds))
                if f"{level}-{i}-en-{lang}" not in done]
        rows.sort(key=lambda x: len(x[1]), reverse=True)
        bos = tokenizer.convert_tokens_to_ids(NLLB_CODES[lang])
        for j in range(0, len(rows), batch_size):
            batch = rows[j:j + batch_size]
            enc = tokenizer([norm(t) for _, t in batch], return_tensors="pt", padding=True,
                            truncation=True, max_length=512).to(model.device)
            with torch.no_grad():
                gen = model.generate(**enc, forced_bos_token_id=bos, max_new_tokens=mnt,
                                     num_beams=1, do_sample=False)
            texts = tokenizer.batch_decode(gen, skip_special_tokens=True)
            out = []
            for (i, _), ids, t in zip(batch, gen, texts):
                n = int((ids != tokenizer.pad_token_id).sum())
                out.append({"id": f"{level}-{i}-en-{lang}", "row_id": i, "trg": lang,
                            "level": level, "hypothesis": t.strip(), "n_gen_tokens": n,
                            "hit_limit": n >= mnt})
            save_outputs(out, out_path, append=True)
            print(f"{lang}: {min(j + batch_size, len(rows))}/{len(rows)}", end="\r")
        print()
    return load_outputs(out_path)


# ----------------------------------------------------------------------------- realignment
def map_rows_to_docs(units_en, docs_en):
    """
    units_en: English side of the test units in order (sentences or chunks).
    docs_en : English side of the full test documents in order.
    Returns doc index per unit. Prints a warning for any document that doesn't match.
    """
    doc_of, i, bad = [], 0, 0
    for d, doc in enumerate(docs_en):
        target, acc = norm(doc), ""
        while i < len(units_en) and len(acc) < len(target):
            acc = (acc + " " + norm(units_en[i])).strip()
            doc_of.append(d)
            i += 1
        if acc != target:
            bad += 1
            print(f"[realign] document {d} does not match its units (check order/separators)")
    if i != len(units_en):
        print(f"[realign] {len(units_en) - i} units left over")
    if bad == 0 and i == len(units_en):
        print(f"[realign] OK: {len(units_en)} units -> {len(docs_en)} documents")
    return doc_of


def realign(hyp_by_row, doc_of, n_docs):
    per_doc = [[] for _ in range(n_docs)]
    for r in sorted(hyp_by_row):
        per_doc[doc_of[r]].append(norm(hyp_by_row[r]))
    return [" ".join(x) for x in per_doc]


# ----------------------------------------------------------------------------- string metrics
def _score(metric, hyps, refs, n_bootstrap=0):
    res = metric.corpus_score(hyps, [refs], n_bootstrap=n_bootstrap) if n_bootstrap \
        else metric.corpus_score(hyps, [refs])
    ci = getattr(res, "_ci", None)
    return round(res.score, 2), (round(ci, 2) if ci is not None else None)


def doc_scores(hyp_by_row, doc_of, docs_ref, n_bootstrap=1000):
    """d-chrF and d-BLEU on realigned documents (SacreBLEU defaults, like AFRIDOC-MT)."""
    hyps = realign(hyp_by_row, doc_of, len(docs_ref))
    refs = [norm(r) for r in docs_ref]
    d_chrf, d_chrf_ci = _score(sacrebleu.metrics.CHRF(), hyps, refs, n_bootstrap)
    d_bleu, d_bleu_ci = _score(sacrebleu.metrics.BLEU(), hyps, refs, n_bootstrap)
    return {"d_chrf": d_chrf, "d_chrf_ci95": d_chrf_ci, "d_bleu": d_bleu, "d_bleu_ci95": d_bleu_ci}


def segment_chrfpp(hyps, refs):
    """Corpus chrF++ over segments + per-segment chrF++ (for error analysis)."""
    m = sacrebleu.metrics.CHRF(word_order=2)
    corpus = m.corpus_score([norm(h) for h in hyps], [[norm(r) for r in refs]]).score
    per_seg = [m.sentence_score(norm(h), [norm(r)]).score for h, r in zip(hyps, refs)]
    return round(corpus, 2), per_seg


# ----------------------------------------------------------------------------- numbers / units
_NUM = re.compile(r"\d+(?:[.,]\d+)*")
UNITS = ["mg", "mcg", "µg", "g", "kg", "ml", "mL", "l", "L", "iu", "IU", "%", "°c", "°C",
         "mmhg", "mmHg", "cm", "mm", "km", "mmol", "mol"]
_UNIT = re.compile(r"(?<![A-Za-z])(" + "|".join(sorted({re.escape(u.lower()) for u in UNITS},
                                                        key=len, reverse=True)) + r")(?![A-Za-z])")


def numbers_in(text):
    """Digits only (number words are not checked). '1,000' / '1.000' / '1000' -> '1000'."""
    out = []
    for m in _NUM.findall(text or ""):
        parts = re.split(r"[.,]", m)
        if len(parts) > 1 and all(len(p) == 3 for p in parts[1:]):   # thousands separators
            out.append("".join(parts))
        else:
            out.append(m.replace(",", "."))
    return Counter(out)


def units_in(text):
    return Counter(_UNIT.findall((text or "").lower()))


def numbers_kept(src, hyp):
    """None if the source has no numbers; else True if every source number appears in hyp."""
    s = numbers_in(src)
    if not s:
        return None
    h = numbers_in(hyp)
    return all(h[k] >= v for k, v in s.items())


def units_kept(src, hyp):
    """Approximate: unit symbols may legitimately be spelled out in the target language."""
    s = units_in(src)
    if not s:
        return None
    h = units_in(hyp)
    return all(h[k] >= v for k, v in s.items())


# ----------------------------------------------------------------------------- coverage
def max_ngram_repeat(text, n=4):
    w = (text or "").split()
    if len(w) < n:
        return 0
    return max(Counter(tuple(w[i:i + n]) for i in range(len(w) - n + 1)).values())


def coverage_flags(hyp, ref, hit_limit=None, ratio_range=(0.5, 2.0), repeat_threshold=3):
    h, r = norm(hyp), norm(ref)
    ratio = len(h) / max(len(r), 1)
    return {"empty": len(h) == 0,
            "hit_limit": bool(hit_limit),
            "length_ratio": round(ratio, 3),
            "bad_length": not (ratio_range[0] <= ratio <= ratio_range[1]),
            "repetition": max_ngram_repeat(h) >= repeat_threshold}


# ----------------------------------------------------------------------------- language ID
def load_glotlid():
    """GlotLID fastText model from Hugging Face (cis-lmu/glotlid)."""
    import fasttext
    from huggingface_hub import hf_hub_download
    return fasttext.load_model(hf_hub_download("cis-lmu/glotlid", "model.bin"))


def glotlid_predict(model, text):
    text = norm(text).replace("\n", " ")
    if not text:
        return None, 0.0
    try:
        labels, probs = model.predict(text, k=1)
        return labels[0].replace("__label__", ""), float(probs[0])
    except ValueError:                        # fastText + numpy 2 bug in .predict
        prob, label = model.f.predict(text, 1, 0.0, "strict")[0]
        return label.replace("__label__", ""), float(prob)


# ----------------------------------------------------------------------------- SSA-COMET
def comet_scores(srcs, hyps, refs=None, model_name=SSA_COMET_QE, batch_size=32, gpus=1):
    """
    Run in an environment with `pip install unbabel-comet` (may clash with the training env).
    refs=None -> QE model (source + translation only).
    Note: COMET models read at most ~512 tokens; use on sentence-level outputs.
    """
    from comet import download_model, load_from_checkpoint
    model = load_from_checkpoint(download_model(model_name))
    data = [{"src": norm(s), "mt": norm(h)} | ({"ref": norm(r)} if refs is not None else {})
            for s, h, r in zip(srcs, hyps, refs if refs is not None else [None] * len(hyps))]
    out = model.predict(data, batch_size=batch_size, gpus=gpus)
    return {"system": round(float(out.system_score), 4), "segments": list(out.scores)}


# ----------------------------------------------------------------------------- evaluation
def evaluate_language(outputs, lang, units_ds, docs_ds, doc_of=None, glotlid=None,
                      comet=None, n_bootstrap=1000, details_path=None):
    """
    outputs : list of output records for ONE level (all languages allowed; filtered by lang)
    units_ds: raw HF test split the outputs came from ("health" or "doc_health_10"), has en + lang
    docs_ds : raw HF "doc_health" test split (61 documents) -> references for d-chrF/d-BLEU
    doc_of  : precomputed map_rows_to_docs result (computed if None)
    glotlid : loaded GlotLID model or None (skip)
    comet   : optional dict {"ssa_comet": {...}, "ssa_comet_qe": {...}} from comet_scores,
              segments in the same order as this language's outputs sorted by row_id
    """
    outs = sorted([o for o in outputs if o["trg"] == lang], key=lambda o: o["row_id"])
    if not outs:
        return None
    if doc_of is None:
        doc_of = map_rows_to_docs(units_ds["en"], docs_ds["en"])

    hyp_by_row = {o["row_id"]: o["hypothesis"] for o in outs}
    srcs = [units_ds[o["row_id"]]["en"] for o in outs]
    refs = [units_ds[o["row_id"]][lang] for o in outs]
    hyps = [o["hypothesis"] for o in outs]

    res = {"lang": lang, "n": len(outs), "complete": len(outs) == len(units_ds)}
    res |= doc_scores(hyp_by_row, doc_of, docs_ds[lang], n_bootstrap) if res["complete"] else \
        {"d_chrf": None, "d_chrf_ci95": None, "d_bleu": None, "d_bleu_ci95": None}
    res["chrfpp"], seg_chrf = segment_chrfpp(hyps, refs)

    num = [numbers_kept(s, h) for s, h in zip(srcs, hyps)]
    num_ref = [numbers_kept(s, r) for s, r in zip(srcs, refs)]
    unit = [units_kept(s, h) for s, h in zip(srcs, hyps)]
    rate = lambda xs: round(100 * sum(x for x in xs if x is not None) /
                            max(sum(x is not None for x in xs), 1), 2)
    res["numbers_kept_%"] = rate(num)
    res["numbers_kept_ref_%"] = rate(num_ref)          # how often the references themselves keep them
    res["units_kept_%"] = rate(unit)
    res["segments_with_numbers"] = sum(x is not None for x in num)

    cov = [coverage_flags(h, r, o.get("hit_limit")) for h, r, o in zip(hyps, refs, outs)]
    for k in ["empty", "hit_limit", "bad_length", "repetition"]:
        res[f"{k}_%"] = round(100 * sum(c[k] for c in cov) / len(cov), 2)
    res["repetition_ref_%"] = round(100 * sum(max_ngram_repeat(norm(r)) >= 3 for r in refs) / len(refs), 2)

    lid = [glotlid_predict(glotlid, h) for h in hyps] if glotlid is not None else None
    if lid is not None:
        ok = GLOTLID_OK[lang]
        res["wrong_lang_%"] = round(100 * sum(l is not None and l not in ok for l, _ in lid) / len(lid), 2)

    if comet:
        for k, v in comet.items():
            res[k] = v["system"]

    if details_path:      # per-segment file for error analysis
        rows = []
        for i, (o, s, r, h, c) in enumerate(zip(outs, srcs, refs, hyps, cov)):
            row = {"id": o["id"], "row_id": o["row_id"], "lang": lang, "source": s,
                   "reference": r, "hypothesis": h, "chrfpp": round(seg_chrf[i], 2),
                   "numbers_kept": num[i], "units_kept": unit[i]} | c
            if lid is not None:
                row["lid"], row["lid_prob"] = lid[i][0], round(lid[i][1], 3)
            if comet:
                for k, v in comet.items():
                    row[k] = round(float(v["segments"][i]), 4)
            rows.append(row)
        save_outputs(rows, details_path, append=True)
    return res


def evaluate_system(outputs, units_ds, docs_ds, langs=("am", "ha", "sw", "yo", "zu"),
                    glotlid=None, comet_by_lang=None, n_bootstrap=1000, details_path=None):
    """All languages for one system and one level -> pandas DataFrame (+ 'avg' row)."""
    import pandas as pd

    if details_path and os.path.exists(details_path):
        os.remove(details_path)
    doc_of = map_rows_to_docs(units_ds["en"], docs_ds["en"])
    rows = []
    for lang in langs:
        r = evaluate_language(outputs, lang, units_ds, docs_ds, doc_of, glotlid,
                              (comet_by_lang or {}).get(lang), n_bootstrap, details_path)
        if r:
            rows.append(r)
    df = pd.DataFrame(rows).set_index("lang")
    num_cols = [c for c in df.columns if df[c].dtype.kind in "fi" and not c.endswith("ci95")]
    df.loc["avg", num_cols] = df[num_cols].mean().round(2)
    return df


# ----------------------------------------------------------------------------- significance
def paired_bootstrap(outputs_a, outputs_b, lang, units_ds, docs_ds, doc_of=None,
                     n=1000, seed=42):
    """
    Document-level paired bootstrap on d-chrF: resample the 61 documents with replacement.
    Returns mean difference (A - B) and the share of samples where A <= B (one-sided p).
    """
    if doc_of is None:
        doc_of = map_rows_to_docs(units_ds["en"], docs_ds["en"])
    ha = realign({o["row_id"]: o["hypothesis"] for o in outputs_a if o["trg"] == lang}, doc_of, len(docs_ds))
    hb = realign({o["row_id"]: o["hypothesis"] for o in outputs_b if o["trg"] == lang}, doc_of, len(docs_ds))
    refs = [norm(r) for r in docs_ds[lang]]
    chrf = sacrebleu.metrics.CHRF()
    rng = random.Random(seed)
    k = len(refs)
    diffs = []
    for _ in range(n):
        idx = [rng.randrange(k) for _ in range(k)]
        sa = chrf.corpus_score([ha[i] for i in idx], [[refs[i] for i in idx]]).score
        sb = chrf.corpus_score([hb[i] for i in idx], [[refs[i] for i in idx]]).score
        diffs.append(sa - sb)
    full = chrf.corpus_score(ha, [refs]).score - chrf.corpus_score(hb, [refs]).score
    return {"lang": lang, "delta_d_chrf": round(full, 2),
            "p_A_not_better": round(sum(d <= 0 for d in diffs) / n, 4)}