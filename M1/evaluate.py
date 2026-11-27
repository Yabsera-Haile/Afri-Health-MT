import hashlib, json, os, re, random, shutil, unicodedata
from collections import Counter
from functools import lru_cache

import sacrebleu    # the only non-stdlib import at module level: the COMET worker env imports this file

LANG_NAMES = {"en": "English", "am": "Amharic", "ha": "Hausa",
              "sw": "Swahili", "yo": "Yoruba", "zu": "Zulu"}
NLLB_CODES = {"en": "eng_Latn", "am": "amh_Ethi", "ha": "hau_Latn",
              "sw": "swh_Latn", "yo": "yor_Latn", "zu": "zul_Latn"}
GLOTLID_OK = {"en": {"eng_Latn"}, "am": {"amh_Ethi"}, "ha": {"hau_Latn"},
              "sw": {"swh_Latn", "swa_Latn"}, "yo": {"yor_Latn"}, "zu": {"zul_Latn"}}

# max_new_tokens per (level, target language): Amharic needs ~3x more tokens per word.
# yo sent 320 -> 512: some correct Yorùbá sentences (tone marks) were cut off at 320.
# Outputs generated under a lower limit are found with stale_limit_ids() and regenerated.
MAX_NEW = {"sent": {"am": 512, "yo": 512, "default": 320},
           "chunk": {"am": 2048, "default": 1536}}

# SSA-COMET checkpoints (Hugging Face, McGill-NLP). Both names checked on the Hub (Nov 2026):
#   ssa-comet-qe  : QE (source + translation)
#   ssa-comet-mtl : unified model, reference-based when a reference is given (EMNLP 2025 paper)
# The "-final" variants use a newer encoder (afro-xlmr-large-114L); keep the pair above for
# all systems so scores stay comparable.
SSA_COMET_QE = "McGill-NLP/ssa-comet-qe"
SSA_COMET_REF = "McGill-NLP/ssa-comet-mtl"

# AFRIDOC-MT, pinned to the revision used to build data/ (data/manifest.json).
AFRIDOC_DATASET = "masakhane/AfriDocMT"
AFRIDOC_REVISION = "8ae06a3f73aa9194a0d29b32e6c1037b3f9420fb"
UNIT_CONFIG = {"sent": "health", "chunk": "doc_health_5"}
DOC_CONFIG = "doc_health"


# ----------------------------------------------------------------------------- utils
def norm(s):
    """NFC + collapse whitespace (applied to hypotheses and references alike)."""
    return " ".join(unicodedata.normalize("NFC", s or "").split())


def _level_key(level):
    return "sent" if level == "sent" else "chunk"


def max_new_tokens_for(level, lang):
    table = MAX_NEW[_level_key(level)]
    return table.get(lang, table["default"])


def load_eval_splits(level, split="test", revision=AFRIDOC_REVISION):
    """(units_ds, docs_ds) at the pinned revision: the split the outputs came from
    ('health' for sent, 'doc_health_5' for chunks) and the 61 'doc_health' documents."""
    from datasets import load_dataset
    units = load_dataset(AFRIDOC_DATASET, UNIT_CONFIG[_level_key(level)], split=split, revision=revision)
    docs = load_dataset(AFRIDOC_DATASET, DOC_CONFIG, split=split, revision=revision)
    return units, docs


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


def _target(r):
    """Target language of a record: 'trg' if present, else 'lang' (older data files)."""
    t = r.get("trg") or r.get("lang")
    if t is None:
        raise KeyError(f"record {r.get('id')} has neither 'trg' nor 'lang'")
    return t


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
        groups.setdefault((_target(r), r["level"]), []).append(r)
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
                new = ids[1:]                          # drop the decoder start token (</s>)
                n = int((new != tokenizer.pad_token_id).sum())
                ended = bool((new[1:] == tokenizer.eos_token_id).any())   # [1:] skips the lang code
                out.append({"id": f"{level}-{i}-en-{lang}", "row_id": i, "trg": lang,
                            "level": level, "hypothesis": t.strip(), "n_gen_tokens": n,
                            "hit_limit": not ended})
            save_outputs(out, out_path, append=True)
            print(f"{lang}: {min(j + batch_size, len(rows))}/{len(rows)}", end="\r")
        print()
    return load_outputs(out_path)


def stale_limit_ids(outputs):
    """ids of outputs that hit a max_new_tokens limit lower than the current MAX_NEW
    (i.e. would get more room if regenerated now). Outputs at the current limit are loops."""
    return {o["id"] for o in outputs
            if o.get("hit_limit") and o.get("n_gen_tokens", 0) < max_new_tokens_for(o["level"], _target(o))}


def drop_outputs(out_path, ids, backup_suffix=".before_regen"):
    """
    Remove rows with these ids from an output file so generate_llm / generate_nllb (which skip
    ids already present) regenerate exactly them. The first call keeps a copy of the original
    file at out_path + backup_suffix (never overwritten). Returns the removed rows.
    """
    ids = set(ids)
    rows = load_outputs(out_path)
    backup = out_path + backup_suffix
    if ids and not os.path.exists(backup):
        shutil.copyfile(out_path, backup)
    removed = [o for o in rows if o["id"] in ids]
    save_outputs([o for o in rows if o["id"] not in ids], out_path)
    print(f"{out_path}: removed {len(removed)} rows (backup: {backup})")
    return removed


# ----------------------------------------------------------------------------- realignment
def map_rows_to_docs(units_en, docs_en, strict=True):
    """
    units_en: English side of the test units in order (sentences or chunks).
    docs_en : English side of the full test documents in order.
    Returns doc index per unit. Any mismatch raises ValueError (strict=False: only warns),
    because a wrong mapping would silently corrupt d-chrF / d-BLEU.
    """
    doc_of, i, problems = [], 0, []
    for d, doc in enumerate(docs_en):
        target, acc = norm(doc), ""
        while i < len(units_en) and len(acc) < len(target):
            acc = (acc + " " + norm(units_en[i])).strip()
            doc_of.append(d)
            i += 1
        if acc != target:
            problems.append(f"document {d} does not match its units (check order/separators)")
    if i != len(units_en):
        problems.append(f"{len(units_en) - i} units left over")
    if problems:
        msg = "[realign] " + "; ".join(problems[:5])
        if strict:
            raise ValueError(msg)
        print(msg)
    else:
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
# Numbers are compared as digit strings after normalisation (_prep_numbers): Ethiopic and
# other-script digits -> ASCII, "165 000" -> "165000", year ranges "2024-25" -> "2024-2025".
# Number words are not recognised (digits only), so reference rates are reported as baselines.
_NUM = re.compile(r"\d+(?:[.,]\d+)*")
_SPACED_THOUSANDS = re.compile(r"(?<![\d.,])\d{1,3}(?:[ \u00a0\u202f\u2009]\d{3})+(?![\d])")
_YEAR_RANGE = re.compile(r"(?<![\d.,])((?:19|20)\d\d)(\s?[-–]\s?)(\d\d)(?!\d|[.,]\d|\s?[-–]\s?\d)")

# Unit symbols directly after a number (optional space). Single-letter units are case-sensitive
# (g, l, L) so that "5G" or "penicillin G" are not read as grams; "T&CM" / "e.g." no longer count
# because a number must precede the unit. Value = canonical unit.
_UNIT_FORMS = {"%": "%", "per cent": "%", "percent": "%",
               "mg": "mg", "milligram": "mg", "milligrams": "mg",
               "mcg": "mcg", "µg": "mcg", "μg": "mcg", "microgram": "mcg", "micrograms": "mcg",
               "kg": "kg", "kilogram": "kg", "kilograms": "kg", "kilo": "kg", "kilos": "kg",
               "gram": "g", "grams": "g", "gramme": "g", "grammes": "g",
               "ml": "ml", "millilitre": "ml", "millilitres": "ml", "milliliter": "ml", "milliliters": "ml",
               "litre": "l", "litres": "l", "liter": "l", "liters": "l",
               "iu": "iu", "°c": "°c", "° c": "°c", "ºc": "°c", "degrees celsius": "°c",
               "mmhg": "mmhg", "mmol": "mmol", "mol": "mol",
               "cm": "cm", "mm": "mm", "km": "km", "kilometre": "km", "kilometres": "km",
               "kilometer": "km", "kilometers": "km", "metre": "m", "metres": "m", "meter": "m",
               "meters": "m", "tonne": "t", "tonnes": "t", "ton": "t", "tons": "t"}
_CASED_UNITS = {"g": "g", "l": "l", "L": "l"}     # no bare "m"/"t": "US$ 100 m" means million
_UNIT_BODY = (r"((?i:" + "|".join(re.escape(u) for u in sorted(_UNIT_FORMS, key=len, reverse=True))
              + r")|" + "|".join(_CASED_UNITS) + r")(?![^\W\d_])")
_UNIT_AFTER = re.compile(r"[ \u00a0\u202f\u2009]?" + _UNIT_BODY)      # right after a number
_UNIT_ANY = re.compile(r"(?<![^\W\d_])" + _UNIT_BODY)                 # anywhere (not inside a word)
UNITS = sorted(set(_UNIT_FORMS.values()) | set(_CASED_UNITS.values()))
NUMBER_CLASSES = ("quantity", "year", "identifier")

# How each target language writes units as words, mined from the AFRIDOC-MT health *train*
# references with mine_unit_words (window 40 chars; counts = train pairs with the word nearby).
# Regexes on lower-case, diacritic-free text (_fold) near the number; Zulu stems match inside
# inflected words (amaphesenti, ngamaphesenti, kumaphesenti). Only words seen >= 3 times are
# listed; other units (mg, kg, ml, °C ...) have <= 2 train examples and rely on the symbol.
UNIT_WORDS = {
    "am": {"%": [r"በመቶ", r"ከመቶ"],                   # 55 + forms በመቶው/በመቶውን; ከመቶ seen in train
           "g": [r"ግራም"],                          # 8/8
           "mm": [r"ሚሊ ?ሜትር"]},                     # 3/4
    "ha": {"%": [r"\bkaso\b", r"\bkashi\b"],         # 137, 121 ("kashi 40 cikin 100/ɗari")
           "g": [r"\bg[i]?ram\b"],                   # gram 4, giram 3
           "m": [r"\bmita\b"]},                      # 3/3
    "sw": {"%": [r"\basilimia\b"],                   # 6 (Swahili refs keep "%" 321/327)
           "g": [r"\bgramu\b"],                      # 7/8
           "m": [r"\bmita\b"]},                      # 3/3
    "yo": {"%": [r"\bida\b", r"ogo+run"],            # ìdá 84, (nínú) ọgọ́ọ̀rún 3
           "g": [r"\bgiramu\b"],                     # 4/6
           "m": [r"\bmita\b"]},                      # 3/3
    "zu": {"%": [r"phesenti"],                       # amaphesenti 9, ngamaphesenti 5, kumaphesenti 2 ...
           "g": [r"gremu"]},                         # namagremu 3/8
}


_ETH_ONES = {chr(0x1369 + i): i + 1 for i in range(9)}      # ፩..፱ = 1..9
_ETH_TENS = {chr(0x1372 + i): (i + 1) * 10 for i in range(9)}  # ፲..፺ = 10..90
_ETH_NUM = re.compile("[\u1369-\u137C]+")


def ethiopic_to_int(s):
    """Ethiopic numerals are additive (no zero): ፲፬=14, ፪፻=200, ፲፪፻፴፬=1234, ፼=10000."""
    total = block = cur = 0
    for ch in s:
        if ch in _ETH_ONES:
            cur += _ETH_ONES[ch]
        elif ch in _ETH_TENS:
            cur += _ETH_TENS[ch]
        elif ch == "\u137B":                     # ፻ hundred
            block += (cur or 1) * 100; cur = 0
        elif ch == "\u137C":                     # ፼ ten thousand
            total += ((block + cur) or 1) * 10000; block = cur = 0
    return total + block + cur


def _ascii_digits(text):
    """Ethiopic numerals -> Arabic digits; any other Unicode decimal digit -> ASCII."""
    text = _ETH_NUM.sub(lambda m: str(ethiopic_to_int(m.group())), text or "")
    return "".join(str(unicodedata.digit(c)) if c.isdigit() and not c.isascii() else c for c in text)


def _expand_year_range(m):
    y, sep, yy = m.groups()
    return y + sep + (y[:2] + yy if int(yy) > int(y[2:]) else yy)


def _prep_numbers(text):
    """Digits to ASCII, '2024-25' -> '2024-2025', '165 000' -> '165000'."""
    text = _YEAR_RANGE.sub(_expand_year_range, _ascii_digits(text))
    return _SPACED_THOUSANDS.sub(lambda m: re.sub(r"[ \u00a0\u202f\u2009]", "", m.group()), text)


def _canon_number(s):
    """'1,000' / '1.000' -> '1000' (thousands separators); '4,7' -> '4.7'."""
    parts = re.split(r"[.,]", s)
    if len(parts) > 1 and all(len(p) == 3 for p in parts[1:]):
        return "".join(parts)
    return s.replace(",", ".")


@lru_cache(maxsize=200_000)
def number_mentions(text):
    """(normalised text, ((value, start, end), ...)) - positions refer to the normalised text."""
    t = _prep_numbers(text or "")
    return t, tuple((_canon_number(m.group()), m.start(), m.end()) for m in _NUM.finditer(t))


def numbers_in(text):
    """Digits only (number words are not checked). '1,000' / '1.000' / '1 000' / '1000' -> '1000'.
    Ethiopic numerals (e.g. ፲፬) and other scripts' digits are converted first."""
    return Counter(v for v, _, _ in number_mentions(text)[1])


def ethiopic_numeral_count(text):
    """How many Ethiopic numerals a text contains (diagnostic)."""
    return len(_ETH_NUM.findall(text or ""))


@lru_cache(maxsize=50_000)
def source_numbers(src):
    """
    Numbers in an English source with a class and an optional unit:
      identifier - part of a name/code: letters attached (COVID-19, R21, SDG3, PM2.5, H5N1, 5G)
      year       - 1900-2099 without a unit (also '1990s')
      quantity   - everything else: counts, percentages, doses, money, ordinals (4th)
    Returns a tuple of dicts {value, cls, unit}; unit is a canonical symbol ('%', 'mg', ...) or None.
    """
    t, ments = number_mentions(src)
    out = []
    for v, s, e in ments:
        m = _UNIT_AFTER.match(t, e)
        unit = _canon_unit(m.group(1)) if m else None
        left = t[s - 1] if s > 0 else ""
        left2 = t[s - 2] if s > 1 else ""
        right = re.match(r"[^\W\d_]+", t[e:])
        right = right.group() if right else ""
        if left.isalpha() or (left in ("-", "‑") and left2.isalpha()) or \
                (right and unit is None and right.lower() not in ("st", "nd", "rd", "th", "s")):
            cls = "identifier"
        elif unit is None and re.fullmatch(r"(19|20)\d\d", v):
            cls = "year"
        else:
            cls = "quantity"
        out.append({"value": v, "cls": cls, "unit": unit})
    return tuple(out)


def _canon_unit(form):
    return _CASED_UNITS.get(form) or _UNIT_FORMS.get(re.sub(r"\s+", " ", form.lower()))


def _covered(need, have):
    return all(have[k] >= c for k, c in need.items())


def numbers_kept(src, hyp, cls=None):
    """None if the source has no numbers (of class cls); else True if every one appears in hyp."""
    need = Counter(n["value"] for n in source_numbers(src) if cls is None or n["cls"] == cls)
    if not need:
        return None
    return _covered(need, numbers_in(hyp))


def numbers_matched(src, hyp):
    """(source numbers found in hyp, source numbers) - for number-level (micro) recall."""
    need, have = Counter(n["value"] for n in source_numbers(src)), numbers_in(hyp)
    return sum(min(c, have[k]) for k, c in need.items()), sum(need.values())


def numbers_added(src, hyp):
    """
    True if hyp contains a number that is not in the source (possible hallucinated value).
    A '100' is allowed once per source percentage ('20%' -> 'kashi 20 cikin 100').
    """
    nums = source_numbers(src)
    extra = numbers_in(hyp) - Counter(n["value"] for n in nums)
    extra["100"] -= sum(n["unit"] == "%" for n in nums)
    return any(c > 0 for c in extra.values())


def _fold(text):
    """Lower-case, no diacritics (Yorùbá tone marks etc.) - for unit-word matching."""
    d = unicodedata.normalize("NFD", (text or "").lower())
    return unicodedata.normalize("NFC", "".join(c for c in d if unicodedata.category(c) != "Mn"))


def _unit_near(text, start, end, unit, lang, window, words):
    w = text[max(0, start - window): end + window]
    if any(_canon_unit(m.group(1)) == unit for m in _UNIT_ANY.finditer(w)):
        return True
    return words and any(re.search(p, _fold(w)) for p in UNIT_WORDS.get(lang, {}).get(unit, ()))


def unit_pairs_kept(src, hyp, lang=None, window=40, words=True):
    """
    None if the source has no number+unit pair; else True if, for every pair (e.g. '30 mg',
    '20%'), the number appears in hyp with the unit within `window` characters - written as a
    symbol, or (words=True) as a unit word of `lang` from UNIT_WORDS.
    """
    pairs = [(n["value"], n["unit"]) for n in source_numbers(src) if n["unit"]]
    if not pairs:
        return None
    t, ments = number_mentions(hyp)
    return all(any(v == hv and _unit_near(t, s, e, u, lang, window, words) for hv, s, e in ments)
               for v, u in pairs)


def mine_unit_words(train_ds, langs=("am", "ha", "sw", "yo", "zu"), window=40, min_count=3,
                    min_share=0.15, min_lift=4.0, top=12):
    """
    Candidate unit words per (language, unit) from a TRAIN split (en + target columns): words
    found near the number in the reference when the English source has 'number + unit', compared
    with words near plain numbers.
    Returns {lang: {unit: {"n": pairs found, "candidates": [(word, count, share, lift), ...]}}}.
    Candidates are reviewed by hand and copied into UNIT_WORDS (never mine the test split).
    """
    words = lambda s: set(re.findall(r"[^\W\d_]+", _fold(s)))
    res = {}
    for lang in langs:
        near, bg, n_unit, n_bg = {}, Counter(), Counter(), 0
        for src, ref in zip(train_ds["en"], train_ds[lang]):
            t, ments = number_mentions(ref)
            for n in source_numbers(src):
                hits = [(s, e) for v, s, e in ments if v == n["value"]]
                if not hits or n["cls"] == "identifier":
                    continue
                s, e = hits[0]
                ws = words(t[max(0, s - window): e + window])
                if n["unit"]:
                    near.setdefault(n["unit"], Counter()).update(ws)
                    n_unit[n["unit"]] += 1
                else:
                    bg.update(ws)
                    n_bg += 1
        res[lang] = {}
        for unit, cnt in near.items():
            cands = []
            for w, c in cnt.items():
                share = c / n_unit[unit]
                lift = share / ((bg[w] + 1) / max(n_bg, 1))
                if c >= min_count and share >= min_share and lift >= min_lift:
                    cands.append((w, c, round(share, 2), round(lift, 1)))
            res[lang][unit] = {"n": n_unit[unit],
                               "candidates": sorted(cands, key=lambda x: (-x[1], -x[3]))[:top]}
    return res


# ----------------------------------------------------------------------------- coverage
def max_ngram_repeat(text, n=4):
    w = (text or "").split()
    if len(w) < n:
        return 0
    return max(Counter(tuple(w[i:i + n]) for i in range(len(w) - n + 1)).values())


_LOOP = re.compile(r"(.{1,30}?)\1{9,}", re.S)      # a 1-30 char unit repeated 10+ times in a row


def is_degenerate(text, ngram_threshold=8):
    """Decoding loop: a short string repeated 10+ times in a row ('ግግግ...', 'ṣíṣe-ṣíṣe-...')
    or one 4-word sequence 8+ times. Stricter than 'repetition', which also fires on references."""
    t = norm(text)
    return bool(_LOOP.search(t)) or max_ngram_repeat(t) >= ngram_threshold


def coverage_flags(hyp, ref, hit_limit=None, ratio_range=(0.5, 2.0), repeat_threshold=3):
    h, r = norm(hyp), norm(ref)
    ratio = len(h) / max(len(r), 1)
    return {"empty": len(h) == 0,
            "hit_limit": bool(hit_limit),
            "length_ratio": round(ratio, 3),
            "bad_length": not (ratio_range[0] <= ratio <= ratio_range[1]),
            "repetition": max_ngram_repeat(h) >= repeat_threshold,
            "degenerate": is_degenerate(h)}


# ----------------------------------------------------------------------------- language ID
# Pinned: the repo's default "model.bin" switched from v3 to v4 on 2026-11-05.
GLOTLID_FILE = "model_v4.bin"
GLOTLID_REVISION = "e9592e0a9bc3b6f2a946de94faaa188d1dd6f522"


def load_glotlid(filename=GLOTLID_FILE, revision=GLOTLID_REVISION):
    """GlotLID fastText model from Hugging Face (cis-lmu/glotlid), pinned version."""
    import fasttext
    from huggingface_hub import hf_hub_download
    return fasttext.load_model(hf_hub_download("cis-lmu/glotlid", filename, revision=revision))


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
# unbabel-comet needs Transformers 4 and clashes with the training stack (Transformers 5,
# Colab's TensorFlow/protobuf). So COMET runs in a private Python env that this module
# creates and calls by itself; the notebook only calls the functions below.
COMET_ENV = "/content/comet-env"


def _comet_python(env_dir=COMET_ENV):
    return os.path.join(env_dir, "bin", "python")


def setup_comet_env(env_dir=COMET_ENV, python_version="3.11"):
    """Create the private COMET environment once per runtime (a few minutes)."""
    import subprocess, sys
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "uv"], check=True)
    if not os.path.exists(_comet_python(env_dir)):
        subprocess.run([sys.executable, "-m", "uv", "venv", env_dir, "--python", python_version], check=True)
    # setuptools<81 provides pkg_resources, which COMET's torchmetrics still imports;
    # uv-created envs don't include setuptools. Re-running is quick and repairs an existing env.
    subprocess.run([sys.executable, "-m", "uv", "pip", "install", "--python", _comet_python(env_dir),
                    "unbabel-comet", "sacrebleu", "setuptools<81"], check=True)
    print("COMET env ready:", env_dir)
    return env_dir


def _comet_predict_inprocess(data, model_name, batch_size, gpus):
    from comet import download_model, load_from_checkpoint
    model = load_from_checkpoint(download_model(model_name))
    out = model.predict(data, batch_size=batch_size, gpus=gpus, progress_bar=False)
    return {"system": float(out.system_score), "segments": [float(x) for x in out.scores]}


def comet_scores(srcs, hyps, refs=None, model_name=SSA_COMET_QE, batch_size=32, gpus=1,
                 env_dir=COMET_ENV):
    """
    SSA-COMET for one list of segments. refs=None -> QE model (source + translation only).
    Runs inside the private env (setup_comet_env) if it exists, otherwise in this process.
    COMET reads at most ~512 tokens, so use it on sentence-level outputs.
    """
    import subprocess, tempfile
    data = [{"src": norm(s), "mt": norm(h)} | ({"ref": norm(r)} if refs is not None else {})
            for s, h, r in zip(srcs, hyps, refs if refs is not None else [None] * len(hyps))]
    if not os.path.exists(_comet_python(env_dir)):
        return _comet_predict_inprocess(data, model_name, batch_size, gpus)

    with tempfile.TemporaryDirectory() as tmp:
        fin, fout = os.path.join(tmp, "in.json"), os.path.join(tmp, "out.json")
        with open(fin, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
        env = os.environ | {"USE_TF": "0", "TRANSFORMERS_NO_TF": "1"}
        proc = subprocess.run([_comet_python(env_dir), os.path.abspath(__file__), "--comet-worker",
                               fin, fout, model_name, str(batch_size), str(gpus)],
                              env=env, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError("COMET worker failed:\n" + proc.stderr[-3000:])
        with open(fout, encoding="utf-8") as f:
            return json.load(f)


def _fingerprint(obj):
    return hashlib.sha1(json.dumps(obj, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


def comet_for_system(outputs, units_ds, langs=("am", "ha", "sw", "yo", "zu"),
                     models=None, cache_path=None, **kw):
    """
    SSA-COMET (+QE) for one system's sentence-level outputs, per language.
    Returns {lang: {metric: {"system", "segments"}}} -> pass as comet_by_lang to evaluate_system.
    Segments follow row_id order (same as evaluate_language).
    Cache (cache_path): one entry per (lang, metric), keyed by the model name and the exact
    (id, hypothesis) list, so regenerated outputs or a new metric are recomputed and nothing
    else is. Cache files in the old format (no keys) are treated as stale.
    """
    models = models or {"ssa_comet_qe": SSA_COMET_QE}
    cache = {}
    if cache_path and os.path.exists(cache_path):
        with open(cache_path, encoding="utf-8") as f:
            cache = json.load(f)
        if cache.get("_format") != 2:
            print(f"{cache_path}: old cache format, recomputing")
            cache = {}
    cache["_format"] = 2
    res = {}
    for lang in langs:
        o = sorted([x for x in outputs if _target(x) == lang], key=lambda x: x["row_id"])
        if not o:
            continue
        srcs = [units_ds[x["row_id"]]["en"] for x in o]
        refs = [units_ds[x["row_id"]][lang] for x in o]
        hyps = [x["hypothesis"] for x in o]
        res[lang] = {}
        for k, m in models.items():
            entry = f"{lang}|{k}"
            key = _fingerprint({"model": m, "outputs": [(x["id"], x["hypothesis"]) for x in o]})
            if cache.get(entry, {}).get("key") != key:
                cache[entry] = {"key": key, "scores": comet_scores(
                    srcs, hyps, None if k.endswith("_qe") else refs, model_name=m, **kw)}
                if cache_path:                                     # save after every entry
                    with open(cache_path, "w", encoding="utf-8") as f:
                        json.dump(cache, f)
            res[lang][k] = cache[entry]["scores"]
        print(lang, {k: round(v["system"], 4) for k, v in res[lang].items()})
    return res


# ----------------------------------------------------------------------------- evaluation
def evaluate_language(outputs, lang, units_ds, docs_ds, doc_of=None, glotlid=None,
                      comet=None, n_bootstrap=1000, details_path=None):
    """
    outputs : list of output records for ONE level (all languages allowed; filtered by lang)
    units_ds: raw HF test split the outputs came from ("health" or "doc_health_5"), has en + lang
    docs_ds : raw HF "doc_health" test split (61 documents) -> references for d-chrF/d-BLEU
    doc_of  : precomputed map_rows_to_docs result (computed if None)
    glotlid : loaded GlotLID model or None (skip)
    comet   : optional dict {"ssa_comet": {...}, "ssa_comet_qe": {...}} from comet_scores,
              segments in the same order as this language's outputs sorted by row_id
    """
    outs = sorted([o for o in outputs if _target(o) == lang], key=lambda o: o["row_id"])
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

    # numbers and units: every metric also on the references (*_ref_%) as a baseline, since
    # references paraphrase ("1 in 3" -> "one in three") and contain their own errors
    rate = lambda xs: round(100 * sum(x for x in xs if x is not None) /
                            max(sum(x is not None for x in xs), 1), 2)
    pct = lambda xs: round(100 * sum(map(bool, xs)) / max(len(xs), 1), 2)
    seg = {}
    for side, texts in (("hyp", hyps), ("ref", refs)):
        seg[side] = {"num": [numbers_kept(s, t) for s, t in zip(srcs, texts)],
                     "added": [numbers_added(s, t) for s, t in zip(srcs, texts)],
                     "units": [unit_pairs_kept(s, t, lang) for s, t in zip(srcs, texts)],
                     "units_sym": [unit_pairs_kept(s, t, lang, words=False) for s, t in zip(srcs, texts)]}
        for c in NUMBER_CLASSES:
            seg[side][c] = [numbers_kept(s, t, c) for s, t in zip(srcs, texts)]
        matched = [numbers_matched(s, t) for s, t in zip(srcs, texts)]
        seg[side]["recall"] = round(100 * sum(m for m, _ in matched) / max(sum(n for _, n in matched), 1), 2)
    for side, sfx in (("hyp", ""), ("ref", "_ref")):
        S = seg[side]
        res[f"numbers_kept{sfx}_%"] = rate(S["num"])                 # overall, segment level
        for c in NUMBER_CLASSES:
            res[f"numbers_kept_{c}{sfx}_%"] = rate(S[c])
        res[f"numbers_recall{sfx}_%"] = S["recall"]                  # number level (micro)
        res[f"numbers_added{sfx}_%"] = pct(S["added"])               # % of all segments
        res[f"unit_pairs_kept{sfx}_%"] = rate(S["units"])            # symbol or unit word
        res[f"unit_pairs_symbol_kept{sfx}_%"] = rate(S["units_sym"])  # symbol only
    res["segments_with_numbers"] = sum(x is not None for x in seg["hyp"]["num"])
    for c in NUMBER_CLASSES:
        res[f"segments_with_{c}"] = sum(x is not None for x in seg["hyp"][c])
    res["segments_with_unit_pairs"] = sum(x is not None for x in seg["hyp"]["units"])
    res["eth_numerals_%"] = round(100 * sum(ethiopic_numeral_count(h) > 0 for h in hyps) / len(hyps), 2)
    res["eth_numerals_ref_%"] = round(100 * sum(ethiopic_numeral_count(r) > 0 for r in refs) / len(refs), 2)

    cov = [coverage_flags(h, r, o.get("hit_limit")) for h, r, o in zip(hyps, refs, outs)]
    for k in ["empty", "hit_limit", "bad_length", "repetition", "degenerate"]:
        res[f"{k}_%"] = round(100 * sum(c[k] for c in cov) / len(cov), 2)
    res["repetition_ref_%"] = round(100 * sum(max_ngram_repeat(norm(r)) >= 3 for r in refs) / len(refs), 2)
    res["degenerate_ref_%"] = pct([is_degenerate(r) for r in refs])

    lid = [glotlid_predict(glotlid, h) for h in hyps] if glotlid is not None else None
    if lid is not None:
        lid_ref = [glotlid_predict(glotlid, r) for r in refs]
        for sfx, preds in (("", lid), ("_ref", lid_ref)):
            wrong = Counter(l for l, _ in preds if l is not None and l not in GLOTLID_OK[lang])
            res[f"wrong_lang{sfx}_%"] = round(100 * sum(wrong.values()) / len(preds), 2)
            res[f"wrong_lang{sfx}_labels"] = ", ".join(f"{l}:{c}" for l, c in wrong.most_common(4))

    if comet:
        for k, v in comet.items():
            res[k] = v["system"]

    if details_path:      # per-segment file for error analysis
        rows = []
        H = seg["hyp"]
        for i, (o, s, r, h, c) in enumerate(zip(outs, srcs, refs, hyps, cov)):
            row = {"id": o["id"], "row_id": o["row_id"], "lang": lang, "source": s,
                   "reference": r, "hypothesis": h, "chrfpp": round(seg_chrf[i], 2),
                   "numbers_kept": H["num"][i], "numbers_kept_ref": seg["ref"]["num"][i],
                   "numbers_added": H["added"][i], "unit_pairs_kept": H["units"][i]} | \
                  {f"numbers_kept_{k}": H[k][i] for k in NUMBER_CLASSES} | c
            if lid is not None:
                row["lid"], row["lid_prob"] = lid[i][0], round(lid[i][1], 3)
                row["lid_ref"] = lid_ref[i][0]
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
    # macro average over languages; counts (n, segments_with_*) and CIs are not averaged
    num_cols = [c for c in df.columns if df[c].dtype.kind in "fi" and not c.endswith("ci95")
                and c != "n" and not c.startswith("segments_with")]
    df.loc["avg", num_cols] = df[num_cols].mean().round(2)
    return df


# Columns for the paper tables (all are in the full evaluate_system output).
QUALITY_COLUMNS = ["d_chrf", "d_chrf_ci95", "d_bleu", "chrfpp", "ssa_comet", "ssa_comet_qe"]
SAFETY_COLUMNS = ["numbers_kept_%", "numbers_kept_ref_%",
                  "numbers_kept_quantity_%", "numbers_kept_quantity_ref_%",
                  "numbers_kept_year_%", "numbers_kept_identifier_%",
                  "numbers_added_%", "numbers_added_ref_%",
                  "unit_pairs_kept_%", "unit_pairs_kept_ref_%",
                  "degenerate_%", "hit_limit_%", "wrong_lang_%", "wrong_lang_ref_%"]


def select_columns(df, cols):
    return df[[c for c in cols if c in df.columns]]


# ----------------------------------------------------------------------------- significance
def paired_bootstrap(outputs_a, outputs_b, lang, units_ds, docs_ds, doc_of=None,
                     n=1000, seed=42):
    """
    Document-level paired bootstrap on d-chrF: resample the 61 documents with replacement.
    Returns mean difference (A - B) and the share of samples where A <= B (one-sided p).
    """
    if doc_of is None:
        doc_of = map_rows_to_docs(units_ds["en"], docs_ds["en"])
    ha = realign({o["row_id"]: o["hypothesis"] for o in outputs_a if _target(o) == lang}, doc_of, len(docs_ds))
    hb = realign({o["row_id"]: o["hypothesis"] for o in outputs_b if _target(o) == lang}, doc_of, len(docs_ds))
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


# ----------------------------------------------------------------------------- COMET worker
if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--comet-worker":
        _, _, fin, fout, model_name, bs, gpus = sys.argv
        with open(fin, encoding="utf-8") as f:
            data = json.load(f)
        result = _comet_predict_inprocess(data, model_name, int(bs), int(gpus))
        with open(fout, "w", encoding="utf-8") as f:
            json.dump(result, f)