# Comparisons with prior work

These two scripts measure ProSan+ in the settings that two earlier papers used. The paper can
then compare ProSan+ with the numbers those papers report. The other methods are **not** re-run
here; their numbers come from the papers as published.

| Script | Reproduces | Compared with (as reported) |
|---|---|---|
| `prosan_protocol.py` | ProSan (Shen et al., IEEE TIFS 2026): sensitive attributes inserted into MedQA, SAMSum and CodeAlpaca prompts; PHR per attribute | HaS, SanText+, CusText(+), a local Llama-7B sanitizer, ProSan |
| `piidocs_leakage.py` | Rap-LI (Liu et al., IEEE TIFS 2026): PII extraction leakage rate (Def. 6) on PIIDocs | SanText+, CusText+, RanText, SnD, HaS, Kan's, Rap-LI |

## Data

All of it is public on the Hugging Face Hub and is not stored in this repository.

```bash
python -c "from huggingface_hub import hf_hub_download as d; [print(d(r, f, repo_type='dataset', local_dir='ext')) for r, f in [('GBaker/MedQA-USMLE-4-options','phrases_no_exclude_test.jsonl'), ('knkarthick/samsum','test.csv'), ('sahil2801/CodeAlpaca-20k','code_alpaca_20k.json'), ('gretelai/gretel-pii-masking-en-v1','data/test-00000-of-00001.parquet')]]"
pip install pyarrow   # for the PIIDocs parquet file
```

## Running

```bash
python comparisons/prosan_protocol.py --data-dir ext --out prosan_protocol.json --dump outputs.jsonl --n 500
python comparisons/piidocs_leakage.py --parquet ext/data/test-00000-of-00001.parquet --out piidocs.json --n 500
```

## Differences from the original setups

**ProSan protocol:**
- **Templates:** ProSan did not publish its insertion templates, so `prosan_protocol.py` writes its
  own (three per task, chosen and placed at random). The prompts are therefore a re-creation, not
  the originals.
- **Sample size:** we use 500 prompts per task; ProSan used 1,000.
- **Hidden:** PHR is reported two ways.
  - *Lenient:* the item no longer appears verbatim.
  - *Strict:* no content token survives, and no half of a credential-like token survives.

**PIIDocs:**
- **Original PII:** the original PII set comes from the dataset's gold entities, not from an NER
  model, so the sanitizer's own detector never grades itself.
- **Sample:** we sample the test split; Rap-LI used 4,119 documents of the train split.

**Both:** the numbers for other methods were measured on the authors' own prompts. A few points of
difference can come from the re-created prompts alone.
