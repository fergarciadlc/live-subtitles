---
tags:
- translation
- ctranslate2
- opus-mt
- int8
---

# Live Subtitles MT Models

Converted CTranslate2/int8 Opus-MT translation models used by the
`live-subtitles` Hugging Face Space.

These files are derived artifacts. For license, citation, training data, and model
details, see the original upstream model cards.

## Source Models

| Folder | Source model |
| --- | --- |
| `opus-mt-fr-en` | https://huggingface.co/Helsinki-NLP/opus-mt-fr-en |
| `opus-mt-en-fr` | https://huggingface.co/Helsinki-NLP/opus-mt-en-fr |
| `opus-mt-en-es` | https://huggingface.co/Helsinki-NLP/opus-mt-en-es |
| `opus-mt-fr-es` | https://huggingface.co/Helsinki-NLP/opus-mt-fr-es |

## Conversion

The folders were converted with CTranslate2:

```bash
ct2-transformers-converter \
  --model Helsinki-NLP/opus-mt-<src>-<tgt> \
  --output_dir models/opus-mt-<src>-<tgt> \
  --quantization int8 \
  --copy_files source.spm target.spm vocab.json tokenizer_config.json
```

The local helper command is:

```bash
pixi run -e convert convert-mt
```

## Consumer

The Space reads this repo through `LIVE_SUBTITLES_MT_REPO` and downloads
`models/opus-mt-<src>-<tgt>/` folders on startup.
