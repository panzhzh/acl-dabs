# DABS for Structured Affective Extraction

This directory contains the journal extension of DABS for end-to-end structured
sentiment extraction.  It is deliberately isolated from the ACL implementation:

- the ACL entry points remain under `scripts/` and import only `src/`;
- the journal entry points live under `journal/` and import only
  `journal.dabs_structured`;
- journal dependencies, configurations, outputs, and data roots are separate;
- only the complete DORA--QCBS model is exposed here.  Research ablations,
  diagnostic sweeps, and reproduced baseline systems are not part of this
  release path.

The shared model performs one encoder pass, constructs a reusable DORA depth
substrate, and applies QCBS readouts to latent aspect spans, opinion spans, and
directed aspect--opinion queries.  The same implementation supports ASTE and
ASQP; ASQP adds the task-required category head and implicit-aspect handling,
not a second DABS architecture.

## Layout

```text
journal/
├── configs/
│   ├── aste_en.json             # English ASTE protocol
│   ├── aste_multilingual.json   # multilingual ASTE protocol
│   └── asqp.json                # Rest15/Rest16 ASQP protocol
├── dabs_structured/
│   ├── model.py                 # complete DORA--QCBS model
│   ├── checkpoint.py            # portable checkpoint I/O
│   ├── aste/                    # ASTE data, batching, and exact decoding
│   └── asqp/                    # ASQP data, batching, and exact decoding
├── data/README.md               # expected local data layout
├── train_aste.py
├── evaluate_aste.py
├── train_asqp.py
├── evaluate_asqp.py
└── requirements.txt
```

## Environment

Use a separate environment so the journal stack cannot alter the ACL runtime:

```bash
python -m venv journal/.venv
source journal/.venv/bin/activate
pip install -r journal/requirements.txt
```

The code is tested with Python 3.12, PyTorch 2.11, and Transformers 5.8.  A
CUDA GPU with BF16 support is recommended.  ASTE includes a CPU smoke mode;
the released ASQP training protocol is GPU-only.

## Data

The datasets are downloaded from their upstream releases rather than
redistributed in this repository:

| Task | Datasets | Official/upstream source |
|---|---|---|
| English ASTE | `14lap`, `14res`, `15res`, `16res` | [ASTE-Data-V2-EMNLP2020](https://github.com/xuuuluuu/SemEval-Triplet-data/tree/master/ASTE-Data-V2-EMNLP2020) |
| Polish ASTE | `pl_hotels`, fold 1 | [Polish-ASTE-Datasets](https://github.com/NaIwo/Polish-ASTE-Datasets) |
| Catalan/Basque ASTE | `ca`, `eu` | [ASTE-Transformer multilingual conversions](https://github.com/NaIwo/ASTE-Transformer/tree/main/dataset/data/multib) |
| ASQP | `rest15`, `rest16` | [ABSA-QUAD](https://github.com/IsakZhang/ABSA-QUAD/tree/master/data) |

Use the ready-to-run download and copy commands in
[`data/README.md`](data/README.md). They create
`journal/data/aste/<dataset>/` and `journal/data/asqp/<dataset>/` with the exact
filenames expected by the loaders. The multilingual ASTE datasets use the same
token-indexed format as English ASTE, so no task-specific model code is
required. Follow the licences and terms of each upstream release.

## English ASTE

Train one of the five paper seeds:

```bash
python -m journal.train_aste \
  --config journal/configs/aste_en.json \
  --dataset 16res \
  --seed 42
```

The five fixed seeds are `42`, `567`, `12`, `2345`, and `5678`.  By default,
the selected checkpoint and run summary are written to
`journal/outputs/aste/<dataset>/seed_<seed>/`.

Evaluate the development-selected checkpoint:

```bash
python -m journal.evaluate_aste \
  --root journal/data/aste \
  --dataset 16res \
  --checkpoint journal/outputs/aste/16res/seed_42/best.pt
```

## Multilingual ASTE

The multilingual protocol uses `microsoft/mdeberta-v3-base`.  Each dataset is
still passed through the same Full DORA--QCBS trainer:

```bash
python -m journal.train_aste \
  --config journal/configs/aste_multilingual.json \
  --root journal/data/aste \
  --dataset ca \
  --seed 42
```

Dataset directory names are not hard-coded; for the paper they are
`pl_hotels`, `ca`, and `eu`.

## ASQP

Train and evaluate Rest15 or Rest16:

```bash
python -m journal.train_asqp \
  --config journal/configs/asqp.json \
  --dataset rest16 \
  --seed 42

python -m journal.evaluate_asqp \
  --dataset rest16 \
  --checkpoint journal/outputs/asqp/rest16/seed_42/best.pt
```

Checkpoint and threshold selection use only the development split.  The ASQP
trainer records the test-access boundary and evaluates the test set once after
selection is frozen.  An optional fixed hard-negative artifact can be supplied
with `--hard-negative-file`; it changes candidate sampling only and does not
change the DABS architecture.

## Smoke check

After placing ASTE data, run a one-epoch tiny-model check without a GPU:

```bash
python -m journal.train_aste \
  --smoke \
  --root journal/data/aste \
  --dataset 14lap
```

This verifies parsing, token alignment, Full-model forward/backward execution,
development decoding, and checkpoint serialization.  It is not a paper run.
