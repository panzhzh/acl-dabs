# Journal data: official downloads and layout

The datasets are not redistributed in this repository.  After obtaining them
from their upstream releases, arrange them as follows. Run the commands below
from the root of the `acl-dabs` repository in Bash.

## Download from upstream releases

Create a temporary download directory and the local data roots:

```bash
DOWNLOAD_ROOT="${TMPDIR:-/tmp}/dabs-journal-data"
mkdir -p "$DOWNLOAD_ROOT" journal/data/aste journal/data/asqp
```

### English ASTE: ASTE-Data-V2

The four English datasets come from the
[SemEval-Triplet-data](https://github.com/xuuuluuu/SemEval-Triplet-data)
release. This work uses the refined `ASTE-Data-V2-EMNLP2020` version.

```bash
git clone --depth 1 \
  https://github.com/xuuuluuu/SemEval-Triplet-data.git \
  "$DOWNLOAD_ROOT/SemEval-Triplet-data"

cp -R \
  "$DOWNLOAD_ROOT/SemEval-Triplet-data/ASTE-Data-V2-EMNLP2020/"{14lap,14res,15res,16res} \
  journal/data/aste/
```

### Polish ASTE: Hotels

The Polish Hotels data come from the
[Polish-ASTE-Datasets](https://github.com/NaIwo/Polish-ASTE-Datasets)
release. The journal experiment uses only Hotels fold 1. Its upstream
`valid.txt` is copied byte-for-byte as the local development split.

```bash
git clone --depth 1 \
  https://github.com/NaIwo/Polish-ASTE-Datasets.git \
  "$DOWNLOAD_ROOT/Polish-ASTE-Datasets"

mkdir -p journal/data/aste/pl_hotels
cp "$DOWNLOAD_ROOT/Polish-ASTE-Datasets/hotels_dataset/folds_list/fold_1/train.txt" \
  journal/data/aste/pl_hotels/train.txt
cp "$DOWNLOAD_ROOT/Polish-ASTE-Datasets/hotels_dataset/folds_list/fold_1/valid.txt" \
  journal/data/aste/pl_hotels/dev.txt
cp "$DOWNLOAD_ROOT/Polish-ASTE-Datasets/hotels_dataset/folds_list/fold_1/test.txt" \
  journal/data/aste/pl_hotels/test.txt
```

### Catalan and Basque ASTE

The exact Catalan (`ca`) and Basque (`eu`) ASTE conversions used here are
released with
[ASTE-Transformer](https://github.com/NaIwo/ASTE-Transformer/tree/main/dataset/data/multib).
Only its `train.txt`, `dev.txt`, and `test.txt` files are needed.

```bash
git clone --depth 1 \
  https://github.com/NaIwo/ASTE-Transformer.git \
  "$DOWNLOAD_ROOT/ASTE-Transformer"

for language in ca eu; do
  mkdir -p "journal/data/aste/$language"
  cp "$DOWNLOAD_ROOT/ASTE-Transformer/dataset/data/multib/$language/"{train,dev,test}.txt \
    "journal/data/aste/$language/"
done
```

### ASQP: Rest15 and Rest16

The ASQP datasets come from the official
[ABSA-QUAD](https://github.com/IsakZhang/ABSA-QUAD) release.

```bash
git clone --depth 1 \
  https://github.com/IsakZhang/ABSA-QUAD.git \
  "$DOWNLOAD_ROOT/ABSA-QUAD"

for dataset in rest15 rest16; do
  mkdir -p "journal/data/asqp/$dataset"
  cp "$DOWNLOAD_ROOT/ABSA-QUAD/data/$dataset/"{train,dev,test}.txt \
    "journal/data/asqp/$dataset/"
done
```

The upstream repositories remain the authoritative source for dataset licences
and terms of use. The commands above copy only the splits used by this code.

## ASTE

```text
journal/data/aste/
├── 14lap/
├── 14res/
├── 15res/
├── 16res/
├── pl_hotels/
├── ca/
└── eu/
```

Each directory must contain either `train_triplets.txt`, `dev_triplets.txt`,
and `test_triplets.txt`, or the shorter accepted names `train.txt`, `dev.txt`,
and `test.txt`. Each non-empty line has the ASTE-Data-V2 form:

```text
tokenized sentence####[([aspect token indices], [opinion token indices], 'POS')]
```

Sentiment labels may be `NEG`, `NEU`, or `POS`.  The same format is used for
the multilingual datasets.

## ASQP

```text
journal/data/asqp/
├── rest15/
│   ├── train.txt
│   ├── dev.txt
│   └── test.txt
└── rest16/
    ├── train.txt
    ├── dev.txt
    └── test.txt
```

Each line follows the ABSA-QUAD surface format:

```text
sentence####[['aspect', 'category', 'positive', 'opinion']]
```

`NULL` is supported for an implicit aspect.  The reader performs deterministic
surface-to-span alignment and reports every unrepresentable annotation.
