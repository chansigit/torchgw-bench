#!/usr/bin/env bash
# Download fastText aligned wiki vectors (en, es, fi) and MUSE bilingual
# dictionaries for the cross-lingual word-embedding alignment benchmark.
# fastText vectors: ~1 GB each — skipped if already present.
# MUSE dicts: ~few MB total — downloaded unconditionally if absent.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
DATA_DIR="$REPO_ROOT/data/core_05_word_embedding"

mkdir -p "$DATA_DIR/vectors" "$DATA_DIR/dicts"

# fastText aligned vectors (first 200k lines is plenty for N<=10k)
for lang in en es fi; do
    f="$DATA_DIR/vectors/wiki.$lang.vec"
    if [[ -s "$f" ]]; then
        echo "[c5-fetch] cached $f"
        continue
    fi
    echo "[c5-fetch] downloading wiki.$lang.vec (~1 GB)"
    curl -sSL -o "$f" \
        "https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.$lang.vec"
done

# MUSE bilingual dictionaries (train: 0-5000, test: 5000-6500)
for pair in en-es en-fi; do
    for split in "0-5000" "5000-6500"; do
        f="$DATA_DIR/dicts/${pair}.${split}.txt"
        if [[ -s "$f" ]]; then
            echo "[c5-fetch] cached $f"
            continue
        fi
        echo "[c5-fetch] downloading ${pair}.${split}.txt"
        curl -sSL -o "$f" \
            "https://dl.fbaipublicfiles.com/arrival/dictionaries/${pair}.${split}.txt"
    done
done

echo "[c5-fetch] done."
