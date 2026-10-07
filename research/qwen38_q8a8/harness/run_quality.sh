#!/bin/zsh
source $Q8A8_ROOT/harness/common.sh
TAG=${1:-quality1}
export PYTHONPATH=$Q/src
[[ -n "$ACT" ]] && export OMLX_OQ_A8_Q8_ACT_MODE=$ACT
$PY $H/guard_run.py --min-available-gib 60 --log $RUNS/$TAG.log -- $PY -u $H/quality_q8.py --tree $Q/src --ckpt $CKPT --corpus $WORKSPACE/artifacts/prod_a8/quality_corpus_36.json --pick all --chunk 2048 --out $RUNS/$TAG.json
echo "exit $?"
