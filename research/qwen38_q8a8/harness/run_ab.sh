#!/bin/zsh
# usage: run_ab.sh TAG "LENGTHS" SAMPLES CONDS [ROT] [extra measure args...]
source $Q8A8_ROOT/harness/common.sh
TAG=$1; LENGTHS=(${=2}); SAMPLES=$3; CONDS=$4; ROT=${5:-0}; shift 5 2>/dev/null
TREE=${TREE:-$Q/src}
$PY $H/guard_run.py --min-available-gib 60 --log $RUNS/$TAG.log -- $PY -u $H/measure_q8.py --python $PY --tree $TREE --workdir $RUNS --model-dir $MODEL_ROOT --checkpoint $CKPT --model $MODEL_ID --corpus $H/corpus.json --tag $TAG --out $RUNS/$TAG.json --conds $CONDS --rot $ROT --lengths $LENGTHS --samples $SAMPLES --settle 1.5 "$@"
echo "exit $?"
