#!/bin/zsh
source $Q8A8_ROOT/harness/common.sh
TREE=$Q/main
$PY $H/guard_run.py --min-available-gib 60 --log $RUNS/passive_main.log -- $PY -u $H/measure_q8.py --python $PY --tree $TREE --workdir $RUNS --model-dir $MODEL_ROOT --checkpoint $CKPT --model $MODEL_ID --corpus $H/corpus.json --tag passive_main --out $RUNS/passive_main.json --passive --conds off --lengths 1024 4096 8192 16384 --samples 2 --model-settings '{}' --settle 2
echo "exit $?"
