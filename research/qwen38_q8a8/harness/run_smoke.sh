#!/bin/zsh
source $Q8A8_ROOT/harness/common.sh
$PY $H/guard_run.py --min-available-gib 60 --log $RUNS/smoke_tools.log -- $PY -u $H/smoke_tools_q8.py --python $PY --tree $Q/src --workdir $RUNS --model-dir $MODEL_ROOT --checkpoint $CKPT --model $MODEL_ID --cases 10 --tag tools1 --out $RUNS/smoke_tools.json
echo "exit $?"
