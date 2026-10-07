#!/bin/zsh
source $Q8A8_ROOT/harness/common.sh
$PY $H/guard_run.py --min-available-gib 60 --log $RUNS/mtp_confirm.log -- $PY -u $H/mtp_confirm.py --python $PY --tree $Q/src --workdir $RUNS --model-dir $MODEL_ROOT --checkpoint $CKPT --model $MODEL_ID --corpus $H/corpus.json --out $RUNS/mtp_confirm${1:-}.json ${2:+--mtp-key} ${2:+$2}
echo "exit $?"
