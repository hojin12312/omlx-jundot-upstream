#!/bin/zsh
# Dense oQ4e model-level check: A8 route counts per request on pristine main vs src.
source $Q8A8_ROOT/harness/common.sh
MR=$WORKSPACE/models_dense; MID=qwen3.8-27b-oQ4e-mtp
for which in main src; do
  if [[ $which == main ]]; then TREE=$Q/main; else TREE=$Q/src; fi
  $PY $H/guard_run.py --min-available-gib 60 --log $RUNS/q4check_$which.log -- $PY -u $H/measure_q8.py --python $PY --tree $TREE --workdir $RUNS --model-dir $MR --checkpoint $MR/$MID --model $MID --corpus $H/corpus.json --tag q4check_$which --out $RUNS/q4check_$which.json --conds on --lengths 4096 --samples 2 --settle 5 --nonce-set q4check
  echo "[$which] exit $?"; sleep 15
done
echo ALL_DONE
