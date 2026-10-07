#!/bin/zsh
source $Q8A8_ROOT/harness/common.sh
export PYTHONPATH=$Q/src
i=0
for mode in off on on off; do
  i=$((i+1))
  $PY $H/guard_run.py --min-available-gib 60 --log $RUNS/mem_${i}_$mode.log -- $PY -u $H/memory_q8.py --tree $Q/src --ckpt $CKPT --corpus $H/corpus.json --mode $mode --lengths 16384 32768 --out $RUNS/mem_${i}_$mode.json
  echo "[$i $mode] exit $?"; sleep 15
done
echo ALL_DONE
