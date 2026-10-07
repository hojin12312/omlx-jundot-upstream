#!/bin/zsh
# Separate-process control: pristine main (setting off, no hook of ours) vs src (setting on, passive hook), ABBA.
# usage: run_sep.sh TAG "LENGTHS" SAMPLES "SEQUENCE" ; SEQUENCE like "main src src main main src"
source $Q8A8_ROOT/harness/common.sh
TAG=$1; LENGTHS=(${=2}); SAMPLES=$3; SEQ=(${=4})
i=0
for which in $SEQ; do
  i=$((i+1))
  if [[ $which == main ]]; then TREE=$Q/main; MS='{}'; else TREE=$Q/src; MS='{"qwen35_oq_a8_enabled": true, "qwen35_oq_a8_min_tokens": 128}'; fi
  $PY $H/guard_run.py --min-available-gib 60 --log $RUNS/${TAG}_${i}_${which}.log -- $PY -u $H/measure_q8.py --python $PY --tree $TREE --workdir $RUNS --model-dir $MODEL_ROOT --checkpoint $CKPT --model $MODEL_ID --corpus $H/corpus.json --tag ${TAG}_${i}_${which} --out $RUNS/${TAG}_${i}_${which}.json --passive --conds off --lengths $LENGTHS --samples $SAMPLES --settle 10 --nonce-set $TAG --model-settings "$MS"
  echo "[$i $which] exit $?"
  sleep 20
done
echo ALL_DONE
