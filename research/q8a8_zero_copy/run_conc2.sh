#!/bin/zsh
# usage: run_conc2.sh TREE ARM OUT [concurrency.py args...]   (model 1)
PY=<HOME>/Projects/mltf-qwen4-routed-a8/venv323/bin/python
HERE=${0:A:h}
MODEL_ROOT=<HOME>/Projects/mltf-qwen4-routed-a8/q8a8/models; MODEL_ID=Qwen3.8-27B-oQ8e-mtp
TREE=$1; ARM=$2; OUT=$3; shift 3
unset PYTHONPATH
for k in ${(k)parameters[(I)OMLX_*]}; do unset $k; done
mkdir -p $HERE/final_validation/results/conc $HERE/work
$PY -u $HERE/harness/concurrency.py --python $PY --tree $TREE --workdir $HERE/work --model-dir $MODEL_ROOT \
  --checkpoint $MODEL_ROOT/$MODEL_ID --model $MODEL_ID --corpus $HERE/harness/corpus.json --arm $ARM \
  --out $HERE/final_validation/results/conc/$OUT "$@"
