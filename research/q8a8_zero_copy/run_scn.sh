#!/bin/zsh
# usage: [MODEL_ROOT=.. MODEL_ID=..] run_scn.sh TREE MODE ARM A8(on|off) MTP(on|off) OUT [extra scenarios.py args...]
PY=<HOME>/Projects/mltf-qwen4-routed-a8/venv323/bin/python
HERE=${0:A:h}
MODEL_ROOT=${MODEL_ROOT:-<HOME>/Projects/mltf-qwen4-routed-a8/q8a8/models}
MODEL_ID=${MODEL_ID:-Qwen3.8-27B-oQ8e-mtp}
TREE=$1; MODE=$2; ARM=$3; A8=$4; MTP=$5; OUT=$6; shift 6
unset PYTHONPATH
for k in ${(k)parameters[(I)OMLX_*]}; do unset $k; done
mkdir -p $HERE/final_validation/results/scn $HERE/work
$PY -u $HERE/harness/scenarios.py --python $PY --tree $TREE --workdir $HERE/work --model-dir $MODEL_ROOT \
  --checkpoint $MODEL_ROOT/$MODEL_ID --model $MODEL_ID --corpus $HERE/harness/corpus.json \
  --mode $MODE --arm $ARM --a8 $A8 --mtp $MTP --out $HERE/final_validation/results/scn/$OUT "$@"
