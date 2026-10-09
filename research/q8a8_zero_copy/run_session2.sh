#!/bin/zsh
# usage: [MODEL_ROOT=.. MODEL_ID=.. REF_MAIN=.. PREFIX=m1] run_session2.sh TREE ARM ROUND MTP(off|on) A8(on|off) [extra session.py args...]
PY=${PY:-python3}
HERE=${0:A:h}
MODEL_ROOT=${MODEL_ROOT:?set MODEL_ROOT to the directory that holds the checkpoint}
MODEL_ID=${MODEL_ID:-Qwen3.8-27B-oQ8e-mtp}
REF_MAIN=${REF_MAIN:-models--Jundot--Qwen3.8-27B-oQ8e-mtp}
PREFIX=${PREFIX:-m1}
TREE=$1; ARM=$2; R=$3; M=$4; A8=$5; shift 5
unset PYTHONPATH
for k in ${(k)parameters[(I)OMLX_*]}; do unset $k; done
D=$HERE/final_validation/results/e2e
mkdir -p $D $HERE/work
$PY -u $HERE/harness/session.py --python $PY --tree $TREE --workdir $HERE/work --model-dir $MODEL_ROOT \
  --checkpoint $MODEL_ROOT/$MODEL_ID --model $MODEL_ID --corpus $HERE/harness/corpus.json --ref-main $REF_MAIN \
  --arm ${PREFIX}_$ARM --cond $M --a8 $A8 --round $R --out $D/${PREFIX}_${ARM}_r${R}_mtp${M}.json "$@"
