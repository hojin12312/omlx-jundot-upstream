Q=$Q8A8_ROOT
PY=$WORKSPACE/venv323/bin/python
MODEL_ROOT=$Q/models
MODEL_ID=Qwen3.8-27B-oQ8e-mtp
CKPT=$MODEL_ROOT/$MODEL_ID
RUNS=$Q/runs
H=$Q/harness
unset PYTHONPATH
for k in ${(k)parameters[(I)OMLX_*]}; do unset $k; done
