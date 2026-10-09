#!/bin/zsh
# Runs after the chunk campaign: concurrency and stability, then the three-arm performance sessions (model 1).
HERE=${0:A:h}; P=<HOME>/Projects/omlx-q8a8-measured/prep; cd $HERE
R=final_validation/results
while [ ! -f $R/scn/chunks.done ]; do sleep 20; done
# --- concurrency and stability
./run_conc2.sh $P c1_native c1_native.json --concurrency 1 > $R/conc_c1_native.log 2>&1
./run_conc2.sh $P c2_native c2_native.json --concurrency 2 > $R/conc_c2_native.log 2>&1
./run_conc2.sh $P c2_transposed c2_transposed.json --concurrency 2 --env OMLX_OQ_A8_Q8_NATIVE_META=0 > $R/conc_c2_transposed.log 2>&1
./run_conc2.sh $P c2cache_native c2cache_native.json --concurrency 2 --cache --extra > $R/conc_c2cache_native.log 2>&1
./run_conc2.sh $P c2cache_transposed c2cache_transposed.json --concurrency 2 --cache --extra --env OMLX_OQ_A8_Q8_NATIVE_META=0 > $R/conc_c2cache_transposed.log 2>&1
./run_conc2.sh $P c4_native c4_native.json --concurrency 4 --extra > $R/conc_c4_native.log 2>&1
./run_conc2.sh $P c2_native_mtp c2_native_mtp.json --concurrency 2 --mtp on > $R/conc_c2_native_mtp.log 2>&1
touch $R/conc.done
# --- performance: arms off / transposed / native; round 1 in this order, round 2 reversed
S="--lengths 256 512 1024 4096 8192 --repeats 3"
LNG="--lengths 16384 32768 --repeats 2"
MT="--lengths 1024 4096 8192 --repeats 2"
for set in short long mtp; do
  for rd in 1 2; do
    if [ $rd = 1 ]; then ORDER=(off transposed native); else ORDER=(native transposed off); fi
    for arm in $ORDER; do
      case $arm in off) A8=off; ENV=();; transposed) A8=on; ENV=(--env OMLX_OQ_A8_Q8_NATIVE_META=0);; native) A8=on; ENV=();; esac
      case $set in short) MTP=off; ARGS=($=S); PFX=m1;; long) MTP=off; ARGS=($=LNG); PFX=m1L;; mtp) MTP=on; ARGS=($=MT); PFX=m1M;; esac
      PREFIX=$PFX ./run_session2.sh $P $arm $rd $MTP $A8 $ARGS $ENV > $R/e2e_${PFX}_${arm}_r${rd}.log 2>&1
    done
  done
done
touch $R/perf.done
