#!/bin/zsh
# prefix-cache scenarios, model 1 (MTP off for all three arms, MTP on for native and transposed), one server per session
HERE=${0:A:h}; P=<HOME>/Projects/omlx-q8a8-measured/prep
cd $HERE
./run_scn.sh $P cache off      off off m1_cache_off.json --lengths 4096 8192 > final_validation/results/scn/m1_cache_off.log 2>&1
./run_scn.sh $P cache native   on  off m1_cache_native.json --lengths 4096 8192 > final_validation/results/scn/m1_cache_native.log 2>&1
./run_scn.sh $P cache transposed on off m1_cache_transposed.json --lengths 4096 8192 --env OMLX_OQ_A8_Q8_NATIVE_META=0 > final_validation/results/scn/m1_cache_transposed.log 2>&1
./run_scn.sh $P cache native_mtp on on m1_cache_native_mtp.json --lengths 4096 > final_validation/results/scn/m1_cache_native_mtp.log 2>&1
./run_scn.sh $P cache transposed_mtp on on m1_cache_transposed_mtp.json --lengths 4096 --env OMLX_OQ_A8_Q8_NATIVE_META=0 > final_validation/results/scn/m1_cache_transposed_mtp.log 2>&1
touch final_validation/results/scn/cache.done
