#!/bin/zsh
HERE=${0:A:h}; P=<HOME>/Projects/omlx-q8a8-measured/prep; cd $HERE
C="--lengths 4096 8192 --chunks 512 1024 2048 4096 8192"
./run_scn.sh $P chunks native on off m1_chunks_native.json $=C > final_validation/results/scn/m1_chunks_native.log 2>&1
./run_scn.sh $P chunks transposed on off m1_chunks_transposed.json $=C --env OMLX_OQ_A8_Q8_NATIVE_META=0 > final_validation/results/scn/m1_chunks_transposed.log 2>&1
./run_scn.sh $P chunks off off off m1_chunks_off.json $=C > final_validation/results/scn/m1_chunks_off.log 2>&1
touch final_validation/results/scn/chunks.done
