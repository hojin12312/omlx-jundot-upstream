#!/bin/zsh
# Upstream main (group-major Q8 A8, merged #4350 + env removal) against the follow-up commit, same prompts, ABBA.
HERE=${0:A:h}; M=<HOME>/Projects/omlx-q8a8-measured/mainbase; F=<HOME>/Projects/omlx-q8a8-measured/followup; cd $HERE
R=final_validation/results; S="--lengths 1024 4096 8192 --repeats 3"
for rd in 1 2; do
  if [ $rd = 1 ]; then ORDER=(main followup); else ORDER=(followup main); fi
  for arm in $ORDER; do
    if [ $arm = main ]; then T=$M; else T=$F; fi
    PREFIX=fu ./run_session2.sh $T $arm $rd off on ${=S} > $R/e2e_fu_${arm}_r${rd}.log 2>&1
  done
done
touch $R/followup.done
