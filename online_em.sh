#!/usr/bin/env bash
# 
#$ -cwd
#$ -j y
#$ -S /bin/bash
#$ -M han.zhao@uwaterloo.ca
#$ -m ea

module add gcc-510
#./online_em --train "data/$DATASET.ts.data" --valid "data/$DATASET.valid.data" --test "data/$DATASET.test.data" \
#--model "models_abdullah/$DATASET.r.spn.txt" --output_model "models_output/$DATASET.r.spn.txt" \
#--num_iters 10 | tee "log/$DATASET.online_em.txt"
./online_em --train "data/$DATASET.ts.data" --test "data/$DATASET.test.data" \
--model "models_abdullah/$DATASET.spn.txt"  --output_model "models_output/$DATASET.em.spn.txt" \
--num_iters 5 | tee "log/$DATASET.online_em.txt"
