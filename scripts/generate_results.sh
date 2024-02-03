#!/bin/bash

datasets=("spo2_hr_lo" "spo2_hr_hi" "rr_lo" "rr_hi" "spo2_lo" "crossmodal" "crowd" "recsys" "spam")
pls=("majorityvote" "snorkel")

cd ../Naturally_Adversarial_Datasets

results=""
for d in "${datasets[@]}"; do
    for pl in "${pls[@]}"; do
        r1=$(python curate_datasets.py --dataset $d --pl $pl --skip_lf_pruning --skip_ci --evaluate)
        r2=$(python curate_datasets.py --dataset $d --pl $pl --skip_ci --evaluate)
        r3=$(python curate_datasets.py --dataset $d --pl $pl --skip_lf_pruning --evaluate)
        r4=$(python curate_datasets.py --dataset $d --pl $pl --evaluate)
        results="$results$r1 & $r2 & $r3 & $r4 \\\\ \n"
    done
done

echo -e "$results"