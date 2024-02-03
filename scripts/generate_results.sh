#!/bin/bash

datasets=("spo2_hr_lo" "spo2_hr_hi" "rr_lo" "rr_hi" "spo2_lo" "crossmodal" "crowd" "recsys" "spam")
default_preds=(1 1 1 1 1 0 0 1 0)
pls=("majorityvote" "snorkel")

cd ..

results=""
for ((i = 0; i < ${#datasets[@]}; i++)); do
    for pl in "${pls[@]}"; do
        r1=$(python main.py --dataset ${datasets[i]} --default_pred ${default_preds[i]} --pl $pl --skip_lf_pruning --skip_ci --evaluate)
        r2=$(python main.py --dataset ${datasets[i]} --default_pred ${default_preds[i]} --pl $pl --skip_ci --evaluate)
        r3=$(python main.py --dataset ${datasets[i]} --default_pred ${default_preds[i]} --pl $pl --skip_lf_pruning --evaluate)
        r4=$(python main.py --dataset ${datasets[i]} --default_pred ${default_preds[i]} --pl $pl --evaluate)
        results="$results${datasets[i]} & $pl & $r1 & $r2 & $r3 & $r4 \\\\\\ \n"
    done
done

echo -e "$results"