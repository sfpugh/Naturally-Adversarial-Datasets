#!/bin/bash

outdir=$1

datasets=("spo2_hr_lo" "spo2_hr_hi" "rr_lo" "rr_hi" "spo2_lo" "crossmodal" "crowd" "recsys" "spam")
default_preds=(1 1 1 1 1 0 0 1 0)
pls=("majorityvote" "snorkel")

cd ..

results=""
for ((i = 0; i < ${#datasets[@]}; i++)); do
    for pl in "${pls[@]}"; do
        r1=$(python main.py --dataset ${datasets[i]} --default_pred ${default_preds[i]} --pl $pl --skip_lf_pruning --skip_ci --evaluate --save --idxs_file $outdir/$pl/pl_conf_all_lfs/${datasets[i]}_dataset_indexes.pkl --y_prob_file $outdir/$pl/pl_conf_all_lfs/${datasets[i]}_y_prob.npy)
        r2=$(python main.py --dataset ${datasets[i]} --default_pred ${default_preds[i]} --pl $pl --skip_ci --evaluate --save --idxs_file $outdir/$pl/pl_conf_indep_lfs/${datasets[i]}_dataset_indexes.pkl --y_prob_file $outdir/$pl/pl_conf_indep_lfs/${datasets[i]}_y_prob.npy)
        r3=$(python main.py --dataset ${datasets[i]} --default_pred ${default_preds[i]} --pl $pl --skip_lf_pruning --evaluate --save --idxs_file $outdir/$pl/ci_lb_all_lfs/${datasets[i]}_dataset_indexes.pkl --y_prob_file $outdir/$pl/ci_lb_all_lfs/${datasets[i]}_y_prob.npy)
        r4=$(python main.py --dataset ${datasets[i]} --default_pred ${default_preds[i]} --pl $pl --evaluate --save --idxs_file $outdir/$pl/our_approach/${datasets[i]}_dataset_indexes.pkl --y_prob_file $outdir/$pl/our_approach/${datasets[i]}_y_prob.npy)
        results="$results${datasets[i]} & $pl & $r1 & $r2 & $r3 & $r4 \\\\\\ \n"
    done
done

echo -e "Case Study & Probabilistic Labeler (PL) & PL Conf with all LFs & PL Conf with indep LFs & CI LB with all LFs & Our Approach \\\\\\"
echo -e "$results"
