#!/bin/bash

module load anaconda3
module load cuda-toolkit-10.1
source activate torch180

dataset=$1
loss=$2
lr=$3
ep=$4
cuda=$5

cd ..

stime=`date +%s`
for i in $(seq 1 1 8);
do
    regs=`cat ./exp_evasion/reg_scale.txt|awk "NR==$i{print}"|awk '{print $1" "$2" "$3" "$4" "$5" "$6" "$7}'`;
    echo $regs;
    for reg in $regs;
    do
        for ptb in $(seq 0.05 0.05 0.5);
        do
            python -m evasion_attack --device $cuda --dataset $dataset --loss_type ${loss} --att_lr ${lr} --perturb_epochs $ep --ptb_rate $ptb --reg_weight $reg --gnn_path "./nat_model_saved/${dataset}_gcn.pt" --sanitycheck no;
        done > ./exp_evasion/result/${dataset}_${loss}_reg${reg}.output
    done
    wait
done
etime=`date +%s`

s=`echo "scale=0; ($etime - $stime)%60" | bc`
m=`echo "scale=0; ($etime - $stime)/60%60" | bc`
h=`echo "scale=0; ($etime - $stime)/60/60" | bc`

echo `date +"%F %T"` end cost $h:$m:$s