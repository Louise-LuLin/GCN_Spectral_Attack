#!/bin/bash

dataset=$1
loss=$2
lr=$3
ep=$4


stime=`date +%s`
for i in $(seq 1 1 8);
do
    regs=`cat reg_scale.txt|awk "NR==$i{print}"|awk '{print $1" "$2" "$3" "$4" "$5" "$6" "$7}'`;
    echo $regs;
    for reg in $regs;
    do
        sbatch --job-name=$dataset$loss$reg --output=./result/${dataset}_${loss}_reg${reg}.out --error=./result/${dataset}_${loss}_reg${reg}.err slurm_run.sh $dataset $loss $lr $ep $reg;
    done
    sleep 20m
done
etime=`date +%s`

s=`echo "scale=0; ($etime - $stime)%60" | bc`
m=`echo "scale=0; ($etime - $stime)/60%60" | bc`
h=`echo "scale=0; ($etime - $stime)/60/60" | bc`

echo `date +"%F %T"` end cost $h:$m:$s
