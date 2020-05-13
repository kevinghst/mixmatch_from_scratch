#!/bin/bash

#SBATCH --gres=gpu:1
#SBATCH --mem=16000
#SBATCH --time=48:00:00
#SBATCH --job-name="mixmatch from scratch"

source activate mixmatch

cd /scratch/wz1232/mixmatch_from_scratch
$1
$2
$3
$4
$5
$6
$7
$8
$9
