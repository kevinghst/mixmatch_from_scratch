#!/bin/bash

#SBATCH --gres=gpu:1
#SBATCH --mem=16000
#SBATCH --time=12:00:00
#SBATCH --job-name="mixmatch from scratch"

source activate mixmatch

cd /scratch/iav225/nlu/mixmatch_from_scratch
$1
$2
$3
