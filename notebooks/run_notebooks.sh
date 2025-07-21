#!/bin/bash

#SBATCH --time=10:00:00
#SBATCH --mem=5G
#SBATCH --job-name=notebooks
#SBATCH -o ./out/%j.out
#SBATCH --gpus=1
#SBATCH --constraint=volta

echo ""
echo "Started on node $HOSTNAME at $(date)."
echo ""
echo "--------------------------------------"
echo ""

module load mamba
module load triton/2024.1-gcc cuda/12.2.1

source activate TDTR-Analysis

for file in $1; do
	echo "Executing notebook $file"
	echo ""
	jupyter nbconvert --to notebook --execute "$file" --inplace
	echo "File $file completed at $(date)"
	echo ""
done
echo "Done"
echo ""
echo "--------------------------------------"
echo ""
