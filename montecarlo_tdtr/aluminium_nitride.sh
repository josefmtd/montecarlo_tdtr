#!/bin/bash
#SBATCH --time=08:00:00
#SBATCH -c 24
#SBATCH --mem=8G
#SBATCH --gpus=1
#SBATCH --constraint=volta
#SBATCH -o ./out/%j.out

# Check if a file name was provided
if [ -z "$1" ] | [ -z "$2" ]; then
  echo "Usage: $0 <file-name> <film-thickness> <number-of cases>"
  exit 1
fi

echo ""
echo "Started on node $HOSTNAME at $(date)."
echo ""
echo "--------------------------------------"
echo ""

# Load recquired modules
module load mamba
module load triton/2024.1-gcc cuda/12.2.1

# Activate the python environment
source activate TDTR-Analysis

# Store file path
file="data/raw/$1"

# Store the name of the measurement
name="${1%@*.mat}"

# Store parameters
pump_radius=12.5e-6
probe_radius=11.5e-6
frequency=6000e3
film_thickness="$2"
transducer_thickness=80
n_draws="${3:-256}"

# Run the python script
echo "Executing file $file"
python montecarlo_aln.py "$file" $pump_radius $probe_radius $frequency $film_thickness -k 200 -t $transducer_thickness -N $n_draws -P "$name"
echo "File $1 completed at $(date)"

# Rename out-file
mv ./out/$SLURM_JOB_ID.out ./out/$name/$name-$SLURM_JOB_ID.out
rm ./out/$SLURM_JOB_ID.out
