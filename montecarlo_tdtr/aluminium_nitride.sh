#!/bin/bash
#SBATCH --time=12:00:00
#SBATCH -c 24
#SBATCH --mem=20G
#SBATCH --gpus=1
#SBATCH --constraint=volta
#SBATCH -o ./out/%j.out

# Check if necessary parameters were provided
if [ -z "$1" ] | [ -z "$2" ] | [ -z "$3" ] | [ -z "$4" ]; then
  echo "Usage: $0 <file-name> <film-thickness> <probe-radius> <pump-radius> optional: <number-of-cases> <sample-name>"
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

# Store the file name
file_name="${1##*/}"

# Store file path
file_path="../data/raw/$file_name"

# Store the name of the measurement
name="${file_name%@*.mat}"

# Store parameters
pump_radius="$4"
probe_radius="$3"
frequency=6000e3
film_thickness="$2"
transducer_thickness=80
n_draws="${5:-256}"

# Run the python script
echo "Executing file $file_name"
python montecarlo_aln.py $file_path $pump_radius $probe_radius $frequency $film_thickness -k 200 -t $transducer_thickness -N $n_draws -P $name
echo "File $1 completed at $(date)"

# Find and store the sample name by truncating the measurement name
sample_name="${name##AlN?}"
sample_name="${sample_name%%?S?*}"
sample_name="${6:-sample_name}"

# Rename and move out-file and folder
mkdir -p ./out/$sample_name
mv ./out/$name ./out/$sample_name
mv ./out/$SLURM_JOB_ID.out ./out/$sample_name/$name/$name-$SLURM_JOB_ID.out
