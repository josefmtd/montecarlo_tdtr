#!/bin/bash
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=12
#SBATCH --mem=12G
#SBATCH --gpus=1
#SBATCH --constraint=volta
#SBATCH -o ../out/%j.out

# Usage: sbatch run_aln_montecarlo_gpu.sbatch <file-name> <film-thickness> \
#            <probe-radius> <pump-radius> [number-of-cases] [sample-name] [n-workers]

if [ -z "$1" ] || [ -z "$2" ] || [ -z "$3" ] || [ -z "$4" ]; then
  echo "Usage: $0 <file-name> <film-thickness> <probe-radius> <pump-radius> optional: <number-of-cases> <sample-name> <n-workers>"
  exit 1
fi

echo ""
echo "Started on node $HOSTNAME at $(date)."
echo "========================================"
nvidia-smi
echo ""

module load mamba
module load triton/2024.1-gcc cuda/12.2.1

source activate montecarlo_tdtr

file_name="${1##*/}"
file_path="../data/raw/$file_name"
name="${file_name%@*.mat}"

pump_radius="$4"
probe_radius="$3"
frequency=6000e3
film_thickness="$2"
transducer_thickness=80
n_draws="${5:-256}"
n_workers="${7:-12}"

echo "Executing file $file_name with $n_draws draws on $n_workers dask workers"
python aln_montecarlo_gpu_parallel.py "$file_path" "$pump_radius" "$probe_radius" \
    "$frequency" "$film_thickness" \
    -t "$transducer_thickness" -N "$n_draws" -P "$name" -w "$n_workers"
rc=$?
echo "File $1 completed at $(date) (exit code $rc)"

sample_name="${name##AlN?}"
sample_name="${sample_name%%?S?*}"
sample_name="${6:-$sample_name}"

mkdir -p "../out/$sample_name"
mv "../out/$name" "../out/$sample_name"
mkdir -p "../out/$sample_name/$name/mc-cases"
mv "../out/mc-cases/$name" "../out/$sample_name/$name/mc-cases"
mv "../out/$SLURM_JOB_ID.out" "../out/$sample_name/$name/$name-$SLURM_JOB_ID.out"

exit $rc
