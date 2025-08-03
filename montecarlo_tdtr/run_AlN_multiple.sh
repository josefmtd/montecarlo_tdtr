#!/bin/bash

# Check if necessary parameters were provided
if [ -z "$1" ] | [ -z "$2" ] | [ -z "$3" ] | [ -z "$4" ]; then
  echo "Usage: $0 <file-name> <film-thickness> <probe-radius> <pump-radius> optional: <number-of-cases> <measurement-name>"
  exit 1
fi

# Store the pattern
pattern="$1"

# Use globbing to match files
shopt -s nullglob
matches=($pattern)
shopt -u nullglob

# Check if any files matched
if [ ${#matches[@]} -eq 0 ]; then
  echo "No files match the pattern '$pattern'"
  exit 0
fi

# Submit batch jobs for all matched files
for file in "${matches[@]}"; do
  echo "$Processing $file"
  sbatch aluminium_nitride.sh "$file" "$2" "$3" "$4" ${5:+$5} ${6:+$6}
done
