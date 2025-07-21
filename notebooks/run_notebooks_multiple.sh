#!/bin/bash

# Check if a file pattern was provided
if [ -z "$1" ]; then
  echo "Usage: $0 <file-pattern>"
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

# Echo matching file names
for file in "${matches[@]}"; do
  echo "$Running $file"
  sbatch run_notebooks.sh "$file"
done