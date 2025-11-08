#!/bin/bash
# Helper to submit `job.sh` with optional overrides
# Usage: ./submit_job.sh [--time HH:MM:SS] [--cpus N] [--mem M]

TIME_OVERRIDE=""
CPUS_OVERRIDE=""
MEM_OVERRIDE=""

while [[ $# -gt 0 ]]; do
  key="$1"
  case $key in
    --time)
      TIME_OVERRIDE="--time=$2"
      shift; shift
      ;;
    --cpus)
      CPUS_OVERRIDE="--cpus-per-task=$2"
      shift; shift
      ;;
    --mem)
      MEM_OVERRIDE="--mem=$2"
      shift; shift
      ;;
    *)
      echo "Unknown option $1"
      exit 1
      ;;
  esac
done

sbatch $TIME_OVERRIDE $CPUS_OVERRIDE $MEM_OVERRIDE job.sh
