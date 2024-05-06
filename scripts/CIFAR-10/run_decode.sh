#!/bin/bash

# Run the training script with different hyperparameters.
export PYTHONPATH=/scratch/gpfs/adityam/GradAttack-Med:$PYTHONPATH
export WORKDIR=/scratch/gpfs/adityam/GradAttack-Med

# Attack with no defenses.
scommand="sbatch -J attack-no-defenses scripts/attack/vanilla.slurm"
echo "submit command: $scommand"
$scommand

