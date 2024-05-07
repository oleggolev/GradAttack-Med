#!/bin/bash

# Run the training script with different hyperparameters.
export PYTHONPATH=/scratch/network/ogolev/GradAttack-Med:$PYTHONPATH
export WORKDIR=/scratch/network/ogolev/GradAttack-Med

# Attack with no defenses.
scommand="sbatch -J attack-no-defenses scripts/attack/vanilla.slurm"
echo "submit command: $scommand"
$scommand

