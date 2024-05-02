#!/bin/bash

# Run the training script with different hyperparameters.
export PYTHONPATH=/scratch/network/ogolev/GradAttack-Med:$PYTHONPATH
export WORKDIR=/scratch/network/ogolev/GradAttack-Med

# Attack with no defenses.
scommand="sbatch -J attack-no-defenses scripts/attack/vanilla.slurm"
echo "submit command: $scommand"
$scommand

# Attack with GradPrune.
for p in 0.5 0.7 0.9 0.95 0.99 0.999
do
    scommand="sbatch -J attack-prune-${p} scripts/attack/prune.slurm $p"
    echo "submit command: $scommand"
    $scommand
done

# Attack with MixUp.
for k in 2 4 6 8
do
    scommand="sbatch -J attack-mixup-${k} scripts/attack/mixup.slurm $k"
    echo "submit command: $scommand"
    $scommand
done

# Attack with InstaHide.
for k in 2 4 6 8
do
    scommand="sbatch -J attack-insta-${k} scripts/attack/insta.slurm $k"
    echo "submit command: $scommand"
    $scommand
done

# Attack with MixUp + GradPrune.
for k in 2 4 6
do
    for p in 0.7 0.9 0.99
    do
        scommand="sbatch -J attack-prune-mixup-${p}-${k} scripts/attack/prune-mixup.slurm $p $k"
        echo "submit command: $scommand"
        $scommand
    done
done

# Attack with InstaHide + GradPrune.
for k in 2 4 6
do
    for p in 0.7 0.9 0.99
    do
        scommand="sbatch -J attack-prune-insta-${p}-${k} scripts/attack/prune-insta.slurm $p $k"
        echo "submit command: $scommand"
        $scommand
    done
done
