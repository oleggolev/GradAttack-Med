#!/bin/bash

# Run the training script with different hyperparameters.
export PYTHONPATH=/scratch/network/ogolev/GradAttack-Med:$PYTHONPATH
export WORKDIR=/scratch/network/ogolev/GradAttack-Med

# Define the batch size for all runs.
batch_size=1

# Attack with no defenses.
scommand="sbatch -J attack-no-defenses-${batch_size} attack/vanilla.slurm $batch_size"
echo "submit command: $scommand"
$scommand

# Attack with GradPrune.
for p in 0.5 0.7 0.9 0.95 0.99 0.999
do
    scommand="sbatch -J attack-prune-${p}-${batch_size} attack/prune.slurm $p $batch_size"
    echo "submit command: $scommand"
    $scommand
done

# Attack with MixUp.
for k in 2 4 6 8
do
    scommand="sbatch -J attack-mixup-${k}-${batch_size} attack/mixup.slurm $k $batch_size"
    echo "submit command: $scommand"
    $scommand
done

# Attack with InstaHide.
for k in 2 4 6 8
do
    scommand="sbatch -J attack-insta-${k}-${batch_size} attack/insta.slurm $k $batch_size"
    echo "submit command: $scommand"
    $scommand
done

# Attack with MixUp + GradPrune.
for k in 2 4 6
do
    for p in 0.7 0.9 0.99
    do
        scommand="sbatch -J attack-prune-mixup-${p}-${k}-${batch_size} attack/prune-mixup.slurm $p $k $batch_size"
        echo "submit command: $scommand"
        $scommand
    done
done

# Attack with InstaHide + GradPrune.
for k in 2 4 6
do
    for p in 0.7 0.9 0.99
    do
        scommand="sbatch -J attack-prune-insta-${p}-${k}-${batch_size} attack/prune-insta.slurm $p $k $batch_size"
        echo "submit command: $scommand"
        $scommand
    done
done
