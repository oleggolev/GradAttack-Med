#!/bin/bash

# Run the training script with different hyperparameters.
export PYTHONPATH=/scratch/gpfs/adityam/GradAttack-Med:$PYTHONPATH
export WORKDIR=/scratch/gpfs/adityam/GradAttack-Med

# Train with no defenses.
sbatch -J no-defenses train/vanilla.slurm

# Train with GradPrune.
for p in 0.5 0.7 0.9 0.95 0.99 0.999
do
    scommand="sbatch -J prune-${p} train/prune.slurm $p"
    echo "submit command: $scommand"
    $scommand
done

# Train with MixUp.
for k in 2 4 6 8
do
    scommand="sbatch -J mixup-${k} train/mixup.slurm $k"
    echo "submit command: $scommand"
    $scommand
done

# Train with InstaHide.
for k in 2 4 6 8
do
    scommand="sbatch -J insta-${k} train/insta.slurm $k"
    echo "submit command: $scommand"
    $scommand
done

# Train with MixUp + GradPrune.
for k in 2 4 6
do
    for p in 0.7 0.9 0.99
    do
        scommand="sbatch -J prune-mixup-${p}-${k} train/prune-mixup.slurm $p $k"
        echo "submit command: $scommand"
        $scommand
    done
done

# Train with InstaHide + GradPrune.
for k in 2 4 6
do
    for p in 0.7 0.9 0.99
    do
        scommand="sbatch -J prune-insta-${p}-${k} train/prune-insta.slurm $p $k"
        echo "submit command: $scommand"
        $scommand
    done
done
