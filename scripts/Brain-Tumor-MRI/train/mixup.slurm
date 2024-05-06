#!/bin/bash
#SBATCH --job-name=train-mixup   # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=4        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=4G         # memory per cpu-core (4G is default)
#SBATCH --gres=gpu:1             # number of gpus per node
#SBATCH --constraint=a100        # specifically run on A100 GPU
#SBATCH --time=06:00:00          # total run time limit (HH:MM:SS)
#SBATCH --mail-type=begin        # send email when job begins
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-type=fail         # send email if job fails
#SBATCH --output=/scratch/gpfs/adityam/GradAttack-Med/output/Brain-Tumor-MRI/train/%x.out
#SBATCH --mail-user=ogolev@princeton.edu

module purge
module load anaconda3/2022.10
conda activate grad-attack-env

# Defend with MixUp
srun python3 examples/train_brain_tumor_mri.py --scheduler ReduceLROnPlateau --n_epoch 50 --tune_on_val 0.02 --lr 0.1 --lr_factor 0.5 --logname Brain-Tumor-MRI/MixUp-$1 --defense_mixup --k $1 --c_1 0 --c_2 0.65  --patience 50
