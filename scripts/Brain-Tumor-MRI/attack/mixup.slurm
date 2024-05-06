#!/bin/bash
#SBATCH --job-name=attack-mixup
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=4        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=4G         # memory per cpu-core (4G is default)
#SBATCH --gres=gpu:1             # number of gpus per node
#SBATCH --constraint=a100        # specifically run on A100 GPU
#SBATCH --time=01:00:00          # total run time limit (HH:MM:SS)
#SBATCH --mail-type=begin        # send email when job begins
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-type=fail         # send email if job fails
#SBATCH --output=/scratch/gpfs/adityam/GradAttack-Med/output/attack/%x.out
#SBATCH --mail-user=ogolev@princeton.edu

module purge
module load anaconda3/2022.10
conda activate grad-attack-env

# Attack the MixUp model.
srun python3 examples/attack_cifar10_gradinversion.py --batch_size $2 --BN_exact --tv 0.1 --bn_reg 0.005 --defense_mixup --k $1 --c_1 0 --c_2 0.65