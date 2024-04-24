#!/bin/bash
#SBATCH --job-name=GradAttack    # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=4        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=4G         # memory per cpu-core (4G is default)
#SBATCH --gres=gpu:1             # number of gpus per node
#SBATCH --constraint=a100        # specifically run on A100 GPU
#SBATCH --time=00:10:00          # total run time limit (HH:MM:SS)
#SBATCH --mail-type=begin        # send email when job begins
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-type=fail         # send email if job fails
#SBATCH --mail-user=ogolev@princeton.edu

module purge
module load anaconda3/2022.10
source /scratch/network/ogolev/GradAttack-Med/grad-attack-env/bin/activate
setenv PYTHONPATH="/scratch/network/ogolev/GradAttack-Med:$PYTHONPATH"
setenv WORKDIR="/scratch/network/ogolev/GradAttack-Med"

# Vanilla Training
python3 examples/train_cifar10.py --scheduler ReduceLROnPlateau --tune_on_val 0.02 --lr 0.05 --lr_factor 0.5 --logname CIFAR10/Vanilla 

# Defend with Gradient Pruning
python3 examples/train_cifar10.py --scheduler ReduceLROnPlateau --tune_on_val 0.02 --lr 0.05 --lr_factor 0.5 --logname CIFAR10/GradPrune --defense_gradprune --p 0.9

# Defend with MixUp
python3 examples/train_cifar10.py --scheduler ReduceLROnPlateau --tune_on_val 0.02 --lr 0.1 --lr_factor 0.5 --logname CIFAR10/MixUp --defense_mixup --k 4 --c_1 0 --c_2 0.65  --patience 50

# Defend with InstaHide
python3 examples/train_cifar10.py --scheduler StepLR --lr_step 50 --tune_on_val 0.02 --lr 0.05 --lr_factor 0.1 --logname CIFAR10/InstaHide --defense_instahide --k 4 --c_1 0 --c_2 0.65 --patience 50

# Gradient Pruning + MixUp 
python3 examples/train_cifar10.py --scheduler ReduceLROnPlateau --tune_on_val 0.02 --lr 0.1 --lr_factor 0.5 --logname CIFAR10/MixUp+GradPrune --defense_mixup --k 6 --c_1 0 --c_2 0.65 --defense_gradprune --p 0.9 --patience 50 

# Gradient Pruning + InstaHide  
python3 examples/train_cifar10.py --scheduler StepLR --lr_step 50 --tune_on_val 0.02 --lr 0.05 --lr_factor 0.1  --logname CIFAR10/InstaHide+GradPrune --defense_instahide --k 6 --c_1 0 --c_2 0.65 --defense_gradprune --p 0.9 --patience 50