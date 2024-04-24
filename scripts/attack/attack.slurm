#!/bin/bash
#SBATCH --job-name=GradAttack    # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=4        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=4G         # memory per cpu-core (4G is default)
#SBATCH --gres=gpu:1             # number of gpus per node
#SBATCH --constraint=a100        # specifically run on A100 GPU
#SBATCH --time=02:00:00          # total run time limit (HH:MM:SS)
#SBATCH --mail-type=begin        # send email when job begins
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-type=fail         # send email if job fails
#SBATCH --mail-user=ogolev@princeton.edu

module purge
module load anaconda3/2022.10
source /scratch/network/ogolev/GradAttack-Med/grad-attack-env/bin/activate
prepend-path PYTHONPATH /scratch/network/ogolev/GradAttack-Med
setenv WORKDIR /scratch/network/ogolev/GradAttack-Med

# No defenses: realistic attacker
python3 examples/attack_cifar10_gradinversion.py --batch_size 16 --tv 0.05 --bn_reg 0.001 --reconstruct_labels

# No defenses: strong attacker (the attacker knows the BatchNorm statistics and private labels)
python3 examples/attack_cifar10_gradinversion.py --batch_size 16 --BN_exact --tv 0.1 --bn_reg 0.005

# With gradient pruning
python3 examples/attack_cifar10_gradinversion.py --batch_size 16 --BN_exact --tv 0.1 --bn_reg 0.005 --defense_gradprune --p 0.5

# With MixUp
python3 examples/attack_cifar10_gradinversion.py --batch_size 16 --BN_exact --tv 0.1 --bn_reg 0.005 --defense_instahide --k 4 --c_1 0 --c_2 0.65

# With InstaHide
python3 examples/attack_cifar10_gradinversion.py --batch_size 16 --BN_exact --tv 0.1 --bn_reg 0.005 --defense_instahide --k 4 --c_1 0 --c_2 0.65
# python3 examples/attack_decode.py --dir PATH_TO_STEP1_RESULTS --instahide --k 4 --dest_dir PATH_TO_STEP2_RESULTS
