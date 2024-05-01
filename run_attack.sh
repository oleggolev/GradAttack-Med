# As done in the paper, we will run the attack in its strongest setting. That is,
# the private BatchNorm statistics and private labels of the victim batch are known.

# Run the attack with no defenses.
python3 examples/attack_cifar10_gradinversion.py --batch_size 16 --BN_exact --tv 0.1 --bn_reg 0.005

# # Run the attack on pruned models.
# python3 examples/attack_cifar10_gradinversion.py --batch_size 16 --BN_exact --tv 0.1 --bn_reg 0.005 --defense_gradprune --p 0.5

# # Run the attack on InstaHide models.
# python3 examples/attack_cifar10_gradinversion.py --batch_size 16 --BN_exact --tv 0.1 --bn_reg 0.005 --defense_instahide --k 4 --c_1 0 --c_2 0.65
# python examples/attack_decode.py --dir PATH_TO_STEP1_RESULTS --instahide --k 4 --dest_dir PATH_TO_STEP2_RESULTS

# /scratch/network/ogolev/GradAttack-Med/results/CIFAR10-16-InstaHideDefense-k\{4\}-c_1\{0.0\}-c_2\{0.65\}/tv\=0.01BN_exact-bn\=0.001/)
