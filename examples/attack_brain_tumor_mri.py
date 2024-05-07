import os
import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
from torch.nn.modules.loss import CrossEntropyLoss

from gradattack.attacks.gradientinversion import GradientReconstructor
from gradattack.datamodules import BrainTumorMRIDataModule
from gradattack.defenses.defense_utils import DefensePack
from gradattack.models import create_lightning_module
from gradattack.trainingpipeline import TrainingPipeline
from gradattack.utils import (cross_entropy_for_onehot, parse_args,
                              patch_image, save_fig)

mri_mean = torch.tensor(
    (0.485, 0.456, 0.406))
mri_std = torch.tensor(
    (0.229, 0.224, 0.225))
dm = mri_mean[:, None, None]
ds = mri_std[:, None, None]


def setup_attack():
    """Setup the pipeline for the attack"""
    args, hparams, attack_hparams = parse_args()
    print(attack_hparams)

    global ROOT_DIR, DEVICE, EPOCH, devices

    DEVICE = torch.device(f"cuda:{args.gpuid}")
    EPOCH = attack_hparams["epoch"]
    devices = [args.gpuid]

    seed = 42 + EPOCH
    pl.utilities.seed.seed_everything(seed)
    torch.backends.cudnn.benchmark = True

    BN_str = ''

    if not args.attacker_eval_mode:
        BN_str += "-attacker_train"
    if not args.defender_eval_mode:
        BN_str += '-defender_train'
    if args.BN_exact:
        BN_str = 'BN_exact'
        attack_hparams['attacker_eval_mode'] = False

    datamodule = BrainTumorMRIDataModule(batch_size=args.batch_size,
                                   augment={
                                       "hflip": False,
                                       "color_jitter": None,
                                       "rotation": -1,
                                       "crop": False
                                   },
                                   num_workers=48,
                                   seed=seed)
                                #    seed=args.data_seed)
    print("Loaded data!")
    if args.defense_instahide or args.defense_mixup:  # Customize loss
        loss = cross_entropy_for_onehot
    else:
        loss = CrossEntropyLoss(reduction="mean")
        
    if args.defense_gradprune:
        if args.defense_instahide:
            dir = f"/scratch/network/ogolev/GradAttack-Med/scripts/Brain-Tumor-MRI/tb_logs/Brain-Tumor-MRI/InstaHide+GradPrune-{args.p}-{args.klam}/SGD/StepLR/version_3/checkpoints/"
            ckpt_file = [file for file in os.listdir(dir) if os.path.isfile(os.path.join(dir, file))][0]
            ckpt_file_abs_path = dir + ckpt_file
            model = create_lightning_module("ResNet18",
                                            datamodule.num_classes,
                                            training_loss_metric=loss,
                                            pretrained=False,
                                            ckpt=ckpt_file_abs_path,
                                            **hparams).to(DEVICE)
        elif args.defense_mixup:
            dir = f"/scratch/network/ogolev/GradAttack-Med/scripts/Brain-Tumor-MRI/tb_logs/Brain-Tumor-MRI/MixUp+GradPrune-{args.p}-{args.klam}/SGD/ReduceLROnPlateau/version_4/checkpoints/"
            ckpt_file = [file for file in os.listdir(dir) if os.path.isfile(os.path.join(dir, file))][0]
            ckpt_file_abs_path = dir + ckpt_file
            model = create_lightning_module("ResNet18",
                                            datamodule.num_classes,
                                            training_loss_metric=loss,
                                            pretrained=False,
                                            ckpt=ckpt_file_abs_path,
                                            **hparams).to(DEVICE)
        else:
            dir = f"/scratch/network/ogolev/GradAttack-Med/scripts/Brain-Tumor-MRI/tb_logs/Brain-Tumor-MRI/GradPrune-{args.p}/SGD/ReduceLROnPlateau/version_4/checkpoints/"
            ckpt_file = [file for file in os.listdir(dir) if os.path.isfile(os.path.join(dir, file))][0]
            ckpt_file_abs_path = dir + ckpt_file
            model = create_lightning_module("ResNet18",
                                            datamodule.num_classes,
                                            training_loss_metric=loss,
                                            pretrained=False,
                                            ckpt=ckpt_file_abs_path,
                                            **hparams).to(DEVICE)
            
    elif args.defense_instahide:
        dir = f"/scratch/network/ogolev/GradAttack-Med/scripts/Brain-Tumor-MRI/tb_logs/Brain-Tumor-MRI/InstaHide-{args.klam}/SGD/StepLR/version_1/checkpoints/"
        ckpt_file = [file for file in os.listdir(dir) if os.path.isfile(os.path.join(dir, file))][0]
        ckpt_file_abs_path = dir + ckpt_file
        model = create_lightning_module("ResNet18",
                                        datamodule.num_classes,
                                        training_loss_metric=loss,
                                        pretrained=False,
                                        ckpt=ckpt_file_abs_path,
                                        **hparams).to(DEVICE)
    elif args.defense_mixup:
        dir = f"/scratch/network/ogolev/GradAttack-Med/scripts/Brain-Tumor-MRI/tb_logs/Brain-Tumor-MRI/MixUp-{args.klam}/SGD/ReduceLROnPlateau/version_1/checkpoints/"
        ckpt_file = [file for file in os.listdir(dir) if os.path.isfile(os.path.join(dir, file))][0]
        ckpt_file_abs_path = dir + ckpt_file
        model = create_lightning_module("ResNet18",
                                        datamodule.num_classes,
                                        training_loss_metric=loss,
                                        pretrained=False,
                                        ckpt=ckpt_file_abs_path,
                                        **hparams).to(DEVICE)
    else:
        model = create_lightning_module(
            "ResNet18",
            datamodule.num_classes,
            training_loss_metric=loss,
            pretrained=False,
            ckpt="/scratch/network/ogolev/GradAttack-Med/scripts/Brain-Tumor-MRI/tb_logs/Brain-Tumor-MRI/Vanilla/SGD/ReduceLROnPlateau/version_13/checkpoints/epoch=43-step=967.ckpt",
            **hparams).to(DEVICE)

    logger = TensorBoardLogger("tb_logs", name=f"{args.logname}")
    trainer = pl.Trainer(gpus=devices, benchmark=True, logger=logger)
    pipeline = TrainingPipeline(model, datamodule, trainer)

    defense_pack = DefensePack(args, logger)
    if attack_hparams["mini"]:
        datamodule.setup("attack_mini")
    elif attack_hparams["large"]:
        datamodule.setup("attack_large")
    else:
        datamodule.setup("attack")

    defense_pack.apply_defense(pipeline)

    ROOT_DIR = f"{args.results_dir}/Brain-Tumor-MRI-{args.batch_size}-{str(defense_pack)}"
    try:
        os.makedirs(ROOT_DIR, exist_ok=True)
    except FileExistsError:
        pass
    print("storing in root dir", ROOT_DIR)

    if "InstaHideDefense" in defense_pack.defense_params.keys():
        cur_lams = defense_pack.instahide_defense.cur_lams.cpu().numpy()
        cur_selects = defense_pack.instahide_defense.cur_selects.cpu().numpy()
        np.savetxt(f"{ROOT_DIR}/epoch_lams.txt", cur_lams)
        np.savetxt(f"{ROOT_DIR}/epoch_selects.txt", cur_selects.astype(int))
    elif "MixupDefense" in defense_pack.defense_params.keys():
        cur_lams = defense_pack.mixup_defense.cur_lams.cpu().numpy()
        cur_selects = defense_pack.mixup_defense.cur_selects.cpu().numpy()
        np.savetxt(f"{ROOT_DIR}/epoch_lams.txt", cur_lams)
        np.savetxt(f"{ROOT_DIR}/epoch_selects.txt", cur_selects.astype(int))

    return pipeline, attack_hparams


def run_attack(pipeline, attack_hparams):
    """Launch the real attack"""
    trainloader = pipeline.datamodule.train_dataloader()
    model = pipeline.model

    for batch_idx, (batch_inputs, batch_targets) in enumerate(trainloader):
        BATCH_ROOT_DIR = ROOT_DIR + f"/{batch_idx}"
        os.makedirs(BATCH_ROOT_DIR, exist_ok=True)
        save_fig(batch_inputs,
                 f"{BATCH_ROOT_DIR}/original.png",
                 save_npy=True,
                 save_fig=False)
        save_fig(patch_image(batch_inputs),
                 f"{BATCH_ROOT_DIR}/original.png",
                 save_npy=False)

        batch_inputs, batch_targets = batch_inputs.to(
            DEVICE), batch_targets.to(DEVICE)

        batch_gradients, step_results = model.get_batch_gradients(
            (batch_inputs, batch_targets),
            batch_idx,
            eval_mode=attack_hparams["defender_eval_mode"],
            apply_transforms=True,
            stop_track_bn_stats=False,
            BN_exact=attack_hparams["BN_exact"])
        batch_inputs_transform, batch_targets_transform = step_results[
            "transformed_batch"]

        save_fig(
            batch_inputs_transform,
            f"{BATCH_ROOT_DIR}/transformed.png",
            save_npy=True,
            save_fig=False,
        )
        save_fig(
            patch_image(batch_inputs_transform),
            f"{BATCH_ROOT_DIR}/transformed.png",
            save_npy=False,
        )

        attack = GradientReconstructor(
            pipeline,
            ground_truth_inputs=batch_inputs_transform,
            ground_truth_gradients=batch_gradients,
            ground_truth_labels=batch_targets_transform,
            reconstruct_labels=attack_hparams["reconstruct_labels"],
            num_iterations=10000,
            signed_gradients=True,
            signed_image=attack_hparams["signed_image"],
            boxed=True,
            total_variation=attack_hparams["total_variation"],
            bn_reg=attack_hparams["bn_reg"],
            lr_scheduler=True,
            lr=attack_hparams["attack_lr"],
            mean_std=(dm, ds),
            attacker_eval_mode=attack_hparams["attacker_eval_mode"],
            BN_exact=attack_hparams["BN_exact"])

        tb_logger = TensorBoardLogger(BATCH_ROOT_DIR, name="tb_log")
        attack_trainer = pl.Trainer(
            gpus=devices,
            logger=tb_logger,
            max_epochs=1,
            benchmark=True,
            checkpoint_callback=False,
        )
        attack_trainer.fit(attack)
        result = attack.best_guess.detach().to("cpu")

        save_fig(result,
                 f"{BATCH_ROOT_DIR}/reconstructed.png",
                 save_npy=True,
                 save_fig=False)
        save_fig(patch_image(result),
                 f"{BATCH_ROOT_DIR}/reconstructed.png",
                 save_npy=False)


if __name__ == "__main__":
    pipeline, attack_hparams = setup_attack()
    run_attack(pipeline, attack_hparams)
