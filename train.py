import os
from copy import deepcopy
import time

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torchvision as tv
import einops
import matplotlib.pyplot as plt
import argparse
from itertools import chain

from diffusion.diffusion_ddpm_pan import GaussianDiffusion
from diffusion.diffusion_ddpm_pan import make_beta_schedule
from models.PanDiM import PanDiM
from data.PanDataset import PanDataset
from data.H5Dataset import H5Dataset
from utils.optim_utils import EmaUpdater
from utils.lr_scheduler import get_lr_from_optimizer, StepsAll
from utils.logger import TensorboardLogger
from utils.misc import grad_clip, model_load
from utils.metrics import AnalysisMetrics

os.environ["CUDA_VISIBLE_DEVICES"] = "4"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def _safe_log_scalar(logger, tag, value, step):
    if value is None:
        return
    if isinstance(value, bool):
        return
    if isinstance(value, (int, float)):
        logger.log_scalar(tag, float(value), step)


def _mean_attr(modules, attr_name):
    values = []
    for module in modules:
        if hasattr(module, attr_name):
            value = getattr(module, attr_name)
            if isinstance(value, (int, float)):
                values.append(float(value))
    if not values:
        return None
    return float(sum(values) / len(values))


def _param_grad_norm(module, attr_path):
    target = module
    for part in attr_path.split('.'):
        target = getattr(target, part, None)
        if target is None:
            return None
    grad = getattr(target, 'grad', None)
    if grad is None:
        return None
    return float(grad.detach().norm().item())


def _mean_grad_norm(modules, attr_path):
    values = []
    for module in modules:
        value = _param_grad_norm(module, attr_path)
        if value is not None:
            values.append(value)
    if not values:
        return None
    return float(sum(values) / len(values))


def _collect_posterior_diagnostics(model):
    mapper_modules = [m for m in model.modules() if m.__class__.__name__ in ('PosteriorControlMapper', 'DetailGuidedAffineModulator')]
    block_modules = [m for m in model.modules() if m.__class__.__name__ == 'CondDiffusionMambaBlock']
    return {
        'posterior/z_deg_norm_avg': _mean_attr(mapper_modules, '_last_z_deg_norm'),
        'posterior/d_map_mean_avg': _mean_attr(mapper_modules, '_last_d_map_mean'),
        'posterior/phys_energy_mean_avg': _mean_attr(mapper_modules, '_last_phys_energy'),
        'posterior/scale_abs_mean_avg': _mean_attr(mapper_modules, '_last_scale_abs_mean'),
        'posterior/shift_abs_mean_avg': _mean_attr(mapper_modules, '_last_shift_abs_mean'),
        'posterior/delta_gamma_abs_avg': _mean_attr(mapper_modules, '_last_delta_gamma_abs'),
        'posterior/delta_ssm_abs_avg': _mean_attr(mapper_modules, '_last_delta_ssm_abs'),
        'hyper/in_scale_avg': _mean_attr(mapper_modules, '_last_hyper_in_scale'),
        'hyper/out_scale_avg': _mean_attr(mapper_modules, '_last_hyper_out_scale'),
        'hyper/delta_in_ratio_avg': _mean_attr(block_modules, '_last_delta_in_ratio'),
        'hyper/delta_out_ratio_avg': _mean_attr(block_modules, '_last_delta_out_ratio'),
        'hyper/delta_in_abs_avg': _mean_attr(block_modules, '_last_delta_in_abs'),
        'hyper/delta_out_abs_avg': _mean_attr(block_modules, '_last_delta_out_abs'),
        'hyper/base_in_abs_avg': _mean_attr(block_modules, '_last_base_in_abs'),
        'hyper/base_out_abs_avg': _mean_attr(block_modules, '_last_base_out_abs'),
        'hyper/gate_mean_avg': _mean_attr(block_modules, '_last_gate_mean'),
        'hyper/ssm_mean_avg': _mean_attr(block_modules, '_last_ssm_mean'),
        'grads/gamma_mlp_avg': _mean_grad_norm(mapper_modules, 'gamma_mlp.1.weight'),
        'grads/hyper_ssm_avg': _mean_grad_norm(mapper_modules, 'hyper_ssm.1.weight'),
        'grads/hyper_in_left_avg': _mean_grad_norm(mapper_modules, 'hyper_in_left.weight'),
        'grads/hyper_out_left_avg': _mean_grad_norm(mapper_modules, 'hyper_out_left.weight'),
        'grads/hyper_in_scale_avg': _mean_grad_norm(mapper_modules, 'hyper_in_scale'),
        'grads/hyper_out_scale_avg': _mean_grad_norm(mapper_modules, 'hyper_out_scale'),
        'debug/num_mapper_modules': float(len(mapper_modules)),
        'debug/num_cond_mamba_blocks': float(len(block_modules)),
    }

def train(
    train_dataset_folder,
    valid_dataset_folder,
    args,
    dataset_name="",
    # image settings
    ms_num_channel = 3,
    pan_num_channel = 1,
    image_size=128,
    patch_size=8,
    # diffusion settings
    schedule_type="cosine",
    n_steps=3_000,
    max_iterations=400_000,
    # optimizer settings
    batch_size=128,
    lr_d=1e-5,
    show_recon=False,
    # pretrain settings
    pretrain_weight=None,
    pretrain_iterations=None,
):  
    denoise_fn = PanDiM(
        in_channel=ms_num_channel,
        out_channel=ms_num_channel,
        image_size=image_size,
        patch_size=patch_size,
        inner_channel=args.inner_channel,
        noise_level_channel=args.noise_level_channel,
        lms_channel=ms_num_channel,
        pan_channel=pan_num_channel,
        time_hidden_ratio=args.time_hidden_ratio,
        num_dit_layers=args.num_dit_layers,
        num_heads=args.num_heads,
        mlp_ratio=args.mlp_ratio,
        use_gated_block=args.use_gated_block,
        all_gated=args.all_gated,
        local_type=args.local_type,
        use_channel_swap=args.use_channel_swap,
        use_pure_mamba=args.use_pure_mamba,
        gate_type=args.gate_type,
        use_cond_mamba=args.use_cond_mamba
    ).to(device)

    
    diffusion = GaussianDiffusion(
        denoise_fn,
        image_size=image_size,
        channels=ms_num_channel,
        pred_mode="x_start",
        # pred_mode="noise",
        loss_type=args.loss_type,
        device=device,
        clamp_range=(0, 1),
        self_condition=args.self_condition,
        # FDDL
        use_fddl=args.use_fddl,
        fddl_weight=args.fddl_weight,
        ms_channels=ms_num_channel,
        # GT-FDDL & Curriculum Learning
        gt_fddl_high_boost=args.gt_fddl_high_boost,
        max_iterations=max_iterations,
        prior_warmup_ratio=args.prior_warmup_ratio,
    )
    diffusion.set_new_noise_schedule(
        betas=make_beta_schedule(schedule=schedule_type, n_timestep=n_steps, cosine_s=8e-3)
    )
    diffusion = diffusion.to(device)
    
    
    if pretrain_weight is not None and pretrain_weight != "":
        if isinstance(pretrain_weight, (list, tuple)):
            model_load(pretrain_weight[0], denoise_fn, strict=True, device=device)
        else:
            model_load(pretrain_weight, denoise_fn, strict=False, device=device)
        print("load pretrain weight from {}".format(pretrain_weight))
        
    # model, optimizer and lr scheduler
    diffusion_dp = (
        diffusion
    )
    ema_updater = EmaUpdater(
        diffusion_dp, deepcopy(diffusion_dp), decay=0.995, start_iter=20_000
    )
    
    opt_d = torch.optim.AdamW(denoise_fn.parameters(), lr=lr_d, weight_decay=args.weight_decay)

    scheduler_d = torch.optim.lr_scheduler.MultiStepLR(
        opt_d, milestones=[60_000, 80_000, 100_000], gamma=0.5
    )
    schedulers = StepsAll(scheduler_d)
    
    if args.use_h5:
         train_dataset = H5Dataset(
            h5_folder=train_dataset_folder,
            dataset_name=dataset_name,
            mode='train'
        )
         valid_dataset = H5Dataset(
            h5_folder=valid_dataset_folder,
            dataset_name=dataset_name,
            mode='valid'
        )
    else:
        train_dataset = PanDataset(
            img_size=image_size,
            ms_folder=os.path.join(train_dataset_folder, 'ms'),
            pan_folder=os.path.join(train_dataset_folder, 'pan'),
            ms_is_GT=True,
            mode='train',
            start_iter=args.pretrain_iterations,
        )
        valid_dataset = PanDataset(
            img_size=image_size,
            ms_folder=os.path.join(valid_dataset_folder, 'ms'),
            pan_folder=os.path.join(valid_dataset_folder, 'pan'),
            ms_is_GT=True,
            mode='test',
        )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=4, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, num_workers=4, shuffle=False)

    stf_time = time.strftime("%m-%d_%H-%M", time.localtime())
    comment = f"{dataset_name}_P{patch_size}C{args.inner_channel}L{args.num_dit_layers}H{args.num_heads}B{batch_size}-{args.model_name}"
    
    if args.use_pure_mamba:
        comment += "_PureMamba"
    if args.use_cond_mamba:
        comment += "_CondMamba"
        
    logger = TensorboardLogger(
        place="./runs",
        file_dir="./logs",
        file_logger_name="{}-{}".format(stf_time, comment),
        random_id=False,
        tb_comment="{}-{}".format(stf_time, comment),
    )
    
    if pretrain_iterations != 0:
        iterations = pretrain_iterations
        logger.print("load previous training with {} iterations".format(iterations))
        schedulers.step(iterations)
    else:
        iterations = 0
        
    # Ensure directories exist
    os.makedirs("./samples/recon_x", exist_ok=True)
    os.makedirs("./samples/valid_samples", exist_ok=True)
    os.makedirs("./checkpoints", exist_ok=True)

    best_psnr = 0.0
    while iterations <= max_iterations:
        for i, batch in enumerate(train_loader):
            pan, lms, wavelets, hr = map(lambda x: x.cuda(), (batch['img_pan'], batch['img_ms_up'], batch['wavelets'], batch['GT']))
            cond, _ = einops.pack(
                [
                    lms,
                    pan,
                    F.interpolate(wavelets, size=lms.shape[-1], mode="bicubic"),
                ],
                "b * h w",
            )
            cond = cond.to(torch.float32)
            opt_d.zero_grad()
            res = hr - lms
            diff_loss, recon_x = diffusion_dp(res, cond=cond, current_iter=iterations, lock_threshold=args.lock_threshold)
            diff_loss.backward()
            recon_x = recon_x + lms

            # do a grad clip on diffusion model
            grad_clip(diffusion_dp.model.parameters(), mode="norm", value=1.0)

            opt_d.step()
            ema_updater.update(iterations)
            schedulers.step()

            iterations += 1
            logger.print(
                f"[iter {iterations}/{max_iterations}: "
                + f"d_lr {get_lr_from_optimizer(opt_d): .8f}] - "
                + f"denoise loss {diff_loss:.6f} "
            )
            
            if iterations > max_iterations:
                break

            # test predicted sr
            if show_recon and iterations % 5_000 == 0:
                # NOTE: only used to validate code
                recon_x = recon_x[:64]

                x = tv.utils.make_grid(recon_x, nrow=8, padding=0).cpu()
                x = x.clip(0, 1)  # for no warning
                fig, ax = plt.subplots(figsize=(x.shape[-1] / 100, x.shape[-2] / 100))
                x_show = (
                    x.permute(1, 2, 0).detach().numpy()
                )
                if x_show.shape[-1] == 8:
                     # WV3: Red=4, Green=2, Blue=1 (0-based indices)
                     x_show = x_show[:, :, [4, 2, 1]]
                elif x_show.shape[-1] > 3:
                     x_show = x_show[:, :, :3]
                ax.imshow(x_show)
                ax.set_axis_off()
                plt.tight_layout(pad=0)
                os.makedirs("./samples/recon_x", exist_ok=True)
                fig.savefig(
                    f"./samples/recon_x/iter_{iterations}.png",
                    dpi=200,
                    bbox_inches="tight",
                    pad_inches=0,
                )

            # validate and save best model
            if iterations % args.val_freq == 0:
                diffusion_dp.model.eval()
                ema_updater.ema_model.model.eval()

                analysis_metrics = AnalysisMetrics()
                with torch.no_grad():
                    for i, batch in enumerate(valid_loader):
                        torch.cuda.empty_cache()
                        pan, lms, wavelets, hr = map(lambda x: x.cuda(), (batch['img_pan'], batch['img_ms_up'], batch['wavelets'], batch['GT']))
                        cond, _ = einops.pack(
                            [
                                lms,
                                pan,
                                F.interpolate(wavelets, size=lms.shape[-1], mode="bicubic"),
                            ],
                            "b * h w",
                        )
                        cond = cond.to(torch.float32)
                        sr = ema_updater.ema_model(cond, mode="ddim_sample", section_counts="ddim25")
                        sr = sr + lms
                        sr = sr.clip(0, 1)
                        hr = hr.to(sr.device)
                        analysis_metrics.batch_metrics(sr, hr, mode=args.mode)
                        
                        # Disabled image saving as requested
                        # hr = tv.utils.make_grid(hr, nrow=4, padding=0).cpu()
                        # x = tv.utils.make_grid(sr, nrow=4, padding=0).detach().cpu()
                        # x = x.clip(0, 1)

                        # s = torch.cat([hr, x], dim=-1)  # [b, c, h, 2*w]
                        # fig, ax = plt.subplots(
                        #     figsize=(s.shape[-1] / 100, s.shape[-2] / 100)
                        # )
                        # s_np = s.permute(1, 2, 0).detach().numpy()
                        # if s_np.shape[-1] == 8:
                        #     # WV3: Red=4, Green=2, Blue=1
                        #     s_np = s_np[:, :, [4, 2, 1]]
                        # elif s_np.shape[-1] > 3:
                        #     s_np = s_np[:, :, :3]
                        # ax.imshow(s_np)
                        # ax.set_axis_off()

                        # plt.tight_layout(pad=0)
                        # os.makedirs("./samples/valid_samples", exist_ok=True)
                        # fig.savefig(
                        #     f"./samples/valid_samples/iter_{iterations}_batch_{i}.png",
                        #     dpi=200,
                        #     bbox_inches="tight",
                        #     pad_inches=0,
                        # )
                        
                    logger.print("---diffusion result---")
                    logger.print(analysis_metrics.get_metrics_str())
                    
                    # Log to tensorboard
                    current_metrics = analysis_metrics.get_metrics()
                    logger.log_scalars("diffusion_perf", current_metrics, iterations)
                    
                    # Save best model
                    current_psnr = current_metrics['PSNR']
                    if current_psnr > best_psnr:
                        best_psnr = current_psnr
                        logger.print(f"New best PSNR: {best_psnr:.4f}, saving best model...")
                        torch.save(
                            ema_updater.on_fly_model_state_dict,
                            f"./checkpoints/best_diffusion_{comment}.pth",
                        )
                        torch.save(
                            ema_updater.ema_model_state_dict,
                            f"./checkpoints/best_ema_diffusion_{comment}.pth",
                        )
                        
                        # Save to extra path if provided
                        if args.extra_checkpoint_path:
                            try:
                                os.makedirs(args.extra_checkpoint_path, exist_ok=True)
                                torch.save(
                                    ema_updater.on_fly_model_state_dict,
                                    os.path.join(args.extra_checkpoint_path, f"best_diffusion_{comment}.pth")
                                )
                                torch.save(
                                    ema_updater.ema_model_state_dict,
                                    os.path.join(args.extra_checkpoint_path, f"best_ema_diffusion_{comment}.pth")
                                )
                                logger.print(f"Saved best model backup to {args.extra_checkpoint_path}")
                            except Exception as e:
                                logger.print(f"Failed to save backup confirmation: {e}")

                diffusion_dp.model.train()
                
            # Regular checkpoint saving
            if iterations % args.save_per_iter == 0:
                os.makedirs("./checkpoints", exist_ok=True)
                torch.save(
                    ema_updater.on_fly_model_state_dict,
                    f"./checkpoints/diffusion_{comment}_iter_{iterations}.pth",
                )
                torch.save(
                    ema_updater.ema_model_state_dict,
                    f"./checkpoints/ema_diffusion_{comment}_iter_{iterations}.pth",
                )
                logger.print("save model")
                logger.print("saved performances")
                
                # Save to extra path if provided
                if args.extra_checkpoint_path:
                    try:
                        os.makedirs(args.extra_checkpoint_path, exist_ok=True)
                        torch.save(
                            ema_updater.on_fly_model_state_dict,
                            os.path.join(args.extra_checkpoint_path, f"diffusion_{comment}_iter_{iterations}.pth")
                        )
                        torch.save(
                            ema_updater.ema_model_state_dict,
                            os.path.join(args.extra_checkpoint_path, f"ema_diffusion_{comment}_iter_{iterations}.pth")
                        )
                    except Exception as e:
                        logger.print(f"Failed to save periodic backup: {e}")

            # log loss and posterior / hyper diagnostics
            if iterations % 500 == 0:
                _safe_log_scalar(logger, "denoised_loss", diff_loss.item(), iterations)

                diagnostics = _collect_posterior_diagnostics(diffusion_dp)
                for tag, value in diagnostics.items():
                    _safe_log_scalar(logger, tag, value, iterations)

                logger.print(
                    "[diag] "
                    + f"z_deg={diagnostics.get('posterior/z_deg_norm_avg', None)} | "
                    + f"d_map={diagnostics.get('posterior/d_map_mean_avg', None)} | "
                    + f"d_ssm={diagnostics.get('posterior/delta_ssm_abs_avg', None)} | "
                    + f"din_ratio={diagnostics.get('hyper/delta_in_ratio_avg', None)} | "
                    + f"dout_ratio={diagnostics.get('hyper/delta_out_ratio_avg', None)} | "
                    + f"g_hssm={diagnostics.get('grads/hyper_ssm_avg', None)}"
                )

def parse_args():
    parser = argparse.ArgumentParser(description="Training script for PanDiM")
    # parser.add_argument('--train_dataset_folder', type=str, required=True, help='Path to the training dataset folder')
    # parser.add_argument('--valid_dataset_folder', type=str, required=True, help='Path to the validation dataset folder')
    parser.add_argument('--model_name', type=str, default='PanDiM', help='Model name')
    parser.add_argument('--dataset_name', type=str, default='WV3', help='Dataset name')
    parser.add_argument('--ms_num_channel', type=int, default=4, help='Number of multispectral channels')
    parser.add_argument('--pan_num_channel', type=int, default=1, help='Number of panchromatic channels')
    parser.add_argument('--train_dataset_folder', type=str, default='PanDataset', help='Path to the dataset folder')
    parser.add_argument('--valid_dataset_folder', type=str, default='PanDataset', help='Path to the validation dataset folder')
    parser.add_argument('--image_size', type=int, default=128, help='Input image size')
    parser.add_argument('--patch_size', type=int, default=8, help='Patch size for processing')
    parser.add_argument('--schedule_type', type=str, default='cosine', choices=['cosine', 'linear'], help='Schedule type for beta schedule')
    parser.add_argument('--n_steps', type=int, default=500, help='Number of diffusion steps')
    parser.add_argument('--max_iterations', type=int, default=200_000, help='Maximum number of training iterations')
    parser.add_argument('--save_per_iter', type=int, default=20_000, help='Save model per iteration')
    parser.add_argument('--val_freq', type=int, default=2_000, help='Validation frequency')
    parser.add_argument('--batch_size', type=int, default=6, help='Batch size for training')
    parser.add_argument('--lr_d', type=float, default=5e-5, help='Learning rate for optimizer')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay for optimizer')
    parser.add_argument('--loss_type', type=str, default='l1', choices=['l1sam', 'l1', 'l1ssim','gtfddl'], help='Loss type for diffusion')
    parser.add_argument('--show_recon', type=bool, default=True, help='Whether to show reconstructed images during training')
    parser.add_argument('--pretrain_weight', type=str, default=None, help='Path to pretrained weights (if any)')
    parser.add_argument('--pretrain_iterations', type=int, default=0, help='Number of iterations for pretrained model')
    parser.add_argument('--inner_channel', type=int, default=16, help='Number of inner channels')
    parser.add_argument('--noise_level_channel', type=int, default=128, help='Number of noise level channels')
    parser.add_argument('--time_hidden_ratio', type=int, default=4, help='Time hidden ratio')
    parser.add_argument('--num_dit_layers', type=int, default=12, help='Number of DIT layers')
    parser.add_argument('--num_heads', type=int, default=16, help='Number of attention heads')
    parser.add_argument('--mlp_ratio', type=int, default=4, help='MLP ratio in DIT Block')
    parser.add_argument('--with_noise_level_emb', type=bool, default=True, help='Whether to include noise level embedding')
    parser.add_argument('--self_condition', type=bool, default=True, help='Whether to self condition')
    parser.add_argument('--mode', type=str, default='CMYK', help='Path to save the output images')
    parser.add_argument('--use_h5', action='store_true', default=True, help='Use H5 dataset')
    parser.add_argument('--use_gated_block', action='store_true', default=True, help='Use GatedHybridBlock')
    parser.add_argument("--all_gated", action="store_true", help="Use GatedHybridBlock in ALL layers (Unified Backbone)")
    parser.add_argument('--local_type', type=str, default='convnext', choices=['classic', 'arconv', 'resblock', 'local_refine', 'convnext', 'fdconv', 'mkpu'], help='Local branch type in GatedDiMBlock')
    parser.add_argument("--use_channel_swap", action="store_true", help="Enable Optional Channel Swap in GatedDiMBlock (Inspired by PanMamba)")
    parser.add_argument('--extra_checkpoint_path', type=str, default=None, help='Extra path to save checkpoints')
    parser.add_argument("--use_pure_mamba", action="store_true", help="Enable Pure Mamba Baseline for ablation study")
    parser.add_argument("--use_cond_mamba", action="store_true", help="Enable Condition-Modulated Mamba (CM-SSM) Block")
    parser.add_argument('--gate_type', type=str, default='spatial', choices=['spatial', 'consensus', 'skfusion'], help='Gate type for GatedDiMBlock')
    parser.add_argument("--lock_threshold", type=int, default=16000, help="Step threshold to lock ARConv kernels")
    # FDDL (Frequency Decoupled Loss)
    parser.add_argument("--use_fddl", action="store_true", help="Enable Spatial_FDDL physics prior with cosine decay")
    parser.add_argument("--fddl_weight", type=float, default=0.05, help="Max weight for Spatial_FDDL prior")
    # GT-FDDL
    parser.add_argument("--gt_fddl_high_boost", type=float, default=2.0, help="High-freq boost factor for GT_FDDL (when loss_type=gtfddl)")
    parser.add_argument("--prior_warmup_ratio", type=float, default=0.5, help="Fraction of training to keep full Spatial_FDDL weight before cosine decay")
    return parser.parse_args()

if __name__ == "__main__":
    py_path = os.path.abspath(__file__) 
    file_dir = os.path.dirname(py_path)
    # dataset_folder = os.path.join(os.path.dirname(file_dir), 'PanDataset')
    args = parse_args()
    train(
        train_dataset_folder=args.train_dataset_folder,
        valid_dataset_folder=args.valid_dataset_folder,
        dataset_name=args.dataset_name,
        ms_num_channel=args.ms_num_channel,
        pan_num_channel=args.pan_num_channel,
        patch_size=args.patch_size,
        image_size=args.image_size,
        schedule_type=args.schedule_type,
        n_steps=args.n_steps,
        max_iterations=args.max_iterations,
        batch_size=args.batch_size,
        lr_d=args.lr_d,
        show_recon=args.show_recon,
        pretrain_weight=args.pretrain_weight,
        pretrain_iterations=args.pretrain_iterations,
        args=args,
    )