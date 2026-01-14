import copy
import math
import os
from contextlib import suppress
from pathlib import Path
from typing import List, Optional, Type, Union
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple
import torch
from diffusers import UNet2DModel
from diffusers.models.unet_2d import UNet2DOutput
from diffusers.utils import randn_tensor
from pytorch_lightning import LightningModule
from torch import nn, optim
from torchmetrics import MeanMetric
from src.utils.transforms import apply_transforms, apply_inverse_transforms
import torch.nn.functional as F
from src.configuration import Config
from src.consistency_model.loss import PerceptualLoss


class Consistency(LightningModule):
    """ Consistency model implementation based on: https://github.com/openai/consistency_models """

    def __init__(
        self,
        config: Config,
        bins_min: int = 2,
        bins_max: int = 150,
        bins_rho: float = 7,
        loss_func: str = 'LPIPS',
        initial_ema_decay: float = 0.9,
        optimizer_type: Type[optim.Optimizer] = optim.RAdam,
        num_samples: int = 16,
        use_ema: bool = True,
        sample_seed: int = 0,
        **kwargs,
    ) -> None:

        """
        Args:
            config: Network configuration.
            bins_min: Minimum number of time steps.
            bins_max: Maximum number of time steps.
            bins_rho: Determines time boundaries.
            loss_func: Loss function.
            initial_ema_decay: Exponential average decay parameter.
            optimizer_type: Gradient decent optimizer.
            num_samples: Number of generated samples per batch.
            use_ema: Enables the EMA model for inference.
            sample_seed: Seed value of the random number generator.
        """

        super().__init__()
        
        self.save_hyperparameters(ignore=['loss_fn'])

        self.config = config

        model = UNet2DModel(
            in_channels=self.config.in_channels + 3,
            out_channels=self.config.out_channels + 1,
            block_out_channels=self.config.channels,
            down_block_types=self.config.down_block_types,
            up_block_types=self.config.up_block_types
        )

        #self.model.train()  # Explicitly set to training mode
        self.model = model
        self.model_ema = copy.deepcopy(model)
        self.image_size = self.config.sample_dimension

        self.model_ema.requires_grad_(False)

        if loss_func == "LPIPS":
            self.loss_fn = PerceptualLoss(net_type="squeeze")
            self.ts_loss_fn = PerceptualLoss(net_type="squeeze")  
        if loss_func == "MSE":
            self.loss_fn = nn.MSELoss(),
        else:
            print("loss function not defined.")

        self.optimizer_type = optimizer_type

        self.learning_rate = self.config.lr
        self.initial_ema_decay = initial_ema_decay

        self.data_std = self.config.data_std
        self.time_min = self.config.time_min 
        self.time_max = self.config.time_max
        self.clip = self.config.clip_output

        self.bins_min = bins_min
        self.bins_max = bins_max
        self.bins_rho = bins_rho

        self._train_loss_tracker = MeanMetric()
        self._val_loss_tracker = MeanMetric()
        self._bins_tracker = MeanMetric()
        self._ema_decay_tracker = MeanMetric()

        self.num_samples = num_samples
        self.use_ema = use_ema
        self.sample_seed = sample_seed
        self.sample_steps = 1
        self.validation_step_data = []
        self.L_low_regions = []
    def configure_optimizers(self):
        return self.optimizer_type(self.parameters(), lr=self.learning_rate)

    def forward(
        self,
        images: torch.Tensor,
        times: torch.Tensor,
        srf: torch.Tensor,
        ins: torch.Tensor,
        ):  # Add srf parameter
        """Forward pass with surface map conditioning."""
        return self._forward(self.model, images, times, srf, ins)  # Pass srf downstream

    def _forward(
        self,
        model: nn.Module,
        noisy_images: torch.Tensor,  # Already contains noised image+ts
        times: torch.Tensor,
        srf: torch.Tensor,
        ins: torch.Tensor,
        ):
        # Split noised input into image and ts components
        srf = srf.float().unsqueeze(1) if srf.dim() == 3 else srf.float()  # [B, 1, H, W]
        ins = ins.float().unsqueeze(1) if ins.dim() == 3 else ins.float()
        
        noisy_image = noisy_images[:, :self.config.in_channels].float()
        noisy_ts = noisy_images[:, self.config.in_channels:].float()
        
        # Model processes combined input
        out = model(
            torch.cat([noisy_image, noisy_ts, srf, ins], dim=1),
            times
        )
        
        # Split output
        image_out = out.sample[:, :self.config.out_channels]
        ts_out = out.sample[:, self.config.out_channels:]
        
        # Time-dependent coefficients
        skip_coef = self.data_std**2 / ((times - self.time_min).pow(2) + self.data_std**2)
        out_coef = self.data_std * times / (times.pow(2) + self.data_std**2).pow(0.5)
        
        # Consistency update for both components
        final_image = self.image_time_product(noisy_image, skip_coef) + \
                    self.image_time_product(image_out, out_coef)
        final_ts = self.image_time_product(noisy_ts, skip_coef) + \
                self.image_time_product(ts_out, out_coef)
        
        if self.clip:
            final_image = final_image.clamp(-1.0, 1.0)
            final_ts = final_ts.clamp(-1.0, 1.0)
            
        return final_image, final_ts

    def training_step(self, batch, *args, **kwargs):
        """Performs a single training step."""
        _bins = self.bins
        x, month, srf, ins, ts = batch
        
        # Ensure proper dimensions and data types
        x = x.unsqueeze(1).float() if x.dim() == 3 else x.float()
        srf = srf.unsqueeze(1).float()
        ins = ins.unsqueeze(1).float()
        ts = ts.unsqueeze(1).float()

        # Create noise with proper gradients
        noise_image = torch.randn_like(x, requires_grad=True)
        noise_ts = torch.randn_like(ts, requires_grad=True)

        # Timing setup
        timesteps = torch.randint(0, _bins-1, (x.size(0),), device=x.device)
        current_times = self.timesteps_to_times(timesteps, _bins)
        next_times = self.timesteps_to_times(timesteps+1, _bins)

        # Create noised inputs
        current_noise_image = x + self.image_time_product(noise_image, current_times)
        current_noise_ts = ts + self.image_time_product(noise_ts, current_times)
        next_noise_image = x + self.image_time_product(
            noise_image,
            next_times)
        next_noise_ts = ts + self.image_time_product(
            noise_ts,
            next_times)
        # Forward passes

        with torch.no_grad():
            target_image, target_ts = self._forward(
                self.model_ema,
                torch.cat([current_noise_image, current_noise_ts], dim=1),
                current_times,
                srf,
                ins,
            )


        concatenated_input = torch.cat([next_noise_image, next_noise_ts], dim=1)
        pred_image, pred_ts = self(concatenated_input, next_times, srf, ins)
        image_loss = self.loss_fn(pred_image, target_image)
        ts_loss = self.ts_loss_fn(pred_ts, target_ts)
        total_loss = image_loss + ts_loss


        # Logging

        self._train_loss_tracker(total_loss)
        self.log(
            "train_loss",
            self._train_loss_tracker,
            on_step=False,
            on_epoch=True,
            logger=True,
            prog_bar=True
        )

        return total_loss



    @torch.no_grad()
    def validation_step(self, images: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """Performs a single validation step with ts consistency."""
        _bins = self.bins
        x, month, srf, ins, ts = images
        srf = srf.float().unsqueeze(1)  # [B, 1, H, W]
        ins = ins.float().unsqueeze(1)  # [B, 1, H, W]
        ts = ts.float().unsqueeze(1)    # [B, 1, H, W]
    
        batch_size = x.shape[0]

        # Initialize validation data buffer
        if not hasattr(self, 'validation_step_data') or not isinstance(self.validation_step_data, list):
            self.validation_step_data = []
        if len(self.validation_step_data) < 16:
            self.validation_step_data.append((x, month, srf, ins, ts))

        # Add channel dimension to TS

        
        # Create noise for both components
        noise_image = torch.randn_like(x)  # [B, C, H, W]
        noise_ts = torch.randn_like(ts)    # [B, 1, H, W]

        # Timing setup
        timesteps = torch.randint(0, _bins - 1, (batch_size,), device=x.device).long()
        current_times = self.timesteps_to_times(timesteps, _bins)
        next_times = self.timesteps_to_times(timesteps + 1, _bins)

        # Create noised inputs with proper dimensions
        current_noise_image = x + self.image_time_product(noise_image, current_times)
        current_noise_ts = ts + self.image_time_product(noise_ts, current_times)
        
        next_noise_image = x + self.image_time_product(noise_image, next_times)
        next_noise_ts = ts + self.image_time_product(noise_ts, next_times)

        # Combined inputs
        current_combined = torch.cat([current_noise_image, current_noise_ts], dim=1)
        next_combined = torch.cat([next_noise_image, next_noise_ts], dim=1)

        # Forward passes
        with torch.no_grad():
            target_image, target_ts = self._forward(
                self.model_ema,
                current_combined,
                current_times,
                srf,
                ins,
            )



       
        concatenated_input = torch.cat([next_noise_image, next_noise_ts], dim=1)
        pred_image, pred_ts = self(concatenated_input, next_times, srf, ins)
        image_loss = self.loss_fn(pred_image, target_image)
        ts_loss = self.ts_loss_fn(pred_ts, target_ts)
        total_loss = image_loss + ts_loss

        self._val_loss_tracker(total_loss)
        self.log(
            "val_loss",
            self._val_loss_tracker,
            on_step=False,
            on_epoch=True,
            logger=True,
            prog_bar=True
        )


        return total_loss

    @torch.no_grad()
    def on_validation_epoch_end(self):
        """Plots generated examples with SRF conditioning and TS predictions."""
        if (self.validation_step_data is not None and 
            (self.config.plot_valid_samples or self.config.show_valid_samples_tensorboard)):
            
            # Concatenate validation data
            x_list, month_list, srf_list, ins_list, ts_list = zip(*self.validation_step_data)
            x = torch.cat(x_list, dim=0).squeeze(1).float()
            ins = torch.cat(ins_list, dim=0)
            srf = torch.cat(srf_list, dim=0)
            month = torch.cat(month_list, dim=0)
            ts_gt = torch.cat(ts_list, dim=0)  # Rename to ts_gt for clarity

            # Random subset selection
            if x.shape[0] > 5:
                idx = np.random.choice(x.shape[0], 5, replace=False)
                x, month, srf, ins, ts_gt = x[idx], month[idx], srf[idx], ins[idx], ts_gt[idx]

            # Generate predictions (returns both images and ts)
            pred_images, pred_ts = self.sample(
                num_samples=x.shape[0],
                steps=10,
                month=month,
                srf_map=srf,
                ins_map=ins,
                x_image_size=x.shape[-2],
                y_image_size=x.shape[-1]
            )
            
            # Create plots with both TS predictions and ground truth
            self.create_plot(
                x_gt=x, 
                pred_images=pred_images,
                ts_gt=ts_gt,
                ts_pred=pred_ts,
                month=month,
                ins=ins,
                srf=srf
            )

        self.validation_step_data = None

    def create_plot(self, 
                x_gt: torch.Tensor, 
                pred_images: torch.Tensor,
                ts_gt: torch.Tensor,
                ts_pred: torch.Tensor,
                month: torch.Tensor, 
                ins: torch.Tensor, 
                srf: torch.Tensor):
        """Plots samples with TS predictions and ground truth."""
        n_samples = len(x_gt)
        fig, axs = plt.subplots(4, n_samples, figsize=(15, 8))

        # Plot ground truth images
        for i in range(n_samples):
            img = x_gt[i].squeeze().cpu().numpy()
            axs[0,i].imshow(img, vmin=-1, vmax=1)
            axs[0,i].set_title(f'GT Image {i}\nMonth: {month.cpu()[i]}', fontsize=8)

        # Plot predicted images
        for i in range(n_samples):
            pred_img = pred_images[i,0].squeeze().cpu().numpy()
            axs[1,i].imshow(pred_img, vmin=-1, vmax=1)
            axs[1,i].set_title(f'Pred Image {i}', fontsize=8)

        # Plot predicted TS
        for i in range(n_samples):
            ts_img = ts_pred[i].squeeze().cpu().numpy()
            axs[2,i].imshow(ts_img, vmin=-3, vmax=3)
            axs[2,i].set_title(f'Pred TS {i}', fontsize=8)

        # Plot ground truth TS
        for i in range(n_samples):
            gt_ts_img = ts_gt[i].squeeze().cpu().numpy()
            axs[3,i].imshow(gt_ts_img, vmin=-3, vmax=3)
            axs[3,i].set_title(f'GT TS {i}', fontsize=8)

        # Formatting
        for ax_row in axs:
            for ax in ax_row:
                ax.axis('off')
        plt.tight_layout()

        # Save and log figures
        if self.config.plot_valid_samples:
            plt.savefig(f"/data/plots/samples_epoch_{self.current_epoch}_tss.png")
        
        # Create histograms
        self._create_histograms(x_gt, pred_images, ts_gt, ts_pred, srf)
        
        if self.logger:
            self.logger.experiment.add_figure('validation_samples', fig, self.current_epoch)
        
        plt.close()

    def _create_histograms(self, x_gt, pred_images, ts_gt, ts_pred, srf):
        """Creates comparison histograms for images and TS."""
        # Image histograms
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        
        # Mask out non-terrain areas
        mask = srf.squeeze().cpu().numpy() != -1
        img_gt_masked = x_gt.squeeze().cpu().numpy()[mask]
        img_pred_masked = pred_images.squeeze().cpu().numpy()[mask]
        
        sns.histplot(img_pred_masked, bins=100, binrange=[-2,2], 
                    stat="density", alpha=0.5, label="Predicted", ax=ax1)
        sns.histplot(img_gt_masked, bins=100, binrange=[-2,2], 
                    stat="density", alpha=0.5, label="Ground Truth", ax=ax1)
        ax1.set_title("Image Values (Masked)")
        ax1.legend()

        # TS histograms
        ts_gt_flat = ts_gt.squeeze().cpu().numpy()[mask]
        ts_pred_flat = ts_pred.squeeze().cpu().numpy()[mask]
        
        sns.histplot(ts_pred_flat, bins=100, binrange=[-4,4], 
                    stat="density", alpha=0.5, label="Predicted", ax=ax2)
        sns.histplot(ts_gt_flat, bins=100, binrange=[-4,4], 
                    stat="density", alpha=0.5, label="Ground Truth", ax=ax2)
        ax2.set_title("TS Values (Masked)")
        ax2.legend()

        plt.tight_layout()
        
        if self.config.plot_valid_samples:
            plt.savefig(f"/data/plots/histograms_epoch_{self.current_epoch}_tss.png")
        
        if self.logger:
            self.logger.experiment.add_figure('histograms', fig, self.current_epoch)
        
        plt.close()

    def optimizer_step(self, *args, **kwargs) -> None:
        super().optimizer_step(*args, **kwargs)
        self.ema_update()


    @torch.no_grad()
    def ema_update(self):
        param = [p.data for p in self.model.parameters()]
        param_ema = [p.data for p in self.model_ema.parameters()]

        torch._foreach_mul_(param_ema, self.ema_decay)
        torch._foreach_add_(param_ema, param, alpha=1 - self.ema_decay)

        self._ema_decay_tracker(self.ema_decay)
        self.log(
            "ema_decay",
            self._ema_decay_tracker,
            on_step=False,
            on_epoch=True,
            logger=True,
        )


    @property
    def ema_decay(self):
        return math.exp(self.bins_min * math.log(self.initial_ema_decay) / self.bins)


    @property
    def bins(self) -> int:
        return math.ceil(
            math.sqrt(
                self.trainer.global_step
                / self.trainer.estimated_stepping_batches
                * (self.bins_max**2 - self.bins_min**2)
                + self.bins_min**2
            )
        )


    def timesteps_to_times(self,
                           timesteps: torch.LongTensor,
                           bins: int):
        return (
            (
                self.time_min ** (1 / self.bins_rho)
                + timesteps
                / (bins - 1)
                * (
                    self.time_max ** (1 / self.bins_rho)
                    - self.time_min ** (1 / self.bins_rho)
                )
            )
            .pow(self.bins_rho)
            .clamp(0, self.time_max)
        )

    @torch.no_grad()
    def sample(
        self,
        num_samples: Optional[int] = 16,
        steps: Optional[int] = 1,
        x_image_size: Optional[int] = None,
        y_image_size: Optional[int] = None,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        use_ema: Optional[bool] = False,
        month: Optional[torch.Tensor] = None,
        srf_map: Optional[torch.Tensor] = None,
        ins_map: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Conditioned sampler returning both images and ts predictions"""
        # Handle input conditioning
        if srf_map is not None:
            if srf_map.dim() == 2:
                srf_map = srf_map.unsqueeze(0)
            if srf_map.shape[0] == 1:
                srf_map = srf_map.repeat(num_samples, 1, 1)
            elif srf_map.shape[0] != num_samples:
                raise ValueError(f"SRF map batch size mismatch: {srf_map.shape[0]} vs {num_samples}")



        # Determine output shapes
        img_channels = self.config.in_channels
        if x_image_size and y_image_size:
            img_shape = (num_samples, img_channels, x_image_size, y_image_size)
            ts_shape = (num_samples, 1, x_image_size, y_image_size)
        else:
            img_shape = (num_samples, img_channels, *self.config.sample_dimension)
            ts_shape = (num_samples, 1, *self.config.sample_dimension)

        # Initialize with combined noise
        time = torch.tensor([self.time_max], device=self.device)
        img_noise = randn_tensor(img_shape, generator=generator, device=self.device) * time
        ts_noise = randn_tensor(ts_shape, generator=generator, device=self.device) * time
        combined_noise = torch.cat([img_noise, ts_noise], dim=1)

        # Initial forward pass
        model = self.model_ema if use_ema else self.model
        pred_images, pred_ts = self._forward(
            model,
            combined_noise,
            time,
            srf=srf_map,
            ins=ins_map,
        )

        # Multi-step sampling
        if steps > 1:
            _timesteps = list(reversed(range(0, self.bins_max, self.bins_max // steps - 1)))[1:]
            _timesteps = [t + self.bins_max // ((steps - 1) * 2) for t in _timesteps]
            times = self.timesteps_to_times(torch.tensor(_timesteps, device=self.device), bins=150)

            for time in times:
                # Add noise to both components
                img_noise = randn_tensor(img_shape, generator=generator, device=self.device)
                ts_noise = randn_tensor(ts_shape, generator=generator, device=self.device)
                combined_noise = torch.cat([
                    pred_images + math.sqrt(time.item()**2 - self.time_min**2) * img_noise,
                    pred_ts + math.sqrt(time.item()**2 - self.time_min**2) * ts_noise
                ], dim=1)

                # Forward pass
                pred_images, pred_ts = self._forward(
                    model,
                    combined_noise,
                    time[None],
                    srf=srf_map,
                    ins=ins_map,
                )

        return pred_images, pred_ts
    def float_to_bin(self, t_float: float, bins: int):
        # 1) lift into the 1/ρ power‐domain
        u      = t_float ** (1/self.bins_rho)
        u_min  = self.time_min ** (1/self.bins_rho)
        u_max  = self.time_max ** (1/self.bins_rho)

        # 2) how far between min and max?
        frac   = (u - u_min) / (u_max - u_min)

        # 3) scale to [0 … bins-1] and round
        idx    = int(round(frac * (bins - 1)))

        # 4) clamp just in case of numerical over/underflow
        idx    = max(0, min(idx, bins - 1))
        return idx

    def merge_lowres_blocks(self,
                            land_counts: torch.Tensor,
                            threshold: int = 16*16
    ) -> torch.LongTensor:
        """
        Returns `regions`  (H_lr × W_lr) where the *entire right-most column* is
        always merged with the column just to its left if it contains land.
        """
        H, W   = land_counts.shape
        device = land_counts.device
        assigned = torch.zeros((H, W), dtype=torch.bool, device=device)
        regions  = -torch.ones((H, W), dtype=torch.long, device=device)
        region_id = 0
        neigh = [(1,0),(-1,0),(0,1),(0,-1)]

        for i in range(H):
            for j in range(W):
                if assigned[i, j]:
                    continue

                # ────────────────────────────────────────────────
                # force a two-column seed if we are at right edge
                # ────────────────────────────────────────────────
                if j == W - 1:          # right margin → always pull (i, j-1)
                    start_cells = [(i, j-1), (i, j)]
                else:
                    start_cells = [(i, j)]

                blocks, total = [], 0
                frontier = set()

                # initialise with the start_cells
                for si, sj in start_cells:
                    if assigned[si, sj]:
                        continue
                    assigned[si, sj] = True
                    blocks.append((si, sj))
                    total += int(land_counts[si, sj].item())

                # standard region-growing
                for si, sj in blocks:
                    for di, dj in neigh:
                        ni, nj = si+di, sj+dj
                        if (0 <= ni < H) and (0 <= nj < W) and not assigned[ni, nj]:
                            frontier.add((ni, nj))

                while total < threshold and frontier:
                    bi, bj = max(frontier, key=lambda x: land_counts[x].item())
                    frontier.remove((bi, bj))
                    assigned[bi, bj] = True
                    blocks.append((bi, bj))
                    total += int(land_counts[bi, bj].item())

                    for di, dj in neigh:
                        ni, nj = bi+di, bj+dj
                        if (0 <= ni < H) and (0 <= nj < W) and not assigned[ni, nj]:
                            frontier.add((ni, nj))

                # label the region
                for bi, bj in blocks:
                    regions[bi, bj] = region_id
                region_id += 1

        return regions


    @torch.no_grad()
    def sample_conditional(
        self,
        conditioning: torch.Tensor,
        x_image_size: int,
        y_image_size: int,
        *,
        steps: int = 10,
        sample_times: List = [None],
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        use_ema: bool = False,
        month: Optional[torch.Tensor] = None,
        srf_map: Optional[torch.Tensor] = None,
        ins_map: Optional[torch.Tensor] = None,
        ts_map: Optional[torch.Tensor]  = None,
        # optional conservation / (de)-normalisation helpers
        constraints: bool = True,
        transform_stats_smb: Optional = None,
        transform_stats_ts: Optional = None
        ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Low-res conditioned sampler producing a high-res field (pred_images)
        and a companion TS field (pred_ts).  Only the image channels are forced
        to conserve their low-resolution regional sums.
        """
        # ---------------------------------------------------------------------
        # 0) … sanity-check and broadcast the static maps ---------------------
        # ---------------------------------------------------------------------
        B = conditioning.shape[0]
        num_steps        = steps      # total number of diffusion steps
        free_tail_steps  = 0                 # how many steps run unconstrained

        ref_mean, ref_std, ref_min, ref_max, log_mean, log_std = (
        transform_stats_smb.ref_mean,
        transform_stats_smb.ref_std,
        transform_stats_smb.ref_min,
        transform_stats_smb.ref_max,
        transform_stats_smb.log_mean,
        transform_stats_smb.log_std,
        )

        ref_mean_ts, ref_std_ts, ref_min_ts, ref_max_ts, log_mean_ts, log_std_ts = (
        transform_stats_ts.ref_mean,
        transform_stats_ts.ref_std,
        transform_stats_ts.ref_min,
        transform_stats_ts.ref_max,
        transform_stats_ts.log_mean,
        transform_stats_ts.log_std,
        )

        #print(conditioning.shape, ts_map.shape)
        def _broadcast(m: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
            if m is None:
                return None
            if m.dim() == 2:
                m = m.unsqueeze(0)
            if m.shape[0] == 1:
                m = m.repeat(B, 1, 1)
            elif m.shape[0] != B:
                raise ValueError(
                    f"Map batch size {m.shape[0]} must be 1 or match conditioning batch size {B}"
                )
            return m

        srf_map = _broadcast(srf_map).unsqueeze(1)
        ins_map = _broadcast(ins_map).unsqueeze(1)
        ts_map  = _broadcast(ts_map).unsqueeze(1)

        # land/sea mask (-1 means ocean in the training data)
        mask_hr = (srf_map != -1) if srf_map is not None else 1.0

        # ---------------------------------------------------------------------
        # 1) prepare initial time, shapes & noise -----------------------------
        # ---------------------------------------------------------------------
        t0 = self.time_max if sample_times[0] is None else float(sample_times[0])
        time = torch.tensor([t0], device=self.device, dtype=torch.float32)

        img_C = self.config.in_channels                # e.g. 1
        img_shape = (B, img_C, x_image_size, y_image_size)
        ts_shape  = (B, 1,    x_image_size, y_image_size)
        #print(img_shape)
        img_noise = randn_tensor(img_shape, generator=generator, device=self.device) * time
        ts_noise  = randn_tensor(ts_shape,  generator=generator, device=self.device) * time
        
        # ---------------------------------------------------------------------
        # 2) build the image part: conditioning + noise -----------------------
        # ---------------------------------------------------------------------
        conditioning_inv = apply_inverse_transforms(
            data      = conditioning,           # (B,H,W) or (B,C,H,W) later unsqueezed
             config=self.config,
            ref_s0=self.transform_stats_smb.s0,
            ref_mean=self.transform_stats_smb.mean_y,
            ref_std=self.transform_stats_smb.std_y

        )
        conditioning_ts_inv = apply_inverse_transforms(
            data      = ts_map,           # (B,H,W) or (B,C,H,W) later unsqueezed
            config=self.config,
            ref_s0=self.transform_stats_ts.s0,
            ref_mean=self.transform_stats_ts.mean_y,
            ref_std=self.transform_stats_ts.std_y

        )

        conditioning = conditioning.unsqueeze(1)      
        if constraints == True:
            B, C, H_hr, W_hr = conditioning.shape
            assert H_hr == x_image_size and W_hr == y_image_size, "Size mismatch"
            r = 16  # known upscale factor
            H_lr   = math.ceil(H_hr / r)   # 2
            W_lr   = math.ceil(W_hr / r)   # 2

            pc = torch.zeros((H_lr, W_lr), device=conditioning.device)

            for i in range(H_lr):
                hs = r if i < H_lr-1 else (H_hr - r*(H_lr-1))
                for j in range(W_lr):
                    ws = r if j < W_lr-1 else (W_hr - r*(W_lr-1))
                    pc[i,j] = hs*ws
            pixel_count_total = pc.unsqueeze(0).unsqueeze(0)  # [1,1,H_lr,W_lr]

            # land‐pixel counts per LR block
            mask_lr = (F.adaptive_avg_pool2d(mask_hr.float(), (H_lr, W_lr)) * pixel_count_total)
    
            pixel_count_land = mask_lr.clamp(min=1)   # avoid divide by zero


            # 1) compute the original low-res sums by sum‐pooling in r×r blocks
            #    avg_pool2d * (r*r) is equivalent to a sum‐pool
            L_low = F.adaptive_avg_pool2d(conditioning_inv, (H_lr, W_lr)) * pixel_count_total
            #    L_low has shape [B, C, H_lr, W_lr]
            L_low_T = F.adaptive_avg_pool2d(conditioning_ts_inv, (H_lr, W_lr)) * pixel_count_total

            land_counts = mask_lr[0,0]  # [H_lr, W_lr]
            regions     = self.merge_lowres_blocks(land_counts)
            num_regions = int(regions.max().item()) + 1
            
            #without region growing: 
            # H_lr, W_lr = land_counts.shape
            # regions = torch.arange(H_lr * W_lr, device=land_counts.device).view(H_lr, W_lr)
            # num_regions = H_lr * W_lr



            # regions: a [H_lr, W_lr] torch.LongTensor of region IDs

            L_low_regions = []
            L_low_regions_ts = []
            pixel_counts   = []
            for rid in range(num_regions):
                m = (regions == rid).float()  # [H_lr,W_lr]
                # sum L_low over this region
                sum_low = (L_low * m.unsqueeze(0).unsqueeze(0)).sum(dim=(2,3))  # [B, C]
                # sum of mask pixels in region
                sum_land = (pixel_count_land[0,0] * m).sum()                 # scalar
                L_low_regions.append(sum_low)
                pixel_counts.append(sum_land)  
            for rid in range(num_regions):
                m = (regions == rid).float()  # [H_lr,W_lr]
                # sum L_low over this region
                sum_low_ts = (L_low_T * m.unsqueeze(0).unsqueeze(0)).sum(dim=(2,3))  # [B, C]
                # sum of mask pixels in region
                sum_land = (pixel_count_land[0,0] * m).sum()                 # scalar
                L_low_regions_ts.append(sum_low_ts)
                pixel_counts.append(sum_land)       # [B,1,H,W]
        conditioning = conditioning * mask_hr + img_noise
        conditioning_ts = ts_map * mask_hr + ts_noise
        #img_field    = conditioning * mask_hr + img_noise     # land masked
        
        combined = torch.cat([conditioning, conditioning_ts], dim=1)    # [B,img_C+1,H,W]
        
        model = self.model_ema if use_ema else self.model
        pred_images, pred_ts = self._forward(
            model,
            combined,
            time,
            srf=srf_map,
            ins=ins_map,
        )

        # ---------------------------------------------------------------------
        # 3) build the multistep schedule -------------------------------------
        # ---------------------------------------------------------------------
        if steps > 1:
            # 1) capture the exact float you used for the very first noising
            t0      = float(sample_times[0])
            init_t  = torch.tensor([t0], device=self.device, dtype=torch.float32)

            # 2) find which bin that corresponds to
            start_bin = self.float_to_bin(t0, self.bins_max)

            # 3) same integer spacing you used in the fallback
            bin_step   = (self.bins_max // steps) - 1
            half_step  = self.bins_max // ((steps - 1) * 2)

            # 4) build the full reversed bin list and keep only bins ≤ start_bin
            all_bins = list(reversed(range(0, self.bins_max, bin_step)))
            usable   = [b for b in all_bins if b <= start_bin]

            # 5) drop the first one (we already applied init_t), then center each
            raw_bins     = usable[1:]
            centered_bin = [min(b + half_step, self.bins_max - 1) for b in raw_bins]

            # 6) map back to floats
            bins_tensor = torch.tensor(centered_bin, device=self.device, dtype=torch.long)
            float_ts    = self.timesteps_to_times(bins_tensor, bins=self.bins_max)

            # 7) build your final list (each is 1-D for the loop)
            times = [init_t] + [t.unsqueeze(0) for t in float_ts]
            times = times[1:]         # list of [1]-tensors
        else:
            times = []

        # ---------------------------------------------------------------------
        # 4) main loop --------------------------------------------------------
        # ---------------------------------------------------------------------
        for idx, time in enumerate(times):
            #print("IDX: ", idx, time)
            # 4-A) OPTIONAL regional-sum conservation on **images only**
            apply_constraints = constraints and (idx < num_steps - free_tail_steps)

            #print("apply constraints", apply_constraints)
            if apply_constraints:
                
                images_inv = apply_inverse_transforms(pred_images, config=self.config,
                            ref_s0=self.transform_stats_smb.s0,
                            ref_mean=self.transform_stats_smb.mean_y,
                            ref_std=self.transform_stats_smb.std_y
                    )
                L_pred = F.adaptive_avg_pool2d(images_inv, (H_lr, W_lr)) * pixel_count_total

                ts_inv = apply_inverse_transforms(pred_ts, config=self.config, ref_s0=self.transform_stats_ts.s0,
                ref_mean=self.transform_stats_ts.mean_y,
                ref_std=self.transform_stats_ts.std_y)
                L_pred_ts = F.adaptive_avg_pool2d(ts_inv, (H_lr, W_lr)) * pixel_count_total
                delta_regions = []
                delta_regions_ts = []
                for rid in range(num_regions):
                    sum_pred = (L_pred * (regions==rid).float().unsqueeze(0).unsqueeze(0)).sum((2,3))
                    delta_regions.append(L_low_regions[rid] - sum_pred)  # [B, C]
                    sum_pred_ts = (L_pred_ts * (regions==rid).float().unsqueeze(0).unsqueeze(0)).sum((2,3))
                    delta_regions_ts.append(L_low_regions_ts[rid] - sum_pred_ts)  # [B, C]

                # build a block‐level delta-per-pixel:
                #   for each block (i,j), lookup its region’s delta and divide by its size
                delta_block = torch.zeros_like(L_pred)  # [B,C,H_lr,W_lr]
                delta_block_ts = torch.zeros_like(L_pred_ts)
                for rid in range(num_regions):
                    mask_r = (regions == rid).float()    # [H_lr,W_lr]
                    dr     = delta_regions[rid].unsqueeze(-1).unsqueeze(-1)  # [B,C,1,1]
                    size_r = pixel_counts[rid]           # scalar
                    delta_block += dr * mask_r / size_r # broadcast into blocks

                    dr_ts     = delta_regions_ts[rid].unsqueeze(-1).unsqueeze(-1)  # [B,C,1,1]
                    size_r = pixel_counts[rid]           # scalar
                    delta_block_ts += dr_ts * mask_r / size_r  # broadcast into blocks
                
                # upsample with nearest to avoid seams
                corr = F.interpolate(delta_block, size=(H_hr,W_hr), mode='area')
    
        
                #small weight across real edges
                k = 49 #11
                σ = 10#5.0

                coords = torch.arange(k, device=corr.device) - (k//2)
                g1 = torch.exp(-0.5 * (coords/σ)**2)
                g1 = g1 / g1.sum()
                g2 = g1[:,None] * g1[None,:]            # 5×5
                kernel = g2.view(1,1,k,k).repeat(corr.shape[1],1,1,1)
                #corr_smooth = self._masked_gaussian_blur(corr, mask_hr.float(), kernel, k//2)
                

                corr_smooth = F.conv2d(corr, kernel, padding=k//2, groups=corr.shape[1])
                plots_dir='/data/plots/'
                
                corr_ts = F.interpolate(delta_block_ts, size=(H_hr,W_hr), mode='area')
                corr_ts = F.conv2d(corr_ts, kernel, padding=k//2, groups=corr_ts.shape[1])
               
                # upsample correction to HR
                inverse_ts = torch.clamp(ts_inv +corr_ts, max=0.0)
                # back to model space
                
                pred_images = apply_transforms(
                    images_inv + corr_smooth, config=self.config,
                    ref_s0=self.transform_stats_smb.s0,
                    ref_mean=self.transform_stats_smb.mean_y,
                    ref_std=self.transform_stats_smb.std_y
                ) * mask_hr
                pred_ts = apply_transforms(
                    inverse_ts,
                    config=self.config,
                    ref_s0=self.transform_stats_ts.s0,
                    ref_mean=self.transform_stats_ts.mean_y,
                    ref_std=self.transform_stats_ts.std_y

                ) * mask_hr
                # apply to image channels, mask to land
                #pred_images = (pred_inv + corr_tr) * mask_hr

            # 4-B) add scaled noise to **both** channels
            sigma = math.sqrt(time.item() ** 2 - self.time_min ** 2)
            img_noise = randn_tensor(img_shape, generator=generator, device=self.device)
            ts_noise  = randn_tensor(ts_shape,  generator=generator, device=self.device)

            combined = torch.cat([
                pred_images + sigma * img_noise,
                pred_ts     + sigma * ts_noise
            ], dim=1)

            # 4-C) denoise one step
            pred_images, pred_ts = self._forward(
                model, combined, time,
                srf=srf_map, ins=ins_map
            )
        # ---------------------------------------------------------------------
        # 5) return -----------------------------------------------------------
        # ---------------------------------------------------------------------
        return pred_images, conditioning, conditioning_ts, pred_ts



    @staticmethod
    def image_time_product(images: torch.Tensor, times: torch.Tensor):
        # Ensure 4D input [B, C, H, W]
        if images.dim() == 3:
            images = images.unsqueeze(1)  # Add channel dimension
        return torch.einsum("b c h w, b -> b c h w", images, times)