from typing import List, Optional, Type, Union, Tuple

import numpy as np
import tqdm
from pytorch_lightning import LightningModule
import torch
import matplotlib
matplotlib.use('Agg')  # Use 'Agg' backend for saving figures without a display
from dataclasses import asdict
import matplotlib.pyplot as plt
import seaborn as sns
from src.configuration import Config
from src.sde_model.ema import ExponentialMovingAverage
from src.sde_model.loss import VELoss
from src.sde_model.net import ScoreUNet

class SDEModel(LightningModule):

    def __init__(self,
                 config: Config,
                 verbose: bool = False) -> None: 
        super().__init__()
        """Includes training and inference sampling of a score based diffusion model.
        
        Args:
            config: Stores hyperparameters and file paths.
            verbose: Prints the training configuration.
        """

        self.save_hyperparameters(asdict(config), ignore=['model'])

        self.config = config
        self.config_checkpoint = None
        self.validation_step_data = []  # Initialize attribute


        if verbose: 
            print('Initializing SDEModel with Network resolution ='+str(config.network_resolution),' and channels='+str(config.channels))

        self.net = ScoreUNet(marginal_prob_std=self.marginal_prob_std,
                             channels=config.channels,
                             in_channels=config.in_channels,
                             out_channels=config.out_channels,
                             resolution=config.network_resolution,
                             down_block_types=config.down_block_types,
                             up_block_types=config.up_block_types 
                             )
        
        self.loss = VELoss(marginal_prob_std=self.marginal_prob_std)

        self.current_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if config.use_ema:
            self.ema = ExponentialMovingAverage(self.net.to(self.current_device).parameters(),
                                                decay=config.ema_rate)


    def configure_optimizers(self)-> torch.optim:
        return torch.optim.Adam(self.net.parameters(), lr=self.config.lr)


    def on_save_checkpoint(self, checkpoint):
        if self.config.use_ema:
            checkpoint['ema_state_dict'] = self.ema.state_dict()


    def training_step(self, x_in, batch_idx) -> torch.Tensor:
        """ Performs a single training step.

        Args:
            x_in: Input batch 

        Returns:
            Training loss
        """
        x, ins, srf = x_in  # Assuming the dataloader returns both x and month

        loss = self.loss(self.net, x, ins, srf)
    
        optimizer = self.optimizers()

        if self.config.warmup > 0:
            for g in optimizer.param_groups:
                  g['lr'] = self.config.lr * np.minimum(self.global_step/ self.config.warmup, 1.0)

        if self.config.use_ema: 
            self.ema.update(self.net.parameters())

        self.log("train_loss",
                  loss.detach(),
                  on_step=False,
                  on_epoch=True,
                  prog_bar=True,
                  logger=True,
                  sync_dist=True)

        return loss


    @torch.no_grad()
    def validation_step(self, x_in, batch_idx)-> torch.Tensor:
        """ Performs a single validation step.

        Args:
            x_in: Input batch 
            month: condition

        Returns:
            Validation loss
        """
        x, ins, srf = x_in  # Assuming the dataloader returns both x and month
        loss_dict = {}
        if not hasattr(self, 'validation_step_data'):
            self.validation_step_data = []
        if batch_idx < 12*8:
            self.validation_step_data.append(x_in)

        if self.config.use_ema:

            self.ema.store(self.net.parameters())
            self.ema.copy_to(self.net.parameters())
            loss = self.loss(self.net, x, ins, srf)
            self.ema.restore(self.net.parameters())
            loss_dict['val_loss'] = loss.detach()

        else:

            loss = self.loss(self.net, x, ins, srf)
            loss_dict['val_loss'] = loss

        loss_dict['gpu-alloc'] = torch.cuda.max_memory_allocated(self.device) / 2**30
        loss_dict['gpu-reserved'] = torch.cuda.max_memory_reserved(self.device) / 2**30

        self.log_dict(loss_dict,
                      on_step=False,
                      on_epoch=True,
                      prog_bar=True,
                      logger=True, 
                      sync_dist=True)

        return loss 

    def on_validation_epoch_end(self):
        """Plots generated example images at the end of every 5th validation epoch."""
        
        # Check if it's the 5th epoch (0, 5, 10, ...) and other conditions
        if (
            self.validation_step_data is not None
            and (self.config.plot_valid_samples or self.config.show_valid_samples_tensorboard)
            and (self.current_epoch % 5 == 0)  # Run every 5 epochs
        ):
            print("len(self.validation_step_data)", len(self.validation_step_data))

            x_list, ins_list, srf_list = zip(*self.validation_step_data)

            # Convert lists of tensors into a single tensor
            x = torch.cat(list(x_list), dim=0)  # Stack all validation inputs
            ins = torch.cat(list(ins_list), dim=0)
            srf = torch.cat(list(srf_list), dim=0)

            x = x.squeeze(1).float()  # Ensure correct shape
            print("shapes", x.shape, ins.shape, srf.shape)

            if x.shape[0] > 5:
                idx = np.random.choice(x.shape[0], 5, replace=False)
                x = x[idx, :]
                print("ins before", ins, "idx", idx)
                ins = ins[idx]
                print("ins after", ins)
                srf = srf[idx, :]
            print("shapes", x.shape, ins.shape, srf.shape)
            pred = self.euler_maruyama_sampler(
                batch_size=x.shape[0],
                num_steps=2000,
                month=ins,
                sample_dimension=(x.shape[-2], x.shape[-1]),
                srf_map=srf,
            )

            print("test", pred.shape, x.shape)
            x_inv, pred_inv = None, None

            self.create_plot(x, x_inv, pred, pred_inv, ins, srf)

        # Free memory after every validation epoch, regardless of whether processing happened
        self.validation_step_data = None



    def create_plot(self, x: torch.tensor, x_inv: torch.tensor, pred: torch.tensor, pred_inv: torch.tensor, ins: torch.tensor, srf: torch.tensor):
        """ 
        Plots samples at the end of the validation.
        Args:
            x: Target batch.
            x_inv: Inverse transformed target batch.
            pred: Prediction batch.
            pred_inv: Inverse transformed prediction batch.
        """

        n_samples =  len(x)  # Ensure we don't exceed batch size

        fig, axs = plt.subplots(2, n_samples, figsize=(15, 6))
        print(f"GT range: {x.min()} - {x.max()}")
        print(f"Pred range: {pred.min()} - {pred.max()}")
        # Plot ground truth
        for i in range(n_samples):
            img = x[i].squeeze().detach().cpu().numpy()
            axs[0,i].imshow(img,vmin=-1, vmax=1)
            axs[0,i].axis('off')
            axs[0,i].set_title(f'GT {i}, month = {ins.cpu()[i]}, \nm = {np.mean(img):.2f}pm{np.std(img):.2f}', fontsize=6, pad=5)

        # Plot predictions
        for j in range(n_samples):
            pred_img = pred[j,0,:].squeeze().detach().cpu().numpy()
            axs[1,j].imshow(pred_img, vmin=-1, vmax=1)
            axs[1,j].axis('off')
            axs[1,j].set_title(f'Pred {j}, m = {np.mean(pred_img):.2f}+-{np.std(pred_img):.2f}',  fontsize=6, pad=5)

        plt.tight_layout()
        if self.config.plot_valid_samples:
            plt.savefig(f"/data/plots/epoch_unc1_{self.current_epoch}.png")  # Save as PNG
        fig2, axs2 = plt.subplots(figsize=(6, 6))
        srf = srf.squeeze().detach().cpu().numpy()
        pred_hist = pred[:].squeeze().detach().cpu().numpy()[srf!=-1].flatten()
        x_hist = x[:].squeeze().detach().cpu().numpy()[srf!=-1].flatten()
        sns.histplot(pred_hist, bins=100, binrange=[-2,2], stat="density", alpha = 0.5, label = "prediction", ax=axs2)
        sns.histplot(x_hist, bins=100, binrange=[-2,2], stat="density", label = "x", alpha = 0.5, ax =axs2)
        axs2.legend()
        plt.tight_layout()
        if self.config.plot_valid_samples:
            plt.savefig(f"/data/plots/hist_epoch_unc1_{self.current_epoch}.png")  # Save as PNG
                    
        fig3, axs3 = plt.subplots(figsize=(6, 6))
        sns.histplot(pred[:].squeeze().detach().cpu().numpy().flatten(), bins=100, binrange=[-2,2], stat="density", alpha = 0.5, label = "prediction", ax=axs3)
        sns.histplot(x[:].squeeze().detach().cpu().numpy().flatten(), bins=100, binrange=[-2,2], stat="density", label = "x", alpha = 0.5, ax =axs3)
        axs3.legend()
        plt.tight_layout()

        if self.config.plot_valid_samples:
            plt.savefig(f"/data/plots/hist0_epoch_unc1_{self.current_epoch}.png")  # Save as PNG
                    
        

        if self.logger:
            self.logger.experiment.add_figure(
                'validation_samples',
                fig,
            global_step=self.current_epoch
            )
            self.logger.experiment.add_figure(
                'histograms',
                fig2,
            global_step=self.current_epoch
            )



        plt.close()


    def marginal_prob_std(self, t: torch.tensor):
        """Compute the standard deviation of $p_{0t}(x(t) | x(0))$.

        Variance exploding (VE): (hyperparameters: $\sigma_{min}$ and $\sigma_{max}$)

            $\sigma^2(t) = \sigma^2_{min} (\frac{\sigma_{max}}{\sigma_{min}})^{2t}$

        Args:    
          t: A vector of time steps.
    
        Returns:
          The standard deviation.
        """   

        std = self.config.sigma_min*(self.config.sigma_max/self.config.sigma_min)**t

        return std


    def diffusion_coeff(self, t: torch.tensor):
        """Compute the diffusion coefficient of the SDE.

        Variance exploding (VE):

            $g(t) = \sigma_{min} (\frac{\sigma_{max}}{\sigma_{min}})^{t} 
            \sqrt{2 \log{\frac{\frac{\sigma_{max}}{\sigma_{min}}}}}$

        Args:
          t: A vector of time steps.
    
        Returns:
          The vector of diffusion coefficients.
        """

        coeff = self.config.sigma_min * (self.config.sigma_max/self.config.sigma_min)**t  \
                    * np.sqrt(2 * np.log(self.config.sigma_max/self.config.sigma_min))
                    
        return coeff
    
    def generate_noise(self, 
                       t: torch.tensor, 
                       batch_size: int,
                       channels: int,
                       sample_dimension: Tuple[int, int]) -> torch.Tensor:

        x_init = torch.randn(batch_size, channels, sample_dimension[0], sample_dimension[1],
                            device=self.device) *self.marginal_prob_std(t)[:, None, None, None]

        return x_init 

    @torch.no_grad()
    def euler_maruyama_sampler2(
        self,
        batch_size: int,
        sample_dimension: Tuple[int, int],
        month: torch.Tensor,  
        srf_map: torch.Tensor, 
        ts: Optional[torch.Tensor]=None,
        eps: Optional[float]=1e-3,
        num_steps: Optional[int]=500, 
        stop_step: Optional[float]=np.inf,
        show_progress: Optional[bool]=False,
        init_x: Optional[torch.tensor]=None 
     ) -> torch.Tensor:
        """Generate samples from score-based models with the Euler-Maruyama scheme.

        Args:
            batch_size: Number of samples in gbatch
            sample dimention: Height and width of generated image
            eps: The smallest time step for numerical stability
            num_steps: number of SDE integration steps
            stop_step: Step number that terminates the SDE integration
            show_progress: show a progress bar for sampling
            init_x: Initial condition for the SDE, randomly generated if None is provided 
            months: Month condition 
        Returns:
            Generated samples
        """

        t = torch.ones(batch_size, device=self.device)

        if init_x is None: 
            init_x = self.generate_noise(t, batch_size, self.config.in_channels, sample_dimension) 

        time_steps = torch.linspace(1., eps, num_steps, device=self.device)
        step_size = time_steps[0] - time_steps[1]
        x = init_x

        if show_progress:
            time_steps = tqdm.notebook.tqdm(time_steps)

        for step, time_step in enumerate(time_steps):      

            batch_time_step = torch.ones(batch_size, device=self.device) * time_step
            g = self.diffusion_coeff(batch_time_step)
            mean_x = (g**2)[:, None, None, None] * self.net(x, batch_time_step, month, srf_map)

            mean_x = x + mean_x * step_size 

            noise = torch.sqrt(step_size) * g[:, None, None, None] * torch.randn_like(x)      

            x = mean_x + noise
            x_cpu = x.cpu()
            plt.imshow(x_cpu[0,0,:,:], vmin=-1, vmax=1, origin='lower')
            plt.colorbar()
            plt.savefig(f"/data/plots/gif/{step}.png", dpi=500)
            plt.clf()
            if step > stop_step:
                break

        return mean_x 

    @torch.no_grad()
    def euler_maruyama_sampler(
        self,
        batch_size: int,
        sample_dimension: Tuple[int, int],
        month: torch.Tensor,
        srf_map: torch.Tensor,
        ts: Optional[torch.Tensor]=None,
        num_steps: int = 500,
        snr: float = 0.16,
        eps: float = 1e-3,
        show_progress: bool = False,
        init_x: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Predictor-Corrector sampler with conditional inputs"""
        
        # Initialize noise
        t = torch.ones(batch_size, device=self.device)
        if init_x is None:
            init_x = self.generate_noise(t, batch_size, self.config.in_channels, sample_dimension)
        
        x = init_x
        time_steps = torch.linspace(1., eps, num_steps, device=self.device)
        step_size = time_steps[0] - time_steps[1]

        # Maintain your progress display convention
        if show_progress:
            time_steps = tqdm.notebook.tqdm(time_steps)

        for time_step in time_steps:
            batch_time_step = torch.ones(batch_size, device=self.device) * time_step

            # --- Corrector Step (Langevin MCMC) ---
            # Get score with conditional inputs
            grad = self.net(x, batch_time_step, month, srf_map)
            
            # Calculate adaptive step size
            grad_norm = torch.norm(grad.reshape(grad.shape[0], -1), dim=-1).mean()
            noise_norm = np.sqrt(np.prod(x.shape[1:]))
            langevin_step_size = 2 * (snr * noise_norm / grad_norm)**2
            
            # Apply correction
            x = x + langevin_step_size * grad + \
                torch.sqrt(2 * langevin_step_size) * torch.randn_like(x)

            # --- Predictor Step (Euler-Maruyama) ---
            g = self.diffusion_coeff(batch_time_step)
            x_mean = x + (g**2)[:, None, None, None] * \
                    self.net(x, batch_time_step, month, srf_map) * step_size
            x = x_mean + torch.sqrt(g**2 * step_size)[:, None, None, None] * torch.randn_like(x)

            # Keep your visualization logic if needed
            if False:  # Toggle this for debugging
                x_cpu = x.cpu()
                plt.imshow(x_cpu[0,0,:,:], vmin=-1, vmax=1, origin='lower')
                plt.colorbar()
                plt.savefig(f"path/{step}.png", dpi=500)
                plt.clf()

        # Return final mean without added noise
        return x_mean

    @torch.no_grad()
    def conditional_euler_maruyama_sampler(
        self,
        batch_size: int,
        sample_dimension: Tuple[int, int],
        month: torch.Tensor,  # Add month as a parameter
        srf_map: torch.Tensor, 
        ts: Optional[torch.Tensor]=None,
        init_x: Optional[torch.Tensor]=None,
        eps: Optional[float]=1e-3,
        num_steps: Optional[int]=500, 
        stop_step: Optional[int]=None,
        step_size: Optional[float]=None,
        show_progress: Optional[bool]=False,
        forward: Optional[bool]=False,
     ) -> torch.Tensor:
        """Generate samples from score-based models with the Euler-Maruyama scheme
        using a conditional starting point for the denoising.

        Args:
            batch_size: Number of samples in gbatch
            sample dimention: Height and width of generated image
            init_x: starting point for SDE integration
            eps: The smallest time step for numerical stability
            num_steps: Number of SDE integration steps
            stop_step: Step number that terminates the SDE integration
            step_size: Time step size for SDE integration
            init_x: Initial condition for the SDE, randomly generated if None is provided 
            forward: Runs SDE forward in time
            months: Month condition 
            ts: surface temperature field
        Returns:
            Generated samples
        """
        print("batch size coniditonal euler", batch_size)
        if init_x is None:
            t = torch.ones(batch_size, device=self.device)
            x = torch.randn(batch_size, 1, sample_dimension[0], sample_dimension[1], device=self.device) \
            * self.marginal_prob_std(t)[:, None, None, None] 
        else:
            mean_x = x = init_x

        time_steps = torch.linspace(1., eps, num_steps, device=self.device)
        if stop_step is not None:
            time_steps = time_steps[-stop_step:]

        if forward:
            time_steps = torch.linspace(1., eps, num_steps, device=self.device).flip(dims=(0,))
            if stop_step is not None:
                time_steps = time_steps[:stop_step]

        if step_size is None:
            step_size = abs(time_steps[0] - time_steps[1])

        if show_progress:
            time_steps = tqdm.notebook.tqdm(time_steps)
      
        for i, time_step in enumerate(time_steps):      

            batch_time_step = torch.ones(batch_size, device=self.device) * time_step
            g = self.diffusion_coeff(batch_time_step)

            if forward:
                mean_x = mean_x + torch.sqrt(step_size) * g[:, None, None, None] * torch.randn_like(mean_x)      
            else:
                mean_x = (g**2)[:, None, None, None] * self.net(x, batch_time_step, month, srf_map)

                mean_x = x + mean_x * step_size 

                noise = torch.sqrt(step_size) * g[:, None, None, None] * torch.randn_like(x)      

                x = mean_x + noise

        # Do not include any noise in the last sampling step.
        return mean_x

