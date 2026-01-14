from dataclasses import asdict
import json
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from src.data import get_dataloaders
from src.configuration import Config
from src.utils.utils import MyProgressBar
import matplotlib.pyplot as plt



from itertools import islice
import numpy as np


def visualize_dataloader(dataloader, num_batches=1):
    """Visualizes data from the dataloader.

    Args:
        dataloader: The DataLoader to visualize data from.
        num_batches: Number of batches to visualize.
    """
    for i, data in enumerate(dataloader):
        if i >= num_batches:
            break

        # Assuming data is a tensor of shape (batch_size, channels, height, width)
        # Here, we visualize the first sample in the batch
        sample = data[0].squeeze(0)  # Remove channel dimension if it's 1

        plt.figure(figsize=(8, 6))
        plt.imshow(sample[0,0,:,:].numpy(), cmap='viridis', vmin=-1, vmax=1)
        plt.title(f'Batch {i+1} - Sample 1')
        plt.colorbar()
        plt.savefig(f"/data/test{i}.png")
        
def training(config: Config,
             model: pl.LightningModule,
             verbose=True,
             resume_ckpt_path= ""):
    """ Main training function.

    Args:
        config: Configuration containing hyperparameters and file paths
        model: The diffusion model to be trained
        verbose: Prints the trainig configuration
        resume_ckpt_path: Resumes training from a saved model checkpoint if provided.
    """

    pl.seed_everything(42, workers=True)

    if verbose:
        print(f'saving checkpoints at: {config.checkpoint_path}')
        print(json.dumps(asdict(config), sort_keys=False, indent=4))
    output_directory = '/data' #
    # save model checkpoints to disk
    checkpoints = ModelCheckpoint(
                    dirpath=output_directory + "/results_oversample_asinh/", 
                    save_top_k=10, 
                    monitor="val_loss")
    callbacks = [checkpoints]

    # custom progress bar
    progressbar = MyProgressBar()
    callbacks.append(progressbar)

    # Log training statistics to Tensorboard
    tb_logger = TensorBoardLogger(config.tensorboard_path,
                                  name=config.name,
                                  default_hp_metric=False,
                                  version=config.date_time)
    model.config.tensorboard_path = config.tensorboard_path

    # Initialize trainer instance
    trainer = pl.Trainer(max_epochs=config.n_epochs,
                         callbacks=callbacks,
                         deterministic=False,
                         check_val_every_n_epoch=config.check_val_every_n_epoch,
                         gradient_clip_val=config.grad_clip_norm,
                         logger=tb_logger,
                         accelerator='gpu',
                         devices=1,
                         sync_batchnorm=True,
                         strategy='ddp')

    # Get dataloaders
    dataloaders = get_dataloaders(config,
                                  n_workers=config.n_workers,
                                  use_mnist=config.use_mnist)


    def expected_summer_fraction(months, summer_months, boost):
        months = np.asarray(months)
        S = np.isin(months, summer_months).sum()
        C = len(months) - S
        # With weights: summer=boost, other=1
        return (boost * S) / (boost * S + C + 1e-9)

    # 1) Make sure we're actually using WeightedRandomSampler
    print("Sampler:", type(dataloaders["train"].sampler))

    # 2) Theoretical expectation
    summer_months = getattr(config, "summer_months", (6,7,8))
    boost = getattr(config, "summer_boost", 5.0)
    p_exp = expected_summer_fraction(dataloaders["train"].dataset.months, summer_months, boost)
    print(f"Expected summer fraction ≈ {p_exp:.3f}")

    # 3) Empirical: draw indices from the sampler and count months
    N = 10000
    idx = list(islice(iter(dataloaders["train"].sampler), N))
    months_drawn = np.asarray(dataloaders["train"].dataset.months)[idx]  # 1..12
    p_emp = np.isin(months_drawn, summer_months).mean()
    print(f"Empirical summer fraction over {N} draws = {p_emp:.3f}")

    visualize_dataloader(dataloaders['val'], num_batches=200)
    # Train the diffusion model
    trainer.fit(model,
                train_dataloaders=dataloaders['train'],
                val_dataloaders=dataloaders['val'],
                ckpt_path=resume_ckpt_path) 

    # Save best performing model
    trainer.save_checkpoint(output_directory + f"/results_oversample_asinh/best_{config.diffusion_model}_model.ckpt")

    return None