from dataclasses import dataclass, field
from typing import List
import argparse
import toml
import copy
from sklearn.model_selection import train_test_split
import numpy as np
@dataclass
class DataConfig:
    """ Contains the dataset configuration:
        - data paths
        - file names
        - transforms
    """

    out_path: str = '/data'

    data_path: str = field(init=False)
    results_path: str = field(init=False)
    tensorboard_path: str = field(init=False)
    checkpoint_path: str = field(init=False)
    config_path: str = field(init=False)

    target_filename: str = "CMIP6/SMB_ST_5km_MAR_merged_masked.nc" #training data
    esm_filename: str = "CMIP6/NorESM/NorESM-MM_hist_126.nc" #not used for training right now
    configuration_filename: str = None
    
    use_mnist: bool = False

    sample_dimension: int = field(default=(None, None))
    
    transforms: List = field(default_factory= lambda: ['asinh', 'standardize'])

    epsilon: float = 0.0001

    predict_variable: str = None

    n_workers: int = 0

    crop_data_latitude: int = field(default=(None, None))
    crop_data_longitude: int = field(default=(None, None))

    use_float16: bool = False

    #pad_input: int = field(default=(3, 3, 0, 0))
    pad_input: int = field(default=(0, 0, 0, 0))
    #pad_input: int = field(default=(1, 1, 1, 1))  #
    lazy: bool = False
    oversample_year_tail = True
    boost_year = 2080     # threshold
    year_boost = 10.0     # 4–20 is a sensible range

    oversample_summer: bool = True
    summer_months: tuple = (6, 7)   
    summer_boost: float = 50.0          # how much more often to draw summer samples

    def __post_init__(self):
        self.data_path = self.out_path + '/datasets'
        self.results_path = self.out_path + '/results_cond'
        self.tensorboard_path = self.out_path + '/tensorboard'
        self.checkpoint_path: str = self.out_path + '/checkpoints'
        self.config_path: str = self.out_path + '/config-files'


@dataclass
class TrainingConfig:
    """ Contains the training configuration:
        - data splits
        - hyperparameters: learning rate, batch size, number of epochs
    """
    np.random.seed(42)
    """
    train_start: int = 1950 # for training this values
    train_end: int = 2070 #2650 for training
    valid_start: int = 2070#2651
    valid_end: int = 2090#2800
    test_start: int = 2015 #2090#2021
    test_end: int = 2100

    """
    years = np.arange(1950, 2101)  # Including 2100

    # Shuffle the years
    np.random.shuffle(years)
    plot_valid_samples = True
    # Define split ratios
    #show_valid_samples_tensorboard = True
    plot_valid_samples = True
    train_ratio = 0.7
    valid_ratio = 0.2
    test_ratio = 0.15

    # Compute split indices
    n_total = len(years)
    n_train = int(n_total * train_ratio)
    n_valid = int(n_total * valid_ratio)

    # Assign years to sets
    #   random split
    train_years = years[:n_train] #  # #  
    valid_years = years[n_train:n_train + n_valid] 
    test_years = years[n_train + n_valid:] 

    # Print or use the splits
    print("Train years:", train_years)
    print("Validation years:", valid_years)
    print("Test years:", test_years)

    batch_size: int  = 8

    lr: float = 2e-4

    grad_clip_norm: float = 1

    warmup: float = 0

    check_val_every_n_epoch: int = 1


@dataclass
class DiffusionConfig:
    """ Contains the diffusion configuration:
        - UNet architechture: number of channels, up and down blocks
        - variance schedule parameters
        - eponential moving average (EMA) decay
    """

    name: str ='test_model_v2'

    network_resolution: int = None
    in_channels: int = 1
    out_channels: int = 1
    channels: tuple = field(default=(128, 128, 256, 256))
    down_block_types: List = field(default=("DownBlock2D",
                                            "DownBlock2D",
                                            "DownBlock2D",
                                            "AttnDownBlock2D"))
    up_block_types: List = field(default=("AttnUpBlock2D",
                                          "UpBlock2D",
                                          "UpBlock2D",
                                          "UpBlock2D",))

    diffusion_model: str = 'consistency' #'ve' # 've' or 'consistency'

    # SDE
    sigma_max: float = 600 
    sigma_min: float = 1e-3

    # CM
    data_std: float = 0.5
    time_min: float = 0.002
    time_max: float = 80
    clip_output: bool = False
    num_batches = 85

    ema_rate: float = 0.999
    use_ema: bool = True



@dataclass
class Config(DataConfig, TrainingConfig, DiffusionConfig):
    """ Wrapper class that combines the configuration. """
    None


def parse_command_line():
    """ Parses the command line options and overwrites the default configuration. """

    parser = argparse.ArgumentParser()

    parser.add_argument("-rw", "--n_worker",
                        help="Number of worker processes", type=int)

    parser.add_argument("-tf", "--target_filename",
                        help="Filename of the target .nc file", type=str)

    parser.add_argument("-ef", "--esm_filename",
                        help="Filename of the ESM .nc file", type=str)

    parser.add_argument("-ts", "--trainig_start",
                        help="Start year of training data", type=int)

    parser.add_argument("-te", "--trainig_end",
                        help="End year of training data", type=int)

    parser.add_argument("-vs", "--valid_start",
                        help="Start year of valid data", type=int)

    parser.add_argument("-ve", "--valid_end",
                        help="End year of validation data", type=int)

    parser.add_argument("-clat", "--crop_data_latitude", nargs='+',
                        help="List of indeces to crop field in latitude dimension", type=int)

    parser.add_argument("-clon", "--crop_data_longitude", nargs='+',
                        help="List of indeces to crop field in longitude dimension", type=int)

    parser.add_argument("-fp", "--use_float_16",
                        help="Convert data to float 16 precision", action='store_true')

    parser.add_argument("-n", "--name",
                        help="The name of the model", type=str)

    parser.add_argument("-dm", "--diffusion_model",
                        help="The name of the model", type=str)

    parser.add_argument("-ep", "--n_epochs",
                        help="Number of training epochs", type=int)

    parser.add_argument("-bs", "--batch_size",
                        help="Training batch size", type=int)

    parser.add_argument("-nr", "--network_resolution",
                        help="Neural network resolution", type=int)

    parser.add_argument("-c", "--channels", nargs='+',
                        help="List of network channels", type=int)

    parser.add_argument("-dbt", "--down_block_types", nargs='+',
                        help="List of network down sampling blocks" , type=str)

    parser.add_argument("-ubt", "--up_block_types", nargs='+',
                        help="List of network up sampling blocks" , type=str)

    parser.add_argument("-sgmin", "--sigma_min",
                        help="Minimum sigma value for stochastic process", type=float)

    parser.add_argument("-sgmax", "--sigma_max",
                        help="Maximum sigma value for stochastic process", type=float)

    parser.add_argument("-wu", "--warmup",
                        help="Number of warmup steps", type=float)

    parser.add_argument("-std", "--standardize",
                        help="Standardize data to zero mean and standard deviation of 1", action='store_true')

    parser.add_argument("-norm", "--normalize",
                        help="Normalize data to [-1, 1]", action='store_true')

    parser.add_argument("-ema", "--use_ema",
                        help="Use EMA model.", action='store_true')

    parser.add_argument("-cp", "--checkpoint_path",
                        help="Path to the model checkpoint .ckpt file", type=str)

    parser.add_argument("-of", "--output_filename",
                        help="Filename of the model output samples file", type=str)

    parser.add_argument("-ns", "--num_sde_steps",
                        help="Number of sde integration steps", type=int)

    parser.add_argument("-nb", "--num_batches",
                        help="Number of batches to sample", type=int)


    args = parser.parse_args()

    config = Config()
   
    if args.normalize and 'normalize_minus1_to_plus1' not in config.transforms:
        config.transforms.append('normalize_minus1_to_plus1')

    for arg in vars(args):
        value = getattr(args, arg)
        if value is not None:
            if type(value) == list:
                value = tuple(value)
            setattr(config, arg, value)

    return config
