import numpy as np

n = 240
psd = 3.8e-6
sigma_min = 0.01
sigma_max = 500

def noise_time(psd: float, N: float, sigma_min:float, sigma_max: float) -> float:
    """Computes the noise time for a given power spectral density wavenumber.

    Args:
        psd: The value of the PSD and k^*
        N: The dimension of the image/field
        sigma_min: The minimum noise standard deviation of the DM
        sigma_max: The maximim noise standard deviation of the DM

    Returns:
        The noise time to stop the the diffusion process for downscaling.

    """
    noise_time = 1/2*np.log(psd*N**2/sigma_min**2)/np.log(sigma_max/sigma_min)
    return noise_time


noise_time(psd, n, sigma_min, sigma_max)
