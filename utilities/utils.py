import numpy as np
import torch
import matplotlib.pyplot as plt
from torchvision import transforms


def PSNR(img1, img2, peak=255):
    """
    Return the PSNR Peak signal-to-noise ratio.
    
    :param peak: maximum possible value, 255 for RGB, 1 for BW
    """
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    return 10 * np.log10(peak ** 2 / mse)
