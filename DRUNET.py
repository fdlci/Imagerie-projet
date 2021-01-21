import numpy as np
import matplotlib.pyplot as plt
from DRUNET_DPIR.network_unet import UNetRes as DRUNet
from FFDNET_IPOL.utils import normalize
import torch
import skimage.io as io
import torch.nn as nn
from torch.autograd import Variable

from utilities.utils import PSNR

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


def gaussian_noise_image(img, sigma):
    """Create a noisy image from a clean image."""
    img_noise = np.random.normal(0, sigma, img.shape)
    return img + img_noise


def prepare_image_shape(img):
    dim1_4 = img.shape[0] % 4
    dim2_4 = img.shape[1] % 4
    if dim1_4 != 0:
        for _ in range(dim1_4):
            img = np.concatenate((img, np.expand_dims(img[-1, :, :], 0)), axis=0)
    if dim2_4 != 0:
        for _ in range(dim2_4):
            img = np.concatenate((img, np.expand_dims(img[:, -1, :], 1)), axis=1)
    return img


def prepare_image(img, add_noise=True, noise_sigma=20):
    dtype = torch.cuda.FloatTensor
    noise_sigma /= 255.
    img = prepare_image_shape(img)
    img_test = img.transpose(2, 0, 1)
    if img_test.max() > 50:
        img_test = normalize(img_test)
    if add_noise:
        img_test = gaussian_noise_image(img_test, noise_sigma)
    img_test = dtype(img_test[np.newaxis, :, :, :])
    img_test = torch.cat((img_test, torch.full((1, 1, img_test.shape[2], img_test.shape[3]),
                                               noise_sigma).type_as(img_test)), dim=1)
    return img_test


def variable_to_image(varim):
    """Converts a torch.autograd.Variable to an OpenCV image
    Args:
    varim: a torch.autograd.Variable
    """
    nchannels = varim.size()[1]
    if nchannels == 1:
        res = (varim.data.cpu().numpy()[0, 0, :] * 255.).clip(0, 255).astype(np.uint8)
    elif nchannels == 3:
        res = varim.data.cpu().numpy()[0]
        res = res.transpose(1, 2, 0)
        res = (res * 255.).clip(0, 255).astype(np.uint8)
    else:
        raise Exception('Number of color channels not supported')
    return res


def load_model(saved_model):
    Denoiser = DRUNet(in_nc=4, out_nc=3, nc=[64, 128, 256, 512], nb=4, act_mode='R', downsample_mode="strideconv",
                      upsample_mode="convtranspose")
    state_dict = torch.load(saved_model)
    Denoiser.load_state_dict(state_dict)
    Denoiser = Denoiser.to(device)
    Denoiser.eval()
    return Denoiser


def DRUNET(img, saved_model, sigma, add_noise=False):
    # Preparing image
    imnoisy = prepare_image(img, add_noise=add_noise, noise_sigma=sigma)

    # Denoised image   
    Denoiser = load_model(saved_model)
    with torch.no_grad():
        denoised_img = Denoiser(imnoisy)

    # noised_img = variable_to_image(imnoisy)
    denoised_img = variable_to_image(denoised_img)

    return denoised_img


if __name__ == '__main__':
    img = plt.imread('FFDNET_IPOL/noisy.png')
    saved_model = "DRUNET_DPIR/drunet_color.pth"
    sigma = 30
    img_noise = gaussian_noise_image(img, sigma / 255)
    img_denoised = DRUNET(img, saved_model, sigma, add_noise=False)
    psnr_out = PSNR(prepare_image_shape(img), img_denoised / 255, peak=1)
    psnr_in = PSNR(img, img_noise, peak=1)

    print(psnr_out)
    print(psnr_in)
    plt.imshow(img_denoised)
    plt.show()
