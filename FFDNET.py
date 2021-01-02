import numpy as np 
import matplotlib.pyplot as plt
from FFDNET_IPOL.models import FFDNet
from FFDNET_IPOL.utils import normalize
import torch
import torch.nn as nn
from torch.autograd import Variable

def gaussian_noise_image(img, sigma):
    """Create a noisy image from a clean image."""
    img_noise = np.random.normal(0, sigma, img.shape)
    return img + img_noise


def prepare_image(img, add_noise=True, noise_sigma=20):
    noise_sigma /= 255
    imorig = img.transpose(2, 0, 1)
    sh_im = imorig.shape
    imorig = np.expand_dims(imorig, 0)
    # Handle odd sizes
    expanded_h = False
    expanded_w = False
    sh_im = imorig.shape
    if sh_im[2]%2 == 1:
        expanded_h = True
        imorig = np.concatenate((imorig, \
            imorig[:, :, -1, :][:, :, np.newaxis, :]), axis=2)

    if sh_im[3]%2 == 1:
        expanded_w = True
        imorig = np.concatenate((imorig, \
            imorig[:, :, :, -1][:, :, :, np.newaxis]), axis=3)
    
    if img.dtype=="uint8":
        imorig = normalize(imorig)
    imorig = torch.Tensor(imorig)
    
    # Add noise
    if add_noise:
        noise = torch.FloatTensor(imorig.size()).\
                normal_(mean=0, std=noise_sigma)
        imnoisy = imorig + noise
    else:
        imnoisy = imorig.clone()
        
    # Test mode
    with torch.no_grad(): # PyTorch v0.4.0
        imorig, imnoisy = Variable(imorig.type(dtype)), \
                        Variable(imnoisy.type(dtype))
        nsigma = Variable(
           torch.FloatTensor([noise_sigma]).type(dtype))
    return imnoisy, nsigma


def variable_to_image(varim):
    """Converts a torch.autograd.Variable to an OpenCV image
    Args:
    varim: a torch.autograd.Variable
    """
    nchannels = varim.size()[1]
    if nchannels == 1:
        res = (varim.data.cpu().numpy()[0, 0, :]*255.).clip(0, 255).astype(np.uint8)
    elif nchannels == 3:
        res = varim.data.cpu().numpy()[0]
        res = res.transpose(1, 2, 0)
        res = (res*255.).clip(0, 255).astype(np.uint8)
    else:
        raise Exception('Number of color channels not supported')
    return res

imnoisy, nsigma = prepare_image(img, add_noise=True, noise_sigma=20)

def load_model(saved_model):
    Denoiser = FFDNet(num_input_channels=3)
    state_dict = torch.load(saved_model)
    Denoiser = nn.DataParallel(Denoiser, device_ids=[0]).cuda()
    Denoiser.load_state_dict(state_dict)
    Denoiser.eval()
    return Denoiser

saved_model = "FFDNET_IPOL/models/net_rgb.pth"

def FFDNET(img, saved_model, sigma):

    # Preparing image
    imnoisy, nsigma = prepare_image(img, add_noise=False, noise_sigma=sigma)

    # Denoised image   
    Denoiser = load_model(saved_model)
    with torch.no_grad():
        im_noise_estim = Denoiser(imnoisy, nsigma)

    outim = torch.clamp(imnoisy-im_noise_estim, 0., 1.)
    # noised_img = variable_to_image(imnoisy)
    denoised_img = variable_to_image(outim)

    return denoised_img