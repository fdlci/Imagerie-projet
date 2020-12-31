import numpy as np 
import matplotlib.pyplot as plt
import math
import pywt
import re

# noisy_img = plt.imread('FFDNET_IPOL/noisy.png')
# red_noise_img = noisy_img[:,:,0]
# green_noise_img = noisy_img[:,:,1]
# blue_noise_img = noisy_img[:,:,2]

def Gamma(q, tau):
    return q**3/(q**2+tau**2)

def denoising_1D_image(noisy_img, tau):

    # wavelet transform
    coeffs = pywt.wavedec(noisy_img, 'haar', level=2)

    # inverse wavelet transform
    coeffs_rec = [coeffs[0]]
    for i in range(1, len(coeffs)):
        coeffs_rec.append(Gamma(coeffs[i],tau))
    return np.clip(pywt.waverec(coeffs_rec, 'haar'), 0, 255)

def reconstruct_3D_image(red, green, blue, tau):
    n, p = red.shape[0], red.shape[1]
    recon_img = np.zeros((n,p,3))
    recon_img[:,:,0] = denoising_1D_image(red, tau)
    recon_img[:,:,1] = denoising_1D_image(green, tau)
    recon_img[:,:,2] = denoising_1D_image(blue, tau)
    return recon_img

def plotting_recon_img(noisy_img, tau):
    red, green, blue = noisy_img[:,:,0], noisy_img[:,:,1], noisy_img[:,:,2]
    recon_img = reconstruct_3D_image(red, green, blue, tau)

    plt.figure(figsize=(12,6))

    plt.subplot(1, 2, 1)
    plt.imshow(noisy_img)
    plt.axis('off')
    plt.title('Original with noise')

    plt.subplot(1, 2, 2)
    plt.imshow(recon_img)
    plt.axis('off')
    plt.title('Denoised')

    plt.show()

    plt.imshow(noisy_img-recon_img)
    plt.show()

tau = 7.5
print(plotting_recon_img(plt.imread('FFDNET_IPOL/noisy.png'), tau))