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

def denoising_3D_image(noisy_img, tau):

    # wavelet transform
    coeffs = pywt.wavedec2(noisy_img, 'haar', axes=(0,1), level=2)

    # inverse wavelet transform
    coeffs_rec = [coeffs[0]]
    for i in range(1, len(coeffs)):
        coeffs_rec.append((Gamma(coeffs[i][0],tau), Gamma(coeffs[i][0],tau), Gamma(coeffs[i][0],tau)))
    return np.clip(pywt.waverec2(coeffs_rec, 'haar', axes = (0, 1)), 0, 255)

def plotting_3D_denoised_img(noisy_img, tau):

    denoised_img = denoising_3D_image(noisy_img, tau)

    plt.figure(figsize=(12,6))

    plt.subplot(1, 2, 1)
    plt.imshow(noisy_img)
    plt.axis('off')
    plt.title('Original with noise')

    plt.subplot(1, 2, 2)
    plt.imshow(denoised_img) #Ajout de ce coeff multiplicatif sinon les coeffs sont trop petits
    plt.axis('off')
    plt.title('Denoised')

    plt.show()

    plt.imshow(noisy_img-denoised_img)
    plt.show()

tau = 7.5
print(plotting_3D_denoised_img(plt.imread('FFDNET_IPOL/noisy.png'), tau))
