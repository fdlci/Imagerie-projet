import matplotlib.pyplot as plt
from FFDNET import gaussian_noise_image, FFDNET
from block_matching import block_matching
from NLF_filtering import inverse_look_up_table, NLF

# Parameters:
# K=2, lambda_k = 1/k, tau_k = (1/4)sigma*lambda_k, N1 = 10, N2 = 32

def NN3D(img, sigma, K, saved_model, N1, N2):

    # Initialisation
    z = gaussian_noise_image(img, sigma)
    y_previous = None

    # Iterative Denoising
    for k in range (1, K+1):
        tau_k = (1/4)*sigma/k
        if k == 1:
            lambd = 1
            # FFDNET
            y_tilde = FFDNET(z, saved_model, lambd*sigma)
            # Block Matching only on y_tilde_1
            patches, look_up_table = block_matching(y_tilde)
            # inverse look-up table
            inv = inverse_look_up_table(patches, look_up_table)
            #NLF
            y_hat = NLF(y_tilde, N1, tau_k, N2, patches, look_up_table, inv)
        else:
            lambd = 1/k
            # Convex Combination
            z_hat = lambd*z + (1-lambd)*y_previous
            # FFDNET
            y_tilde = FFDNET(z_hat, saved_model, lambd*sigma)
            # NLF
            y_hat = NLF(y_tilde, N1, tau_k, N2, patches, look_up_table, inv) 
        y_previous = y_hat         
    return y_hat

img = 'FFDNET_IPOL/input.png'
saved_model = "FFDNET_IPOL/models/net_rgb.pth"
sigma, K, N1, N2 = 30, 2, 10, 32
denoised_image = NN3D(img, sigma, K, saved_model, N1, N2)

print(denoised_image)
plt.imshow(denoised_image)
plt.show()
