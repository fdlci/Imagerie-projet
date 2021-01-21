import matplotlib.pyplot as plt
from tqdm import tqdm

from DRUNET import gaussian_noise_image, DRUNET, prepare_image_shape
from block_matching import block_matching
from NLF_filtering import inverse_look_up_table, NLF
from utilities.utils import PSNR


# Parameters:
# K=2, lambda_k = 1/k, tau_k = (1/4)sigma*lambda_k, N1 = 10, N2 = 32

def NN3D(img, sigma, K, saved_model, N1, N2):
    # Initialisation
    z = gaussian_noise_image(img, sigma / 255)
    y_previous = None

    # Iterative Denoising
    for k in tqdm(range(1, K + 1)):
        tau_k = (1 / 4) * sigma / k
        if k == 1:
            lambd = 1
            # FFDNET
            y_tilde = DRUNET(z, saved_model, lambd * sigma, add_noise=False)
            # Block Matching only on y_tilde_1
            patches, look_up_table = block_matching(y_tilde)
            # inverse look-up table
            inv = inverse_look_up_table(patches, look_up_table)
            # NLF
            y_hat = NLF(y_tilde, N1, tau_k, N2, patches, look_up_table, inv)
        else:
            lambd = 1 / k
            # Convex Combination
            z_hat = lambd * z + (1 - lambd) * y_previous
            # FFDNET
            y_tilde = DRUNET(z_hat, saved_model, lambd * sigma, add_noise=False)
            # NLF
            y_hat = NLF(y_tilde, N1, tau_k, N2, patches, look_up_table, inv)
        y_previous = y_hat
    return y_hat


if __name__ == '__main__':
    img_path = 'FFDNET_IPOL/input.png'
    img = plt.imread(img_path)
    img = prepare_image_shape(img)
    saved_model = "DRUNET_DPIR/drunet_color.pth"
    sigma, K, N1, N2 = 30, 2, 10, 32
    denoised_image = NN3D(img, sigma, K, saved_model, N1, N2)

    psnr = PSNR(prepare_image_shape(img), denoised_image/255, peak=1)
    print(psnr)

    plt.imshow(denoised_image)
    plt.savefig('Images/NN3D_DRU_30.png')
    plt.show()
