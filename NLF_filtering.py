import numpy as np 
import matplotlib.pyplot as plt
import pywt
import re
from block_matching import block_matching

def separating_rgb(patches):
    """Separates the 3D patches into 1D patches
    to apply the 1D wavelet"""
    red = patches[:,:,:,0]
    green = patches[:,:,:,1]
    blue = patches[:,:,:,2]
    return red, green, blue

def Gamma(q, tau):
    """Shrinkage Operator"""
    return q**3/(q**2+tau**2)

def building_gkj_array(patch_ind, look_up_table, patches, N1, N2):
    """For each S_j, we build a group gkj_tilde which is the 3D
    array of size N1xN1xN2 formed by stacking the blocks extracted
    from y_tilde_k"""
    gkj_tilde = np.zeros((N1,N1,N2))
    idx = look_up_table['patch ' + str(patch_ind)]
    for i in range(len(idx)):
        gkj_tilde[:,:,i] = patches[idx[i]]
    return gkj_tilde

def reconstruct(tau,coeffs):
    """Applies the shrinkage operator and then the 
    inverse wavelet transform"""
    coeffs_rec = [coeffs[0]]
    for i in range(1, len(coeffs)):
        coeffs_rec.append(Gamma(coeffs[i],tau))
    return np.clip(pywt.waverec(coeffs_rec, 'haar'), 0, 255)

def transform_haar_wavelet(gkj_tilde, tau):
    """Computes gkj_hat by applying the wavelet transform,
    the shrinkage operator and the inverse wavelet transform"""
    # gkj_tilde = building_gkj_array(patch_ind, look_up_table, patches)
    n, p = gkj_tilde.shape[0], gkj_tilde.shape[1]
    gkj_hat = np.zeros_like(gkj_tilde)
    for i in range(n):
        for j in range(p):
            coeffs = pywt.wavedec(gkj_tilde[i,j,:], 'haar', level=2)
            recon = reconstruct(tau, coeffs)
            gkj_hat[i,j,:] = recon
    return gkj_hat

def transform_over_all_img(look_up_table,patches,tau, N1, N2):
    """Applies the haar wavelet to all the image patches"""
    n = patches.shape[0]
    transform = []
    for i in range(n):
        gkj_tilde = building_gkj_array(i, look_up_table, patches, N1, N2)
        gkj_hat = transform_haar_wavelet(gkj_tilde, tau)
        transform.append(gkj_hat)
    return np.array(transform)

def weight_j(patch_ind, look_up_table, patches, tau, N1, N2):
    """Computes the weight defined in the article for a given gkj_tilde"""
    gkj_tilde = building_gkj_array(patch_ind, look_up_table, patches, N1, N2)
    n, p = gkj_tilde.shape[0], gkj_tilde.shape[1]
    summing = 0
    for i in range(n):
        for j in range(p):
            coeffs = pywt.wavedec(gkj_tilde[i,j,:], 'haar', level=2)
            for i in range(len(coeffs)):
                summing = np.linalg.norm(coeffs[i])**2
    wkj = summing/(summing+tau**2)
    return wkj**(-2)


def all_weights(look_up_table, patches, tau, N1, N2):
    """Computes all weights"""
    n = patches.shape[0]
    w = []
    for i in range(n):
        w.append(weight_j(i, look_up_table, patches, tau, N1, N2))
    return w

def inverse_look_up_table(patches, look_up_table):
    """Inverses the look-up table to have access for a given patch
    to all the patches this particular patch was similar to"""
    inv = {}
    n = patches.shape[0]
    for i in range(n):
        for patch in look_up_table:
            if i in look_up_table[patch]:
                num = int(re.findall(r'\d+', patch)[0])
                if ('patch '+str(i)) in inv.keys():
                    inv['patch '+str(i)] += [(num,np.where(look_up_table[patch] == i)[0][0])]
                else: 
                    inv['patch '+str(i)] = [(num,np.where(look_up_table[patch] == i)[0][0])]
    return inv

def new_patches(look_up_table, inv, patches, tau, N1, N2):
    """Computes the new patches using the found weights and gkj_hats"""
    weights = all_weights(look_up_table, patches, tau, N1, N2)
    all_transforms = transform_over_all_img(look_up_table,patches,tau, N1, N2)
    new_patches = []
    for patch in inv:
        summing = np.zeros((N1,N1))
        normalization = 0
        for pat in inv[patch]:
            (ind_patch, position) = pat
            new_patch = all_transforms[ind_patch,:,:,position]
            summing += weights[ind_patch]*new_patch
            normalization += weights[ind_patch]
        new_patches.append(summing / normalization)
    return np.array(new_patches)

def image_estimate(new_patches, img, N1):
    """Computes the denoised image thanks to the new computed patches"""
    n, p = img.shape[0], img.shape[1]
    new_image = np.zeros((n, p))
    n_10, p_10 = n//10, p//10
    cpt = 0
    for i in range(n_10+1):
        for j in range(p_10+1):
            if i < n_10 and j < p_10:
                new_image[i*N1:i*N1+N1, j*N1:j*N1+N1] = new_patches[cpt]
            elif i == n_10 and j < p_10:
                new_image[i*N1:n, j*N1:j*N1+N1] = new_patches[cpt][N1-(n-i*N1):,:]
            elif i < n_10 and j == p_10:
                new_image[i*N1:i*N1+N1, j*N1:p] = new_patches[cpt][:,N1-(p-j*N1):]
            elif i == n_10 and j == p_10:
                new_image[i*N1:n, j*N1:p] = new_patches[cpt][N1-(n-i*N1):,N1-(p-j*N1):]
            cpt += 1
    return new_image
    
def denoising_image(img, N1, tau, patches, look_up_table, inv, N2):

    # # inverse look-up table
    # inv = inverse_look_up_table(patches, look_up_table)

    # New patches
    new_patch = new_patches(look_up_table, inv, patches, tau, N1, N2)

    # Denoised image
    new_img = image_estimate(new_patch, img, N1)

    return new_img

def NLF(noisy_img, N1, tau, N2, patches, look_up_table, inv):
    
    # Building 1D patches for filtering
    red_patch, green_patch, blue_patch = separating_rgb(patches)

    # NLF_filtering
    red_estimate = denoising_image(noisy_img, N1, tau, red_patch, look_up_table, inv, N2)
    green_estimate = denoising_image(noisy_img, N1, tau, green_patch, look_up_table, inv, N2)
    blue_estimate = denoising_image(noisy_img, N1, tau, blue_patch, look_up_table, inv, N2)

    # Denoised 3D image
    image_est = np.zeros_like(noisy_img)
    image_est[:,:,0], image_est[:,:,1], image_est[:,:,2] = red_estimate, green_estimate, blue_estimate

    return image_est

if __name__ == '__main__':
    N1, tau, N2 = 10, 7.5, 32
    
    noisy_img = plt.imread('FFDNET_IPOL/noisy.png')
    patches, look_up_table = block_matching(noisy_img)
    inv = inverse_look_up_table(patches, look_up_table)
    
    img_est = NLF(noisy_img, N1, tau, N2, patches, look_up_table, inv)
    plt.imshow(img_est)
    plt.show()