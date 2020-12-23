# At each iteration k and for each Sj in S, a group g_tilde_kj is the 3D array of size N1xN1xN2
# The CNNF has already operated smoothing, therefore we only perform shrinkage, only along the 
# third dimension of the group, with respect to a 1D transform T1d of length N2

from dividing_rgb_img import *
from block_matching import *

def Gamma(q, tau):
    return q**3/(q**2+tau**2)

def building_gkj_array(patch_ind, look_up_table, patches):
    """Builds gkj_ttilde given the index of a patch"""
    gkj = np.zeros((N1,N1,N2))
    idx = look_up_table['patch ' + str(patch_ind)]
    for i in range(len(idx)):
        gkj[:,:,i] = patches[idx[i]]
    return gkj

def transform_haar_wavelet(gkj_tilde, k, sigma):
    """Performs the Haar transform for a given gkj_tilde"""
    tau_k = (1/4)*sigma/k
    a_k, b_k = pywt.dwt(gkj_tilde, 'haar')
    gam_a_k = Gamma(a_k, tau_k)
    gam_b_k = Gamma(b_k, tau_k)
    gkj_hat = pywt.idwt(gam_a_k,gam_b_k, 'haar')
    return gkj_hat

def transform_over_all_img(look_up_table,patches,k,sigma):
    """Computes the coeffictients gkj_hat for all patches"""
    n = patches.shape[0]
    transform = []
    for i in range(n):
        gkj_tilde = building_gkj_array(i, look_up_table, patches)
        gkj_hat = transform_haar_wavelet(gkj_tilde, k, sigma)
        transform.append(gkj_hat)
    return np.array(transform)

def weight_j(patch_ind, look_up_table, patches, tau):
    """Computes the weight wkj for a given patch_ind"""
    gkj_tilde = building_gkj_array(patch_ind, look_up_table, patches)
    a_k, b_k = pywt.dwt(gkj_tilde, 'haar')
    wkj = (np.linalg.norm(a_k**2/(a_k**2+tau**2)) + np.linalg.norm(b_k**2/(b_k**2+tau**2)))**(-2)
    return wkj



img = plt.imread('FFDNET IPOL\input.png')

patches, look_up_table = block_matching(img)

plt.figure(figsize=(12,6))

plt.subplot(1, 2, 1)
plt.imshow(img)
plt.axis('off')
plt.title('Original')

plt.subplot(1, 2, 2)
plt.imshow(patches[0])
plt.axis('off')
plt.title('Patch of size N1xN1 (top left of original)')

plt.show()

print(look_up_table['patch 0'])

