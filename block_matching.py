# In NN3D, the BM is done once on an estimate of y. We use the output
# y_tilde_1 of the CNNF.

# Expected result: A look-up-table of group coordinates S = {S1, ..., SN}
# Each Sj contains the coordinates of N2 mutually similar blocks of size N1xN1
# Each pixel in the image is covered by at least one block.

# Main steps of block matching:
#       - divide the image into patches of size N1xN1 (non overlapping patches
#         except at the edges because the image does not have a size which is a 
#         multiple of N1=10)
#       - Build the look up table (choose the N2 most similar blocks for every 
#         patch: I included the actual patch in the similar blocks to ensure the
#         presence of every patch in the look up table). The table is a dictionnary
#         of the form: S = {'patch 0': [0,24,37,...,458...], 'patch 1' : [1, ...]}
#         where the elements of the list are the indices of the patches most similar
#         to the one considered.

import numpy as np 
import matplotlib.pyplot as plt
import math
import pywt
import re

N1, N2 = 10, 32

def dividing_into_patches(img):
    """Divides an input image into patches of size N1xN1"""
    n, p = img.shape[0], img.shape[1]
    patches = []
    n_10, p_10 = n//10, p//10
    for i in range(n_10+1):
        for j in range(p_10+1):
            if i < n_10 and j < p_10:
                patches.append(img[i*N1:i*N1+10, j*N1:j*N1+10, :])
            elif i == n_10 and j < p_10:
                patches.append(img[n-10:n, j*N1:j*N1+10, :])
            elif i < n_10 and j == p_10:
                patches.append(img[i*N1:i*N1+10, p-10:p, :])
            elif i == n_10 and j == p_10:
                patches.append(img[n-10:n, p-10:p, :])
    return np.array(patches) 

def similarity_matrix(patches):
    """Computes the similarity matrix between the patches
    Returns a matrix of size N1xN1"""
    size = patches.shape[0]
    similarity = np.zeros((size,size))
    sim = 0

    # similarity computed with the Frobenius norm
    # leave the 0 similarity with itself to make sure every patch will be in S
    for i in range(size):
        for j in range(size):
            sim = np.linalg.norm(patches[i] - patches[j])
            similarity[i][j] = sim
    return similarity    

def building_the_look_up_table(similarity):
    """Builds a dictionnary by taking for each patch, the
    N2 most similar patches (including itself to make sure
    that all patches are in S"""
    S = {}
    n = similarity.shape[0]
    for i in range(n):
        idx = np.argpartition(similarity[i], N2)
        S['patch ' + str(i)] = idx[:N2]
    return S

def block_matching(img):

    # Dividing image into patches
    patches = dividing_into_patches(img)

    # Computing similarity matrix
    similarity = similarity_matrix(patches)

    # Building look-up-table
    look_up_table = building_the_look_up_table(similarity)

    return patches, look_up_table

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