# We have to separate the RGB values to have a 2D matrix and not a 3D

import matplotlib.pyplot as plt

def separating_rgb(img):
    red = img[:,:,0]
    green = img[:,:,1]
    blue = img[:,:,2]
    return red, green, blue

img = plt.imread('FFDNET IPOL\input.png')

red, green, blue = separating_rgb(img)

plt.figure(figsize=(12,6))

plt.subplot(1, 3, 1)
plt.imshow(red)
plt.axis('off')
plt.title('Red')

plt.subplot(1, 3, 2)
plt.imshow(green)
plt.axis('off')
plt.title('Green')

plt.subplot(1, 3, 3)
plt.imshow(blue)
plt.axis('off')
plt.title('Blue')

plt.show()