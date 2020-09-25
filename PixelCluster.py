# Setup prerequisites
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import PIL
from scipy.cluster.vq import kmeans, vq, whiten
import scipy

# Loading 28x28 pixels image data
im = PIL.Image.open("d9.png")
# Display loaded image
# Transfering into numpy array
im = np.array(im)
# Normalizing image pixel
im = im / 255
# Partition the image into number of M X M blocks
block = 7
# Normalizing Image
K = np.uint8(im.shape[1]/block)

# Feature consists of each image pixle color
# The 3D vector storing RGB data.
features = []

# Scanning through each row and column of the image
# Calculating the mean value of the pixel RGB values
for row in range(block):
    for col in range(block):
        red = np.mean(im[row*K:(row+1)*K, col*K:(col+1)*K,0])
        green = np.mean(im[row*K:(row+1)*K, col*K:(col+1)*K,1])
        blue = np.mean(im[row*K:(row+1)*K, col*K:(col+1)*K,2])
        features.append([red,green,blue])
# Transforming into numpy array
features = np.array(features)

# Calculating centroids and variance from Kmean 
centroids, variance = kmeans(features,3)
# Assigning observations to cluster
code, distance = vq(features, centroids)

# Creating image with cluster labels
codeim = code.reshape(partition,partition)
codeim = tf.constant(codeim)
print (codeim)
codeim = codeim[tf.newaxis,...,tf.newaxis]
print (codeim.shape)
# Resizing the image to render
codeim = tf.image.resize(codeim,[28,28], method='nearest')
fig = plt.figure()
ax = fig.add_subplot()
ax.imshow(codeim[0])
plt.show()
