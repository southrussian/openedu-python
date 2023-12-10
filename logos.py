import numpy as np

scores_matrix = np.loadtxt('X_reduced_513.csv', delimiter=';')
weights_matrix = np.loadtxt('X_loadings_513.csv', delimiter=';')

image_size = (100, 100)

reconstructed_image = np.dot(scores_matrix[:, :10], weights_matrix[:, :10].T)
reconstructed_image = reconstructed_image.reshape(image_size)

import matplotlib.pyplot as plt

plt.imshow(reconstructed_image, cmap='gray')
plt.axis('off')
plt.show()
