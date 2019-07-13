from pymli.models.ae import AutoEncoder
from pymli.visualization.graphs import scatter2d

import matplotlib.pyplot as plt
from keras.datasets import mnist
import numpy as np
from scipy.stats import norm
from sklearn.decomposition import PCA


original_dim = 784
intermediate_dim = 256
latent_dim = 2
batch_size = 128
epochs = 10

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(-1, original_dim) / 255.
x_test = x_test.reshape(-1, original_dim) / 255.

ae = AutoEncoder(intermediate_dim=intermediate_dim, latent_dim=latent_dim, epochs=epochs)
ae.fit(x_train)

n = 15
quantile_min = 0.01
quantile_max = 0.99
img_rows, img_cols = 28, 28

if latent_dim == 2:
    z1 = norm.ppf(np.linspace(quantile_min, quantile_max, n))
    z2 = norm.ppf(np.linspace(quantile_max, quantile_min, n))
    z_grid = np.dstack(np.meshgrid(z1, z2))
    z_grid = z_grid.reshape(n*n, latent_dim)

    x_pred_grid = ae.decoder_.predict(z_grid).reshape(n, n, img_rows, img_cols)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(np.block(list(map(list, x_pred_grid))), cmap='gray')
    ax.set_xticks(np.arange(0, n*img_rows, img_rows) + .5 * img_rows)
    ax.set_xticklabels(map('{:.2f}'.format, z1), rotation=90)
    ax.set_yticks(np.arange(0, n*img_cols, img_cols) + .5 * img_cols)
    ax.set_yticklabels(map('{:.2f}'.format, z2))
    ax.set_xlabel('$z_1$')
    ax.set_ylabel('$z_2$')
    plt.show()

z_test = ae.encoder_.predict(x_test)
if latent_dim > 2:
    z_test = PCA(n_components=2).fit_transform(z_test)
scatter2d(z_test[:, 0], z_test[:, 1], cs=y_test)
