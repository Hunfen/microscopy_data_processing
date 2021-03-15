import numpy as np

from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch
from matplotlib import rcParams
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
rcParams['mathtext.fontset'] = 'stix'
from skimage.feature import peak_local_max
from skimage.feature import canny
from skimage import filters
from skimage import measure
from skimage.io import imread
from skimage.color import rgb2gray


# color map
cdict = {
    'red': [[0.0, 0.0, 0.0], [0.34, 168 / 256, 168 / 256],
            [0.67, 243 / 256, 243 / 256], [1.0, 1.0, 1.0]],
    'green': [[0.0, 0.0, 0.0], [0.34, 40 / 256, 40 / 256],
              [0.67, 194 / 256, 194 / 256], [1.0, 1.0, 1.0]],
    'blue': [[0.0, 0.0, 0.0], [0.34, 15 / 256, 15 / 256],
             [0.67, 93 / 256, 93 / 256], [1.0, 1.0, 1.0]]
}
gwyddion = LinearSegmentedColormap('gwyddion', segmentdata=cdict, N=256)

import inspect
import os
import string
import struct
import math
import re
import cv2 as cv
import sati
import Nanonis as nano

import ipywidgets as widgets


def sxm2figuer(fname=''):
    if fname == '':
        print('No path?')
    else:
        f = nano.read_file(fname)
        fig, axes = plt.subplots(2, 2, figsize=(6, 6))
        for i in range(len(f.data)):
            for j in range(len(f.data[i])):
                axes[i][j].imshow(f.data[i][j], cmap=gwyddion)
                axes[i][j].axis('off')
        fig.tight_layout()


def dat2figure(fname):
    if fname == '':
        print('No path?')
    else:
        f = nano.read_file(fname)
        data = f.data
        fig, axis = plt.subplots(1, 2, figsize=(10, 5), sharex=True)
        for i in range(len(data)):
            axis[0].plot(data[i][0], (data[i][4] + data[i][1]) / 2 * 1e12,
                         '.',
                         color='black')
            axis[1].plot(data[i][0], (data[i][2] + data[i][5]) / 2 * 1e12,
                         '.',
                         color='black')
        axis[0].axhline(y=0, ls='--', color='black')
        axis[1].axhline(y=0, ls='--', color='black')
        axis[0].set_xlabel(r'$Bias\ \mathrm{[V]}$', fontsize=14)
        axis[1].set_xlabel(r'$Bias\ \mathrm{[V]}$', fontsize=14)
        axis[0].set_ylabel(r'$Current\ \mathrm{[pA]}$', fontsize=14)
        axis[1].set_ylabel(r'$dI/dV\ \mathrm{[a.u.]}$', fontsize=14)
        fig.tight_layout()


def edge_detect(fname, sig):
    f = nano.read_file(fname)

    topo_fwd = f.data[0][0]
    topo_fwd = (topo_fwd - topo_fwd.min()) * 1e9
    edge = canny(topo_fwd, sigma=sig)

    fig, axis = plt.subplots(1, 2, figsize=(10, 5), sharex=True, sharey=True)
    im_handle = axis[0].imshow(topo_fwd, cmap=gwyddion)
    axis[0].axis('off')
    cbar = plt.colorbar(im_handle,
                        ax=axis[0],
                        orientation='vertical',
                        fraction=0.046,
                        pad=0.04,
                        use_gridspec=True)

    axis[1].imshow(edge, cmap='binary')

    fig.tight_layout()
