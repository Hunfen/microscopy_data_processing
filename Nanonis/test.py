from utility import *
from matplotlib import animation as anm

bias_7, specs_7, z_7 = mapping('/Users/hunfen/OneDrive/General Files/ゼミー/20210401/2021-03-15/Grid Spectroscopy017.3ds', full = True)
bias, specs, z = mapping('/Users/hunfen/OneDrive/General Files/ゼミー/20210401/2021-03-15/Grid Spectroscopy007.3ds', full = True)

fig, axis = plt.subplots(1, 2, figsize = (10, 5))

def update(i):
    if i != 0:
        plt.cla()                      # 現在描写されているグラフを消去

    axis[0].imshow(specs_7[i], cmap = 'viridis')
    axis[0].text(0, 5, str(round(bias_7[i] * 1e3 - 8, 1)) + r'$\mathrm{mV}$', fontsize = 'large', color = 'white')

    if i in [0, 67, 133, 200, 267, 333, 401]:
        axis[1].imshow(specs[i], cmap = 'viridis')
        axis[1].text(0, 5, str(round(bias[i] * 1e3 - 8, 1)) + r'$\mathrm{mV}$', fontsize = 'large', color = 'white')
    axis[0].axis('off')
    axis[1].axis('off')

fig.tight_layout()
ani = anm.FuncAnimation(fig, update, interval = 50, frames = 401, repeat_delay = 5000)
ani.save('/Users/hunfen/OneDrive/General Files/ゼミー/20210401/0T_mapping.gif', writer = 'Pillow');


import numpy as np

d = 0
delta_h = 0
Lambda = 1.0
pi = 3.1415926
0.5 * np.log((np.pi * 0.5) * (0.5 * np.log((np.pi * d) / (2 * Lambda)) + np.log(delta_h))) + np.log(delta_h) - d / Lambda