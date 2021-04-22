import numpy as np

from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch
from matplotlib import rcParams
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from numpy.lib.stride_tricks import as_strided
rcParams['mathtext.fontset'] = 'stix'
from skimage.feature import peak_local_max
from skimage.feature import canny
from skimage import filters
from skimage import measure
from skimage.io import imread
from skimage.color import rgb2gray

from scipy.optimize import curve_fit
from scipy.signal import convolve
from scipy.signal import deconvolve
from scipy.signal import savgol_filter

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
import Nanonis as nano

import ipywidgets as widgets


def sxm_path_list(folder_path):
    """
    parameter
    ---------
    foler_path : path of folder

    return
    ------
    path_ls : list of paths of files inside folder 
    """
    extensiton_list = ['.sxm']
    path_ls = []
    # path_dict = {}
    file_ls = os.listdir(folder_path)
    for i in range(len(file_ls)):
        if file_ls[i][-4:] not in extensiton_list:
            continue
        else:
            path_ls.append(os.path.join(folder_path, file_ls[i]))
    del file_ls
    path_ls.sort()
    return path_ls


def dat_path_list(folder_path):
    """
    parameter
    ---------
    foler_path : path of folder

    return
    ------
    path_ls : list of paths of files inside folder 
    """
    extensiton_list = ['.dat']
    path_ls = []
    # path_dict = {}
    file_ls = os.listdir(folder_path)
    for i in range(len(file_ls)):
        if file_ls[i][-4:] not in extensiton_list:
            continue
        else:
            path_ls.append(os.path.join(folder_path, file_ls[i]))
    del file_ls
    path_ls.sort()
    return path_ls


def grid_path_list(folder_path):
    """
    parameter
    ---------
    foler_path : path of folder

    return
    ------
    path_ls : list of paths of files inside folder 
    """
    extensiton_list = ['.3ds']
    path_ls = []
    # path_dict = {}
    file_ls = os.listdir(folder_path)
    for i in range(len(file_ls)):
        if file_ls[i][-4:] not in extensiton_list:
            continue
        else:
            path_ls.append(os.path.join(folder_path, file_ls[i]))
    del file_ls
    path_ls.sort()
    return path_ls


def file_sort(folder_path):
    """
    Parameter
    ---------
    folder_path : path of folder
    
    return
    ------
    sxm_ls : list of paths of '.sxm' files inside folder 
    dat_ls : list of paths of '.dat' files inside folder
    grid_ls : list of paths of '.3ds' files inside folder
    """
    # Initialization
    sxm_ls, dat_ls, grid_ls = [], [], []
    file_ls = os.listdir(folder_path)

    # Classification
    for i in range(len(file_ls)):
        if os.path.splitext(file_ls[i])[1] == '.sxm':
            sxm_ls.append(os.path.join(folder_path, file_ls[i]))
        elif os.path.splitext(
                file_ls[i])[1] == '.dat' and not (os.path.splitext(
                    file_ls[i])[0][:8] == 'Spectrum'):
            dat_ls.append(os.path.join(folder_path, file_ls[i]))
        elif os.path.splitext(file_ls[i])[1] == '.3ds':
            grid_ls.append(os.path.join(folder_path, file_ls[i]))
        else:
            continue
    #  Sort
    del file_ls
    sxm_ls.sort()
    dat_ls.sort()
    grid_ls.sort()

    return sxm_ls, dat_ls, grid_ls


def topo_extent(header):
    """
    Calculate position of topograph.
    
    Parameter
    ---------
    header : reformed header of .sxm
    
    Return
    ------
    position tuple (left[X], right[X], bottom[Y], top[Y]) 
    """

    center_X = header['SCAN_FILED']['X_OFFSET']
    center_Y = header['SCAN_FILED']['Y_OFFSET']
    range_X = header['SCAN_FILED']['X_RANGE']
    range_Y = header['SCAN_FILED']['Y_RANGE']
    return (center_X - range_X / 2, center_X + range_X / 2,
            center_Y - range_Y / 2, center_Y + range_Y / 2)


def grid_extent(header):
    """
    Calculate position of 3ds mapping.
    
    Parameter
    ---------
    header : reformed header of .3ds
    
    Return
    ------
    position tuple (left[X], right[X], bottom[Y], top[Y]) 
    """
    center_X = header['Grid settings']['X_OFFSET']
    center_Y = header['Grid settings']['Y_OFFSET']
    range_X = header['Grid settings']['X_RANGE']
    range_Y = header['Grid settings']['Y_RANGE']
    return (center_X - range_X / 2, center_X + range_X / 2,
            center_Y - range_Y / 2, center_Y + range_Y / 2)


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


# def edge_detect(fname, sig):
#     f = nano.read_file(fname)

#     topo_fwd = f.data[0][0]
#     topo_fwd = (topo_fwd - topo_fwd.min()) * 1e9
#     edge = canny(topo_fwd, sigma=sig)

#     fig, axis = plt.subplots(1, 2, figsize=(10, 5), sharex=True, sharey=True)
#     im_handle = axis[0].imshow(topo_fwd, cmap=gwyddion)
#     axis[0].axis('off')
#     cbar = plt.colorbar(im_handle,
#                         ax=axis[0],
#                         orientation='vertical',
#                         fraction=0.046,
#                         pad=0.04,
#                         use_gridspec=True)

#     axis[1].imshow(edge, cmap='binary')

#     fig.tight_layout();


def find_nearest(array, value):
    idx, val = min(enumerate(array), key=lambda x: abs(x[1] - value))
    return idx


def mapping(fname, bias=8e-3, full=False):
    f = nano.read_file(fname)
    dim = f.header['Grid dim']
    points = f.header['Points']

    sweep = np.linspace(round(f.Parameters[0][0], 3),
                        round(f.Parameters[0][1], 3), f.header['Points'])

    # z mapping
    z = np.zeros(dim)
    for row in range(dim[0]):
        for col in range(dim[1]):
            z[row][col] = f.Parameters[(dim[0] - row - 1) * dim[1] + col][4]

    # if full:
    #     sliced_data = np.zeros((f.header['Points'], f.header['Grid dim'][0], f.header['Grid dim'][1]))
    #     for i in range(f.header['Points']):
    #         for row in range(f.header['Grid dim'][0]):
    #             for col in range(f.header['Grid dim'][1]):
    #                 average = (f.data[f.header['Grid dim'][1] * row + col][1][i] + f.data[f.header['Grid dim'][1] * row + col][4][i]) / 2
    #                 sliced_data[i][f.header['Grid dim'][0] - row - 1][col] = average
    #     return sweep, sliced_data, z
    if full:
        sliced_data = np.zeros((points, dim[0], dim[1]))
        for i in range(points):  # points of sweep for each position
            for row in range(dim[0]):  # number of row for matplotlib
                for col in range(dim[1]):  # number of column for matplotlib
                    ave = (f.data[(dim[0] - row - 1) * dim[1] + col][1][i] +
                           f.data[(dim[0] - row - 1) * dim[1] + col][4][i]
                           ) / 2  # average value for fwd & bwd
                    sliced_data[i][row][col] = ave  # re-distribution of data
        return sweep, sliced_data, z

    # else:
    #     sliced_data = np.zeros((f.header['Grid dim'][0], f.header['Grid dim'][1]))
    #     bias_idx = find_nearest(sweep, bias)

    #     for row in range(f.header['Grid dim'][0]):
    #         for col in range(f.header['Grid dim'][1]):
    #             average = (f.data[f.header['Grid dim'][1] * row + col][1][bias_idx] + f.data[f.header['Grid dim'][1] * row + col][4][bias_idx]) / 2
    #             sliced_data[f.header['Grid dim'][0] - row - 1][col] = average
    #     return sweep[bias_idx], sliced_data, z
    else:
        sliced_data = sliced_data = np.zeros(dim)
        bias_idx = find_nearest(sweep, bias)

        for row in range(dim[0]):
            for col in range(dim[1]):
                ave = (f.data[(dim[0] - row - 1) * dim[1] + col][1][bias_idx] +
                       f.data[
                           (dim[0] - row - 1) * dim[1] + col][4][bias_idx]) / 2
                sliced_data[row][col] = ave
        return sweep[bias_idx], sliced_data, z


def dat_curve(topo_path, dat_path_ls):
    # 读取文件
    f = nano.read_file(topo_path)
    # 文件夹地址
    n_folder = os.path.splitext(topo_path)[0] + '/'
    # 新建文件夹
    if not os.path.exists(n_folder):
        os.makedirs(n_folder)
    # 初始化图像
    fig, axis = plt.subplots(figsize=(5, 5))

    for j in range(len(dat_path_ls)):
        f_l = nano.read_file(dat_path_ls[j])

        spec = (f_l.data[:, 2] + f_l.data[:, 5]) / 2
        IV = (f_l.data[:, 1] + f_l.data[:, 4]) / 2
        bias = f_l.data[:, 0]

        axis.plot(bias, spec, '.', alpha=0.5, color='r')
        axis.plot(bias, savgol_filter(spec, 5, 3), '-', color='black')
        axis.set_xlabel(r'$Bias\ \mathrm{[V]}$', fontsize=14)
        axis.set_ylabel(r'$dI/dV\ \mathrm{[a.u.]}$', fontsize=14)
        fig.savefig(n_folder + 'spec_' + '{}.png'.format(j))
        plt.cla()

        axis.plot(bias, IV, '.', alpha=0.5, color='r')
        axis.plot(bias, savgol_filter(IV, 5, 3), '-', color='black')
        axis.set_xlabel(r'$Bias\ \mathrm{[V]}$', fontsize=14)
        axis.set_ylabel(r'$I\ \mathrm{[A]}$', fontsize=14)
        fig.savefig(n_folder + 'IV_' + '{}.png'.format(j))
        plt.cla()

    axis.imshow(f.data[0][0], extent=topo_extent(f.header), cmap=gwyddion)
    axis.set_title('Set Point: {}V'.format(f.header['BIAS']) + ' ' +
                   f.header['CONTROLLER_INFO']['Setpoint'] +
                   '{}nm x {}nm'.format(
                       int(f.header['SCAN_FILED']['X_RANGE'] *
                           1e9), int(f.header['SCAN_FILED']['Y_RANGE'] * 1e9)))
    for i in range(len(dat_path_ls)):
        f_l = nano.read_file(dat_path_ls[i])
        axis.text(float(f_l.header['X (m)']), float(f_l.header['Y (m)']),
                  str(i))
    axis.axis('off')
    fig.savefig(n_folder + 'topo' + os.path.basename(topo_path)[-7:-4] +
                '.png')
    plt.cla()


def spec_map(f_path, topo_path):
    # 读取文件
    f = nano.read_file(f_path)
    f_t = nano.read_file(topo_path)
    # 文件夹地址
    n_folder = os.path.splitext(f_path)[0] + '/'
    # 新建文件夹
    if not os.path.exists(n_folder):
        os.makedirs(n_folder)
    # 初始化图像
    fig, axis = plt.subplots(figsize=(5, 5))

    axis.imshow(f_t.data[0][0], extent=topo_extent(f_t.header), cmap=gwyddion)
    axis.set_title('Set Point: {}V'.format(f_t.header['BIAS']) +
                   f_t.header['CONTROLLER_INFO']['Setpoint'])
    for i in range(len(f.data)):
        axis.text(f.Parameters[i][2], f.Parameters[i][3], str(i))
    axis.axis('off')
    fig.savefig(n_folder + 'topo' + os.path.basename(topo_path)[-7:-4] +
                '.png')
    plt.cla()


def ds_curve(f_path):
    # from Nanonis import read_file
    # 读取文件
    f = nano.read_file(f_path)
    # 判断文件完整与否
    if f.integrity:
        # 文件夹地址
        n_folder = os.path.splitext(f_path)[0] + '/'
        # 新建文件夹
        if not os.path.exists(n_folder):
            os.makedirs(n_folder)
        # 初始化图像
        fig, axis = plt.subplots(figsize=(5, 5))
        # -------------全部的平均--------------- #
        tol_specs = np.zeros(f.header['Points'])  # 初始化存储用np数组
        tol_IV = np.zeros(f.header['Points'])
        for i in range(len(f.data)):
            tol_specs += (f.data[i][1] + f.data[i][4]) / 2
            tol_IV += (f.data[i][0] + f.data[i][3])
        axis.plot(np.linspace(f.Parameters[0][0], f.Parameters[0][1],
                              f.header['Points']),
                  tol_specs / len(f.data),
                  color='black')
        axis.set_xlabel(r'$Bias\ \mathrm{[V]}$', fontsize=14)
        axis.set_ylabel(r'$dI/dV\ \mathrm{[a.u.]}$', fontsize=14)
        fig.savefig(n_folder + 'tol_ave_specs.png')
        plt.cla()  # 清理

        axis.plot(np.linspace(f.Parameters[0][0], f.Parameters[0][1],
                              f.header['Points']),
                  tol_IV / len(f.data) / 2,
                  color='black')
        axis.set_xlabel(r'$Bias\ \mathrm{[V]}$', fontsize=14)
        axis.set_ylabel(r'$I\ \mathrm{[A]}$', fontsize=14)
        fig.savefig(n_folder + 'tol_ave_IV.png')
        plt.cla()
        # ----------------单条curve+平滑------------------ #
        for j in range(len(f.data)):
            spec = np.zeros(f.header['Points'])
            IV = np.zeros(f.header['Points'])
            spec = (f.data[j][1] + f.data[j][4]) / 2
            axis.plot(np.linspace(f.Parameters[j][0], f.Parameters[j][1],
                                  f.header['Points']),
                      spec,
                      '.',
                      alpha=0.5,
                      color='r')
            axis.plot(np.linspace(f.Parameters[j][0], f.Parameters[j][1],
                                  f.header['Points']),
                      savgol_filter(spec, 5, 3),
                      '-',
                      color='black')
            axis.set_xlabel(r'$Bias\ \mathrm{[V]}$', fontsize=14)
            axis.set_ylabel(r'$dI/dV\ \mathrm{[a.u.]}$', fontsize=14)
            fig.savefig(n_folder + 'spec_' + '{}.png'.format(j))
            plt.cla()
            IV = (f.data[j][0] + f.data[j][3]) / 2
            axis.plot(np.linspace(f.Parameters[j][0], f.Parameters[j][1],
                                  f.header['Points']),
                      IV,
                      '.',
                      alpha=0.5,
                      color='r')
            axis.plot(np.linspace(f.Parameters[j][0], f.Parameters[j][1],
                                  f.header['Points']),
                      savgol_filter(IV, 5, 3),
                      '-',
                      color='black')
            axis.set_xlabel(r'$Bias\ \mathrm{[V]}$', fontsize=14)
            axis.set_ylabel(r'$I\ \mathrm{[A]}$', fontsize=14)
            fig.savefig(n_folder + 'IV_' + '{}.png'.format(j))
            plt.cla()
        # ---------------标准差统计------------------- #
        # error = np.zeros(f.header['Points'])
        # error = tol_specs - spec


def fast_generate(folder_path):
    sxm_ls, dat_ls, grid_ls = file_sort(folder_path)

    for i in range(len(sxm_ls)):
        f_topo = nano.read_file(sxm_ls[i])
        t_extent = topo_extent(f_topo.header)
        spec_idx = []
        if len(dat_ls) == 0:
            continue
        else:
            for j in range(len(dat_ls)):
                f_dat = nano.read_file(dat_ls[j])
                x = float(f_dat.header['X (m)'])
                y = float(f_dat.header['Y (m)'])
                if (x > t_extent[0]) and (x < t_extent[1]) and (
                        y > t_extent[2]) and (y < t_extent[3]):
                    spec_idx.append(dat_ls[j])
                else:
                    continue
        dat_curve(sxm_ls[i], spec_idx)

    if not (len(grid_ls) == 0):
        for i in range(len(grid_ls)):
            ds_curve(grid_ls[i])
            f_3ds = nano.read_file(grid_ls[i])
            x = f_3ds.Parameters[:, 2]
            y = f_3ds.Parameters[:, 3]
            for j in range(len(sxm_ls)):
                f_topo = nano.read_file(sxm_ls[j])
                t_extent = topo_extent(f_topo.header)
                x_confirmation = x > t_extent[0]
                x_confirmation_ = x < t_extent[1]
                y_confirmation = y > t_extent[2]
                y_confirmation_ = y < t_extent[3]
                if all(x_confirmation) and all(x_confirmation_) and all(
                        y_confirmation) and all(y_confirmation_):
                    spec_map(grid_ls[i], sxm_ls[j])

def map_pos(cOordinate = (0, 0), dim = (30, 30), dim_p = (50, 50),mod = 'nm'):
    x = dim[0] / dim_p[0]
    y = dim[1] / dim_p[1]
    if mod == 'pixel':
        return  int(cOordinate[0] + cOordinate[1] * dim_p[0])
    elif mod == 'nm':
        return int(cOordinate[0] // x + cOordinate[1] // y * dim_p[0])