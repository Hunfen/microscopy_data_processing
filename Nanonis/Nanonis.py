import math
import os
import re

import numpy as np


def read_file(f_path):
    if os.path.splitext(f_path)[1] == '.sxm':
        return __NanonisFile_sxm__(f_path)
    elif os.path.splitext(f_path)[1] == '.dat':
        return __NanonisFile_dat__(f_path)
    elif os.path.splitext(f_path)[1] == '.3ds':
        return __NanonisFile_3ds__(f_path)
    else:
        print("File type not supported.")
    # switch = {'.sxm': lambda x:__NanonisFile_sxm__(x),
    #           '.dat': lambda x:__NanonisFile_dat__(x)}
    # try:
    #     switch[os.path.splitext(f_path)[1]](f_path)
    # except KeyError as e:
    #     print('File type not supported.')


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


# def read_save_time(fname_ls=[]):
#     times = {}
#     for i in range(len(fname_ls)):
#         if fname_ls[i][-4:] == '.dat':
#             f = read_file(fname_ls[i])
#             times['{}'.format(i + 1).zfill(3)] = f.header['Saved Date']
#         elif fname_ls[i][-4:] == '.3ds':
#             f = read_file(fname_ls[i])
#             times['{}'.format(i + 1).zfill(3)] = f.header['End time']
#         else:
#             f = read_file(fname_ls[i])
#             times['{}'.format(i + 1).zfill(3)] = f.header['REC_TIME']
#     return times


class __NanonisFile_sxm__:
    """
    Nanonis File Class
    """
    def __init__(self, f_path):
        self.file_path = os.path.split(f_path)[0]
        self.fname = os.path.split(f_path)[1]
        self.__sxm_header_reader__(f_path)
        self.__sxm_header_reform__(self.raw_header)
        self.__sxm_channel_counts__(self.header)
        self.__sxm_data_reader__(f_path, self.header)

    def __sxm_header_reader__(self, f_path):
        key = ''
        contents = ''
        raw_header = {}
        with open(f_path, 'rb') as f:
            header_end = False
            while not header_end:
                line = f.readline().decode(encoding='utf-8', errors='replace')
                if re.match(':SCANIT_END:\n', line):
                    header_end = True
                # ':.+:' is the Nanonis .sxm file header entry regex, fuck.
                elif re.match(':.+:', line):
                    key = line[1:-2]  # Read header_entry
                    contents = ''  # Clear
                else:
                    contents += line
                    # remove EOL
                    raw_header[key] = contents.strip('\n')
        self.raw_header = raw_header

    def __sxm_header_reform__(self, raw_header):
        scan_info_float = [
            'ACQ_TIME', 'BIAS', 'Scan>lines', 'Scan>pixels/line',
            'Scan>speed backw. (m/s)', 'Scan>speed forw. (m/s)'
        ]
        table = ['DATA_INFO', 'Scan>Scanfield', 'Z-CONTROLLER']
        # The order of scan_field_key should not be changed
        scan_field_key = [
            'X_OFFSET', 'Y_OFFSET', 'X_RANGE', 'Y_RANGE', 'ANGLE'
        ]
        trash_bin = [
            'NANONIS_VERSION', 'REC_TEMP', 'SCANIT_TYPE', 'SCAN_ANGLE',
            'SCAN_OFFSET', 'SCAN_PIXELS', 'SCAN_RANGE', 'SCAN_TIME',
            'Scan>channels'
        ]
        scan_info_str = [
            'COMMENT', 'REC_DATE', 'REC_TIME', 'SCAN_DIR', 'SCAN_FILE'
        ]
        header_dict = {}
        keys = list(raw_header.keys())
        for i in range(len(keys)):
            if keys[i] in trash_bin:
                # Abandon redundant header entries
                continue
            else:
                header_dict[keys[i]] = raw_header[keys[i]]
        # Clear redundant space in scan_info_str
        for j in range(len(scan_info_str)):
            header_dict[scan_info_str[j]] = header_dict[
                scan_info_str[j]].strip(' ')
        # Transform scan_info_float from str to float
        for k in range(len(scan_info_float)):
            header_dict[scan_info_float[k]] = float(
                header_dict[scan_info_float[k]])
        # Transform table from str to dict
        # SCAN_FIELD
        scan_field = header_dict['Scan>Scanfield'].split(';')
        # SCAN_FIELD dict
        SCAN_FIELD = {}
        for m in range(len(scan_field_key)):
            SCAN_FIELD[scan_field_key[m]] = float(scan_field[m])
        # CHANNEL_INFO
        data_info = header_dict['DATA_INFO'].split('\n')
        DATA_INFO = []
        for row in data_info:
            DATA_INFO.append(row.strip('\t').split('\t'))
        # CHANNEL_INFO dict
        key_list = DATA_INFO[0][1:]
        channels = []
        values = []
        CHANNEL_INFO = {}
        for n in range(1, len(DATA_INFO)):
            channels.append(DATA_INFO[n][0])
            values.append(DATA_INFO[n][1:])
        for u in range(len(channels)):
            chan_dict = {}
            for v in range(len(key_list)):
                chan_dict[key_list[v]] = values[u][v]
            CHANNEL_INFO[channels[u]] = chan_dict
        # Z_CONTROLLER_INFO
        Z_Controller = header_dict['Z-CONTROLLER'].split('\n')
        Controller_config = []
        for row in Z_Controller:
            Controller_config.append(row.strip('\t').split('\t'))
        # CONTROLLER_INFO dict
        CONTROLLER_INFO = {}
        for o in range(len(Controller_config[0])):
            CONTROLLER_INFO[Controller_config[0][o]] = Controller_config[1][o]
        # Substitute table dict
        for p in range(len(table)):
            header_dict.pop(table[p])
        header_dict['SCAN_FILED'] = SCAN_FIELD
        header_dict['CONTROLLER_INFO'] = CONTROLLER_INFO
        header_dict['CHANNEL_INFO'] = CHANNEL_INFO
        self.header = header_dict

    def __sxm_channel_counts__(self, header):
        num_channels = 0
        dims = ()
        channels = list(header['CHANNEL_INFO'].keys())
        for i in range(len(channels)):
            if header['CHANNEL_INFO'][channels[i]]['Direction'] == 'both':
                num_channels += 2
            else:
                num_channels += 1
        dims = (num_channels, int(header['Scan>pixels/line']),
                int(header['Scan>lines']))
        self.num_channels = num_channels
        self.dims = dims

    def __sxm_data_reader__(self, f_path, header):
        # Find the start of data
        with open(f_path, 'rb') as f:
            read_all = f.read()
            offset = read_all.find('\x1A\x04'.encode(encoding='utf-8'))
            # print('found start at {}'.format(offset))
            f.seek(offset + 2)
            data = np.fromfile(f, dtype='>f')
        # dimension check
        check = False
        for i in range(len(header['CHANNEL_INFO'])):
            if header['CHANNEL_INFO'][list(
                    header['CHANNEL_INFO'].keys())[i]]['Direction'] == 'both':
                check = True
                break
        if check:
            data_shaped = data.reshape(
                (len(header['CHANNEL_INFO']), 2, int(math.sqrt(data.size / 4)),
                 int(math.sqrt(data.size / 4))))
        else:
            data_shaped = data.reshape(
                (len(header['CHANNEL_INFO']), 1, int(math.sqrt(data.size / 4)),
                 int(math.sqrt(data.size / 4))))
        for i in range(len(data_shaped)):
            for j in range(len(data_shaped[i])):
                if not j % 2 == 0:
                    data_shaped[i][j] = np.fliplr(data_shaped[i][j])
        self.data = data_shaped


class __NanonisFile_dat__:
    def __init__(self, f_path):
        self.file_path = os.path.split(f_path)[0]
        self.fname = os.path.split(f_path)[1]
        self.__dat_header_reader__(f_path)
        self.__dat_header_reform__(self.raw_header)
        self.__dat_data_reader__(f_path)

    def __dat_header_reader__(self, f_path):
        """

        """
        header = []
        with open(f_path, 'r') as f:
            header_end = False
            while not header_end:
                line = f.readline()
                if re.match(r'\[DATA\]', line):
                    header_end = True
                else:
                    header.append(line)
        self.raw_header = header

    def __dat_header_reform__(self, raw_header):
        header_str = []
        trash_bin = [
            '', 'Cutoff frq', 'Date', 'Experiment', 'Filter type',
            'Final Z (m)', 'Order', 'User'
        ]
        for i in range(len(raw_header)):
            header_str.append(raw_header[i].strip('\n'))
        for j in range(len(header_str)):
            header_str[j] = header_str[j].split('\t')
        key = ''
        content = ''
        header = {}
        for i in range(len(header_str)):
            if header_str[i][0] in trash_bin:
                continue
            else:
                key = header_str[i][0]
                for j in range(len(header_str[i]) - 1):
                    if not header_str[i][j + 1] == '':
                        content = header_str[i][j + 1]
                    else:
                        continue
            header[key] = content
        self.header = header

    def __dat_data_reader__(self, f_path):
        data_str = ''
        data_list = []
        with open(f_path, 'r') as f:
            while True:
                if re.match(r'\[DATA\]', f.readline()):
                    f.readline()
                    data_str = f.read()
                    break
                else:
                    continue
        # Notification: data_str type changed
        data_str = data_str.split('\n')
        del data_str[-1]
        for i in range(len(data_str)):
            data_list.append(data_str[i].split('\t'))
        self.data = np.array(data_list).astype(float)


class __NanonisFile_3ds__:
    def __init__(self, f_path):
        self.file_path = os.path.split(f_path)[0]
        self.fname = os.path.split(f_path)[1]
        self.__3ds_header_reader__(f_path)
        self.__3ds_header_reform__(self.raw_header)
        self.__3ds_data_reader__(f_path, self.header)

    def __3ds_header_reader__(self, f_path):
        key = ''
        contents = ''
        raw_header = {}
        with open(f_path, 'rb') as f:
            header_end = False
            while not header_end:
                line = f.readline().decode(encoding='utf-8', errors='replace')
                if re.match(':HEADER_END:', line):
                    header_end = True
                else:
                    key, contents = line.split('=')
                    contents = contents.strip('"\r\n')
                raw_header[key] = contents
        self.raw_header = raw_header

    def __3ds_header_reform__(self, raw_header):
        # table = ['Grid settings', 'Channels']
        parameters = ['Fixed parameters', 'Experiment parameters']
        # spec_info_str = ['Filetype', 'Sweep Signal', 'Experiment', 'Start time', 'End time', 'User', 'Comment']
        spec_info_int = [
            '# Parameters (4 byte)', 'Experiment size (bytes)', 'Points'
        ]
        grid_settings = ['X_OFFSET', 'Y_OFFSET', 'X_RANGE', 'Y_RANGE', 'ANGLE']
        header_dict = {}
        keys = list(raw_header.keys())
        for i in range(len(keys)):
            header_dict[keys[i]] = raw_header[keys[i]]
        # spec_info_int
        for j in range(len(spec_info_int)):
            header_dict[spec_info_int[j]] = int(header_dict[spec_info_int[j]])
        # Delay before measuring (s)
        header_dict['Delay before measuring (s)'] = float(
            header_dict['Delay before measuring (s)'])
        # Grid dim
        dims = header_dict['Grid dim'].split('x')
        grid_tuple = (int(dims[0]), int(dims[1]))
        grid_check = True
        for k in (0, 1):
            if dims[k] == 1:
                grid_check = False
        header_dict['Grid dim'] = grid_tuple
        # Grid settings
        if grid_check:
            grid_settings_dict = {}
            grid_settings_ls = header_dict['Grid settings'].split(';')
            for i in range(len(grid_settings_ls)):
                grid_settings_dict[grid_settings[i]] = float(
                    grid_settings_ls[i])
            header_dict['Grid settings'] = grid_settings_dict
        else:
            header_dict['Grid settings'] = None
        # Parameters
        fixed_parameter = header_dict['Fixed parameters'].split(';')
        experiment_parameters = header_dict['Experiment parameters'].split(';')
        Parameter = []
        for i in range(len(fixed_parameter)):
            Parameter.append(fixed_parameter[i])
        for j in range(len(experiment_parameters)):
            Parameter.append(experiment_parameters[j])
        header_dict['Parameters'] = Parameter
        for k in parameters:
            del header_dict[k]
        # Channels
        header_dict['Channels'] = header_dict['Channels'].split(';')
        # Number of channels
        header_dict['num_Channels'] = len(header_dict['Channels'])
        self.header = header_dict

    def __3ds_data_reader__(self, f_path, header):
        """
        __3ds_data_reader__:
        read the .3ds file
        
        Parameters
        ----------
        f_path : path of .3ds file
        header: reformed header of .3ds file
        
        Return
        ------
        Parameters : values of parameters of every position, return (position, num_parameters) np.array
        data : specscopies, returns (position, channels, points) np.array
        """
        with open(f_path, 'rb') as f:
            read_all = f.read()
            offset = read_all.find(
                ':HEADER_END:\x0d\x0a'.encode(encoding='utf-8'))
            # print('found start at {}'.format(offset))
            f.seek(offset + 14)
            data = np.fromfile(f, dtype='>f')
        Parameters = np.zeros((header['Grid dim'][0] * header['Grid dim'][1],
                               header['# Parameters (4 byte)']))
        spec_data = np.zeros((header['Grid dim'][0] * header['Grid dim'][1],
                              header['num_Channels'], header['Points']))
        for i in range(header['Grid dim'][0] * header['Grid dim'][1]):
            # Read Parameters
            for j in range(header['# Parameters (4 byte)']):
                Parameters[i][j] = data[
                    i * int(header['# Parameters (4 byte)'] +
                            header['Experiment size (bytes)'] / 4) + j]
            # Read spec data
            for k in range(header['num_Channels']):
                for l in range(header['Points']):
                    spec_data[i][k][l] = data[
                        int(i * (header['Experiment size (bytes)'] / 4 +
                                 header['# Parameters (4 byte)']) +
                            (k * header['Points'] +
                             header['# Parameters (4 byte)']) + l)]
        self.Parameters = Parameters
        self.data = spec_data