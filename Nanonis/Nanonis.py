import math
import os
import re

import numpy as np


def read_file(f_path):
    if os.path.splitext(f_path)[1] == '.sxm':
        return __NanonisFile_sxm__(f_path)
    elif os.path.splitext(f_path)[1] == '.dat':
        return __NanonisFile_dat__(f_path)
    else:
        print("File type not supported.")


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
        # # Transform scan_info_float from str to float
        for k in range(len(scan_info_float)):
            header_dict[scan_info_float[k]] = float(
                header_dict[scan_info_float[k]])
        # Transform table from str to dict
        # SCAN_FIELD
        scan_field = header_dict['Scan>Scanfield'].split(';')
        #  SCAN_FIELD dict
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
        # self.raw_header = self.__dat_header_reader(f_path)
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
