import h5py
import numpy as np
import pyUSID as usid
import matplotlib.pyplot as plt
import sidpy
import re

# read sxm file header
def header_reader(fname):
    '''
    Read the sxm file header into dictionary. Header entries
    as dict keys, and header contents as dict values.

    Parameters
    ----------------
    fname : .sxm file path.

    Return
    ----------------
    header : header dict file. 
    '''
    header_end = False # file_handler
    key = '' # Header_dict_key
    contents = '' # Header_data_buffer
    header = {} # Header_dict
    with open(fname, 'rb') as f:
        while not header_end:
            line = f.readline().decode(encoding = 'utf-8', errors = 'replace')
            if re.match(':SCANIT_END:\n',
                        line
                        ):
                header_end = True
            elif re.match(':.+:', # ':.+:' is the Nanonis .sxm file header entry regex, fuck.
                          line
                          ):
                key = line[1:-2] # Read header_entry
                content = ''     # Clear 
            else:
                contents += line
                # remove EOL
                header[key] = contents.strip('\n') 
    return header


# Header reform
def header_reform(header):
    """ 
    Reform the header which is obtained from NANONIS .sxm file.
    
    Parameter
    ---------
    header : header dict
    
    Returns
    -------
    header : reformed header dict
    """
    # HEADER_CLASSIFICATION
    trash_bin = ['NANONIS_VERSION',
                 'SCANIT_TYPE',
                 'REC_TEMP',
                 'SCAN_PIXELS',
                 'SCAN_TIME',
                 'SCAN_RANGE',
                 'SCAN_OFFSET',
                 'SCAN_ANGLE',
                 'Scan>channels'
                 ]
    scan_info_str = ['REC_DATE',
                     'REC_TIME',
                     'SCAN_FILE',
                     'SCAN_DIR',
                     'COMMENT'
                     ]
    scan_info_float = ['BIAS',
                       'ACQ_TIME',
                       'Scan>pixels/line',
                       'Scan>lines',
                       'Scan>speed forw. (m/s)',
                       'Scan>speed backw. (m/s)'
                       ]
    table = ['Scan>Scanfield',
             'Z-CONTROLLER',
             'DATA_INFO'
             ]
    scan_field_key = ['X_OFFSET',
                      'Y_OFFSET',
                      'X_RANGE',
                      'Y_RANGE',
                      'ANGLE'
                      ]
    # Clear redundant header entries
    for i in range(len(trash_bin)):
        header.pop(trash_bin[i])

    # Clear redundant space in scan_info_str
    for j in range(len(scan_info_str)):
        header[scan_info_str[j]] = header[scan_info_str[j]].strip(' ')
    
    # Transform scan_info_float from str to float
    for k in range(len(scan_info_float)):
        header[scan_info_float[k]] = float(header[scan_info_float[k]])

    # Transform table from str to dict
    # SCAN_FIELD
    scan_field = header['Scan>Scanfield'].split(';')

    #  SCAN_FIELD dict
    SCAN_FIELD = {}
    for m in range(len(scan_field_key)):
        SCAN_FIELD[scan_field_key[m]] = float(scan_field[m])

    # CHANNEL_INFO
    data_info = header['DATA_INFO'].split('\n')
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
    Z_Controller = header['Z-CONTROLLER'].split('\n')
    Controller_config = []
    for row in Z_Controller:
        Controller_config.append(row.strip('\t').split('\t'))
    # CONTROLLER_INFO dict
    CONTROLLER_INFO = {}
    for o in range(len(Controller_config[0])):
        CONTROLLER_INFO[Controller_config[0][o]] = Controller_config[1][o]
    # Substitute table dict
    for p in range(len(table)):
        header.pop(table[p])
    header['SCAN_FILED'] = SCAN_FIELD
    header['CONTROLLER_INFO'] = CONTROLLER_INFO
    header['CHANNEL_INFO'] = CHANNEL_INFO
    
    return header


# read_sxm_file
def sxm_read(fname):
    ''' 
    Read the .sxm data
    
    Parameter
    ---------
    fname : .sxm file path
    
    Returns
    -------
    header : reshaped .sxm file header(dict)
    raw_data : reshaped .sxm file dataset(np.array)
    dimension : dimension of dataset    
    '''
    
    header = header_reform(header_reader(fname))
    dimension = channels_counts(header)
    # read data
    with open(fname, 'rb') as f:
        file = f.read() # read the whole file into buffer
        # .sxm file header is end with \x1A\x04
        offset = file.find('\x1A\x04'.encode(encoding = 'utf-8'))
        f.seek(offset + 2) # Data starts 2 bytes after header end
        # read all data from file, MSB first
        data = np.fromfile(f, dtype = '>f')
        # reshape the based on header information
        raw_data = data.reshape(dimension)
        # flip if there are two directions
        if dimension[1] == 2:
            for i in range(len(raw_data)):
                for j in range(len(raw_data[i])):
                    if not j % 2 == 0:
                        raw_data[i][j] = np.fliplr(raw_data[i][j])
    return header, raw_data, dimension

def channels_counts(header):
    '''
    Determine the dimensions of raw_data.
    
    Parameter
    ---------
    header : reformed .sxm file header
    
    Return
    ------
    dimension : return dimension of raw_data (tuple)
    '''
    
    channels = header['CHANNEL_INFO']
    keys = channels.keys()
    dir_l = []
    counts = 0
    for i in keys:
        dir_l.append(channels[i]['Direction'])
    if 'both' in dir_l:
        counts = 2
    else:
        counts = 1
    dimension = (len(keys), counts, int(header['Scan>pixels/line']), int(header['Scan>lines']))
    return dimension