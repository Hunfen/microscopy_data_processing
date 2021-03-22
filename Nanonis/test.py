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
        offset = read_all.find(':HEADER_END:\x0d\x0a'.encode(encoding='utf-8'))
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
                spec_data[i][k][l] = data[int(
                    i * (header['Experiment size (bytes)'] / 4 +
                         header['# Parameters (4 byte)']) +
                    (k * header['Points'] + header['# Parameters (4 byte)']) +
                    l)]
    self.Parameters = Parameters
    self.data = spec_data
