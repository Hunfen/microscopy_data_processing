import numpy as np

class substraction:
    def __init__(self, nanonis_file):
        self.__topo_fwd__ = nanonis_file.data[0][0]
        self.__topo_bwd__ = nanonis_file.data[0][0]
    
    def fit_line(self, topo):
        