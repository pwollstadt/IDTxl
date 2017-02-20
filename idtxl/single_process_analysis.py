"""Parent class for analysis of single processes in the network.

@author: patricia
"""
import numpy as np
from .network_analysis import Network_analysis


class Single_process_analysis(Network_analysis):
    def __init__(self):
        super().__init__()
