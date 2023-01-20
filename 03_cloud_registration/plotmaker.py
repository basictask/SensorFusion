#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 20 13:56:46 2023

@author: daniel
"""

#%% Libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

#%% Main program
if(__name__ == '__main__'):
    df = pd.read_csv('runlogs.csv', sep=';', header=0)