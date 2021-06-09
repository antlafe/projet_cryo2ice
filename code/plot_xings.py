#! /home/antlafe/anaconda3/bin/python

#
# plot_xings.py
#
# ----------------------------------------------------------------------
# Copyright (c) 2020 LEGOS/SERCO
# All rights reserved.
#



"""
DESCRIPTION:

     Program to plot statistics at crossing points

USAGE:

     plot_xings.py [options]

optional arguments:


EXAMPLES:

    python plot_xings.py -f file_name -p sla

COMMENTS:

    - Only one product at once  

"""
import sys
import h5py
import netCDF4 as nc
import numpy as np
from numpy import ma 
import matplotlib.pyplot as plt
import glob
from datetime import date, timedelta, datetime
import argparse
import cs2_dict
import is2_dict
import common_functions as cf
import warnings
import scipy.spatial
from scipy.stats import gaussian_kde
import pickle
from scipy.stats import pearsonr
import matplotlib as mpl


# Global attributs
###########################################
PATH_INPUT = "/home/antlafe/Documents/work/data/xings/"

###########################################
#
#              Functions
#
###########################################

###########################################
#
#              Main
#
###########################################


if __name__ == '__main__':

    # Define programme description
    description ='Program to plot statistics at crossing points'

    # Initiate the parser with a description
    parser = argparse.ArgumentParser(description=description)

    # Add long and short arguments
    parser.add_argument("-f","--inputfile",help="provide input pickle data file",required=True)

    parser.add_argument("-p","--parameter",default='list',help="provide parameter to be tested")

    # Read arguments
    # ----------------------------------------------------------
    args = parser.parse_args()

    # Open xings file
    # ---------------------------------------------------------
    inputfile = args.inputfile
    inputfilepattern = PATH_INPUT + inputfile
    filename = glob.glob(inputfilepattern)
    if len(filename)==0: sys.exit("\n%s: No found" %(inputfilepattern))
    
    f = nc.Dataset(filename)

    # Get parameter
    # ----------------------------------------------------------
    #pnames = args.parameter
    param = [p for p in args.parameter.split(',')]

    #param_opts=
    
    if 'list' in param:
        print("\nChoose param -p within:",param_opts)
        sys.exit()
    if param not in param_opts:
        print("Provide -p parameters within",param_opts)
        sys.exit()

    dtime = ma.ravel(f.variables['d_time'][:])
    
    
    
