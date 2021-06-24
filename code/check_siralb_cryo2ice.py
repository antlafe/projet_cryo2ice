#! /home/antlafe/anaconda3/bin/python

#
# check_siralb_cryo2ice.py
#
# ----------------------------------------------------------------------
# Copyright (c) 2021 LEGOS/SERCO
# All rights reserved.
#

"""
DESCRIPTION:

       Programm to compare along-track SLA and freeboard from IS2/CS2 collocated tracks from Cryo2Ice project plus cross-over with other missions

USAGE:

    check_siralb_cryo2ice.py [options]

optional arguments:

    -pi IS2 input path name
    -pc CS2 input path name
    -p parameter to check

EXAMPLES:


"""

import json
import sys
import h5py
import netCDF4 as nc
import numpy as np
from numpy import ma 
from mpl_toolkits.basemap import Basemap
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt
from matplotlib import cm
import glob
from datetime import date, timedelta, datetime
import argparse
import cs2_dict
import is2_dict
import path_dict
import saral_dict
import common_functions as cf
import warnings
import scipy.spatial
from scipy.stats import gaussian_kde
import pickle
import os
from scipy.interpolate import interp1d
from parserObjects import ParentAction,ChildAction
import xlrd
import sortnsave_cryo2ice_xings as snsc

# Global attributs
###########################################

spreadsheetpath = '../CRYO2ICE_tracks.xlsx'

# list of days for which CS2/IS2 are collocated with one day apart
list_midnight_dates = {
    'ATL07 ': ['20201005','20201018','20201022','20201108','20201125','20201129','20201216','20200102','20200119','20200123'],
    'ATL10': ['20201005','20201018','20201022','20201108','20201125','20201129','20201216','20200102','20200119','20200123'],
    'ATL12': ['20201014','20201031'],
    }

REF_GDR_CS2 = 'ESA_BD_GDR'
REF_GDR_IS2 = 'ATL10'


# Functions
###########################################

#---------------------------------
# Get CS2 dates
#--------------------------------

def is2date_2_cs2date(date,is2gdr):
    """
    this function deals with cases when IS2/CS2 collocated tracks are one day apart
    the function simplys add one day to CS2 date for these specific cases
    """
    
    #if date in [datetime.strptime(d,'%Y%m%d') for d in list_midnight_dates]:
    datestr = date.strftime('%Y%m%d')
    if datestr in list_midnight_dates[is2gdr]:
        date_1 =  date + timedelta(days=1)
        print("IS2 date: %s correspond to CS2 date: %s" %(date.strftime('%d/%m/%Y'),date_1.strftime('%d/%m/%Y')))
        #date_1 = date_1.strftime('%Y%m%d')
    else:
        date_1 = date

    return date_1

# Main
###########################################



if __name__ == '__main__':

    # Define programme description
    description ='rogramm to compare along-track SLA and freeboard from IS2/CS2 collocated tracks from Cryo2Ice project plus cross-over with other missions'

    # Initiate the parser with a description
    parser = argparse.ArgumentParser(description=description)

    # Save commande
    cmd = ' '.join(sys.argv)

    # Add long and short arguments
    # -------------------------------------------------------
    parser.add_argument("-d","--date",required=True,help="provide CS2 track date")

    parser.add_argument("-pi","--inputpathIS2",required=True,help="provide IS2 inputpath")

    parser.add_argument("-pc","--inputpathCS2",required=True,help="provide CS2 inputpath")

    parser.add_argument("-p","--parameter",required=True,help="parameter to test")

    # Read arguments
    # ----------------------------------------------------------
    args = parser.parse_args()

    date = [d for d in args.date.split(',')]
    if len(date) != 2: print('Provide date -d as YYYYMMDD,YYYYMMDD in arguments');sys.exit()

    start_date = datetime.strptime(date[0], '%Y%m%d')
    end_date = datetime.strptime(date[1], '%Y%m%d')
    
    print("\nAll CRYO2ICE tracks from %s to %s requested: \n" %(start_date.strftime("%d/%m/%y"),end_date.strftime("%d/%m/%y")))
    
    date_list=[start_date + timedelta(n) for n in range(int((end_date - start_date).days)+1)]


    # Read spreadsheet of collocated tracks: Don't forget to update spread sheet
    #-----------------------------------------------------------
    loc = (spreadsheetpath)
    wb = xlrd.open_workbook(loc)
    sheet = wb.sheet_by_index(0)
    
    is2_colloc_files = list()
    for i in range(sheet.nrows):
        is2_colloc_files.append(sheet.cell_value(i, 8))

    # Get IS2 files
    # -------------------------------------------------------
    path_IS2 = args.inputpathIS2
    file_list = list()
    
    for dt in date_list:
        datestr = dt.strftime("%y%m%d")
        filepattern = path_IS2 + "*%s*.h5" %(datestr)
        filename = glob.glob(filepattern)
        file_list.extend(filename)
        date_colloc_list.append(dt)
            
    if len(file_list)==0:
        print("No file found in %s for %s,%s" %(path_IS2,date[0],date[-1]))
        sys.exit()

    
    """
    IS2_files = list()
    index_list = list()
    for fileN in file_list:
        if fileN in is2_colloc_files:
            index_list.append(is2_colloc_files.index(fileN))
            IS2_files.append(fileN)

    if len(fileN)==0:
        print("No COLLOCATED file found in %s for %s,%s, check CRYO2ICE_tracks spreadsheet" %(path_IS2,date[0],date[-1]))
        sys.exit()
    """

    

    # Get CS2 files
    # -------------------------------------------------------
    path_CS2 = args.inputpathCS2

    file_list_cs2 = list()
    file_list_is2 = list()
    date_list = list()
    for dt in date_colloc_list:
        date_cs2 = is2date_2_cs2date(dt,'ATL10')
        datestr = date_cs2.strftime("%y%m%d")
        filepattern = path_CS2 + "*_%s*.nc" %(datestr)
        filename = glob.glob(filepattern)
        if len(filename)==0:
            continue
        else:
            idx = date_colloc_list.index(dt)
            file_list_cs2.append(filename)
            file_list_is2.append(date_colloc_list[idx])
            date_list.append(dt)
            
            
    if len(file_list)==0:
        print("No file found in %s for %s,%s" %(path_IS2,date[0],date[-1]))
        sys.exit()

    # creating dictionnaries
    #---------------------------
    file_dict = {}
    file_dict['CS2'] = {}
    file_dict[REF_GDR_CS2] = {}
    file_dict['IS2'] = {}
    file_dict[REF_GDR_IS2] = {}
    for ndate,dt in enumerate(date_list):

        # filling IS2 dictionnary
        date_is2 = dt.strftime("%y%m%d")
        file_dict['IS2'][REF_GDR_IS2][date_is2] = file_list_is2[ndate]

        # filling CS2 dictionnary
        date_cs2 = is2date_2_cs2date(dt,REF_GDR_IS2)
        file_dict['CS2'][REF_GDR_CS2][date_cs2] = file_list_cs2[ndate]
        
    
    # Check if files are well collocated
    common_data_list = snsc.get_collocated_data(date_list,file_dict)

    # Get aligned data
    data_dict = dict()
    data_dict['CS2'],cs2_info_dict = snsc.concatenate_cs2_data(date_list,file_dict['CS2'],common_data_list)
    data_dict['IS2'],is2_info_dict = snsc.concatenate_is2_data(date_list,file_dict['IS2'],common_data_list)
    data_dict['IS2'] =  snsc.get_beamwise_mean(date_list,data_dict['CS2'][REF_GDR_CS2],data_dict['IS2'],is2_info_dict)

    snsc.print_status_params(cs2_info_dict,is2_info_dict,outpath,'status')
    
    # compare and display the data
    

    
