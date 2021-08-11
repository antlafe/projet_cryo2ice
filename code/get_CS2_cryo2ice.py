#!/usr/bin/python3

#
# get_CS2_cryo2ice_files.py
#
# ----------------------------------------------------------------------
# Copyright (c) 2020 LEGOS/SERCO
# All rights reserved.
#


"""
DESCRIPTION:

     Programm to find collocated tracks file name based on provided spreadsheet

USAGE:

   python get_CS2_cryo2ice.py [options]

MAIN OPTIONS:

  -h, --help            show this help message and exit
  -p PRODUCT            provide product name
  -pn PATHNAME          provide path where CS2 files are stored
  -d month              month to retrieve data from


EXAMPLES:

python -m pdb get_CS2_cryo2ice.py -p CPOM -pn ~/Documents/work/projet_cryo2ice/data/CS2/CPOM/202011/ -d 202011

"""

import sys
import numpy as np
import glob
import argparse
from datetime import datetime,timedelta
import xlrd
import path_dict
import netCDF4 as nc
import os

# Global attributs
###########################################

varHome = os.environ['HOME']
spreadsheetpath = path_dict.PATH_DICT[varHome]['PATH_SPREADSHEET']+ '/CRYO2ICE_tracks.xlsx'


filepattern={
    'AWI': 'awi-siral-l2i-sithick-cryosat2-rep_nh_*_v2p3.nc',

    'UOB': 'ubristol_trajectory_rfb_ESA-SAR_*.txt',

    'CPOM': 'cry_NO_*.dat',

    'LEGOS_SAM': 'fb_SRL_GPS*.nc',

    'ESA_BD': 'CS_OFFL_SIR_GDR_2__*.nc',
    }



def get_date_in_file(product,fileN):

    if product=='AWI':
        date = fileN.split("_")[-3]
        timefile = datetime.strptime(date,"%Y%m%dT%H%M%S")
    elif product=='UOB':
        date = '_'.join(fileN.split(".")[-2].split("_")[-4:])
        timefile = datetime.strptime(date,"%Y_%m_%d_%H%M%S")
    elif product=='CPOM':
        date = fileN.split("_")[-1]
        timefile = datetime.strptime(date,"%Y%m%dT%H%M%S")
    elif 'LEGOS' in product:
        date = fileN.split(".")[-2].split("_")[-1]
        timefile = datetime.strptime(date,"%Y%m%dT%H%M%S")
    elif 'ESA_BD' in product:
        date = fileN.split(".")[-2].split("_")[-3]
        timefile = datetime.strptime(date,"%Y%m%dT%H%M%S")
    else:
        sys.exit("Unknow product: %s" %(product))

    return timefile
    



# Main
##########################################

if __name__ == '__main__':


    # Define programme description
    description =''

    # Initiate the parser with a description
    parser = argparse.ArgumentParser(description=description)

    # Add long and short arguments
    parser.add_argument("-d","--month",required=True,help="provide parameters to read. If absent or given all as arg, function list all available parameters")
    #parser.add_argument('-ds', '--datasource',required=True)
    parser.add_argument("-pn","--pathname",required=True,help="provide path where CS2 files are stored")
    parser.add_argument("-p","--product",required=True,help="provide path where CS2 files are stored")
    
    # Read arguments
    args = parser.parse_args()


    # Get product name
    product = args.product
    print("Getting collocated tracks for: %s\n" %(product))
    if product not in filepattern.keys():
        print("Choose product with list",filepattern.keys())
        sys.exit()


    # get month
    month = args.month
    date = datetime.strptime(month,"%Y%m")
    

    # Get filename where file are located
    pathname = args.pathname


    # read spreadsheet
    loc = (spreadsheetpath)
    wb = xlrd.open_workbook(loc)
    sheet = wb.sheet_by_index(0)


    # For the case ESA with can use absolute orbit number
    abs_orb_list  = np.arange(57361,58779,19).tolist()
    #abs_orb_list = list()
    if 'ESA_BD' in product:
        
        """
        for i in range(sheet.nrows):
            abs_orb = sheet.cell_value(i, 0)
            if isinstance(abs_orb, str): continue
            abs_orb_list.append(int(abs_orb))
        """

        abs_orb_list.sort()
            
        fileColloc = list()
        idx = list()
        for fileN in glob.glob(pathname+'*.nc'):
            
            f = nc.Dataset(fileN)
            abs_orb_file = f.getncattr('abs_orbit_number')
            if abs_orb_file in abs_orb_list:
                fileColloc.append(fileN)
                idx.append(abs_orb_list.index(abs_orb_file))
        
        fileColloc=np.array(fileColloc)[np.argsort(idx)]
        #fileColloc = np.array(fileColloc)
         
    # /!\ Need equator time in spreadsheet to find values
    else:
        # Get equator time of collocated tracks
        equator_time = list()
        print("find time at equator of collocated tracks")
        for i in range(sheet.nrows):
            time = sheet.cell_value(i, 3)
            if time[:2] != '20': continue
            print(time)
            time = datetime.strptime(sheet.cell_value(i, 3),"%Y-%m-%dT%H:%M")
            if time.strftime("%Y%m") != month: continue
            else:
                equator_time.append(time)


        # Get collocated file in provided folder
        filenames = glob.glob(pathname + filepattern[product])

        if len(filenames) > 0:
            print("\nFile found: %s \n" %(filenames))
        else:
            print("No file found in provided folder %s with file pattern %s" %(pathname,filepattern[product]) )
            sys.exit()


        # Find collocated file in provided folder
        index_colloc = list()
        for N,fileN in enumerate(filenames):
            timefile = get_date_in_file(product,fileN)
            for eqtime in equator_time:
                if timefile >= eqtime and timefile < eqtime + timedelta(minutes=50):
                    index_colloc.append(N)


        fileColloc= np.array(filenames)[index_colloc]


    if fileColloc.size==0:
        print("No collocated files found")
    else:
        print("#----------------------------")
        print("# CRYO2ICE collocated tracks")
        print("#---------------------------- \n\n")

        print("pathname: %s\n" %(pathname))
        for fileN in fileColloc:
            print(fileN.split('/')[-1])

        for fileN in fileColloc:
            print("cp %s ./" %(fileN))



    
