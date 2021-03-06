#! /home/antlafe/anaconda3/bin/python

#
# sortnsave_cryo2ice_xings.py
#
# ----------------------------------------------------------------------
# Copyright (c) 2020 LEGOS/SERCO
# All rights reserved.
#



"""
DESCRIPTION:

     Programm to compare along-track data from IS2/CS2 collocated tracks from Cryo2Ice project plus cross-over with other missions

USAGE:
     1/ Sort monthly collocated tracks files in specific folder for CS2 and IS2

     2/ Add non corresponding dates btw CS2/IS2 in list_midnight_dates for each IS2 GDR [see spreadsheet: CRYO2ICE_tracks.xlsx] 
     3/ Modify path_dict.py with correct path repertories 

     4/ If new product: add sat+gdr info in file_pattern function

     5/ Check global attributs

     6/ Check parameters to be aligned in cs2_dict.py and is2_dict.py

     sortnsave_cryo2ice.py [options]

optional arguments:

EXAMPLES:

    python sortnsave_cryo2ice.py -s CS2 -g LEGOS_SAM,AWI -s SARAL,S3 -d20201001,20201007

    python -m ipdb sortnsave_cryo2ice_xings.py -s CS2 -g ESA_BD_GDR -s SARAL -gLEGOS_T50 -sIS2 -gATL07,ATL10 -d20201103,20201103 -ofn test
 
    python sortnsave_cryo2ice_xings.py -s CS2 -g ESA_BD_GDR,CPOM,AWI,LEGOS_SAM,UOB,LEGOS_T50,LEGOS_PLRM -s SARAL -gLEGOS_T50 -sIS2 -gATL07,ATL10 -d20201103,20201111 -ofn 202011_all



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


# Global attributs
###########################################

varHome = os.environ['HOME']

# Get paths
PATH_COLLOC=path_dict.PATH_DICT[varHome]['PATH_COLLOC']
PATH_ALL=path_dict.PATH_DICT[varHome]['PATH_DATA']
PATH_OUT = path_dict.PATH_DICT[varHome]['PATH_OUT']

# ref GDR: serving as reference for all other data-sets
REF_GDR = 'ESA_BD_GDR'

# IS2 gdr required
#is2_gdrs = ['ATL10']

matrixParamList = ['coherence','ph_diff','wvf']

# Add date when IS2 and CS2 are collocated on a different day

list_midnight_dates = path_dict.MIDNIGHT_DATES

# True only for ESA_BD
flag_1hz = False # alignement in 1Hz or in 20Hz
flag_IS2_mean = True # compute mean weighted value for each IS2 parameter (to get a same size 1D arrays between CS2 and IS2); if False: provide a matrix that contain each individual IS2 granules sorted by order of closest to beam centre

MAX_DIST_OF_COLLOC_DATA= 5.5 #5.5 # km
LAT_MIN = 55 # deg North

# relevant if flag_IS2_mean is False
N_IS2PTS_IN_CS2BEAMS = 200 # number of rows in IS2 data matrix
N_MAX_IS2PTS_IN_CS2BEAMS = 1000 # max number of rows informing that there is an issue

# xings points
ref_date={
    'IS2': datetime(2018,1,1),
    'CS2': datetime(2000,1,1),
    'SARAL': datetime(2000,1,1), #TAI
    'S3' : datetime(2000,1,1), # to check XXX 
    }


N_MAX_CROSSPTS_IN_CS2BEAMS = 50 #50
delay_xings = 4 #h
xing_delay = 2 # days to find files before and after first data
MAX_DIST_INTER = 40 # km security to find out if intersections found by algo are correct
TRACK_REDUCTION_IS2 = 80 # only consider 100th of track to look at interp (avoid memory issues)
TRACK_REDUCTION_SAR = 20 
N_NEIGHBORS_SAR = 10
N_NEIGHBORS_IS2 = 40

# encoding beam names into intergers
beam_dict={
    'gt1r': 11,
    'gt1l': 12,
    'gt2r': 21,
    'gt2l': 22,
    'gt3r': 31,
    'gt3l': 32,   
          }

#---------------------------------
# Get CS2 dates
#--------------------------------

def get_weighted_stats(weight1,weight2,val):
    """
    Get double weigted mean and std for an array of data [val]
    """
    val = ma.masked_invalid(val)
    weight1 = ma.masked_where(val.mask,weight1,copy=True)
    weight2 = ma.masked_where(val.mask,weight2,copy=True)
    npts = np.sum(~val.mask)
    
    mean = np.ma.sum(weight1*weight2*val)/np.ma.sum(weight1*weight2)
    
    std = np.ma.sqrt(np.ma.sum(weight1*weight2*(val-mean)**2) / np.ma.sum(weight1*weight2) )

    return mean,std,npts


#---------------------------------
# Get CS2 dates
#--------------------------------

def is2date_2_cs2date(date,is2gdr):
    """
    this function deals with cases when IS2/CS2 collocated tracks are one day apart
    the function simply adds one day to CS2 date for these specific cases
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


def align_sla_swath_seg(date_list,file_dict,common_data_list,cs2_gdr,is2_gdr,is2_b,LAT_MIN):

    """
    function to align CS2 SLA on IS2 SLA over SWATH segments (10Km along-track segments)
    20Hz and 1Hz CS2 data are averaged over these SWATH segments to match the correct size

    Comments:
    Only works for 1 icesat-2 beam at a time

    """

    # parameters to align
    plist_cs2_20hz = ['sla']
    plist_cs2_1hz = ['mss','dac','earth','pole','lpe','ocean','load']
    plist_is2_granules = ['mss','dac','earth','pole','lpe','ocean','load']
    plist_is2_swath = ['sla']

    cs2_pDict = {}
    is2_pDict = {}
    swath_data_array = {}
    swath_data_array['CS2'] = {}; swath_data_array['IS2'] = {};

    for n,date in enumerate(date_list):

        # initiate lists
        if n==0:
            
            for p in plist_cs2_20hz + plist_cs2_1hz + ['lat','lon','dist','delay']:
                cs2_pDict[p] = list()
                swath_data_array['CS2'][p] = list()
                
            for p in plist_is2_granules + plist_is2_swath + ['lat','lon','dist','delay']:
                is2_pDict[p] = list()
                swath_data_array['IS2'][p] = list()

        # Get CS2 sla + corrs
        #--------------------------
        date_cs2 = is2date_2_cs2date(date,is2_gdr)
        date_str_cs2 = date_cs2.strftime('%Y%m%d')

        print("\n%s" %(date.strftime('%d/%m/%Y')))
        filename = file_dict['CS2'][cs2_gdr][date_str_cs2]
        file_format = filename.split('.')[-1]
        
        data_desc_cs2_20hz = cs2_dict.init_dict(cs2_gdr,False)
        data_desc_cs2_1hz = cs2_dict.init_dict(cs2_gdr,True)
        
        if file_format=='nc':
            lat_c_20,lon_c_20,time_c_20,x_dist,valid_idx = cf.get_coord_from_netcdf(filename,data_desc_cs2_20hz,'01',LAT_MIN)
            lat_c_01,lon_c_01,time_c_01,x_dist,valid_idx = cf.get_coord_from_netcdf(filename,data_desc_cs2_1hz,'01',LAT_MIN)
        else:
            print("Unknown file format")
            sys.exit()

        cs2_pDict['lat'].append(lat_c_20)
        cs2_pDict['lon'].append(lon_c_20)
        #cs2_pDict['lat_01'].append(lat_c_01)
        #cs2_pDict['lon_01'].append(lon_c_01)
        
        for pname in plist_cs2_20hz:
            param,units,param_is_flag = cf.get_param_from_netcdf(filename,data_desc_cs2_20hz,pname,'01',LAT_MIN)
            cs2_pDict[pname].append(param)
            
        for pname in plist_cs2_1hz:
            param,units,param_is_flag = cf.get_param_from_netcdf(filename,data_desc_cs2_1hz,pname,'01',LAT_MIN)
            if param is None:
                param_hf = ma.masked_array(np.zeros((time_c_20.size,)),mask=np.ones((time_c_20.size,)))
            else:
                param_hf = cf.interp_1hz_to_20hz(param,time_c_01,time_c_20)
            cs2_pDict[pname].append(param_hf)
        


        # Get IS2 sla + corrs
        #------------------------
        
        date_str = date.strftime('%Y%m%d')
        filename = file_dict['IS2'][is2_gdr][date_str]
        is2_beam = get_strong_beams(filename,[is2_b])

        # Granule data
        data_desc_is2_gran = is2_dict.init_dict(is2_gdr,is2_beam[is2_b],'granules')
        lat_i_gran,lon_i_gran,time_i_gran,x_dist,valid_idx = cf.get_coord_from_hf5(filename,data_desc_is2_gran,'01',LAT_MIN)

        # Swath data
        data_desc_is2_sw = is2_dict.init_dict(is2_gdr,is2_beam[is2_b],'swath')
        lat_i_sw,lon_i_sw,time_i_sw,x_dist,valid_idx = cf.get_coord_from_hf5(filename,data_desc_is2_sw,'01',LAT_MIN)

        is2_pDict['lat'].append(lat_i_sw)
        is2_pDict['lon'].append(lon_i_sw)
        
        for pname in plist_is2_granules:
            param,units,param_is_flag = cf.get_param_from_hf5(filename,data_desc_is2_gran,pname,'01',LAT_MIN)
            is2_pDict[pname].append(param)
            
        for pname in plist_is2_swath:
            param,units,param_is_flag = cf.get_param_from_hf5(filename,data_desc_is2_sw,pname,'01',LAT_MIN)
            is2_pDict[pname].append(param)
        


        # get common sections
        #------------------------
        #selected_idx = common_data_list[n][gdr][b]['idx_is2']
        ref_lat = common_data_list[n]['ref_lat']
        ref_lon = common_data_list[n]['ref_lon']
        coord_ref = np.vstack((ref_lon,ref_lat)).T
        tree = scipy.spatial.cKDTree(coord_ref)
        
        # coordinates SWATH
        coord_sw = np.vstack((lon_i_sw,lat_i_sw)).T
        # Looking for collocated section in SWATH tracks
        distance,closest_ind = tree.query(coord_sw,1)
        flag_colloc = distance < 1 #km

        # ref SWATH collocs
        lat_sw = lat_i_sw[flag_colloc]; lon_sw = lon_i_sw[flag_colloc]
        ref_size = lat_sw.shape[0]
        
        x_sw,y_sw,z_sw = cf.lon_lat_to_cartesian(lon_sw, lat_sw)
        coord_sw = np.vstack((x_sw,y_sw,z_sw)).T
        tree = scipy.spatial.cKDTree(coord_sw)


        # Associate CS2 20hz data association to 10km SWATH-SEG coordinate
        x_cs2,y_cs2,z_cs2 = cf.lon_lat_to_cartesian(lon_c_20,lat_c_20)
        coord_cs2 = np.vstack((x_cs2,y_cs2,z_cs2)).T
        distance,closest_ind = tree.query(coord_cs2,1)
        
        flag_colloc = distance < 1
        selected_idx = closest_ind[flag_colloc]
        
        data_array_cs2 = dict()
        for p in cs2_pDict.keys():
            data_array_cs2[p] = ma.masked_array(np.zeros((ref_size,)),mask=np.ones((ref_size,)))
            
        for idx in np.unique(selected_idx):
            argidx = np.argwhere(closest_ind==idx)
            for p in cs2_pDict.keys():
                if p=='dist':
                    data_array_cs2[p][idx] = np.ma.mean(distance[argidx])
                elif p=='delay':
                    time_c =  np.ma.mean(time_c_20[argidx])
                    delta_ref_time = (ref_date['IS2'] - ref_date['CS2']).total_seconds()
                    delta_t_min = (time_i_sw[idx] + delta_ref_time - time_c)/60
                    data_array_cs2[p][idx] = delta_t_min
                else:
                    data_array_cs2[p][idx] = np.ma.mean(cs2_pDict[p][n][argidx]) 
            
            
        # Associate IS2 granule data to 10km SWATH-SEG coordinates
        x_is2,y_is2,z_is2 = cf.lon_lat_to_cartesian(lon_i_gran,lat_i_gran)
        coord_is2 = np.vstack((x_is2,y_is2,z_is2)).T
        distance,closest_ind = tree.query(coord_is2,1)

        flag_colloc = distance < 1
        selected_idx = closest_ind[flag_colloc]
        idx_swath = np.unique(selected_idx)
        
        data_array_is2 = dict()
        for p in is2_pDict.keys():
            data_array_is2[p] = ma.masked_array(np.zeros((ref_size,)),mask=np.ones((ref_size,)))

        data_array_is2['lat'] = lat_sw #[flag_colloc]
        data_array_is2['lon'] = lon_sw #[flag_colloc]
        #data_array_is2['sla'] = is2_pDict['sla'][n][idx_swath]
            
        for idx in idx_swath:
            argidx = np.argwhere(closest_ind==idx)
            for p in is2_pDict.keys():
                if p in ['lat','lon']: continue
                if 'sla' in p:
                    data_array_is2[p][idx] = is2_pDict[p][n][idx]
                elif p=='dist':
                    data_array_is2[p][idx] = data_array_cs2[p][idx]
                elif p=='delay':
                    data_array_is2[p][idx] = -data_array_cs2[p][idx]
                else:
                    data_array_is2[p][idx] = np.ma.mean(is2_pDict[p][n][argidx])


        # Adding data to final array
        for p in is2_pDict.keys():
            swath_data_array['IS2'][p].append(data_array_is2[p])

        # Adding data to final array
        for p in cs2_pDict.keys():
            swath_data_array['CS2'][p].append(data_array_cs2[p])

        
    return swath_data_array  


            

        

def get_avail_files(satName,prod_list,date_list,flag_colloc):


    """
    Find available data in PATH_DATA folder

    Note: Only one collocated data per day in folder
          Need all data for one day to record files for that day

    Args:
    satName   (str)          : satellite name IS2 or CS2
    prod_list list(str)      : list of Cryosat-2 L2 products 
    date_list list(datetime) : list of required datetime objects  
    flag_colloc  : If False multiple files accepted 
    
    """
    print("\nMISSION: %s\n#########" %(satName))
    file_dict = {}
    missing_dates = list()

    if flag_colloc:
        PATH=PATH_COLLOC
    else:
        PATH=PATH_ALL
        
    for gdr in prod_list:

        file_dict[gdr] ={}
        print("\n%s:\n#----------\n" %gdr)
        #path = '%s%s/%s/' %(PATH_DATA,satName,gdr)
        #print("%s\n" %path)
        filenum = 0 # number of file found
        
        for nd,d in enumerate(date_list):

            d_str_folder = d.strftime("%Y%m")
            d_str = d.strftime("%Y%m%d")
            d_str_print = d.strftime("%d/%m/%y")
            #satpath,filepattern = get_sat_filepattern(satName,gdr,d_str,d_str_folder)

            # Get CS2 file pattern
            if satName=='CS2':

                dict_cs2 = cs2_dict.CS2_DATA_DESC
                
                if gdr not in dict_cs2.keys():
                    print("\nError: Missing %s in CS2 in file dictionnary cs2_dict.py \n" %(gdr));sys.exit()
                filepattern = dict_cs2[gdr]['file_pattern'].replace('yyyymmdd',d_str)
                satpath = dict_cs2[gdr]['path'].replace('yyyymm',d_str_folder)

            # Get IS2 file pattern
            elif satName=='IS2':

                dict_is2 = is2_dict.init_dict(gdr,'','fileinfo')
                
                filepattern = dict_is2['file_pattern'].replace('yyyymmdd',d_str)
                satpath = dict_is2['path'].replace('yyyymm',d_str_folder)

            # Unknown satellite
            else:
                print("\nError: Unknown satellite %s \n create new dictionnary" %(satName));sys.exit()
            
            if nd==0: print("%s%s\n" %(PATH,satpath))
            file_pattern = PATH + satpath + filepattern
            filename = glob.glob(file_pattern)

            if len(filename)==0:
                if flag_colloc: print("%s: No file found" %(d_str_print))
                missing_dates.append(d_str)

            elif len(filename)==1:
                if flag_colloc: print("%s: File found: %s" %(d_str_print,filename[0].split('/')[-1]))
                file_dict[gdr][d_str] = filename[0]
                filenum = filenum+1
            else:
                if flag_colloc:
                    print("%s: /!\ WARNING: Too many files found:" %(d_str_print),filename)
                    print("Using first file: %s" %(filename[0].split('/')[-1]))
                    sys.exit()
                    
                else:
                    file_dict[gdr][d_str] = filename
                    filenum = filenum+len(filename)

        found_date_list = list(file_dict[gdr].keys())

    new_date_list = date_list.copy()
    print("\nMissing some %s data at:" %(satName))    
    for d in date_list:
        d_str = d.strftime("%Y%m%d")
        if d_str in missing_dates:
            new_date_list.remove(d)
            print(d.strftime("%d/%m/%Y"))
    if len(new_date_list)==len(date_list): print("None")
    date_list = new_date_list.copy()

    if len(date_list)==0:
        print("\n\nNo file found for %s products:" %(satName),file_dict)
        sys.exit()
        
    print("# of files found: %i\n" %(filenum))

    return file_dict,date_list


def get_strong_beams(filename,is2Beams):

    """
    IceSAT-2 strong beams depend on the satellite orientation. This function checks the orientation in the filename to then provide the right beam name.

    """
    f=h5py.File(filename,'r')
    flag_orientation = np.array(f.get('orbit_info/sc_orient'))
    if flag_orientation==0: #backward
        if is2Beams==['b1']:
            beamN = {'b1':'gt1l'}
        elif is2Beams==['b2']:
            beamN = {'b2':'gt2l'}
        elif is2Beams==['b3']:
            beamN = {'b3':'gt3l'}
        else:
            beamN = {'b1':'gt1l','b2':'gt2l','b3':'gt3l'}
    elif flag_orientation==1: #forward
        if is2Beams==['b1']:
            beamN = {'b1':'gt1r'}
        elif is2Beams==['b2']:
            beamN = {'b2':'gt2r'}
        elif is2Beams==['b3']:
            beamN = {'b3':'gt3r'}
        else:
            beamN = {'b1':'gt1r','b2':'gt2r','b3':'gt3r'}
    else:
        beamN = None
    return beamN


#---------------------------------
# Get collocated tracks
#---------------------------------


def get_collocated_data(date_list,is2_gdrs,file_dict,is2Beams):

    """
    Associate and sort collocated track points 


    Args:
    date_list list(datetime) : list of required datetime objects
    file_dict dict()         : dictionnary of found files
    is2Beams                 : list of required IS2 beams from ['b1','b2','b3']
    """


    common_data_list = list() #retreive common sections
    #first_coord = list()
    #last_coord = list()
    print("\nReading collocated data\n #------------------")
    print("\n max distance between tracks: %i km" %MAX_DIST_OF_COLLOC_DATA)
    
    for n,date in enumerate(date_list):
        
        dict_common_data = {}
        date_str = date.strftime('%Y%m%d')
        print("\n\n%s: " %(date.strftime('%d/%m/%Y')))

        # Get CS2 time & coordinates
        # WARNING always ESA_BD
        cs2_gdr =  next(iter(file_dict['CS2']))
        if cs2_gdr!= REF_GDR: sys.exit("First CS2 gdr must be %s" %(REF_GDR))
        data_desc_cs2 = cs2_dict.init_dict(cs2_gdr,flag_1hz)

        #if date_str not in file_dict['CS2'][cs2_gdr].keys():
        date_cs2 = is2date_2_cs2date(date,is2_gdrs[0])
        date_str_cs2 = date_cs2.strftime('%Y%m%d')
        filename = file_dict['CS2'][cs2_gdr][date_str_cs2]

        # get coordinates
        lat_c_full,lon_c_full,time_c,x_dist,valid_idx = cf.get_coord_from_netcdf(filename,data_desc_cs2,'01',LAT_MIN)
        
        
        #lat_c,lon_c = cf.interp_coord_1hz_to_20hz(lon_c_01,lat_c_01,time_c_01,time_c_20)
        
        # Eliminating land and ocean from collocated tracks
        #---------------------------
        lons,lats,OSISAF_ice_type = cf.get_osisaf_ice_type(date.year,date.month,date.day,'01')
        OSISAF_ice_type[OSISAF_ice_type.mask] = 1
        
        icetype_alongtrack = cf.grid_to_track(OSISAF_ice_type,lons,lats,lon_c_full,lat_c_full)
        flag_change_value = (icetype_alongtrack[:-1] != icetype_alongtrack[1:])
        
        first_idx = np.argmax(flag_change_value) if icetype_alongtrack[0]==1 else 0
        last_idx  = flag_change_value.size - np.argmax(flag_change_value[::-1]) if icetype_alongtrack[0]==1 else flag_change_value.size
        
        lat_c,lon_c = lat_c_full[np.arange(first_idx,last_idx)],lon_c_full[np.arange(first_idx,last_idx)]

        """
        plt.plot(lat_c_full,lon_c_full,'o')
        plt.plot(lat_c,lon_c,'*')
        plt.show()
        """

        #--------------------------

        dist_btw_coord = cf.dist_btw_two_coords(lat_c[:-1],lat_c[1:],lon_c[:-1],lon_c[1:])
        idx_gaps = np.argwhere(dist_btw_coord > 2*np.mean(dist_btw_coord))
       
        coord_polar_cs2 = np.vstack((lon_c, lat_c))

        # initialise min,max idx of cs2 for collocated data
        min_idx_cs2=lon_c.size; max_idx_cs2=0

        first_coord = list()
        last_coord = list()
        

        print("Associating tracks\n-----------------------")
        for ngdr,is2_gdr in enumerate(is2_gdrs):

            #if is2_gdr=='ATL07': beam_list = beamName
            #else: beam_list = ['swath']; continue # XXX to modify
            dict_common_data[is2_gdr] = {}
            filename = file_dict['IS2'][is2_gdr][date_str]
            beamName = get_strong_beams(filename,is2Beams)
            if beamName is None: continue

            print(beamName)
            
            for beam in beamName.keys():

                print("\n Beam %s: %s\n-----------" %(beam,is2_gdr))

                dict_common_data[is2_gdr][beam] = {}
                data_desc_is2 = is2_dict.init_dict(is2_gdr,beamName[beam],'granules')    
                lat_i,lon_i,time_i,x_dist,valid_idx = cf.get_coord_from_hf5(filename,data_desc_is2,'01',LAT_MIN)

                # check collocation
                #plt.plot(lat_i,lon_i,'*')
                #plt.plot(lat_c,lon_c,'o')
                #plt.plot(lat_c_01,lon_c_01,'.')
                #plt.show()
                
                # to avoid killing process for wrong files
                if lat_i is None: continue

                coord_polar_is2 = np.vstack((lon_i, lat_i))
        
                # Convert to cartesien coordinates
                x_c,y_c,z_c = cf.lon_lat_to_cartesian(lon_c, lat_c)
                x_i,y_i,z_i = cf.lon_lat_to_cartesian(lon_i, lat_i)
                coord_cart_cs2 = np.vstack((x_c,y_c,z_c)).T
                coord_cart_is2 = np.vstack((x_i,y_i,z_i)).T
                
                    
                # Find closest match of IS2 in CS2 track
                tree = scipy.spatial.cKDTree(coord_cart_cs2)
                distance,closest_ind = tree.query(coord_cart_is2,1)
                
                # Test if closest match algo work properly
                increase_ind = all(i <= j for i, j in zip(closest_ind, closest_ind[1:]))
                if not increase_ind:
                    print("Warning: %s beam %s track %s is not increasing, check map" %(is2_gdr,beam,date_str))
                    idx_decr, = np.where(np.diff(closest_ind) <0)
                    val_decr = (closest_ind[1:] - closest_ind[:-1])[idx_decr]
                    print("Indexes",idx_decr,"not increasing by",val_decr) 
                
                # time delay of CryoSat-2
                time_cs2 = time_c
                delta_ref_time = (ref_date['IS2'] - ref_date['CS2']).total_seconds()
                delta_t_sec = (time_i + delta_ref_time - time_c[closest_ind])
                delta_t_min = delta_t_sec/60
                
                # Select collocated data
                flag_colloc = distance < MAX_DIST_OF_COLLOC_DATA

                if not np.any(flag_colloc):
                    # showing track on map
                    print(file_dict['CS2'][cs2_gdr][date_str_cs2])
                    
                    coord_list = [coord_polar_cs2.T,coord_polar_is2.T]
                    name_list = ['CS2','IS2']
                    cf.plot_tracks_map(coord_list,name_list)
                    print("\n No collocated data closer than %.2f km found for %s beam %s \n" %(MAX_DIST_OF_COLLOC_DATA,date_list[n],beam))
                    print("Closest data are: %.2fm" %(np.min(distance)))
                    sys.exit()

                idx_is2 = np.arange(lat_i.size)[flag_colloc]
                idx_first_pt_is2 = idx_is2[0]
                idx_last_pt_is2 = idx_is2[-1]
                
                idx_first_pt_cs2 = closest_ind[idx_first_pt_is2]
                idx_last_pt_cs2 = closest_ind[idx_last_pt_is2]

                print("Writing data in dict_common_data")
                

                # Store data
                dict_common_data[is2_gdr][beam]['idx_is2'] = idx_is2
                dict_common_data[is2_gdr][beam]['idx_cs2'] = np.arange(idx_first_pt_cs2,idx_last_pt_cs2+1)
                dict_common_data[is2_gdr][beam]['dist'] = distance[flag_colloc]
                dict_common_data[is2_gdr][beam]['delay'] = delta_t_min[flag_colloc]
                dict_common_data[is2_gdr][beam]['cs2_idx_in_is2'] = closest_ind[flag_colloc]
                

                print("Mean time difference: %i min" %(np.mean(delta_t_min[flag_colloc])))
                print("Mean distance: %i km" %(np.mean(distance[flag_colloc])))
                
            

            # consider longest CS2 track section to includes all colocated beams 
            #dict_common_data['idx_cs2_wide'] = np.arange(min_idx_cs2,max_idx_cs2+1)            

            # Selection of common track section
            # initiate and end all data with same CS2 beam
            #cs2_inter = np.arange(lat_c.size)[1:-1] #initialize with all indexes to avoid associating data from far before first point
            cs2_union = dict_common_data[is2_gdr][is2Beams[0]]['idx_cs2'][0:-1]
            # WARNING: beware of small IS25 beam that constrain other
            for beam in beamName.keys():
                #cs2_inter = list(set(cs2_inter)&set(dict_common_data[is2_gdr][beam]['idx_cs2']))
                cs2_union = list(set(cs2_union)|set(dict_common_data[is2_gdr][beam]['idx_cs2'][0:-1]))

            #cs2_inter = np.sort(np.array(cs2_inter))[1:-1]
            cs2_union = np.sort(np.array(cs2_union))
            #if len(cs2_inter)==0: print("No common data CS2 indexes between",beamName,"for track %s" %(date_str)," Check code, because highly unlikely")

            # Reference CS2 track
            first_coord.append(coord_polar_cs2[:,cs2_union[0]])
            last_coord.append(coord_polar_cs2[:,cs2_union[-1]])
            dict_common_data['ref_lat'] = coord_polar_cs2[1,cs2_union]
            dict_common_data['ref_lon'] = coord_polar_cs2[0,cs2_union]
            dict_common_data['ref_time'] = time_c

            # record idx of gaps in ref track coordinates data
            idx_data_gaps = idx_gaps - cs2_union[0]
            flag_gaps = idx_data_gaps > 0
            idx_gaps_pos = idx_data_gaps[flag_gaps]
            idx_gaps_fin = np.concatenate([idx_gaps_pos,np.array([i+1 for i in idx_gaps_pos])]) 
            dict_common_data['idx_gaps'] =  idx_gaps_fin
            #idx_gaps = None
            
            print("\n Data gaps in ref track %s \n %s :\n" %(REF_GDR,date_str), dict_common_data['idx_gaps'])
            
            # For each beam apply selection
            for beam in beamName.keys():
                dict_common_data[is2_gdr][beam]['ref_idx'] = dict_common_data[is2_gdr][beam]['cs2_idx_in_is2'] - cs2_union[0]

        print("\nShowing tracks on map")
        coord_list = [coord_polar_cs2.T,coord_polar_is2[:,flag_colloc].T]
        name_list = ['CS2','IS2']
        #cf.plot_tracks_map(coord_list,name_list)

          
        # Save first and last coordinates
        dict_common_data['ref_lon_full'] = lon_c_full
        dict_common_data['ref_lat_full'] = lat_c_full     
       
        common_data_list.append(dict_common_data)

            
    return common_data_list


#---------------------------------
# Concatenate data
#---------------------------------


def concatenate_cs2_data(date_list,is2_gdrs,file_dict,common_data_list):

    """
    Align various CRYOSAT-2 products (AWI,GPOD...etc) on reference track (usually ESA BASELINE-D official product) defined in function get_collocated_data 


    Args:
    date_list list(datetime) : list of required datetime objects
    file_dict dict()         : dictionnary of found files
    common_data_list         : daily list of common_track information (output of common_data_list)
    """

    # Init CS2 data dictionnary
    cs2_data_dict = dict()
    flag_data_dict = dict()
    print("\nConcatenating CS2 data\n#---------------\n")
    
    # Unwrap CS2 data
    ref_track_full = list()
    ref_track_seg  = list()
    ref_size = list()

   
    
    for ngdr,gdr in enumerate(file_dict.keys()):

        print("%s:\n---------" %(gdr))
        cs2_data_dict[gdr] = dict()
        flag_data_dict[gdr] = dict()
        for n,date in enumerate(date_list):

            date_cs2 = is2date_2_cs2date(date,is2_gdrs[0])
            date_str_cs2 = date_cs2.strftime('%Y%m%d')
            flag_data_dict[gdr][date_str_cs2] = dict()
            
            print("\n%s" %(date.strftime('%d/%m/%Y')))
            filename = file_dict[gdr][date_str_cs2]
            file_format = filename.split('.')[-1]

            # Get parameters
            data_desc_cs2_1hz = cs2_dict.init_dict(gdr,True)
            data_desc_cs2_20hz = cs2_dict.init_dict(gdr,False)
            param_list = list(set([pname for pname in data_desc_cs2_1hz.keys()] + [pname for pname in data_desc_cs2_20hz.keys()]))

            # Get coords in NetCDF
            if file_format=='nc':
                lat_hf,lon_hf,time_hf,x_dist,valid_idx = cf.get_coord_from_netcdf(filename,data_desc_cs2_20hz,'01',LAT_MIN)
                lat_lf,lon_lf,time_lf,x_dist,valid_idx = cf.get_coord_from_netcdf(filename,data_desc_cs2_1hz,'01',LAT_MIN)

            # Get coords in txt file
            elif file_format=='txt': #XXX      
                lat_hf,lon_hf,time_hf,x_dist,valid_idx = cf.get_coord_from_uob(filename,data_desc_cs2_20hz,'01',LAT_MIN)
                lat_lf,lon_lf,time_lf,x_dist,valid_idx = cf.get_coord_from_uob(filename,data_desc_cs2_1hz,'01',LAT_MIN)

            # Get coords from .dat file
            elif file_format=='dat': #XXX
                lat_hf,lon_hf,time_hf,x_dist,valid_idx = cf.get_coord_from_cpom(filename,data_desc_cs2_20hz,'01',LAT_MIN)
                lat_lf,lon_lf,time_lf,x_dist,valid_idx = cf.get_coord_from_cpom(filename,data_desc_cs2_1hz,'01',LAT_MIN) 
            else:
                print("Unknown file format")

            # For this case need to find position of 20hz in 1hz
            if flag_1hz:
                lat,lon,time = lat_lf,lon_lf,time_lf
            else:
                lat,lon,time = lat_hf,lon_hf,time_hf

            # Check if available data for mode chosen 1Hz/20Hz
            if lat is None:
                print("No lat,lon data for %s with flag_1hz=%i" %(filename,flag_1hz))
                sys.exit()

                
            # convert to cartesien
            x_c,y_c,z_c = cf.lon_lat_to_cartesian(lon, lat)
            coordinates = np.vstack((x_c,y_c,z_c)).T
            #tree = scipy.spatial.KDTree(coordinates)
            
            # ref coordinates
            lat_ref,lon_ref = common_data_list[n]['ref_lat'],common_data_list[n]['ref_lon']
            ref_size = lon_ref.size
            x_ref,y_ref,z_ref = cf.lon_lat_to_cartesian(lon_ref, lat_ref)
            coordinates_ref = np.vstack((x_ref,y_ref,z_ref)).T
            tree = scipy.spatial.KDTree(coordinates_ref)
            distance,idx_in_ref = tree.query(coordinates,1)

            
            # For this case need to find position of 20hz in 1hz ref track
            if flag_1hz:
                x_hf,y_hf,z_hf = cf.lon_lat_to_cartesian(lon_hf, lat_hf) 
                coordinates_hf = np.vstack((x_hf,y_hf,z_hf)).T
                dist,idx_lf_in_hf = tree.query(coordinates_hf,1)
                flag_20hz = dist < 3.2 # max limit of 20hz dist to 1hz center
                idx_lf_in_hf = idx_lf_in_hf[flag_20hz]

            
            # associate data pts with toward ref track  
            selected_idx = np.argwhere(distance < 0.3)
            selected_idx = selected_idx.reshape((selected_idx.size,))
            ref_idx =  idx_in_ref[selected_idx]

            if not np.any(selected_idx):
                # showing track on map
                #print(file_dict[cs2_gdr][date_str_cs2])
                coord_polar_ref = np.vstack((lon_ref, lat_ref))
                coord_polar_gdr = np.vstack((lon, lat))
                coord_list = [coord_polar_gdr.T,coord_polar_ref.T]
                name_list = [gdr,'ESA_BD_GDR']
                print(filename)
                cf.plot_tracks_map(coord_list,name_list)
                #print("\n No collocated data closer than %.2f km found for %s beam %s \n" %(MAX_DIST_OF_COLLOC_DATA,date_list[n],beam))
                #print("Closest data are: %.2fm" %(np.min(distance)))
                #sys.exit()
            
            new_lat = lat[selected_idx]
            new_lon = lon[selected_idx]
            new_time = time[selected_idx]
            
            # initiation masked array
            data_lat = ma.masked_array(np.zeros((ref_size,)),mask=np.ones((ref_size,)))
            data_lon = ma.masked_array(np.zeros((ref_size,)),mask=np.ones((ref_size,)))
            data_time = ma.masked_array(np.zeros((ref_size,)),mask=np.ones((ref_size,)))
            data_id = ma.masked_array(np.zeros((ref_size,)),mask=np.ones((ref_size,)))
            data_cs2_idx = ma.masked_array(np.zeros((ref_size,)),mask=np.ones((ref_size,)))

            # filling masked array
            data_lat[ref_idx] = new_lat
            data_lon[ref_idx] = new_lon
            data_time[ref_idx] = new_time
            data_id[ref_idx] = n*np.ones((ref_idx.size,))
            data_ref_idx = ref_idx
            data_cs2_idx[ref_idx] = selected_idx

            # initiating list
            list_param = ['id','ref_idx','cs2_idx','latref','lonref','latref_full','lonref_full'] + param_list
            if n==0:
                for p in list_param: cs2_data_dict[gdr][p] = list()

            # Adding new track data to list
            cs2_data_dict[gdr]['time'].append(data_time)
            cs2_data_dict[gdr]['id'].append(data_id)
            cs2_data_dict[gdr]['ref_idx'].append(data_ref_idx)
            cs2_data_dict[gdr]['cs2_idx'].append(data_cs2_idx)
            
            if gdr==REF_GDR:
                cs2_data_dict[gdr]['latref'].append(lat_ref)
                cs2_data_dict[gdr]['lonref'].append(lon_ref)
                cs2_data_dict[gdr]['latref_full'].append(common_data_list[n]['ref_lat_full'])
                cs2_data_dict[gdr]['lonref_full'].append(common_data_list[n]['ref_lon_full'])

            # Adding new track parameters data to list
            for freq in ['hf','lf']:

                flag_freq = True if freq=='lf' else False
                data_desc_cs2 = cs2_dict.init_dict(gdr,flag_freq)

                print("\nRecording %s data:\n" %(freq))
            
                for pname,prodname in data_desc_cs2.items():
                    
                    # Not recording time/coordinate variables in hf
                    if pname=='time': continue
                    if pname in ['lat','lon','hour','minute','second'] and freq=='hf': continue
                    print("%s" %(pname))
                    
                    flag_data_dict[gdr][date_str_cs2][pname] = dict()                
                    if prodname is None: flag_data_dict[gdr][date_str_cs2][pname]['status']='NOK'  ; continue
                    if file_format=='txt':
                        param,units,param_is_flag = cf.get_param_from_uob(filename,data_desc_cs2,pname,'01',LAT_MIN)                
                    elif file_format=='nc': #XXX
                        param,units,param_is_flag = cf.get_param_from_netcdf(filename,data_desc_cs2,pname,'01',LAT_MIN)
                        #if np.sum(param.mask)==param.size: param=None
                    elif file_format=='dat': #XXX
                        param,units,param_is_flag = cf.get_param_from_cpom(filename,data_desc_cs2,pname,'01',LAT_MIN)  
                    else:
                        print("Unknown file format")

                    # param status
                    if param is None:
                        flag_data_dict[gdr][date_str_cs2][pname]['status']='NOK'
                        
                        #continue
                    else:
                        flag_data_dict[gdr][date_str_cs2][pname]['status']='OK'

                    # initiate masked array to record data
                    if param is None:
                        data_param = ma.masked_array(np.zeros((ref_size,)),mask=np.ones((ref_size,)))
                        cs2_data_dict[gdr][pname].append(data_param)
                        flag_data_dict[gdr][date_str_cs2][pname]['is_flag'] = param_is_flag
                        flag_data_dict[gdr][date_str_cs2][pname]['name'] = prodname
                        continue
                        
                    elif pname in matrixParamList:
                        ngate = param.shape[1]
                        data_param = ma.masked_array(np.zeros((ref_size,ngate)),mask=np.ones((ref_size,ngate)))
                    elif flag_1hz and freq=='hf':
                        max_freq = 25 #should be 20 for 20hz but small secu
                        data_param = ma.masked_array(np.zeros((max_freq,ref_size)),mask=np.ones((max_freq,ref_size)))
                    else:
                        data_param = ma.masked_array(np.zeros((ref_size,)),mask=np.ones((ref_size,)))
                    
                    # Case recording 20hz params with 1hz ref (matrix)  
                    if flag_1hz and freq=='hf':

                        # Adding data for each 1hz ref column
                        for idx in np.unique(idx_lf_in_hf):
                            ind = np.argwhere(idx_lf_in_hf==idx)
                            data_param[:ind.size,idx] = param[ind].reshape((ind.size,))

                    # Case recording 1hz data with 20hz ref (interpolation)
                    elif not flag_1hz and freq=='lf':
                        
                        data_lr_list = [param]
                        # interpolate
                        param_hf = cf.interp_1hz_to_20hz(param,time_lf,time_hf)
                        
                        data_param[ref_idx] = param_hf[selected_idx]
                        
                    # Other cases: 20hz in 20hz/ 1hz in 1hz
                    else:
                        data_param[ref_idx] = param[selected_idx]

                    cs2_data_dict[gdr][pname].append(data_param)

                    flag_data_dict[gdr][date_str_cs2][pname]['units'] = units
                    flag_data_dict[gdr][date_str_cs2][pname]['is_flag'] = param_is_flag
                    flag_data_dict[gdr][date_str_cs2][pname]['name'] = prodname
                
    
    return cs2_data_dict,flag_data_dict


#---------------------------------
# Find crossings with other missions
#---------------------------------


# Finding crossings and collocated data of other sats
def find_xings2_sat(satName,date_list,file_dict,common_data_list):

    """
    Find cross-overs between required satellite and collocated tracks trajectories. It takes mean value arround cross-over point.


    Args:
    satName                  : Name of the satellite [CS2, SARAL, S3]
    date_list list(datetime) : list of required datetime objects
    file_dict dict()         : dictionnary of found files
    common_data_list         : daily list of common_track information (output of common_data_list)
    """

    

    # Init CS2 data dictionnary
    data_dict = dict()
    data_param = dict()
    param_list = ['lat','lon','time','delay','dist','weight','idx_ref']
    delta_reftime = (ref_date[satName] - ref_date['CS2']).total_seconds()

    print("\nFind Crossings points with %s \n#---------------\n" %(satName))

    for ngdr,gdr in enumerate(file_dict.keys()):

        print("%s:\n---------" %(gdr))
        data_dict[gdr] = dict()
       
        data_list = dict()

        # For each collocated tracks - dates (every 1.5 days)
        #-------------------------------------------------------
        ndates = len(date_list)
        for n,date in enumerate(date_list):
            
            date_str = date.strftime('%Y%m%d')
            
            # if no data for this date continue
            #if date_str not in file_dict[gdr].keys(): continue
            
            print("\n%s" %(date.strftime('%d/%m/%Y')))

            # ref coordinates for this date
            lat_ref,lon_ref = common_data_list[n]['ref_lat'],common_data_list[n]['ref_lon']
            if any(np.abs(np.diff(lon_ref)) > 20): lon_ref[lon_ref > 180] = lon_ref[lon_ref > 180] - 360 
            time_ref = common_data_list[n]['ref_time']
            ref_size = lon_ref.size
            coord_ref = np.vstack((lat_ref,lon_ref)).T
            x_ref,y_ref,z_ref = cf.lon_lat_to_cartesian(lon_ref, lat_ref)
            coordinates_ref = np.vstack((x_ref,y_ref,z_ref)).T
            tree = scipy.spatial.KDTree(coordinates_ref)

            # maximun length list
            max_len = 0
            
            # initiation array of lists
            if n==0:
                for p in  param_list:
                    data_list[p] = np.frompyfunc(list, 0, 1)(np.empty((ndates,), dtype=object))
            
            nfiles = 0
            for ite,val in file_dict[gdr].items():
                if isinstance(val, str): val=[val]
                nfiles = nfiles + len(val)

            # For each file around date
            #-------------------------------------------------------
            nfile = 0
            for datefile,filelist in file_dict[gdr].items():

                # for case one file per day
                if isinstance(filelist, str): filelist=[filelist]
                for filename in filelist:

                    nfile = nfile+1
                    print("%i/%i" %(nfile,nfiles))

                    # Get dictionnary
                    if satName=='SARAL':
                        data_desc = saral_dict.init_dict(gdr,flag_1hz)
                    elif satName=='S3':
                        data_desc = s3_dict.init_dict(gdr,flag_1hz)
                    elif satName=='CS2':
                        data_desc = cs2_dict.init_dict(gdr,flag_1hz)
                    else:
                        print("No dictionnary for %s" %(satName))

                    if nfile==1 and n==0:
                        list_p = [p for p in data_desc.keys()] + param_list
                        for p in data_desc.keys():
                            if p not in param_list:
                                data_list[p] = np.frompyfunc(list, 0, 1)(np.empty((ndates,), dtype=object))
                    
                    lat,lon,time,x_dist,valid_idx = cf.get_coord_from_netcdf(filename,data_desc,'01',LAT_MIN)
                    if lat is None: continue

                    if any(np.abs(np.diff(lon)) > 20): lon[lon > 180] = lon[lon > 180] - 360 
                    lat_sub = lat[::TRACK_REDUCTION_SAR]
                    lon_sub = lon[::TRACK_REDUCTION_SAR]
                    time_sub = time[::TRACK_REDUCTION_SAR]

                    coord_is2 = np.vstack((lat_sub,lon_sub)).T

                    # get intersections
                    from intersection import find_intersections
                    lat_inter, lon_inter, idx_ref, idx_sat = find_intersections(coord_ref,coord_is2,1)
                    

                    # test intersection
                    if lat_inter==None:
                         print("No intersection found for this track")
                         nfile = nfile+1
                         continue

                    data_list['lat'][n].append(lat_inter)
                    data_list['lon'][n].append(lon_inter)
                    data_list['time'][n].append(time_sub[idx_sat])
                    data_list['delay'][n].append(time_ref[idx_ref] - time_sub[idx_sat])
                    data_list['idx_ref'][n].append(idx_ref)
                    #data_list['dist'].append()

                    index = TRACK_REDUCTION_SAR*idx_sat
                    first_index = index-N_NEIGHBORS_SAR if index-N_NEIGHBORS_SAR>0 else 0
                    last_index = index+N_NEIGHBORS_SAR if index+N_NEIGHBORS_SAR< lat.size else lat.size-1
                    idx_sub = np.arange(first_index,last_index)
                    
                    for pname in data_desc.keys():
                        if pname in ['lat','lon','time','lat01','lon01','time01','hour','minute','second'] + param_list: #add 2D params later
                            continue

                        param,units,param_is_flag = cf.get_param_from_netcdf(filename,data_desc,pname,'01',LAT_MIN)
                        if param is None:
                            mean_p = np.nan
                            std_p = np.nan
                        else:
                            mean_p = np.ma.mean(param[idx_sub])
                            std_p = np.ma.std(param[idx_sub])
                            
                        data_list[pname][n].append(mean_p)
                        #data_list[pname]['std'].append(mean_p)
                        
    print("stop")
    for p in list_p:
        data_dict[gdr][p] = np.ma.array(data_list[p])
                        
                   
            
    return data_dict
        
                    

# Finding crossings and collocated data of other sats
def find_xings_sat(satName,date_list,file_dict,common_data_list):
    
    """
    Find cross-overs between required satellite and collocated tracks trajectories. It associates one by one measurements arround cross-over point using kdtree algorithm.


    Args:
    satName                  : Name of the satellite [CS2, SARAL, S3]
    date_list list(datetime) : list of required datetime objects
    file_dict dict()         : dictionnary of found files
    common_data_list         : daily list of common_track information (output of common_data_list)
    """
    
    # Init CS2 data dictionnary
    data_dict = dict()
    data_param = dict()
    param_list = ['lat','lon','time','delay','dist','weight']
    delta_reftime = (ref_date[satName] - ref_date['CS2']).total_seconds()

    print("\nFind Crossings points with %s \n#---------------\n" %(satName))

    for ngdr,gdr in enumerate(file_dict.keys()):

        print("%s:\n---------" %(gdr))
        data_dict[gdr] = dict()
        
        data_list = dict()

        # For each collocated tracks - dates (every 1.5 days)
        #-------------------------------------------------------
        for n,date in enumerate(date_list):
            
            date_str = date.strftime('%Y%m%d')
            
            # if no data for this date continue
            #if date_str not in file_dict[gdr].keys(): continue
            
            print("\n%s" %(date.strftime('%d/%m/%Y')))

            # ref coordinates for this date
            lat_ref,lon_ref = common_data_list[n]['ref_lat'],common_data_list[n]['ref_lon']
            if any(np.abs(np.diff(lon_ref)) > 20): lon_ref[lon_ref > 180] = lon_ref[lon_ref > 180] - 360 
            time_ref = common_data_list[n]['ref_time']
            ref_size = lon_ref.size
            coord_ref = np.vstack((lat_ref,lon_ref)).T
            x_ref,y_ref,z_ref = cf.lon_lat_to_cartesian(lon_ref, lat_ref)
            coordinates_ref = np.vstack((x_ref,y_ref,z_ref)).T
            tree = scipy.spatial.KDTree(coordinates_ref)

            # maximun length list
            max_len = 0
            
            # initiation array of lists
            for p in  param_list:
                data_list[p] = np.frompyfunc(list, 0, 1)(np.empty((ref_size,), dtype=object))

            nfiles = 0
            for ite,val in file_dict[gdr].items():
                if isinstance(val, str): val=[val]
                nfiles = nfiles + len(val)

            # For each file around date
            #-------------------------------------------------------
            nfile = 0
            for datefile,filelist in file_dict[gdr].items():

                # for case one file per day
                if isinstance(filelist, str): filelist=[filelist]
                for filename in filelist:

                    print("%i/%i" %(nfile,nfiles))

                    # Get dictionnary
                    if satName=='SARAL':
                        data_desc = saral_dict.init_dict(gdr,flag_1hz)
                    elif satName=='S3':
                        data_desc = s3_dict.init_dict(gdr,flag_1hz)
                    elif satName=='CS2':
                        data_desc = cs2_dict.init_dict(gdr,flag_1hz)
                    else:
                        print("No dictionnary for %s" %(satName))

                    # initiation array of lists for specific params
                    """
                    if nfile==0:
                        for p in data_desc.keys():
                            if p not in param_list:
                                data_list[p] = np.frompyfunc(list, 0, 1)(np.empty((ref_size,), dtype=object))

                    """
                    lat,lon,time,x_dist,valid_idx = cf.get_coord_from_netcdf(filename,data_desc,'01',LAT_MIN)

                    if any(np.abs(np.diff(lon)) > 20): lon[lon > 180] = lon[lon > 180] - 360 
                    lat_sub = lat[::TRACK_REDUCTION_SAR]
                    lon_sub = lon[::TRACK_REDUCTION_SAR]
                    time_sub = time[::TRACK_REDUCTION_SAR]

                    coord_is2 = np.vstack((lat_sub,lon_sub)).T

                    # get intersections
                    from intersection import find_intersections
                    lat_inter, lon_inter, idx_ref, idx_is2 = find_intersections(coord_ref,coord_is2,1)

                    # test intersection
                    if lat_inter==None:
                         print("No intersection found for this track")
                         nfile = nfile+1
                         continue


                  
                    
                    # test
                    """
                    plt.plot(lat,lon,'.')
                    plt.plot(lat_ref,lon_ref,'*')
                    plt.plot(lat_inter, lon_inter,'*')
                    plt.plot(lat_sub[idx_ref].flatten(),lon_sub[idx_ref].flatten(),'.')
                    plt.plot(new_lat, new_lon,'*')
                    plt.show()
                    """
                    
                    
                    # check if intersection is correct
                    dist1 = cf.dist_btw_two_coords(lat_inter,lat_sub[idx_is2],lon_inter,lon_sub[idx_is2])
                    dist2 = cf.dist_btw_two_coords(lat_inter,lat_ref[idx_ref],lon_inter,lon_ref[idx_ref])
                    if dist2 > MAX_DIST_INTER or dist1 > MAX_DIST_INTER:
                        print("No or wrong intersection found for this track")
                        nfile = nfile+1
                        continue

                    # get subsection of track
                    index = TRACK_REDUCTION_SAR*idx_is2
                    first_index = index-int(index/TRACK_REDUCTION_SAR) if index-int(index/TRACK_REDUCTION_SAR)>0 else 0
                    last_index = index+int(index/TRACK_REDUCTION_SAR) if index+int(index/TRACK_REDUCTION_SAR)< lat.size else lat.size-1
                    idx_sub = np.arange(first_index,last_index)
                    lat_sect = lat[idx_sub]
                    lon_sect = lon[idx_sub]
                    time_sect = time[idx_sub]

                    # convert to cartesien
                    x,y,z = cf.lon_lat_to_cartesian(lon_sect, lat_sect)
                    coordinates = np.vstack((x,y,z)).T            
                    distance,idx_in_ref = tree.query(coordinates,1)

                    # Selection close points
                    selected_idx = np.argwhere(distance < MAX_DIST_OF_COLLOC_DATA) # XXX same as IS2?
                    selected_idx = selected_idx.reshape((selected_idx.size,))
                    ref_idx =  idx_in_ref[selected_idx]

                    new_lat = lat_sect[selected_idx]
                    new_lon = lon_sect[selected_idx]
                    new_time = time_sect[selected_idx]
                    new_dist = distance[selected_idx]
                    weight = np.exp(-(new_dist**2)/(MAX_DIST_OF_COLLOC_DATA**2))

                    # Add data to array of lists
                    for idx in np.unique(ref_idx):
                        size = np.argwhere(ref_idx==idx).size
                        data_list['lat'][idx].extend(new_lat[np.argwhere(ref_idx==idx).flatten()].tolist())
                        data_list['lon'][idx].extend(new_lon[np.argwhere(ref_idx==idx).flatten()].tolist())
                        data_list['time'][idx].extend(new_time[np.argwhere(ref_idx==idx).flatten()].tolist())
                        delay = time_ref[idx] - new_time[np.argwhere(ref_idx==idx).flatten()].tolist() -delta_reftime
                        data_list['delay'][idx].extend(delay)
                        data_list['dist'][idx].extend(new_dist[np.argwhere(ref_idx==idx).flatten()].tolist())
                        data_list['weight'][idx].extend(weight[np.argwhere(ref_idx==idx).flatten()].tolist())

                    # Adding new track parameters data to list
                    #-------------------------------------------------
                    for pname,prodname in data_desc.items():

                        if pname in ['lat','lon','time','lat01','lon01','time01','hour','minute','second'] + param_list: #add 2D params later
                            continue

                        param,units,param_is_flag = cf.get_param_from_netcdf(filename,data_desc,pname,'01',LAT_MIN)
                        param_sub = param[idx_sub]
                        for idx in np.unique(ref_idx):
                            data_list[pname][idx].extend(param[np.argwhere(ref_idx==idx).flatten()].tolist())
                            max_len = max(len(data_list[pname][idx]),max_len)
                            
                    # next file
                    nfile = nfile+1

            # converting arrays of lists to matrix
            #---------------------------------------------
                
            # initiating masked arrays
            data_ar = {}
            print("\nMax data at one crossing point: %i" %(max_len))
            for p in data_list.keys():
                data_ar[p] = ma.masked_array(np.zeros((N_MAX_CROSSPTS_IN_CS2BEAMS,ref_size)),mask=np.ones((N_MAX_CROSSPTS_IN_CS2BEAMS,ref_size)))

            # save into masked array
            #--------------------------------
            #data_param_ar = {}
            for p in data_list.keys():
                for ncol in range(ref_size):
                    # limit size of array
                    if len(data_list[p][ncol])>= N_MAX_CROSSPTS_IN_CS2BEAMS:
                        max_idx = N_MAX_CROSSPTS_IN_CS2BEAMS
                    else:
                        max_idx=len(data_list[p][ncol])

                    # save in column of masked array
                    data_ar[p][:max_idx,ncol] =np.array(data_list[p][ncol][:max_idx]).T
                    if max_idx>0: print(data_ar[p][:max_idx,ncol])

            # initialize track lists to save per colloc ref track the data
            if n==0:
                for pname in data_list.keys():
                    data_dict[gdr][pname] = list()

            # saving the data
            for pname in data_list.keys():
                data_ar[pname] = np.ma.masked_invalid(data_ar[pname],copy=True)
                data_dict[gdr][pname].append(data_ar[pname])                
    
    return data_dict



# Finding crossings and collocated data of IS2
def find_xings2_is2(date_list,file_dict,file_dict_colloc,common_data_list,is2Beams):

    """
    Find cross-overs between IS2 and collocated tracks trajectories. It takes mean value arround cross-over point.


    Args:
    date_list list(datetime) : list of required datetime objects
    file_dict dict()         : dictionnary of found files
    common_data_list         : daily list of common_track information (output of common_data_list)
    is2Beams                 : list of required IS2 beams from ['b1','b2','b3'] 
    """
    
    # Init CS2 data dictionnary
    data_dict = dict()
    data_param = dict()
    param_list = ['lat','lon','time','delay','beam','weight','idx_ref']
    delta_reftime = (ref_date['IS2'] - ref_date['CS2']).total_seconds()
 
    print("\nFind Crossings points with IceSat-2 \n#---------------\n")

    for ngdr,gdr in enumerate(file_dict.keys()):

        print("%s:\n---------" %(gdr))
        data_dict[gdr] = dict()
       
        data_list = dict()

        # For each collocated tracks - dates (every 1.5 days)
        #-------------------------------------------------------
        ndates = len(date_list)
        for n,date in enumerate(date_list):
            
            date_str = date.strftime('%Y%m%d')
            
            # if no data for this date continue
            #if date_str not in file_dict[gdr].keys(): continue
            
            print("\n%s" %(date.strftime('%d/%m/%Y')))

            # ref coordinates for this date
            lat_ref,lon_ref = common_data_list[n]['ref_lat'],common_data_list[n]['ref_lon']
            if any(np.abs(np.diff(lon_ref)) > 20): lon_ref[lon_ref > 180] = lon_ref[lon_ref > 180] - 360 
            time_ref = common_data_list[n]['ref_time']
            ref_size = lon_ref.size
            coord_ref = np.vstack((lat_ref,lon_ref)).T
            x_ref,y_ref,z_ref = cf.lon_lat_to_cartesian(lon_ref, lat_ref)
            coordinates_ref = np.vstack((x_ref,y_ref,z_ref)).T
            tree = scipy.spatial.KDTree(coordinates_ref)

            # maximun length list
            max_len = 0
            
            # initiation array of lists
            if n==0:
                for p in  param_list:
                    data_list[p] = np.frompyfunc(list, 0, 1)(np.empty((ndates,), dtype=object))
            
            nfiles = 0
            for ite,val in file_dict[gdr].items():
                if isinstance(val, str): val=[val]
                nfiles = nfiles + len(val)

            # For each file around date
            #-------------------------------------------------------
            nfile = 0
            for datefile,filelist in file_dict[gdr].items():

                # for case one file per day
                if isinstance(filelist, str): filelist=[filelist]
                for filename in filelist:

                    nfile = nfile+1
                    print("%i/%i" %(nfile,nfiles))
                    
                    # get orientation
                    beamName = get_strong_beams(filename,is2Beams)
                    if beamName is None: continue
                    
                    # to avoid considering collocated tracks
                    if filename.split('/')[-1]==file_dict_colloc[gdr][date_str].split('/')[-1]:
                        print("colloc files: Not used for intersections")
                        continue

                    # for each beam
                    for beam in beamName.keys():

                        print("%s" %(beam))
                        data_desc = is2_dict.init_dict(gdr,beamName[beam],'granules')

                        # initiation array of lists for specific params
                        if nfile==1 and n==0:
                            list_p = [p for p in data_desc.keys()] + param_list
                            for p in data_desc.keys():
                                if p not in param_list:
                                    data_list[p] = np.frompyfunc(list, 0, 1)(np.empty((ndates,), dtype=object))
                                    #data_list[p] = np.frompyfunc(list, 0, 1)(np.empty((ref_size,), dtype=object))
                    

                        lat,lon,time,x_dist,valid_idx = cf.get_coord_from_hf5(filename,data_desc,'01',LAT_MIN)
                        if lat is None: continue
                        
                        if any(np.abs(np.diff(lon)) > 20): lon[lon > 180] = lon[lon > 180] - 360 
                        lat_sub = lat[::TRACK_REDUCTION_IS2]
                        lon_sub = lon[::TRACK_REDUCTION_IS2]
                        time_sub = time[::TRACK_REDUCTION_IS2]

                        coord_is2 = np.vstack((lat_sub,lon_sub)).T
                        # get intersections
                        from intersection import find_intersections
                        lat_inter, lon_inter, idx_ref, idx_is2 = find_intersections(coord_ref,coord_is2,1)
                        # test intersection
                        if lat_inter==None:
                            print("No intersection found for this track")
                            continue

                        data_list['lat'][n].append(lat_inter)
                        data_list['lon'][n].append(lon_inter)
                        data_list['time'][n].append(time_sub[idx_is2])
                        data_list['delay'][n].append(time_ref[idx_ref] - time_sub[idx_is2])
                        data_list['idx_ref'][n].append(idx_ref)
                        data_list['beam'][n].append(beam)
                        #data_list['dist'].append()

                        index = TRACK_REDUCTION_IS2*idx_is2
                        first_index = index-N_NEIGHBORS_IS2 if index-N_NEIGHBORS_IS2>0 else 0
                        last_index = index+N_NEIGHBORS_IS2 if index+N_NEIGHBORS_IS2< lat.size else lat.size-1
                        idx_sub = np.arange(first_index,last_index)
                    
                        for pname in data_desc.keys():
                            if pname in ['lat','lon','time','lat01','lon01','time01','hour','minute','second'] + param_list: #add 2D params later
                                continue

                            param,units,param_is_flag = cf.get_param_from_hf5(filename,data_desc,pname,'01',LAT_MIN)
                            if param is None:
                                mean_p = np.nan
                                std_p = np.nan
                            else:
                                mean_p = np.ma.mean(param[idx_sub])
                                std_p = np.ma.std(param[idx_sub])
                                
                            data_list[pname][n].append(mean_p)

    
    for p in list_p:
        data_dict[gdr][p] = np.ma.array(data_list[p])

    return data_dict

                        

        
# Finding crossings and collocated data of IS2
def find_xings_is2(date_list,file_dict,file_dict_colloc,common_data_list,is2Beams):

    """
    Finds cross-overs between IS2 collocated tracks trajectories. It associates one by one measurements arround cross-over point using kdtree algorithm.


    Args:
    date_list list(datetime) : list of required datetime objects
    file_dict dict()         : dictionnary of found files
    common_data_list         : daily list of common_track information (output of common_data_list)
    is2Beams                 : list of required IS2 beams from ['b1','b2','b3'] 
    """
    
    # Init CS2 data dictionnary
    data_dict = dict()
    data_param = dict()
    param_list = ['lat','lon','time','delay','dist','beam','weight']
    delta_reftime = (ref_date['IS2'] - ref_date['CS2']).total_seconds()
 
    print("\nFind Crossings points with IceSat-2 \n#---------------\n")

    for ngdr,gdr in enumerate(file_dict.keys()):

        print("%s:\n---------" %(gdr))
        data_dict[gdr] = dict()
       
        data_list = dict()

        # For each collocated tracks - dates (every 1.5 days)
        #-------------------------------------------------------
        for n,date in enumerate(date_list):
            
            date_str = date.strftime('%Y%m%d')
            
            # if no data for this date continue
            #if date_str not in file_dict[gdr].keys(): continue
            
            print("\n%s" %(date.strftime('%d/%m/%Y')))

            # ref coordinates for this date
            lat_ref,lon_ref = common_data_list[n]['ref_lat'],common_data_list[n]['ref_lon']
            if any(np.abs(np.diff(lon_ref)) > 20): lon_ref[lon_ref > 180] = lon_ref[lon_ref > 180] - 360 
            time_ref = common_data_list[n]['ref_time']
            ref_size = lon_ref.size
            coord_ref = np.vstack((lat_ref,lon_ref)).T
            x_ref,y_ref,z_ref = cf.lon_lat_to_cartesian(lon_ref, lat_ref)
            coordinates_ref = np.vstack((x_ref,y_ref,z_ref)).T
            tree = scipy.spatial.KDTree(coordinates_ref)

            # maximun length list
            max_len = 0
            
            # initiation masked array
            for p in  param_list:
                data_list[p] = np.frompyfunc(list, 0, 1)(np.empty((ref_size,), dtype=object))

            nfiles = 0
            for ite,val in file_dict[gdr].items():
                if isinstance(val, str): val=[val]
                nfiles = nfiles + len(val)

            # For each file around date
            #-------------------------------------------------------
            nfile = 0
            for datefile,filelist in file_dict[gdr].items():

                # for case one file per day
                if isinstance(filelist, str): filelist=[filelist]
                for filename in filelist:

                   
                    print("%i/%i" %(nfile,nfiles))
                    nfile = nfile+1

                    # get orientation
                    beamName = get_strong_beams(filename,is2Beams)
                    if beamName is None: continue
                    
                    # to avoid considering collocated tracks
                    if filename.split('/')[-1]==file_dict_colloc[gdr][date_str].split('/')[-1]:
                        print("colloc files: Not used for intersections")
                        continue

                    # for each beam
                    for beam in beamName.keys():

                        print("%s" %(beam))
                        data_desc = is2_dict.init_dict(gdr,beamName[beam],'granules')

                        # initiation of data list for params
                        if nfile==1:
                            for p in data_desc.keys():
                                if p not in param_list:
                                    data_list[p] = np.frompyfunc(list, 0, 1)(np.empty((ref_size,), dtype=object))

                        lat,lon,time,x_dist,valid_idx = cf.get_coord_from_hf5(filename,data_desc,'01',LAT_MIN)
                        if lat is None: continue
                        
                        if any(np.abs(np.diff(lon)) > 20): lon[lon > 180] = lon[lon > 180] - 360 
                        lat_sub = lat[::TRACK_REDUCTION_IS2]
                        lon_sub = lon[::TRACK_REDUCTION_IS2]
                        time_sub = time[::TRACK_REDUCTION_IS2]

                        coord_is2 = np.vstack((lat_sub,lon_sub)).T
                        # get intersections
                        from intersection import find_intersections
                        lat_inter, lon_inter, idx_ref, idx_is2 = find_intersections(coord_ref,coord_is2,1)

                        
                        # test intersection
                        if lat_inter==None:
                            print("No intersection found for this track")
                            continue

                        
                        # check if intersection is correct
                        dist1 = cf.dist_btw_two_coords(lat_inter,lat_sub[idx_is2],lon_inter,lon_sub[idx_is2])
                        dist2 = cf.dist_btw_two_coords(lat_inter,lat_ref[idx_ref],lon_inter,lon_ref[idx_ref])
                        if dist2 > MAX_DIST_INTER or dist1 > MAX_DIST_INTER:
                            print("No or wrong intersection found for this track")
                            continue

                        # get subsection of track
                        index = TRACK_REDUCTION_IS2*idx_is2
                        first_index = index-int(index/TRACK_REDUCTION_IS2) if index-int(index/TRACK_REDUCTION_IS2)>0 else 0
                        last_index = index+int(index/TRACK_REDUCTION_IS2) if index+int(index/TRACK_REDUCTION_IS2)< lat.size else lat.size-1
                        idx_sub = np.arange(first_index,last_index)
                        lat_sect = lat[idx_sub]
                        lon_sect = lon[idx_sub]
                        time_sect = time[idx_sub]

                        # convert to cartesien
                        x,y,z = cf.lon_lat_to_cartesian(lon_sect, lat_sect)
                        coordinates = np.vstack((x,y,z)).T            
                        distance,idx_in_ref = tree.query(coordinates,1)

                        # Selection close points
                        selected_idx = np.argwhere(distance < MAX_DIST_OF_COLLOC_DATA) # XXX same as IS2?
                        selected_idx = selected_idx.reshape((selected_idx.size,))
                        ref_idx =  idx_in_ref[selected_idx]

                        new_lat = lat_sect[selected_idx]
                        new_lon = lon_sect[selected_idx]
                        new_time = time_sect[selected_idx]
                        new_dist = distance[selected_idx]
                        weight = np.exp(-(new_dist**2)/(MAX_DIST_OF_COLLOC_DATA**2))

                        # Add data to array of lists
                        for idx in np.unique(ref_idx):
                            size = np.argwhere(ref_idx==idx).size
                            data_list['lat'][idx].extend(new_lat[np.argwhere(ref_idx==idx).flatten()].tolist())
                            data_list['lon'][idx].extend(new_lon[np.argwhere(ref_idx==idx).flatten()].tolist())
                            data_list['time'][idx].extend(new_time[np.argwhere(ref_idx==idx).flatten()].tolist())
                            delay = time_ref[idx] - new_time[np.argwhere(ref_idx==idx).flatten()].tolist() - delta_reftime
                            data_list['delay'][idx].extend(delay)
                            #if len(data_list['delay'][idx])>0:
                                #print('delay',idx,np.array(data_list['delay'][idx]))
                            data_list['dist'][idx].extend(new_dist[np.argwhere(ref_idx==idx).flatten()].tolist())
                            data_list['beam'][idx].extend((beam_dict[beam]*np.ones(size)).tolist())

                            data_list['weight'][idx].extend(weight[np.argwhere(ref_idx==idx).flatten()].tolist())
                        # Adding new track parameters data to list
                        #-------------------------------------------------
                        for pname,prodname in data_desc.items():

                            if pname in ['lat','lon','time','lat01','lon01','time01','hour','minute','second'] + param_list: #add 2D params later
                                continue

                            param,units,param_is_flag = cf.get_param_from_hf5(filename,data_desc,pname,'01',LAT_MIN)
                            if param is None: continue
                            param_sub = param[idx_sub]
                            for idx in np.unique(ref_idx):
                                data_list[pname][idx].extend(param[np.argwhere(ref_idx==idx).flatten()].tolist())
                                max_len = max(len(data_list[pname][idx]),max_len)

                                
            # converting arrays of lists to matrix
            #---------------------------------------------
                
            # initiating masked arrays
            data_ar = {}
            print("\nMax data at one crossing point: %i" %(max_len))
            for p in data_list.keys():
                data_ar[p] = ma.masked_array(np.zeros((N_MAX_CROSSPTS_IN_CS2BEAMS,ref_size)),mask=np.ones((N_MAX_CROSSPTS_IN_CS2BEAMS,ref_size)))

            # save into masked array
            #--------------------------------
            #data_param_ar = {}
            for p in data_list.keys():
                for ncol in range(ref_size):
                    # limit size of array
                    if len(data_list[p][ncol])>= N_MAX_CROSSPTS_IN_CS2BEAMS:
                        max_idx = N_MAX_CROSSPTS_IN_CS2BEAMS
                    else:
                        max_idx=len(data_list[p][ncol])

                    # save in column of masked array
                    data_ar[p][:max_idx,ncol] =np.array(data_list[p][ncol][:max_idx]).T
                    #if len(data_list[p][ncol][:max_idx])>0:
                    #    print(p,data_list[p][ncol][:max_idx])
                    #if max_idx>0: print(data_ar[p][:max_idx,ncol])

            # initialize track lists to save per colloc ref track the data
            if n==0:
                for pname in data_list.keys():
                    data_dict[gdr][pname] = list()

            # saving the data
            for pname in data_list.keys():
                data_ar[pname] = np.ma.masked_invalid(data_ar[pname],copy=True)                   
                data_dict[gdr][pname].append(data_ar[pname])                
    
    return data_dict


def concatenate_is2_data(date_list,is2_gdrs,file_dict,common_data_list,is2Beams):

    """
    Align IceSat-2 data products (ATL10,ATL07 ...etc) on reference track (usually ESA BASELINE-D official product) defined in function get_collocated_data 


    Args:
    date_list list(datetime) : list of required datetime objects
    file_dict dict()         : dictionnary of found files
    common_data_list         : daily list of common_track information (output of common_data_list)
    is2Beams                 : list of required IS2 beams from ['b1','b2','b3']
    """

   

    # Init IS2 data dictionnary
    is2_data_dict = dict()
    flag_data_dict = dict()
    print("\nConcatenating IS2 data\n#---------------\n")

    # Unwrap IS2 data beamwise
    for ngdr,gdr in enumerate(is2_gdrs):

        print("\n%s\n----------" %(gdr))
        flag_data_dict[gdr] = dict()
        is2_data_dict[gdr] = dict()
        
        for n,date in enumerate(date_list):

            date_str = date.strftime('%Y%m%d')
            flag_data_dict[gdr][date_str] = dict()
            print("\n%s" %(date.strftime('%d/%m/%Y')))
            filename = file_dict[gdr][date_str]

            # get beam names
            beamName = get_strong_beams(filename,is2Beams)
            
            for b in beamName.keys():

                flag_data_dict[gdr][date_str][b] = dict()
                
                # init data dict for beam b
                if n==0:is2_data_dict[gdr][b]= {}
            
                data_desc_is2 = is2_dict.init_dict(gdr,beamName[b],'granules')
                lat_i,lon_i,time_i,x_dist,valid_idx = cf.get_coord_from_hf5(filename,data_desc_is2,'01',LAT_MIN)

                # XXXX find start end from lat start/end like CS2
                selected_idx = common_data_list[n][gdr][b]['idx_is2']
                #selected_idx = np.arange(bound_idx[0],bound_idx[1]+1)
                ref_idx = common_data_list[n][gdr][b]['ref_idx']
                delay = common_data_list[n][gdr][b]['delay']
                dist = common_data_list[n][gdr][b]['dist']
                weight = np.exp(-(dist**2)/(MAX_DIST_OF_COLLOC_DATA**2))
                new_lat = lat_i[selected_idx]
                new_lon = lon_i[selected_idx]
                new_time = time_i[selected_idx]
            
                # initiating list
                list_param = ['id','ref_idx','dist','delay','latfull','lonfull','is2_idx','weight'] + [pname for pname in data_desc_is2.keys()]
                if n==0:
                    for p in list_param: is2_data_dict[gdr][b][p] = list()

                # Adding new track data to list
                is2_data_dict[gdr][b]['latfull'].append(lat_i)
                is2_data_dict[gdr][b]['lonfull'].append(lon_i)
                is2_data_dict[gdr][b]['lat'].append(new_lat)
                is2_data_dict[gdr][b]['lon'].append(new_lon)
                is2_data_dict[gdr][b]['time'].append(new_time)
                is2_data_dict[gdr][b]['id'].append(n*np.ones((selected_idx.size,)))
                is2_data_dict[gdr][b]['ref_idx'].append(ref_idx)
                is2_data_dict[gdr][b]['is2_idx'].append(selected_idx)
                is2_data_dict[gdr][b]['dist'].append(dist)
                is2_data_dict[gdr][b]['weight'].append(weight)
                is2_data_dict[gdr][b]['delay'].append(delay)
                
                # Look for parameters
                for pname,prodname in data_desc_is2.items():
                    
                    if pname in ['lat','lon','time']: continue
                    flag_data_dict[gdr][date_str][b][pname] = dict()                
                    if prodname is None: flag_data_dict[gdr][date_str][b][pname]['status']='NOK'; continue

                    param,units,param_is_flag = cf.get_param_from_hf5(filename,data_desc_is2,pname,'01',LAT_MIN)
                    
                    if param is None: flag_data_dict[gdr][date_str][b][pname]['status']='NOK'; continue
                    else: flag_data_dict[gdr][date_str][b][pname]['status']='OK'
                    
                    data_param = param[selected_idx]
                    is2_data_dict[gdr][b][pname].append(data_param)

                    flag_data_dict[gdr][date_str][b][pname]['units'] = units
                    flag_data_dict[gdr][date_str][b][pname]['is_flag'] = param_is_flag
                    flag_data_dict[gdr][date_str][b][pname]['name'] = prodname
                    
    return is2_data_dict,flag_data_dict



def get_beamwise_mean(date_list,ref_data_dict,is2_data_dict,is2_info_dict,is2Beams): #,common_data_list):

    """
    Compute weighted mean and std of IS2 data points sorted in each CS2 reference beams. The weighted coefficients are (1) the granules size (IS2 150 photons aggregates) and (2) the gaussian weight based on the distance of the IS2 granule from the CS2 beam center.

    The output is a array of IS2 data the same size as CS2 reference track.

    Args:
    date_list list(datetime) : list of required datetime objects
    file_dict dict()         : dictionnary of found files
    common_data_list         : daily list of common_track information (output of common_data_list)
    is2Beams                 : list of required IS2 beams from ['b1','b2','b3'] 

    """


    print("\n\nComputing mean IS2 values for each CS2 beams \n----------------------------")
    is2_data_list = dict()
    is2_list_param = dict()
    is2_data = dict() # transitory dictionnary
    
    list_params_coords = ['lat','lon','time']
    list_params_all = ['dist','delay','weight'] #id
    list_param_add = ['sla','isa']
    for gdr in is2_gdrs:

        print("%s \n----------" %(gdr))
        is2_data[gdr] = dict()
        data_desc_is2 = is2_dict.init_dict(gdr,'gt1r','granules')
        #is2_list_param[gdr] =  [pname for pname in data_desc_is2.keys() if pname not in list_params_coords] + list_params_all
        is2_list_param[gdr] =  [pname for pname in data_desc_is2.keys()] + list_params_all

        # check if all parameters are available
        if not all(pname in is2_list_param[gdr] for pname in ['surface_type','flag_leads','gaussian_w']):
            print("Missing some needed parameters ",['surface_type','flag_leads','gaussian_w'],"\n")
            sys.exit()

    print("IS2 data sorting for parameters",is2_list_param[gdr])
    
    # initiate matrix lists
    for n,date in enumerate(date_list):

      
        for gdr in is2_gdrs:
            is2_data_dict[gdr]['ndata'] = list()
            is2_data[gdr]['ndata'] = list()
            for p in [pname for pname in data_desc_is2.keys()]+list_param_add+list_params_all:

                if p=='flag_leads':
                    is2_data_dict[gdr][p] = list()
                    is2_data_dict[gdr][p+'_dist'] = list()
                    is2_data[gdr][p] = list()
                    is2_data[gdr][p+'_dist'] = list()
                    
                else:
                    is2_data_dict[gdr][p+'_mean'] = list()
                    is2_data_dict[gdr][p+'_std'] = list()
                    is2_data[gdr][p+'_mean'] = list()
                    is2_data[gdr][p+'_std'] = list()


    # fill matrices
    for n,date in enumerate(date_list):

        n_idx = 0
        date_str = date.strftime('%Y%m%d')
        print("\n%s" %(date.strftime('%d/%m/%Y')))
        ref_idx_list = ref_data_dict['ref_idx'][n]

        # For each CS2 beam
        #---------------------------------------
        for ref_idx in ref_idx_list:

            for gdr in is2_gdrs:


                # Gather all beam data in one list for each ref_id CS2 beams
                #-------------------------------------------------------------
                for p in is2_list_param[gdr]: is2_data_list[p] = list();
                # For each beam
                #beamName = is2_data_dict[gdr]['beamName']
                for b in is2Beams:

                    idx, = np.where(is2_data_dict[gdr][b]['ref_idx'][n]==ref_idx)
                    #if idx.size==0: continue
                    list_found_param = is2_list_param[gdr].copy()
                    #list_found_param_ATL10 = list_param_ATL10.copy()

                    for p in is2_list_param[gdr]:
                        
                        # if in list of common params
                        if p in list_params_all +list_params_coords:
                            is2_data_list[p].append(is2_data_dict[gdr][b][p][n][idx])
                        # if in list of specific params
                        elif p in is2_list_param[gdr]:
                            # check if data exists
                            # if not remove from list
                            if is2_info_dict[gdr][date_str][b][p]['status']=='NOK':
                                list_found_param.remove(p)
                                continue
                            # if yes: Add to the list of parameters
                            else: is2_data_list[p].append(is2_data_dict[gdr][b][p][n][idx])
                        else:
                            sys.exit("\nUnknown parameter %s for %s:%s" %(p,gdr,b))
                            #is2_data_list[p].append(is2_data_dict[gdr][b][p][n][idx])
                        
                list_param = list_found_param.copy()
                for p in list_param:
                    is2_data_list[p] = np.ma.concatenate(is2_data_list[p], axis=0 )

                
                # compute parameters for each IS2 beams
                #------------------------------------------------------------
                for nparam,p in enumerate(list_param+list_param_add+list_params_coords):

                    #if p in list_params_all: continue
                    if p=='weight': continue
                    
                    elif p=='flag_leads':
                        idx_leads = np.argwhere(is2_data_list[p]==2)
                        if idx_leads.size==0:
                            flagLeads=False
                            min_dist = np.nan
                        else:
                            flagLeads=True
                            min_dist = min(is2_data_list['dist'][idx_leads])[0]
                        is2_data[gdr][p].append(flagLeads)
                        is2_data[gdr][p+'_dist'].append(min_dist)
                        
                    elif p=='gaussian_w':
                        idx_floes = is2_data_list['surface_type']==1
                        mean,std,npts = get_weighted_stats(is2_data_list['Lseg'][idx_floes],is2_data_list['weight'][idx_floes],is2_data_list['gaussian_w'][idx_floes])
                        is2_data[gdr][p+'_mean'].append(mean)
                        is2_data[gdr][p+'_std'].append(std)

                    # warning interbeam sla not good!!
                    elif p=='sla':
                        idx_leads = is2_data_list['flag_leads']==2
                        mean,std,npts = get_weighted_stats(is2_data_list['Lseg'][idx_leads],is2_data_list['weight'][idx_leads],is2_data_list['surface_h'][idx_leads])
                        is2_data[gdr][p+'_mean'].append(mean)
                        is2_data[gdr][p+'_std'].append(std)
                        
                    elif p=='isa':
                        idx_floes = is2_data_list['surface_type']==1
                        mean,std,npts = get_weighted_stats(is2_data_list['Lseg'][idx_floes],is2_data_list['weight'][idx_floes],is2_data_list['surface_h'][idx_floes])
                        is2_data[gdr][p+'_mean'].append(mean)
                        is2_data[gdr][p+'_std'].append(std)

                    elif p=='laser_fb':
                        #if is2_data_list[p].size >0 and not all(np.isnan(is2_data_list[p])):
                        mean,std,npts = get_weighted_stats(is2_data_list['Lseg'],is2_data_list['weight'],is2_data_list[p])
                        is2_data[gdr][p+'_mean'].append(mean)
                        is2_data[gdr][p+'_std'].append(std)
                        is2_data[gdr]['ndata'].append(npts)

                    else:
                        mean,std,npts = get_weighted_stats(is2_data_list['Lseg'],is2_data_list['weight'],is2_data_list[p])
                        is2_data[gdr][p+'_mean'].append(mean)
                        is2_data[gdr][p+'_std'].append(std)

        # add daily tracks and mask invalid
        for gdr in is2_gdrs:
            for p in is2_data[gdr].keys():
                array = ma.masked_invalid(np.array(is2_data[gdr][p]))
                is2_data_dict[gdr][p].append(array)
                is2_data[gdr][p] = list()
                    
        n_idx += 1
        #print("stop")
                    

    
    
    return is2_data_dict

            


def sort_IS2_in_CS2_beam(date_list,is2_gdrs,ref_data_dict,is2_data_dict,is2_info_dict,common_data_list):

    """
    sort IS2 granule points associated with each CS2 beam in matrix where each columns correspond to the beams of CS2 reference track. The rows of the matrix are sorted by order of smallest distance to CS2 beam center.

     Args:
    date_list list(datetime) : list of required datetime objects
    file_dict dict()         : dictionnary of found files
    ref_data_dict            : dictionnary of reference data (outpout of concatenate_cs2 function)
    common_data_list         : daily list of common_track information (output of common_data_list)
    is2Beams                 : list of required IS2 beams from ['b1','b2','b3'] 

    """
    is2_data_list = dict()
    is2_list_param = dict()
    
    list_params_coords = ['lat','lon','time']
    list_params_all = ['beam','ref_idx','dist','delay','weight'] #id
    for gdr in is2_gdrs:
        data_desc_is2 = is2_dict.init_dict(gdr,'gt1r','granules')
        is2_list_param[gdr] = list_params_all + [pname for pname in data_desc_is2.keys()]
        

    # initiate matrix lists
    for n,date in enumerate(date_list):
        for gdr in is2_gdrs:
            for p in is2_list_param[gdr]: is2_data_dict[gdr][p] = list()
            #for p in is2_list_param['ATL10']: is2_data_dict['ATL10'][p] = list()        
    
    n_idx = 0
    # construct empty matrices
    for n,date in enumerate(date_list):
        
        n_cs2_beam = ref_data_dict['lat'][n].size
        # XXX Add also lat,lon, time in ATL07 or eliminate ATL10 or ATL07 distinction
        for gdr in is2_gdrs:
            for p in is2_list_param[gdr]:
                is2_data_dict[gdr][p].append(ma.masked_array(np.zeros((N_IS2PTS_IN_CS2BEAMS,n_cs2_beam)),mask=np.ones((N_IS2PTS_IN_CS2BEAMS,n_cs2_beam)),dtype=np.float64))

    for n,date in enumerate(date_list):
         for gdr in is2_gdrs:
             for b in beamName.keys():
                 
                 unique, counts = np.unique(is2_data_dict[gdr][b]['ref_idx'][n], return_counts=True)
                 print("%i:%s:%s - %i" %(n,gdr,b,np.max(counts)))
    
    # fill matrices
    for n,date in enumerate(date_list):

        n_idx = 0
        date_str = date.strftime('%Y%m%d')
        ref_idx_list = ref_data_dict['ref_idx'][n]

        # For each CS2 beam
        for ref_idx in ref_idx_list:

            # skip where REF_TRACK contains gaps in the data
            # because two many IS2 data linked to it
            if ref_idx in common_data_list[n]['idx_gaps'] or ref_idx in [np.min(ref_idx_list),np.max(ref_idx_list)]:
                continue
            
            for gdr in is2_gdrs:

                for p in is2_list_param[gdr]: is2_data_list[p] = list();
                # For each beam
                for b in beamName.keys():

                    idx, = np.where(is2_data_dict[gdr][b]['ref_idx'][n]==ref_idx)
                    #print("%s: %s - %i" %(gdr,b,idx.size))
                    list_found_param = is2_list_param[gdr].copy()
                    #list_found_param_ATL10 = list_param_ATL10.copy()

                    for p in is2_list_param[gdr]:
                        # if param is beam
                        if p=='beam':
                            #is2_data_list[p].append(np.array([b for n in np.arange(idx.size)]))
                            id_beam = beam_dict[b]
                            is2_data_list[p].append(np.array([id_beam for n in np.arange(idx.size)]))
                        # if in list of common params
                        elif p in list_params_all +list_params_coords:
                            is2_data_list[p].append(is2_data_dict[gdr][b][p][n][idx])
                        # if in list of specific params
                        elif p in is2_list_param[gdr]:
                            # check if data exists
                            # if not remove from list
                            if is2_info_dict[gdr][date_str][b][p]['status']=='NOK':
                                list_found_param.remove(p)
                                continue
                            # if yes: Add to the list of parameters
                            else: is2_data_list[p].append(is2_data_dict[gdr][b][p][n][idx])
                        else:
                            sys.exit("\nUnknown parameter %s for %s:%s" %(p,gdr,b))
                            #is2_data_list[p].append(is2_data_dict[gdr][b][p][n][idx])
                        
                list_param = list_found_param.copy()
                for p in list_param:
                    is2_data_list[p] = np.concatenate(is2_data_list[p], axis=0 )

                # Sort the data wrt distance to CS2 beam center
                distance = is2_data_list['dist']
                sorted_idx = np.argsort(distance)
                
                # Store data
                fill_idx = np.arange(sorted_idx.size)
                for p in list_param:
                    #if fill_idx.size > N_IS2PTS_IN_CS2BEAMS/2: print("%i:%s:%s - %i" %(n,gdr,ref_idx,fill_idx.size))

                    # Keeping matrix
                    #-------------------------                    
                    if fill_idx.size > N_MAX_IS2PTS_IN_CS2BEAMS:
                        print("\n WARNING: too many IS2 data (%s) for size %s for date #%i ref_id= %i" %(fill_idx.size,N_IS2PTS_IN_CS2BEAMS,n,ref_idx))
                        print("Increase N_IS2PTS_IN_CS2BEAMS or check ")
                        sys.exit()
                    elif fill_idx.size > N_IS2PTS_IN_CS2BEAMS:
                        is2_data_dict[gdr][p][n][:,n_idx][fill_idx[:N_IS2PTS_IN_CS2BEAMS]] =  is2_data_list[p][sorted_idx[:N_IS2PTS_IN_CS2BEAMS]]
                        #print("delete this data point")
                    else:
                        is2_data_dict[gdr][p][n][:,n_idx][fill_idx] =  is2_data_list[p][sorted_idx]
                        
            n_idx += 1
                    
    return is2_data_dict    



def print_status_params(cs2_info_dict,is2_info_dict,outpath,filename):

    """
    print parameter of sorted data for each collocated tracks

    """
     
    for n,date in enumerate(date_list):
        
        date_str = date.strftime('%Y%m%d')
        print('\nDate: %s\n' %(date.strftime('%d/%m/%Y')))
        print(" {:<8}".format('CS2')+'# ---------------------')
        fileN= outpath + filename+'.txt'
        with open(fileN, 'w') as file:

            file.write("\n     Data-status    \n------------------------\n")
            file.write("\n"+" {:<8}".format('CS2')+'#--------------------------- ')
            for ngdr,cs2_gdr in enumerate(cs2_info_dict.keys()):

                print('         # ', end='')
                date_cs2 = is2date_2_cs2date(date,is2_gdrs[0])
                date_str_cs2 = date_cs2.strftime('%Y%m%d')
                print("|".join(map(str,cs2_info_dict[cs2_gdr][date_str_cs2].keys())))
                file.write("\n         # "+"|".join(map(str,cs2_info_dict[cs2_gdr][date_str_cs2].keys())))
                
                
                status_list = list()
                print('         # ')
                for p in cs2_info_dict[cs2_gdr][date_str_cs2].keys():
                    status_list.append(cs2_info_dict[cs2_gdr][date_str_cs2][p]['status'])
                print(" {:<8}#      ".format(cs2_gdr), end='')
                print("   |   ".join(map(str,status_list)))
                file.write("\n"+" {:<8}#    ".format(cs2_gdr)+"   |    ".join(map(str,status_list)))
                
                print('         # ')
            print(" {:<8}".format('IS2')+'#--------------------------- ')
            file.write("\n"+" {:<8}".format('IS2')+'#--------------------------- ')
            for ngdr,is2_gdr in enumerate(is2_info_dict.keys()):

                beam = next(iter(is2_info_dict[is2_gdr][date_str].keys()))
                print('         # ', end='')
                print("|".join(map(str,is2_info_dict[is2_gdr][date_str][beam].keys())))
                file.write("\n         # "+"|".join(map(str,is2_info_dict[is2_gdr][date_str][beam].keys())))

                print('         # ')
                for b in is2_info_dict[is2_gdr][date_str].keys():
                    status_list = list()
                    for p in is2_info_dict[is2_gdr][date_str][b].keys():
                        status_list.append(is2_info_dict[is2_gdr][date_str][beam][p]['status'])
                    print(" {:<8}#      ".format(b), end='')
                    print("   |   ".join(map(str,status_list)))
                    file.write("\n"+" {:<8}#    ".format(b)+"   |    ".join(map(str,status_list)))
                    print('         # ')





def daterange(date1, date2):
    for n in range(int ((date2 - date1).days)+1):
        yield date1 + timedelta(n)
    




###########################################
#
#              Main
#
###########################################


if __name__ == '__main__':

    # Define programme description
    description ='Programm to read, select, compare and save collocated Icesat-2 and CryoSat-2 along-track data'

    # Initiate the parser with a description
    parser = argparse.ArgumentParser(description=description)

    # Save commande
    cmd = ' '.join(sys.argv)

    # Add long and short arguments
    # -------------------------------------------------------
    parent = parser.add_argument('-s', '--satellite',required=True,action=ParentAction)
    
    parser.add_argument("-g","--gdrs",help="set desired gdr (only for CryoSat-2)",action=ChildAction, parent=parent)
    
    
    parser.add_argument("-b","--is2Beams",required=True,help="provide IS2 strongs beams to plot")

    parser.add_argument("-d","--date",required=True,help="provide CS2 track date")

    parser.add_argument("-o","--outpath",default=PATH_OUT,help="[optionnal] provide outpath")

    parser.add_argument("-ofn","--outFolderName",default=None,help="[optionnal] provide outpath")

    parser.add_argument("-x","--xings",action="store_true",help="option to add data at crossovers")

    parser.add_argument("-sw","--swath",action="store_true",help="option to add swath aligned data")

    # Read arguments
    # ----------------------------------------------------------
    args = parser.parse_args()

    # Get satellite list
    # ---------------------------------------------------------
    sats = [sat for sat in args.satellite.items()]

    # Get gdr list for CS2 XXX only gdr for CS2
    # --------------------------------------------------------
    if sats[0][0] != 'CS2':
        print("\n First Sat -s must be CryoSat-2")
        sys.exit()
    
    cs2_gdrs = [gdr for gdr in sats[0][-1].gdrs.split(',')]

    if not cs2_gdrs[0]==REF_GDR: sys.exit("First CS2 prod provided must be ref track, in this case: %s" %REF_GDR)
    
    global avail_cs2_gdrs
    avail_cs2_gdrs = cs2_dict.get_gdr_list()

    for gdr in cs2_gdrs:
        if gdr not in avail_cs2_gdrs:
            print("\n WARNING:%s not found in:" %(gdr),avail_cs2_gdrs)
            print("Removing %s from list" %(gdr))
            cs2_gdrs.remove(gdr)
    if len(cs2_gdrs)==0:
        print("No available data in ",cs2_gdrs)
        print("Check CS2 dictionnary")
        sys.exit()


    # print
    idx_is2 = np.argwhere(np.array([sat[0] for sat in sats])=='IS2')[0]
    if idx_is2.size==0:
        print("\n IS2 must be provided in option -s")
        sys.exit()
    elif idx_is2.size==1:
        is2_gdrs = sats[idx_is2[0]][-1].gdrs.split(',')
    else:
        print("\n Too many -s IS2 provided")
        sys.exit()

    # get Is2 beams
    is2Beams = [b for b in args.is2Beams.split(',')]

    # Xings options
    #-----------------------------------------------------
    flag_xings = args.xings

    # Get GDR list for other satellites
    #------------------------------------------------------
    other_missions = dict()
    if flag_xings:
        for sat in sats:
            if sat[0]=='CS2': gdr_list = [sat[-1].gdrs.split(',')[0]]
            else:
                gdr_list = [gdr for gdr in sat[-1].gdrs.split(',')]
            other_missions[sat[0]]= gdr_list
        print("XINGS:",other_missions,"\n#----------------\n")
    else:
        print("NO XINGS: add option -x \n#----------------\n")


    # SWATH option
    #----------------------------------------------------
    flag_swath = args.swath

    # Get required dates
    # ---------------------------------------------------------
    date = [d for d in args.date.split(',')]
    if len(date) != 2: print('Provide date -d as YYYYMMDD,YYYYMMDD in arguments');sys.exit()

    start_date = datetime.strptime(date[0], '%Y%m%d')
    end_date = datetime.strptime(date[1], '%Y%m%d')
    
    print("\nAll CRYO2ICE tracks from %s to %s requested: \n" %(start_date.strftime("%d/%m/%y"),end_date.strftime("%d/%m/%y")))
    
    date_list=[start_date + timedelta(n) for n in range(int((end_date - start_date).days)+1)]

    
    # Get available files
    #------------------------------------------------------------
    file_dict_colloc ={}
    file_dict_all ={}
    ## Test if you always have corresponding data CS2 and IS2

    print("\n#Looking for available files\n#######################")
    
    # For Cryosat-2
    #----------------------
    avail_file_dict_cs2,avail_date_list_cs2 =get_avail_files('CS2',cs2_gdrs,date_list,True)
    file_dict_colloc['CS2'] = avail_file_dict_cs2
    
    # For Icesat-2
    #----------------------
    avail_file_dict_is2,avail_date_list_is2 =get_avail_files('IS2',is2_gdrs,date_list,True)
    file_dict_colloc['IS2'] = avail_file_dict_is2

    # For Xings points
    #-----------------
    for sat in other_missions.keys():
        gdr = other_missions[sat]
        date_list_xings = [dt for dt in daterange(date_list[0]+timedelta(-xing_delay),date_list[-1]+timedelta(xing_delay))]
        avail_file_dict,avail_date_list =get_avail_files(sat,other_missions[sat],date_list_xings,False)
        file_dict_all[sat] = avail_file_dict

    
    # Find corresponding dates for IS2/CS2 colocated tracks
    print("\nKeeping available dates")
    avail_date_list = avail_date_list_is2.copy()
    for d_is2 in avail_date_list_is2:
        #date_str = d.strftime('%Y%m%d')
        d_cs2 = is2date_2_cs2date(d_is2,is2_gdrs[0])
        if d_cs2 not in avail_date_list_cs2:
            avail_date_list.remove(d_is2)
            print("%s from IS2 not found for CS2 %s" %(d_is2.strftime('%d/%m/%Y'),d_cs2.strftime('%d/%m/%Y')))
    date_list = avail_date_list.copy()

    if len(date_list)==0:
        print("\nNo data found for period: %s - %s" %(start_date.strftime("%d/%m/%y"),end_date.strftime("%d/%m/%y"))); sys.exit()
    else:
        print("\nAll GDR data found for:")
        [print(d.strftime("%d/%m/%y")) for d in date_list]
        
    
    # Get position of matching data between IS2 and CS2
    # -----------------------------------------------------------
    print("\n# Sorting data\n##################")
    common_data_list = get_collocated_data(date_list,is2_gdrs,file_dict_colloc,is2Beams)

    # Concatanate data
    # -----------------------------------------------------------
    # Check available data within dictionnary
    # Select collocated section of each track
    # Concatanate data

    
    data_dict = dict()
    
    
    # Align find xings points with collocated tracks
    data_dict['CS2'],cs2_info_dict = concatenate_cs2_data(date_list,is2_gdrs,file_dict_colloc['CS2'],common_data_list)
    data_dict['IS2'],is2_info_dict = concatenate_is2_data(date_list,is2_gdrs,file_dict_colloc['IS2'],common_data_list,is2Beams)

    # get swath data
    if flag_swath:
        REF_GDR_IS2 = is2_gdrs[0]
        data_dict['CS2']['swath'] = dict()
        data_dict['IS2']['swath'] = dict()
        
        for b in is2Beams:
            data_dict['CS2']['swath'][b] = dict()
            data_dict['IS2']['swath'][b] = dict()
            swath_data_array = align_sla_swath_seg(date_list,file_dict_colloc,common_data_list,REF_GDR,REF_GDR_IS2,b,LAT_MIN)
        
            data_dict['CS2']['swath'][b] = swath_data_array['CS2']
            data_dict['IS2']['swath'][b] = swath_data_array['IS2']
    

    # Get only mean values or full matrices
    if flag_IS2_mean:
        data_dict['IS2'] = get_beamwise_mean(date_list,data_dict['CS2'][REF_GDR],data_dict['IS2'],is2_info_dict,is2Beams)

        # OPTION to eliminate granule data [to reduce size]
        for ngdr,gdr in enumerate(is2_gdrs):
            for b in is2Beams:
                data_dict['IS2'][gdr][b] = {}
    else:
        data_dict['IS2'] = sort_IS2_in_CS2_beam(date_list,data_dict['CS2'][REF_GDR],data_dict['IS2'],is2_info_dict,common_data_list)

    
    # find crossing for IS2 with function for each beam
    #for beam in beamName: # add beam
    #    data_dict['IS2'] = find_xings_data(satName,date_list,file_dict,common_data_list)
    for sat in other_missions.keys():
        #data_dict[sat]= {}
        data_dict[sat]['xings'] = {}
        if sat=='IS2':
            data_dict['IS2']['xings'] = find_xings2_is2(date_list,file_dict_all['IS2'],file_dict_colloc['IS2'],common_data_list,is2Beams)
        else:
            data_dict[sat]['xings'] = find_xings2_sat(sat,date_list,file_dict_all[sat],common_data_list)

    
    # add dates
    data_dict['dates'] = date_list
    

    # merge dictionnaries
    info_dict = dict()
    info_dict.update(cs2_info_dict)
    info_dict.update(is2_info_dict)
   
    
    # Save data
    # -----------------------------------------------------------.
    outfolder = args.outFolderName
    if outfolder==None: outfolder = start_date.strftime("%Y%m")
    outpath = args.outpath + outfolder+'/'
    if not os.path.exists(outpath):
        os.makedirs(outpath)
    #outpath = args.outpath + outfolder+'/'
    
    # Print parameter status
    print_status_params(cs2_info_dict,is2_info_dict,outpath,'status')

    # record filename in info_dict
    for gdr in info_dict.keys():
        for date in info_dict[gdr].keys():
            if 'ATL' in gdr:
                info_dict[gdr][date]['file'] =  file_dict_colloc['IS2'][gdr][date]
            else:
                info_dict[gdr][date]['file'] =  file_dict_colloc['CS2'][gdr][date]

    # Save info params dictionnary as text
    fileN = outpath + 'info_params'
    with open(fileN+'.txt', 'w') as file:
        file.write("Command: \n\n %s \n\n" %(cmd))
        file.write(json.dumps(info_dict, indent=4, sort_keys=True))
        
    # Save info params dictionnary as pikle
    dataFile = open(fileN+'.pkl', 'wb')
    pickle.dump(info_dict, dataFile)
    dataFile.close()

    # Save data dictionnary as pikle
    outFile = outpath + 'data_dict.pkl'
    dataFile = open(outFile, 'wb')
    pickle.dump(data_dict, dataFile)
    dataFile.close()
    print("\nfile: data_dict.pkl saved at: %s" %(outpath))
    
    

