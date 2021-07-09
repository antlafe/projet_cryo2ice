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

     python -m pdb check_siralb_cryo2ice.py -pi /work/ALT/odatis/seaice/users/laforga/data/CRYO2ICE/IS2/ATL10/202011/  -pc /work/ALT/odatis/seaice/users/laforga/data/CRYO2ICE/CS2/ESA_BD_GDR/202011/ -b b1 -p ssh -d20201101,20201105


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
import stats_tools as st

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


def align_sla_swath_seg(common_data_list,cs2_gdr,is2_gdr,is2_b,LAT_MIN):

    # parameters to align
    plist_cs2_20hz = ['sla']
    plist_cs2_1hz = ['mss','dac','earth','pole','lpe','ocean','load']
    plist_is2_granules = ['mss','dac','earth','pole','lpe','ocean','load']
    plist_is2_swath = ['slasw']

    cs2_pDict = {}
    is2_pDict = {}
    swath_data_array = {}
    swath_data_array['CS2'] = {}; swath_data_array['IS2'] = {};

    for n,date in enumerate(date_list):

        # initiate lists
        if n==0:
            
            for p in plist_cs2_20hz + plist_cs2_1hz + ['lat','lon']:
                cs2_pDict[p] = list()
                swath_data_array['CS2'][p] = list()
                
            for p in plist_is2_granules + plist_is2_swath + ['lat','lon']:
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
            param_hf = cf.interp_1hz_to_20hz(param,time_c_01,time_c_20)
            cs2_pDict[pname].append(param_hf)
        


        # Get IS2 sla + corrs
        #------------------------
        
        date_str = date.strftime('%Y%m%d')
        filename = file_dict['IS2'][is2_gdr][date_str]
        is2_beam = snsc.get_strong_beams(filename,[is2_b])

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
        lat_swath = lat_i_sw[flag_colloc]; lon_swath = lon_i_sw[flag_colloc]
        ref_size = lat_swath.shape[0]
        coord_sw = np.vstack((lon_swath,lat_swath)).T
        tree = scipy.spatial.cKDTree(coord_sw)


        # Associate CS2 20hz data association to 10km SWATH-SEG coordinate
        coord_cs2 = np.vstack((lon_c_20,lat_c_20)).T
        distance,closest_ind = tree.query(coord_cs2,1)

        
        flag_colloc = distance < 1
        selected_idx = closest_ind[flag_colloc]
        
        data_array_cs2 = dict()
        for p in cs2_pDict.keys():
            data_array_cs2[p] = ma.masked_array(np.zeros((ref_size,)),mask=np.ones((ref_size,)))
            
        for idx in np.unique(selected_idx):
            argidx = np.argwhere(closest_ind==idx)
            for p in cs2_pDict.keys():
                data_array_cs2[p][idx] = np.ma.mean(cs2_pDict[p][n][argidx]) 
            
            
        # Associate IS2 granule data to 10km SWATH-SEG coordinates
        coord_is2 = np.vstack((lon_i_gran,lat_i_gran)).T
        distance,closest_ind = tree.query(coord_is2,1)

        flag_colloc = distance < 1
        selected_idx = closest_ind[flag_colloc]
        idx_swath = np.unique(selected_idx)
        
        data_array_is2 = dict()
        for p in is2_pDict.keys():
            data_array_is2[p] = ma.masked_array(np.zeros((ref_size,)),mask=np.ones((ref_size,)))

        data_array_is2['lat'] = lat_swath #[flag_colloc]
        data_array_is2['lon'] = lon_swath #[flag_colloc]
        #data_array_is2['slasw'] = is2_pDict['slasw'][n][idx_swath]
            
        for idx in idx_swath:
            argidx = np.argwhere(closest_ind==idx)
            for p in is2_pDict.keys():
                if p in ['lat','lon']: continue
                if 'sla' in p:
                    data_array_is2[p][idx] = is2_pDict[p][n][idx]
                else:
                    data_array_is2[p][idx] = np.ma.mean(is2_pDict[p][n][argidx])


        # Adding data to final array
        for p in is2_pDict.keys():
            swath_data_array['IS2'][p].append(data_array_is2[p])

        # Adding data to final array
        for p in cs2_pDict.keys():
            swath_data_array['CS2'][p].append(data_array_cs2[p])

        
    return swath_data_array  


            

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

    #parser.add_argument("-p","--parameter",required=True,help="parameter to test")

    parser.add_argument("-b","--is2Beams",required=True,help="provide IS2 strongs beams to plot")

    parser.add_argument("-o","--pathout",default=None,help="provide pathout to save figues")

    # Read arguments
    # ----------------------------------------------------------
    args = parser.parse_args()

    date = [d for d in args.date.split(',')]
    if len(date) != 2: print('Provide date -d as YYYYMMDD,YYYYMMDD in arguments');sys.exit()

    start_date = datetime.strptime(date[0], '%Y%m%d')
    end_date = datetime.strptime(date[1], '%Y%m%d')
   
    print("\nAll CRYO2ICE tracks from %s to %s requested: \n" %(start_date.strftime("%d/%m/%y"),end_date.strftime("%d/%m/%y")))
    
    date_list=[start_date + timedelta(n) for n in range(int((end_date - start_date).days)+1)]
   
    mid_date = (date_list[0]+(date_list[1]-date_list[0])/2)
    mid_month =  mid_date.strftime('%Y%m')
    
    #parameter = args.parameter
    pathout = args.pathout

    # Read spreadsheet of collocated tracks: Don't forget to update spread sheet
    #-----------------------------------------------------------
    """
    loc = (spreadsheetpath)
    wb = xlrd.open_workbook(loc)
    sheet = wb.sheet_by_index(0)
    
    is2_colloc_files = list()
    for i in range(sheet.nrows):
        is2_colloc_files.append(sheet.cell_value(i, 8))
    """

    # Get IS2 files
    # -------------------------------------------------------
    #print("IS2 files \n ##########")
    path_IS2 = args.inputpathIS2
    file_list = list()
    date_colloc_list = list()
    for dt in date_list:
        #print("%s\n" %(dt))
        datestr = dt.strftime("%Y%m%d")
        filepattern = path_IS2 + "*%s*.h5" %(datestr)
        filename = glob.glob(filepattern)
        if len(filename)==0: continue
        file_list.extend(filename)
        #print("%s" %(filename))
        date_colloc_list.append(dt)
            
    if len(file_list)==0:
        #print("No file found in %s for %s,%s" %(path_IS2,date[0],date[-1]))
        sys.exit()

    # get Is2 beams
    is2Beams = [b for b in args.is2Beams.split(',')]

    
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
    #print("IS2 files \n ##########")
    path_CS2 = args.inputpathCS2

    file_list_cs2 = list()
    file_list_is2 = list()
    date_list = list()
    for dt in date_colloc_list:
        #print("%s\n" %(dt))
        date_cs2 = is2date_2_cs2date(dt,REF_GDR_IS2)
        datestr = date_cs2.strftime("%Y%m%d")
        filepattern = path_CS2 + "*_%s*.nc" %(datestr)
        filename = glob.glob(filepattern)
        if len(filename)==0:
            #print("No file found")
            continue
        else:
            #print("%s" %(filename))
            idx = date_colloc_list.index(dt)
            file_list_cs2.extend(filename)
            file_list_is2.append(file_list[idx])
            date_list.append(dt)
            
            
    if len(file_list)==0:
        print("No file found in %s for %s,%s" %(path_IS2,date[0],date[-1]))
        sys.exit()

    # creating dictionnaries
    #---------------------------
    file_dict = {}
    file_dict['CS2'] = {}
    file_dict['CS2'][REF_GDR_CS2] = {}
    file_dict['IS2'] = {}
    file_dict['IS2'][REF_GDR_IS2] = {}
    for ndate,dt in enumerate(date_list):

        # filling IS2 dictionnary
        date_is2 = dt.strftime("%Y%m%d")
        file_dict['IS2'][REF_GDR_IS2][date_is2] = file_list_is2[ndate]

        # filling CS2 dictionnary
        datetime_cs2 = is2date_2_cs2date(dt,REF_GDR_IS2)
        date_cs2 = datetime_cs2.strftime("%Y%m%d")
        file_dict['CS2'][REF_GDR_CS2][date_cs2] = file_list_cs2[ndate]

   
    print("Found files: \n#------------------------")
    print(json.dumps(file_dict,sort_keys=True, indent=4))
        
    
    # Check if files are well collocated
    common_data_list = snsc.get_collocated_data(date_list,file_dict,is2Beams)

    

    
   
    is2_gdr='ATL10'
    cs2_gdr='ESA_BD_GDR'
    is2_b = 'b2'
    LAT_MIN=60
    swath_data_array = align_sla_swath_seg(common_data_list,cs2_gdr,is2_gdr,is2_b,LAT_MIN)

    
        
    print('[strop')
        
    

    # Get aligned data
    data_dict = dict()
    data_dict['CS2'],cs2_info_dict = snsc.concatenate_cs2_data(date_list,file_dict['CS2'],common_data_list)
    data_dict['IS2'],is2_info_dict = snsc.concatenate_is2_data(date_list,file_dict['IS2'],common_data_list,is2Beams)
    data_dict['IS2'] =  snsc.get_beamwise_mean(date_list,data_dict['CS2'][REF_GDR_CS2],data_dict['IS2'],is2_info_dict)

    #snsc.print_status_params(cs2_info_dict,is2_info_dict,outpath,'status')
    
    # compare and display the data
    #if parameter=='fb':

    lon = np.ma.concatenate(data_dict['CS2'][REF_GDR_CS2]['lonref'],axis=0) 
    lat = np.ma.concatenate(data_dict['CS2'][REF_GDR_CS2]['latref'],axis=0)
        
    radar_fb = np.ma.concatenate(data_dict['CS2'][REF_GDR_CS2]['radar_fb'],axis=0)
    laser_fb = np.ma.concatenate(data_dict['IS2'][REF_GDR_IS2]['laser_fb_mean'],axis=0)
    delta_fb = laser_fb - radar_fb 


    # histogramm
    #-------------------
    f2, axh = plt.subplots(1, 1,figsize=(8,8))
    f2.suptitle('Histogram of FB', fontsize=14)
    xylim = [-0.1,0.5]
    label_list = list()
    data_list = list()
    label_is2 = 'laser FB IS2',is2Beams,'(m)'
    legend_list = [label_is2,'radar FB CS2 (m)']
    data_list = [laser_fb,radar_fb]
    xlabel = 'freeboard (m)'
    st.plot_histo(axh,xylim,'m',xlabel,legend_list,data_list,True)

    if pathout is not None:
        plt.savefig(pathout+'histo_fb_%s.png' %(mid_month))
    else:
        plt.show()


    # scatter plot
    #---------------
    xylim = [[-0.1,0.5], [-0.1,0.5]]
    f1, ax = plt.subplots(1, 1, sharey=True)
    f1.suptitle('Scatter plot', fontsize=12)
    y_data = laser_fb
    y_label = 'laser FB IS2 (m)'
    x_label='radar FB CS2 (m)'
    x_data = radar_fb
    st.plot_scatter(ax,xylim,'','m',x_data,x_label,y_data,y_label,None)
    #plt.savefig(pathout+'histo_ssh_%s.png' %(mid_month))
    
    if pathout is not None:
        plt.savefig(pathout+'scatter_fb_%s.png' %(mid_month))
    else:
        plt.show()
    
    # map
    #-----------------
    f1, ax = plt.subplots(1, 1,figsize=(8,6))
    xylim = [0,0.4]
    f1.suptitle('Delta fb (la-ku) %s' %(mid_month), fontsize=12)
    bmap,cmap = st.plot_track_map(f1,ax,lon,lat,delta_fb,'Delta fb(La-ku)',xylim,None,'m',True,alpha=1)

    if pathout is not None:
        plt.savefig(pathout+'map_delta_fb_%s.png' %(mid_month))
    else:
        plt.show()
    


    #elif parameter=='ssh':

    # SLA CS2
    sla_cs2 = np.ma.concatenate(data_dict['CS2'][REF_GDR_CS2]['sla'],axis=0)
    mss_cs2 = np.ma.concatenate(data_dict['CS2'][REF_GDR_CS2]['mss'],axis=0)

    # corrections
    #array_model = np.ma.zeros(sla_cs2.shape)
    earth = np.ma.concatenate(data_dict['CS2'][REF_GDR_CS2]['solid_earth_tide'],axis=0)
    lpe = np.ma.concatenate(data_dict['CS2'][REF_GDR_CS2]['ocean_eq_tide'],axis=0)
    dac = np.ma.concatenate(data_dict['CS2'][REF_GDR_CS2]['dac'],axis=0)
    pole = np.ma.concatenate(data_dict['CS2'][REF_GDR_CS2]['pole_tide'],axis=0)
    ocean = np.ma.concatenate(data_dict['CS2'][REF_GDR_CS2]['ocean_tide'],axis=0)
    load = np.ma.concatenate(data_dict['CS2'][REF_GDR_CS2]['load_tide'],axis=0)
    #list_corrs = [earth,lpe,dac,pole,ocean,load]
    corrections_cs2 = earth + lpe + dac + ocean + load +pole
    
    # corrections
    earth = np.ma.concatenate(data_dict['IS2'][REF_GDR_IS2]['earth_mean'],axis=0)
    lpe = np.ma.concatenate(data_dict['IS2'][REF_GDR_IS2]['lpe_mean'],axis=0)
    dac = np.ma.concatenate(data_dict['IS2'][REF_GDR_IS2]['dac_mean'],axis=0)
    pole = np.ma.concatenate(data_dict['IS2'][REF_GDR_IS2]['pole_mean'],axis=0)
    ocean = np.ma.concatenate(data_dict['IS2'][REF_GDR_IS2]['ocean_mean'],axis=0)
    load = np.ma.concatenate(data_dict['IS2'][REF_GDR_IS2]['load_mean'],axis=0)
    corrections_is2 = earth + lpe + dac + ocean + load  +pole

    # SLA IS2
    sla_is2 = np.ma.concatenate(data_dict['IS2']['ATL10']['sla_mean'],axis=0)
    mss_is2 = np.ma.concatenate(data_dict['IS2']['ATL10']['mss_mean'],axis=0)
    sla_is2 = sla_is2 + mss_is2 - mss_cs2 + corrections_is2 - corrections_cs2

    delta_sla = sla_is2 - sla_cs2
        
    # histogramm
    #-------------------
    f2, axh = plt.subplots(1, 1,figsize=(8,8))
    f2.suptitle('Histogram of SSH', fontsize=14)
    xylim = [-0.1,0.5]
    legend_list = ['SLA IS2 (m)','SLA CS2 (m)']
    data_list = [sla_is2,sla_cs2]
    xlabel = 'SLA (m)'
    st.plot_histo(axh,xylim,'m',xlabel,legend_list,data_list,True)
    #plt.savefig(pathout+'histo_ssh_%s.png' %(mid_month))

    if pathout is not None:
        plt.savefig(pathout+'histo_ssh_%s.png' %(mid_month))
    else:
        plt.show()

    # scatter plot
    #---------------
    xylim = [[-0.1,0.5], [-0.1,0.5]]
    f1, ax = plt.subplots(1, 1, sharey=True)
    f1.suptitle('Scatter plot', fontsize=12)
    x_data = sla_is2
    x_label = 'SLA IS2 (m)'
    y_label='SLA CS2 (m)'
    y_data = sla_cs2
    st.plot_scatter(ax,xylim,'','m',x_data,x_label,y_data,y_label,None)
    #plt.savefig(pathout+'scatter_ssh_%s.png' %(mid_month))

    if pathout is not None:
        plt.savefig(pathout+'scatter_ssh_%s.png' %(mid_month))
    else:
        plt.show()


    # map
    #-----------------
    f1, ax = plt.subplots(1, 1,figsize=(8,6))
    xylim = [-0.1,0.1]
    f1.suptitle('Delta sla (la-ku) %s' %(mid_month), fontsize=12)
    bmap,cmap = st.plot_track_map(f1,ax,lon,lat,delta_fb,'Delta sla(La-ku)',xylim,mid_date,'m',True,alpha=1)

    if pathout is not None:
        plt.savefig(pathout+'map_delta_sla_%s.png' %(mid_month))
    else:
        plt.show()
        

    
