#! /home/antlafe/anaconda3/bin/python

#
# statistics_cryo2ice.py
#
# ----------------------------------------------------------------------
# Copyright (c) 2020 LEGOS/SERCO
# All rights reserved.
#



"""
DESCRIPTION:

     Program for carrying out statistical analysis between Icesat-2 and Cryosat-2 over collocated track sections for various parameters: surface_type, surface_height, roughness, freeboard

USAGE:

     statistics_cryo2ice.py [options]

optional arguments:



EXAMPLES:

    python statistics_cryo2ice.py -f 202003 -g ESA_BD,AWI -p surface_classif -d20200301,20200303

    python -m pdb statistics_cryo2ice.py -f 202010_ESABD -g ESA_BD -p snow_depth -d20201010,20201111

    python -m pdb statistics_cryo2ice_mean.py -f NovJan_all -g ESA_BD_GDR,LEGOS_SAM,LEGOS_T50,CPOM,UOB -p comp_grid -d20201101,20201130

    python -m pdb statistics_cryo2ice_mean.py -f NovMar_ESA -g ESA_BD_GDR -p simba -d20201101,20201130


COMMENTS:

    - Only one product at once  

"""

import sys
import h5py
import netCDF4 as nc
import numpy as np
from numpy import ma 
from mpl_toolkits.basemap import Basemap
from mpl_toolkits.axes_grid import make_axes_locatable
import matplotlib.pyplot as plt
from matplotlib import cm
import glob
from datetime import date, timedelta, datetime
import argparse
import cs2_dict
import is2_dict
import path_dict
import common_functions as cf
import warnings
import scipy.spatial
import pickle
from scipy.stats import pearsonr, gaussian_kde,linregress
import matplotlib as mpl
import stats_tools as st
import cs2_dict
import pandas as pd
import seaborn as sns
import matplotlib.dates as mdates
import os
        

# Global attributs
###########################################

varHome = os.environ['HOME']

PATH_DATA = path_dict.PATH_DICT[varHome]['PATH_DATA']
PATH_INPUT = path_dict.PATH_DICT[varHome]['PATH_OUT']
#PATH_OUT = "/home/antlafe/Documents/work/projet_cryo2ice/outputs/"

PATH_GRID = path_dict.PATH_DICT[varHome]['PATH_GRID']

param_opts = ['sd_month','find_regions','simba','mean_grid','comp_grid','roughness','data_maps','show_data','sd_comp','xings','xings2','simba_grid']

beamList=['gt1r','gt2r','gt3r'] #,'gt1l','gt2l','gt3l']
MAX_ACROSS_DIST = 1.5 #KM
MAX_NADIR_DIST = 0.1 #KM


###########################################
#
#              Main
#
###########################################


if __name__ == '__main__':

    # Define programme description
    description ='Program for carrying out statistical analysis between Icesat-2 and Cryosat-2 over collocated track sections for various parameters: surface_type, surface_height, roughness, freeboard'

    # Initiate the parser with a description
    parser = argparse.ArgumentParser(description=description)

    # Add long and short arguments
    # -------------------------------------------------------
    parser.add_argument("-f","--inputfolder",help="provide input pickle data file",required=True)

    parser.add_argument("-g","--gdrs",help="set desired CS2 products to analysised",required=True)

    parser.add_argument("-d","--date",required=True,help="provide CS2 track date")

    parser.add_argument("-p","--parameter",default='list',help="provide parameter to be tested")

    #parser.add_argument("-o","--outpath",default=PATH_OUT,help="[optionnal] provide outpath")

    #parser.add_argument("-fn","--fileName",default='data_dict',help="[optionnal] provide outpath")

    # Read arguments
    # ----------------------------------------------------------
    args = parser.parse_args()

    # Open pkl dictionnary file
    # ---------------------------------------------------------
    inputfolder = args.inputfolder+'/'
    inputfilepattern = PATH_INPUT + inputfolder + 'data_dict.pkl'
    filename = glob.glob(inputfilepattern)
    if len(filename)==0: sys.exit("\n%s: No found" %(inputfilepattern))

    pkl_file = open(filename[0], 'rb')
    data_dict = pickle.load(pkl_file)

    # Open info params file
    #-----------------------------------------------------------
    inputfilepattern = PATH_INPUT + inputfolder + 'info_params.pkl'
    filename = glob.glob(inputfilepattern)
    if len(filename)==0:
        print("\n%s: Not found" %(inputfilepattern))
        sys.exit()
    pkl_file = open(filename[0], 'rb')
    info_params = pickle.load(pkl_file)
    
    # Get gdr list
    # ---------------------------------------------------------
    prod_L2P = [g for g in args.gdrs.split(',')]
    #prod_L2P = ["LEGOS_T50","CPOM","ESA_BD_GDR","UOB","LEGOS_SAM"]

    # test if prod is available in pkl file
    if not all([g is not data_dict['CS2'].keys() for g in prod_L2P]):
        print("No %s in" %(prod_L2P),data_dict['CS2'].keys(),"in file %s" %(filename))
        sys.exit()

    N_prod = len(prod_L2P)
    REF_GDR = next(iter(data_dict['CS2']))

    # Get dates
    # ---------------------------------------------------------
    date = [d for d in args.date.split(',')]
    if len(date) != 2: print('Provide date -d as YYYYMMDD,YYYYMMDD in arguments');sys.exit()

    start_date = datetime.strptime(date[0], '%Y%m%d')
    end_date = datetime.strptime(date[1], '%Y%m%d')

    requested_dates =  [start_date + timedelta(n) for n in range(int((end_date - start_date).days)+1)]
    available_dates = data_dict['dates']
    
    mid_date = data_dict['dates'][0] + (data_dict['dates'][-1] - data_dict['dates'][0])/2
    enddate = data_dict['dates'][-1]
    found_dates = [d for d in requested_dates if d in available_dates]
    date_period_str = [found_dates[0].strftime('%d/%m/%Y'),found_dates[-1].strftime('%d/%m/%Y')]
    if len(found_dates)==0: print('Unavailable dates',date,'\nAvailable dates are:',[d.strftime("%Y%m%d") for d in available_dates]);sys.exit()
    ndates = len(found_dates)
    idx_dates = np.array([available_dates.index(date) for date in found_dates])
    month_list = list(np.unique([d.strftime('%Y%m') for d in found_dates]))
    
    monthstr_list = [d.strftime('%b-%Y') for d in found_dates]
    _, idx = np.unique(monthstr_list, return_index=True)
    monthstr_list = np.array(monthstr_list)[np.sort(idx)]

    idx_dates_monthly = dict()
    for month in month_list:
        idx_dates_monthly[month] = np.array([found_dates.index(date) for date in found_dates if date.strftime('%Y%m')==month])

    start_date = found_dates[0].strftime('%d/%m/%Y')
    end_date = found_dates[-1].strftime('%d/%m/%Y')

    
    # Get parameters
    # ---------------------------------------------------------
    param = [p for p in args.parameter.split(',')]

    if 'list' in param:
        print("\nChoose param -p within:",param_opts)
        sys.exit()
        
    if len(param)>1:
        print("WARNING: Only one parameter -p expected within",param_opts)
        sys.exit()
    else: param = param[0]
        
    if param not in param_opts:
        print("Provide -p parameters within",param_opts)
        sys.exit()

    # Get coordinates
    #------------------------------------------

    # ref coordinates
    ref_seg_lat = list(np.array(data_dict['CS2'][REF_GDR]['latref'],dtype=object)[idx_dates])
    ref_seg_lon = list(np.array(data_dict['CS2'][REF_GDR]['lonref'],dtype=object)[idx_dates])

    ref_seg_time = list(np.array(data_dict['CS2'][REF_GDR]['time'],dtype=object)[idx_dates])

    # mean dist between points
    lat1 = np.ma.concatenate(ref_seg_lat,axis=0)[:-1]; lat2=np.ma.concatenate(ref_seg_lat,axis=0)[1:]
    lon1 = np.ma.concatenate(ref_seg_lon,axis=0)[:-1]; lon2=np.ma.concatenate(ref_seg_lon,axis=0)[1:]
    mean_dist_btw_data = np.median(cf.dist_btw_two_coords(lat1,lat2,lon1,lon2))
    #window_size = int(mean_dist_btw_data*25) #km


    # Get Osisaf Ice Type
    #--------------------------------------------------------
    lons_icetype = list()
    lats_icetype = list()
    #icetype = list()
    icetype_al = list()
    icetype = list()
    
    for n in range(ndates):

        date = found_dates[n]
        lons,lats,OSISAF_ice_type = cf.get_osisaf_ice_type(date.year,date.month,date.day,'01')
        OSISAF_ice_type[OSISAF_ice_type==1] = ma.masked # masked ocean
        OSISAF_ice_type[OSISAF_ice_type==3] = 4 # ambigous becomes multi-year ice
        OSISAF_ice_type[560,380] = 2 # to keep colorbar

        lons_icetype.append(lons)
        lats_icetype.append(lats)
        icetype.append(OSISAF_ice_type)
        lon = ref_seg_lon[n]
        if any(np.abs(np.diff(lon)) > 20): lon[lon > 180] = lon[lon > 180] - 360
        icetype_alongtrack = cf.grid_to_track(OSISAF_ice_type,lons,lats,lon,ref_seg_lat[n])
        icetype_al.append(icetype_alongtrack)

    
        


    # Summarizing required analysis
    # --------------------------------------------------------
    print("\nStatistical analysis of %s between %s-%s for" %(prod_L2P,start_date,end_date),param)

    # Statistical analyses
    #---------------------------------------------------------

    #if param=='ST_cryo2ice':

    #meanSnow_MYI = []
    #snow_FYI = []

    if param=='show_data':

        pname='radar_fb'
        LAT_MIN = 55
        param_list = list()
        lat_list = list()
        lon_list = list()

        for n,date in enumerate(month_list):

            for nprod,cs2_gdr in enumerate(prod_L2P):

                # Get parameters
                data_desc_cs2 = cs2_dict.init_dict(cs2_gdr,False)
                
                path_data = '/home/antlafe/Documents/work/projet_cryo2ice/data/CS2/%s/%s/' %(cs2_gdr,date)
                filepattern = path_data+'*%s*.*' %(date[:4])
                filename = glob.glob(filepattern)
                if len(filename)==0: sys.exit("\n%s: No found" %(filepattern))        
                file_format = filename[0].split('.')[-1]

                for fileN in filename:

                    print(fileN)
                    # Get coords in NetCDF
                    if file_format=='nc':
                        lat,lon,time,x_dist,valid_idx = cf.get_coord_from_netcdf(fileN,data_desc_cs2,'01',LAT_MIN)
                        param,units,param_is_flag = cf.get_param_from_netcdf(fileN,data_desc_cs2,pname,'01',LAT_MIN)        

                    # Get coords in txt file
                    elif file_format=='txt': #XXX      
                        lat,lon,time,x_dist,valid_idx = cf.get_coord_from_uob(fileN,data_desc_cs2,'01',LAT_MIN)
                        param,units,param_is_flag = cf.get_param_from_uob(fileN,data_desc_cs2,pname,'01',LAT_MIN)        

                    # Get coords from .dat file
                    elif file_format=='dat': #XXX
                        lat,lon,time,x_dist,valid_idx = cf.get_coord_from_cpom(fileN,data_desc_cs2,'01',LAT_MIN)
                        param,units,param_is_flag = cf.get_param_from_cpom(fileN,data_desc_cs2,pname,'01',LAT_MIN)        

                    else:
                        print("Unknown file format")

                    param_list.append(param)
                    lat_list.append(lat)
                    lon_list.append(lon)
                
                param_list = np.ma.concatenate(param_list,axis=0)
                lat_list = np.ma.concatenate(lat_list,axis=0)
                lon_list = np.ma.concatenate(lon_list,axis=0)

        f1, ax = plt.subplots(1, 1,figsize=(9,8))
        bmap,cmap = st.plot_track_map(f1,ax,lon_list,lat_list,param_list,pname,[0,0.5],mid_date,'m',False,alpha=1,size=5)
        
        plt.show()
        
    if param=='xings':


        # global param
        show_colloc=True
        delay_path = 24 #hours
        to_sec = 60*60
        delay_inc = delay_path*to_sec

        lon = np.ma.concatenate(list(np.array(data_dict['CS2']['ESA_BD_GDR']['lonref'],dtype=object)[idx_dates]),axis=0)
        lat = np.ma.concatenate(list(np.array(data_dict['CS2']['ESA_BD_GDR']['latref'],dtype=object)[idx_dates]),axis=0)

        # calculating absolute orbit number
        ndata_per_tracks = [0]
        abs_ref_idx_cs2 = list()
        abs_ref_idx_is2 = list()
        for nidx,idx in enumerate(idx_dates):
            ndata_tracks = data_dict['CS2']['ESA_BD_GDR']['lat'][idx].size
            ndata_per_tracks.append(ndata_per_tracks[nidx]+ndata_tracks)

            abs_ref_idx_cs2.append(np.array(data_dict['CS2']['xings']['ESA_BD_GDR']['idx_ref'][idx])+ ndata_per_tracks[nidx])
            
            abs_ref_idx_is2.append(np.array(data_dict['IS2']['xings']['ATL10']['idx_ref'][idx])+ ndata_per_tracks[nidx])
            

        
        abs_ref_idx_cs2 = np.ma.concatenate(abs_ref_idx_cs2,axis=0)
        abs_ref_idx_is2 = np.ma.concatenate(abs_ref_idx_is2,axis=0)

        # collocated data
        #-------------------------------------------
        laser_fb_colloc = np.ma.concatenate(list(np.array(data_dict['IS2']['ATL10']['laser_fb_mean'],dtype=object)[idx_dates]),axis=0)
        radar_fb_colloc = np.ma.concatenate(list(np.array(data_dict['CS2']['ESA_BD_GDR']['radar_fb'],dtype=object)[idx_dates]),axis=0)
        delta_fb_colloc = laser_fb_colloc- radar_fb_colloc
        delay_colloc = 3.5*60*60
        delayC = np.ma.ones(laser_fb_colloc.shape)*delay_colloc

       
        # Get CS2 cross-overs
        #-------------------------------------------
        lat_xings = np.ma.concatenate(list(np.array(data_dict['CS2']['xings']['ESA_BD_GDR']['lat'],dtype=object)[idx_dates]),axis=0)
        lon_xings = np.ma.concatenate(list(np.array(data_dict['CS2']['xings']['ESA_BD_GDR']['lon'],dtype=object)[idx_dates]),axis=0)
        radar_fb_xings = np.ma.concatenate(list(np.array(data_dict['CS2']['xings']['ESA_BD_GDR']['radar_fb'],dtype=object)[idx_dates]),axis=0)
        ref_idx_cs2 =  np.ma.concatenate(list(np.array(data_dict['CS2']['xings']['ESA_BD_GDR']['idx_ref'],dtype=object)[idx_dates]),axis=0)
        delay_cs2 =  np.ma.concatenate(list(np.array(data_dict['CS2']['xings']['ESA_BD_GDR']['delay'],dtype=object)[idx_dates]),axis=0)


        # Get IS2 cross-overs
        #-------------------------------------------
        
        lat_is2xings = np.ma.concatenate(list(np.array(data_dict['IS2']['xings']['ATL10']['lat'],dtype=object)[idx_dates]),axis=0)
        lon_is2xings = np.ma.concatenate(list(np.array(data_dict['IS2']['xings']['ATL10']['lon'],dtype=object)[idx_dates]),axis=0)
        laser_fb_xings = np.ma.concatenate(list(np.array(data_dict['IS2']['xings']['ATL10']['laser_fb'],dtype=object)[idx_dates]),axis=0)
        ref_idx_is2 =  np.ma.concatenate(list(np.array(data_dict['IS2']['xings']['ATL10']['idx_ref'],dtype=object)[idx_dates]),axis=0)
        
        delay_is2 =  np.ma.concatenate(list(np.array(data_dict['IS2']['xings']['ATL10']['delay'],dtype=object)[idx_dates]),axis=0)


        maxtime = np.max(np.abs(delay_cs2))

        
        #laser_fb = data_dict['IS2']['xings']['ATL10']['laser_fb']
        delay_0 = 0
        delay=delay_inc

        dt = list()
        ndatax = list()

        ndata_colloc = list()
        std_delta_fb_colloc = list()
        mean_delta_fb_colloc = list()

        std_delta_fb = list()
        mean_delta_fb = list() 
        
        while delay<maxtime:

            
            
            dt.append(delay/(60*60*24)) #in days
            print("Dtime %.1f days" %(delay/(60*60*24)))

            # Xings from IS2
            #---------------------
            
            idx_time_is2 = np.abs(delay_is2.data)>delay
            laser_fb_is2x = np.ma.masked_where(idx_time_is2,laser_fb_xings,copy=True)
            lat_xings_is2 = np.ma.masked_where(idx_time_is2,lat_is2xings,copy=True)
            lon_xings_is2 = np.ma.masked_where(idx_time_is2,lon_is2xings,copy=True)
            abs_ref_idx_is2x = np.ma.masked_where(idx_time_is2,abs_ref_idx_is2,copy=True)

            radar_fb_is2x = radar_fb_colloc[abs_ref_idx_is2x]
            lat_is2x = lat[abs_ref_idx_is2x[abs_ref_idx_is2x.mask==False]]
            lon_is2x = lon[abs_ref_idx_is2x[abs_ref_idx_is2x.mask==False]]
            
            delta_fb_is2x = laser_fb_is2x - radar_fb_is2x

            ndata_is2x = np.sum(~delta_fb_is2x.mask)
            print("N xings IS2 = %i" %(ndata_is2x))
            

            # Xings from CS2
            #--------------------
            idx_time_cs2 = np.abs(delay_cs2.data)>delay
            radar_fb_cs2x = np.ma.masked_where(idx_time_cs2,radar_fb_xings,copy=True)
            lat_xings_cs2 = np.ma.masked_where(idx_time_cs2,lat_xings,copy=True)
            lon_xings_cs2 = np.ma.masked_where(idx_time_cs2,lon_xings,copy=True)
            abs_ref_idx_cs2x = np.ma.masked_where(idx_time_cs2,abs_ref_idx_cs2,copy=True)

            laser_fb_cs2x = laser_fb_colloc[abs_ref_idx_cs2x]
            lat_cs2x = lat[abs_ref_idx_cs2x[abs_ref_idx_cs2x.mask==False]]
            lon_cs2x = lon[abs_ref_idx_cs2x[abs_ref_idx_cs2x.mask==False]]

            """
            plt.plot(lat_cs2x,lon_cs2x,'*')
            plt.plot(lat_xings_cs2,lon_xings_cs2,'.')
            plt.show()
            """
           
            delta_fb_cs2x = laser_fb_cs2x - radar_fb_cs2x

            ndata_cs2x = np.sum(~delta_fb_cs2x.mask)
            print("N xings CS2 = %i" %(ndata_cs2x))

            # concatenate both
            delta_fb = delta_fb_cs2x
            ndata_x = np.sum(~delta_fb.mask)
            ndatax.append(ndata_x)
            # to change XXX delta_fb = np.ma.concatenate((delta_fb_cs2xings,delta_fb_is2xings))

            # Get collocated tracks
            #--------------------------------
            
            if delay > delay_colloc:
                delta_fb_colloc_all = np.ma.concatenate((delta_fb,delta_fb_colloc))
                nonmasked_data_colloc = ~delta_fb_colloc_all.mask
                print('ndata_colloc',np.sum(nonmasked_data_colloc))
                ndata_colloc.append(np.sum(nonmasked_data_colloc))
                std_delta_fb_colloc.append(np.nanstd(delta_fb_colloc_all[nonmasked_data_colloc]))
                mean_delta_fb_colloc.append(np.nanmean(delta_fb_colloc_all[nonmasked_data_colloc]))
            

            nonmasked_data = ~delta_fb.mask
            nbeams = np.sum(nonmasked_data)

            if nbeams==0:
                std_delta_fb.append(np.nan)
                mean_delta_fb.append(np.nan)
            else:
                std_delta_fb.append(np.nanstd(delta_fb[nonmasked_data]))
                mean_delta_fb.append(np.nanmean(delta_fb[nonmasked_data]))


            #lat_xings = np.ma.concatenate((lat_cs2x,lat_is2x))
            #lon_xings = np.ma.concatenate((lon_cs2x,lon_is2x))


            # increment delay
            delay_0 = delay
            delay = delay + delay_inc    

        
        print("stop")
        
        # Show Xo
        #-------------------
        
        
        f1, ax = plt.subplots(1, 1,figsize=(9,8))
        bmap,cmap = st.plot_track_map(f1,ax,lon_xings,lat_xings,delta_fb,'',[0,0.5],mid_date,'m',False,alpha=1,size=5)
        
        #lt.show()
        

        # Show colloc
        #-----------------
        
        f2, ax = plt.subplots(1, 1,figsize=(9,8))
        bmap,cmap = st.plot_track_map(f2,ax,lon,lat,delta_fb_colloc,'',[0,0.5],mid_date,'m',False,alpha=1,size=5)
        
        plt.show()
        

        
        # style plots
        plt.style.use('seaborn-darkgrid')
 
        # create a color palette
        palette = plt.get_cmap('Set1')
    
        # plot data
        f1, ax = plt.subplots(1, 1,figsize=(8,8))
        f1.suptitle('Statistics at crossings points between CS2 and IS2',size=12)
        
            
        #ax.plot(dt,mean_delta_fb,label=r'Mean $\Delta fb$',linestyle='-',color=palette(0))
        rel_delta_fb = np.array(std_delta_fb)/np.array(mean_delta_fb)
        rel_delta_fb_colloc = np.array(std_delta_fb_colloc)/np.array(mean_delta_fb_colloc)
        ax.plot(dt,rel_delta_fb,label=r'Std $\Delta fb$',linestyle='-',color=palette(1))
        ax.plot(dt,rel_delta_fb_colloc,label=r'Std $\Delta fb[colloc]$',linestyle='-',color=palette(2))

        ax.set_xlabel("delay [days]")
        ax.set_ylabel(r'$\Delta fb$ [m]')
        ax.set_ylim([0,2])
        ax.grid()


        ax2 = ax.twinx()  # instantiate a second axes that shares the same x-axis
        ax2.plot(dt,ndatax,label="ndata",linestyle='--',color=palette(1))
        ax2.plot(dt,ndata_colloc,label="ndata[colloc]",linestyle='--',color=palette(2))
        ax2.set_ylabel('ndata')
        ax.legend()

        plt.show()

        
        

    if param=='data_maps':

        laser_fb = np.ma.concatenate(list(np.array(data_dict['IS2']['ATL10']['laser_fb_mean'],dtype=object)[idx_dates]),axis=0)
        lon = np.ma.concatenate(list(np.array(data_dict['CS2']['ESA_BD_GDR']['lon'],dtype=object)[idx_dates]),axis=0)
        lat = np.ma.concatenate(list(np.array(data_dict['CS2']['ESA_BD_GDR']['lat'],dtype=object)[idx_dates]),axis=0)

        
            

        """
        f1, ax = plt.subplots(1, 1,figsize=(9,8))
        bmap,cmap = st.plot_track_map(f1,ax,lon,lat,laser_fb,'laser freeboard IS2',[0,0.5],mid_date,'m',False,alpha=1,size=5)
        
        plt.show()
        """

        # Get CS2 freeboard
        npts = lon.shape[0]
        radar_fb_list = list()
        radar_fb_matrix = ma.masked_array(np.zeros((len(prod_L2P),npts)),mask=np.ones((len(prod_L2P),npts)),dtype='float')
        for nprod,cs2_gdr in enumerate(prod_L2P):
            radar_fb = list(np.array(data_dict['CS2'][cs2_gdr]['radar_fb'],dtype=object)[idx_dates])
            fb_full=ma.masked_invalid(np.ma.concatenate(radar_fb,axis=0))
            radar_fb_list.append(fb_full)
            radar_fb_matrix[nprod,:] = fb_full

        radar_fb_matrix = ma.masked_invalid(radar_fb_matrix,copy=True)
        radar_fb_mean = np.ma.mean(radar_fb_matrix,axis=0)
        radar_fb_std = np.ma.std(radar_fb_matrix,axis=0)


        # Maps
        #------------------------
        lon = np.ma.concatenate(list(np.array(data_dict['CS2']['ESA_BD_GDR']['lon'],dtype=object)[idx_dates]),axis=0)
        lat = np.ma.concatenate(list(np.array(data_dict['CS2']['ESA_BD_GDR']['lat'],dtype=object)[idx_dates]),axis=0)
        
        for nprod,cs2_gdr in enumerate(prod_L2P):

           
            radar_fb_list
            f1, ax = plt.subplots(1, 1,figsize=(9,8))
            bmap,cmap = st.plot_track_map(f1,ax,lon,lat,radar_fb_list[nprod],'elevation %s' %(cs2_gdr),[0,0.5],mid_date,'m',False,alpha=1,size=5)
        
            plt.show()

    if param=='sd_comp':

        nkm = 50
        
        sd_products = ["ASD","AMSR","W99m","PIOMAS"]
        mean_df = pd.DataFrame(index=prod_L2P,columns=sd_products)
        rmsd_df = pd.DataFrame(index=prod_L2P,columns=sd_products)
        
        laser_fb = np.ma.concatenate(list(np.array(data_dict['IS2']['ATL10']['laser_fb_mean'],dtype=object)[idx_dates]),axis=0)

        common_mask = np.zeros(laser_fb.shape)
        radar_list = list()
        for nprod,cs2_gdr in enumerate(prod_L2P):
            
            radar_fb = np.ma.concatenate(list(np.array(data_dict['CS2'][cs2_gdr]['radar_fb'],dtype=object)[idx_dates]),axis=0)
            radar_list.append(radar_fb)
            common_mask = np.logical_or(common_mask,radar_fb.mask)

        latref = list()
        lonref = list()
        icetyperef = list()
        for month,idx in idx_dates_monthly.items():
            latref.append(np.ma.concatenate(list(np.array(data_dict['CS2'][REF_GDR]['latref'],dtype=object)[idx_dates[idx]]),axis=0))
            lonref.append(np.ma.concatenate(list(np.array(data_dict['CS2'][REF_GDR]['lonref'],dtype=object)[idx_dates[idx]]),axis=0))
            icetyperef.append(np.ma.concatenate(np.array(icetype_al)[idx_dates[idx]],axis=0))
            

        """
        icetype_al = list()
        for ndate,date in enumerate(month_list):
            lon = lonref[ndate]
            if any(np.abs(np.diff(lon)) > 20): lon[lon > 180] = lon[lon > 180] - 360
            icetype_alongtrack = cf.grid_to_track(OSISAF_ice_type,lons,lats,lon,ref_seg_lat[n])
            icetype_al.append(icetype_alongtrack)
        """

        
        # Get snow depth products
        #-------------------------------------------

        # ASD
        
        sd_ASD_track = list()
        for ndate,date in enumerate(month_list):
            #datestr = date.strftime('%Y%m')
            lat_grid,lon_grid,sd_grid,sd_grid_unc = cf.get_ASD(50,date)
            sd_al = cf.grid_to_track(sd_grid,lon_grid,lat_grid,lonref[ndate],latref[ndate])
            sd_ASD_track.append(sd_al)
        sd_ASD_full = np.ma.concatenate(sd_ASD_track,axis=0)
        


        # AMSR
        
        SD_AMSR_al_full = list()
        for ndate,date in enumerate(found_dates):
            idx_date = available_dates.index(date)
            lat_grid,lon_grid,SD_AMSR = cf.get_SD_AMSR(date)
            lon_AMSR = np.array(data_dict['CS2']['ESA_BD_GDR']['lon'],dtype=object)[idx_date]
            lat_AMSR = np.array(data_dict['CS2']['ESA_BD_GDR']['lat'],dtype=object)[idx_date]
            
            SD_AMSR_al = cf.grid_to_track(SD_AMSR,lon_grid,lat_grid,lon_AMSR,lat_AMSR)
            SD_AMSR_al = SD_AMSR_al/100
            SD_AMSR_al_full.append(SD_AMSR_al)

        SD_AMSR_full = np.ma.concatenate(SD_AMSR_al_full,axis=0)
        
        
        # PIOMAS
        PIOMAS_SD_track = list()
        for ndate,date in enumerate(month_list):
            datet = datetime.strptime(date, '%Y%m')
            lat_grid,lon_grid,sd_grid = cf.get_PIOMAS_SD(datet)
            #lon_grid,lat_grid = np.meshgrid(lon_ist,lat_ist)
            sd_al = cf.grid_to_track(sd_grid,lon_grid,lat_grid,lonref[ndate],latref[ndate])
            PIOMAS_SD_track.append(sd_al)
        PIOMAS_SD_full = np.ma.concatenate(PIOMAS_SD_track,axis=0)
        

        # W99m
        sd_w99 = list()
        for ndate,date in enumerate(month_list):
            lat_grid,lon_grid,sd_grid = cf.get_W99(date)
            lon1 = lonref[ndate]
            if any(np.abs(np.diff(lon1)) > 20): lon1[lon1 > 180] = lon1[lon1 > 180] - 360
            SD_W99 = cf.grid_to_track(sd_grid,lon_grid,lat_grid,lon1,latref[ndate])
            SD_W99[icetyperef[ndate]==2]= 0.5*SD_W99[icetyperef[ndate]==2]
            SD_W99 = SD_W99/100
            sd_w99.append(SD_W99)
        SD_W99_full = np.ma.concatenate(sd_w99,axis=0)
        

        # Slow down factor
        ds = 0.300
        ns = (1 + 0.51*ds)**(-1.5)

        mean_bias_ASD = list()
        RMSD_ASD = list()
        mean_bias_AMSR = list()
        RMSD_AMSR = list()
        mean_bias_PIOMAS = list()
        RMSD_PIOMAS = list()
        mean_bias_W99m =list()
        RMSD_W99m = list()
            
        for nprod,cs2_gdr in enumerate(prod_L2P):
            
            delta_laku = (laser_fb - radar_list[nprod])*ns
            sd_laku = ma.masked_where(np.isnan(delta_laku),delta_laku,copy=True)
            sd_laku_mask = sd_laku.mask
            sd_laku_smooth = st.rolling_stats(sd_laku,int(nkm/mean_dist_btw_data), stats=['mean'])[0]
            sd_laku_smooth = ma.masked_where(sd_laku_mask,sd_laku_smooth,copy=True)

            sd_laku_smooth = ma.masked_where(common_mask,sd_laku_smooth,copy=True)
            
            # ASD
            delta = sd_laku_smooth-sd_ASD_full
            ndata = np.sum(~delta.mask)
            mean_bias_ASD.append(np.ma.mean(delta)*100)
            RMSD_ASD.append(np.sqrt((1/(ndata-1))*np.ma.sum((delta)**2))*100)

            # AMSR
            delta = sd_laku_smooth-SD_AMSR_full
            ndata = np.sum(~delta.mask)
            mean_bias_AMSR.append(np.ma.mean(delta)*100)
            RMSD_AMSR.append(np.sqrt((1/(ndata-1))*np.ma.sum((delta)**2))*100)

            # PIOMAS
            delta = sd_laku_smooth-PIOMAS_SD_full
            ndata = np.sum(~delta.mask)
            mean_bias_PIOMAS.append(np.ma.mean(delta)*100)
            RMSD_PIOMAS.append(np.sqrt((1/(ndata-1))*np.ma.sum((delta)**2))*100)

            # W99m TODO
            delta = sd_laku_smooth-SD_W99_full
            delta = ma.masked_invalid(delta,copy=True)
            ndata = np.sum(~delta.mask)
            mean_bias_W99m.append(np.ma.mean(delta)*100)
            RMSD_W99m.append(np.sqrt((1/(ndata-1))*np.ma.sum((delta)**2))*100)
            
            # Add data in DataFrame
            #list2add = [mean_bias_ASD,mean_bias_AMSR,mean_bias_W99m,mean_bias_PIOMAS]
            
            #mean_df.loc[cs2_gdr] = pd.Series({'ASD':mean_bias_ASD, 'AMSR':mean_bias_AMSR, 'W99m':mean_bias_W99m, 'PIOMAS':mean_bias_PIOMAS})
            #rmsd_df.loc[cs2_gdr] = pd.Series({'ASD':RMSD_ASD, 'AMSR':RMSD_AMSR, 'W99m':RMSD_W99m, 'PIOMAS':RMSD_PIOMAS})


        # statistics
        #---------------------------------
        mean_std_mean = np.mean([np.std(mean_bias_W99m),np.std(mean_bias_PIOMAS),np.std(mean_bias_AMSR),np.std(mean_bias_ASD)])
        mean_std_rmsd = np.mean([np.std(RMSD_ASD),np.std(RMSD_PIOMAS),np.std(RMSD_W99m),np.std(RMSD_ASD)])
            
        # mean figure
        #--------------------------------
        columns = ['products'] + sd_products
        data_mean = [(prod,mean_bias_ASD[nprod],mean_bias_AMSR[nprod],mean_bias_W99m[nprod],mean_bias_PIOMAS[nprod]) for nprod,prod in enumerate(prod_L2P)]
        df = pd.DataFrame.from_records(data_mean,columns=columns)

        df_melted = pd.melt(df, id_vars=columns[0],var_name="prods", value_name="value_numbers")

        # Plotpalette='mako', 
        fig, ax1 = plt.subplots()
        g = sns.barplot(x=columns[0], y="value_numbers", hue="prods",data=df_melted,palette='mako', ax=ax1)
        
        # Create a second y-axis with the scaled ticks
        ax1.set_ylabel('mean bias [cm]')
        plt.show()

        # std figure
        #----------------------------
        data_rmsd = [(prod,RMSD_ASD[nprod],RMSD_AMSR[nprod],RMSD_W99m[nprod],RMSD_PIOMAS[nprod]) for nprod,prod in enumerate(prod_L2P)]
        df = pd.DataFrame.from_records(data_rmsd,columns=columns)

        df_melted = pd.melt(df, id_vars=columns[0],var_name="prods", value_name="value_numbers")  

        fig, ax2 = plt.subplots()
        g = sns.barplot(x=columns[0], y="value_numbers", hue="prods",data=df_melted,palette='mako', ax=ax2)
        
        # Create a second y-axis with the scaled ticks
        ax2.set_ylabel('RMSD [cm]')

        plt.show()
        
    
    if param=='sd_month':

        laser_fb_full = list()
        radar_fb_full = list()
        snow_depth_full = list()

        # snow depth products
        sd_AMSR_all = list()
        sd_W99_all= list()
        sd_PIOMAS_all = list()
        sd_ASD_all = list()
        sd_Laku_all = list()

        max_smoothing = 200
        maplim = [0,0.4]
        xylim = [[-0.05,0.4],[-0.05,0.4]]
        alpha=0.7
        sizepixmap=1
        nkm = 75 #km
        pathout='/home/antlafe/Documents/work/figures/cryo2ice/comp_sd/'

        
        SD_dict ={}
        SD_list = ['LaKu','ASD','AMSR','W99','PIOMAS']
        for SDtype in SD_list:
            SD_dict[SDtype] = {}
            for Itype in ['MYI','FYI','ALL']:
                SD_dict[SDtype][Itype] = {}
                for Istat in ['mean','std','rmsd','dmean','R','ndata']:
                    SD_dict[SDtype][Itype][Istat] = list()
    
        for month,idx in idx_dates_monthly.items():

            date_list = np.array(found_dates)[idx]
            mid_date = datetime.strptime(month,'%Y%m') + timedelta(days=20)
            mid_data_idx = np.argmin(abs(date_list-mid_date))
            ref_lat = list(np.array(data_dict['CS2'][REF_GDR]['latref'],dtype=object)[idx_dates[idx]])
            ref_lon = list(np.array(data_dict['CS2'][REF_GDR]['lonref'],dtype=object)[idx_dates[idx]])
            #ref_lat = ref_seg_lat[idx]
            #ref_lon = ref_seg_lon[idx]
            
            lat_full = np.ma.concatenate(ref_lat,axis=0)
            lon_full = np.ma.concatenate(ref_lon,axis=0)
            npts = lon_full.shape[0]
            # Get CS2 freeboard
            radar_fb_list = list()
            radar_fb_matrix = ma.masked_array(np.zeros((len(prod_L2P),npts)),mask=np.ones((len(prod_L2P),npts)),dtype='float')
            mask_list_all = np.zeros((npts,))
            
            for nprod,cs2_gdr in enumerate(prod_L2P):
                radar_fb = list(np.array(data_dict['CS2'][cs2_gdr]['radar_fb'],dtype=object)[idx_dates[idx]])
                fb_full=ma.masked_invalid(np.ma.concatenate(radar_fb,axis=0))
                mask_list_all = np.logical_or(mask_list_all,fb_full.mask)
                radar_fb_list.append(fb_full)
                radar_fb_matrix[nprod,:] = fb_full

            radar_fb_matrix = ma.masked_invalid(radar_fb_matrix,copy=True)
            radar_fb_mean = np.ma.mean(radar_fb_matrix,axis=0)
            radar_fb_std = np.ma.std(radar_fb_matrix,axis=0)

            
            # Get IS2 freeboard
            laser_fb =  np.ma.concatenate(list(np.array(data_dict['IS2']['ATL10']['laser_fb_mean'],dtype=object)[idx_dates[idx]]),axis=0)
            laser_fb_full.append(laser_fb)

            print("index in all data")
            print(idx_dates[idx])
            
            
            # Slow down factor
            ds = 0.300
            ns = (1 + 0.51*ds)**(-1.5)

            delta_fb_LaKu = laser_fb - radar_fb_mean
            snow_depth = delta_fb_LaKu*ns
            snow_depth = ma.masked_invalid(snow_depth)
            sd_mask = snow_depth.mask


            # statistics CRYO2ICE snow
            #----------------------------------
            mask_FYI = np.ma.concatenate(np.array(icetype_al)[idx],axis=0) == 2
            mask_MYI = np.ma.concatenate(np.array(icetype_al)[idx],axis=0) == 4

            SD_dict['LaKu']['FYI']['mean'].append(np.ma.mean(snow_depth[mask_FYI]))
            SD_dict['LaKu']['MYI']['mean'].append(np.ma.mean(snow_depth[mask_MYI]))
            SD_dict['LaKu']['ALL']['mean'].append(np.ma.mean(snow_depth))
            SD_dict['LaKu']['FYI']['std'].append(np.ma.std(snow_depth[mask_FYI]))
            SD_dict['LaKu']['MYI']['std'].append(np.ma.std(snow_depth[mask_MYI]))
            SD_dict['LaKu']['ALL']['std'].append(np.ma.std(snow_depth))
            

            # smoothing
           
            snow_depth_smooth = st.rolling_stats(snow_depth,int(nkm/mean_dist_btw_data), stats=['mean'])[0]
            snow_depth_smooth =  ma.masked_where(~sd_mask,snow_depth_smooth,copy=True)
            
            snow_depth_full.append(snow_depth_smooth)

            # Plot monthly map
            #--------------------------------
            """
            f1, ax = plt.subplots(1, 1,figsize=(8,8))
            f1.suptitle('Snow depth LaKu from ESA_BD from %s' %(month), fontsize=12)
            st.plot_track_map(f1,ax,lon_full,lat_full,snow_depth,'Snow depth LaKu',maplim,mid_date,'m',False,alpha=1,size=10)
            #plt.show()

            f2, ax = plt.subplots(1, 1,figsize=(8,8))
            f2.suptitle('Snow depth smooth LaKu from ESA_BD from %s' %(month), fontsize=12)
            st.plot_track_map(f1,ax,lon_full,lat_full,snow_depth_smooth,'Snow depth LaKu',maplim,mid_date,'m',False,alpha=1,size=10)
            plt.show()
            """
            # Compare to external products
            #--------------------------------
            
            #------------------------------
            # Get ASD
            #-----------------------------
            """
            print("\n#  ASD \n#############")
            
            #sd_ASD = list()
            sd_ASD_track = list()
            pixsize = 50
            #months = [date.strftime('%Y%m') for date in found_dates]
            lat_grid,lon_grid,sd_grid,sd_grid_unc = cf.get_ASD(pixsize,month)
            if lat_grid is not None:
                sd_grid = np.squeeze(sd_grid)


                for ndate,date in enumerate(date_list):

                    #lon_grid,lat_grid = np.meshgrid(lon_ist,lat_ist)
                    print(date_list[ndate])
                    sd_al = cf.grid_to_track(sd_grid,lon_grid,lat_grid,ref_lon[ndate],ref_lat[ndate])
                    #print(sd_al.shape)
                    sd_ASD_track.append(sd_al)
                sd_ASD_full = np.ma.concatenate(sd_ASD_track,axis=0)
                print(sd_ASD_full.shape)
                sd_ASD_all.append(sd_ASD_full)

                

                
                # Delta fb vs W99 snow depth
                x_data =  sd_ASD_full
                x_label = 'Snow depth ASD [m]'
                #y_label= r'$\Delta fb$'
                y_data = snow_depth
                y_label= 'snow depth LaKu [m]'
                x_data = ma.masked_where(np.isnan(x_data),x_data,copy=True)
                y_data = ma.masked_where(np.isnan(y_data),y_data,copy=True)
                mask_data = np.logical_and(~x_data.mask,~y_data.mask)
                
                

                
                # find smoothing radius
                #----------------------
                '''
                print("measuring smoothing radius\n")
                f16, ax = plt.subplots(1, 1,figsize=(6,6))
                f16.suptitle('Determination of smoothing radius ASD', fontsize=12)

                R_list,RMSD_list,smoothmin = cf.find_smoothing_radius(x_data,y_data,mean_dist_btw_data,max_smoothing,True)
                
                plt.show()
                '''
                

                # scatter plot
                #--------------------
                nkm = 50
                for nprod,cs2_gdr in enumerate(prod_L2P):
              
                    snow_depth_smooth = st.rolling_stats(snow_depth,int(nkm/mean_dist_btw_data), stats=['mean'])[0]
                    snow_depth_smooth_ASD =  ma.masked_where(~mask_data,snow_depth_smooth,copy=True)
                    y_data = ma.masked_where(~mask_data,y_data,copy=True)
                    

                
                f15, ax = plt.subplots(1, 1,figsize=(8,9))
                f15.suptitle('comparison with snow depth ASD', fontsize=12)
                R,RMSD,slope,dmean,ndata = st.plot_scatter(ax,xylim,'ASD','m',x_data,x_label,snow_depth_smooth_ASD,y_label,None)
                plt.clf()
                #plt.show()
                

                # statistics
                #--------------------
                SD_dict['ASD']['ALL']['rmsd'].append(RMSD)
                SD_dict['ASD']['ALL']['dmean'].append(dmean)
                SD_dict['ASD']['ALL']['R'].append(R)

                R,RMSD,slope,dmean,ndata = st.plot_scatter(ax,xylim,'ASD','m',x_data[mask_FYI],x_label,snow_depth_smooth_ASD[mask_FYI],y_label,None)
                 
                SD_dict['ASD']['FYI']['rmsd'].append(RMSD)
                SD_dict['ASD']['FYI']['dmean'].append(dmean)
                SD_dict['ASD']['FYI']['R'].append(R)

                R,RMSD,slope,dmean,ndata = st.plot_scatter(ax,xylim,'ASD','m',x_data[mask_MYI],x_label,snow_depth_smooth_ASD[mask_MYI],y_label,None)
                
                SD_dict['ASD']['MYI']['rmsd'].append(RMSD)
                SD_dict['ASD']['MYI']['dmean'].append(dmean)
                SD_dict['ASD']['MYI']['R'].append(R)
                
                # map
                #--------------------
                '''
                f1, ax = plt.subplots(1, 1,figsize=(8,9))
                bmap,cmap = st.plot_track_map(f1,ax,lon_grid,lat_grid,sd_grid,'Snow depth',[0.,0.5],mid_date,'m',False,alpha,size=sizepixmap)
                st.add_data_track(bmap,cmap,lon_full,lat_full,snow_depth_smooth,maplim)
                #plot_track_map(f1,ax,lon_full,lat_full,snow_depth,'Snow depth',maplim,None,'m',False)
                plt.savefig(pathout+'Snow_depth_ASD_%s.png' %(month))
                #plt.show()
                '''
                
            else:
                
                for Itype in ['MYI','FYI','ALL']:
                    for Istat in ['mean','std','rmsd','dmean','R']:
                        SD_dict['ASD'][Itype][Istat].append(np.nan)
            
            """    
            #-------------------------
            # Get SD PIOMAS
            #-------------------------
            """
            print("#  PIOMAS \n#############")
            
            PIOMAS_SD_track = list()
            lat_grid,lon_grid,sd_grid = cf.get_PIOMAS_SD(date_list[0])
            if lat_grid is not None:
            
                for ndate,date in enumerate(date_list):
                    
                    print(date_list[ndate])
                    #lon_grid,lat_grid = np.meshgrid(lon_ist,lat_ist)
                    sd_al = cf.grid_to_track(sd_grid,lon_grid,lat_grid,ref_lon[ndate],ref_lat[ndate])
                    #print(sd_al.shape)
                    PIOMAS_SD_track.append(sd_al)
                PIOMAS_SD_full = np.ma.concatenate(PIOMAS_SD_track,axis=0)
                print(PIOMAS_SD_full.shape)
                sd_PIOMAS_all.append(PIOMAS_SD_full)

                # Delta fb vs W99 snow depth
                x_data =  PIOMAS_SD_full
                x_label = 'Snow depth PIOMAS [m]'
                #y_label= r'$\Delta fb$'
                y_data = snow_depth
                y_label= 'snow depth LaKu [m]'
                x_data = ma.masked_where(np.isnan(x_data),x_data,copy=True)
                y_data = ma.masked_where(np.isnan(y_data),y_data,copy=True)
                mask_data = np.logical_and(~x_data.mask,~y_data.mask)

                # find smoothing radius
                #----------------------
                
                '''
                f16, ax = plt.subplots(1, 1,figsize=(8,9))
                f16.suptitle('Determination of smoothing radius PIOMAS', fontsize=12)
                
                R_list,RMSD_list,smoothmin = cf.find_smoothing_radius(x_data,y_data,mean_dist_btw_data,max_smoothing,True)
                '''
                #plt.show()

                # scatter plot
                #--------------------
                snow_depth_smooth = st.rolling_stats(snow_depth,int(nkm/mean_dist_btw_data), stats=['mean'])[0]
                snow_depth_smooth_PIOMAS =  ma.masked_where(~mask_data,snow_depth_smooth,copy=True)
                y_data = ma.masked_where(~mask_data,y_data,copy=True)
                
                
                
                f15, ax = plt.subplots(1, 1,figsize=(8,9))
                f15.suptitle('comparison with snow depth PIOMAS', fontsize=12)
                R,RMSD,slope,dmean,ndata = st.plot_scatter(ax,xylim,'PIOMAS','m',x_data,x_label,snow_depth_smooth_PIOMAS,y_label,None)
                
            
                # statistics
                #--------------------
                SD_dict['PIOMAS']['ALL']['rmsd'].append(RMSD)
                SD_dict['PIOMAS']['ALL']['dmean'].append(dmean)
                SD_dict['PIOMAS']['ALL']['R'].append(R)
                SD_dict['PIOMAS']['ALL']['ndata'].append(ndata)

                R,RMSD,slope,dmean,ndata = st.plot_scatter(ax,xylim,'PIOMAS','m',x_data[mask_FYI],x_label,snow_depth_smooth_PIOMAS[mask_FYI],y_label,None)
                 
                SD_dict['PIOMAS']['FYI']['rmsd'].append(RMSD)
                SD_dict['PIOMAS']['FYI']['dmean'].append(dmean)
                SD_dict['PIOMAS']['FYI']['R'].append(R)
                SD_dict['PIOMAS']['FYI']['ndata'].append(ndata)

                R,RMSD,slope,dmean,ndata = st.plot_scatter(ax,xylim,'PIOMAS','m',x_data[mask_MYI],x_label,snow_depth_smooth_PIOMAS[mask_MYI],y_label,None)
                
                SD_dict['PIOMAS']['MYI']['rmsd'].append(RMSD)
                SD_dict['PIOMAS']['MYI']['dmean'].append(dmean)
                SD_dict['PIOMAS']['MYI']['R'].append(R)
                SD_dict['PIOMAS']['MYI']['ndata'].append(ndata)

                
                # map
                #--------------------
                '''
                f1, ax = plt.subplots(1, 1,figsize=(8,9))
                bmap,cmap = st.plot_track_map(f1,ax,lon_grid,lat_grid,sd_grid,'Snow depth',maplim,mid_date,'m',False,alpha=alpha,size=sizepixmap)
                st.add_data_track(bmap,cmap,lon_full,lat_full,snow_depth_smooth_PIOMAS,maplim)
                plt.savefig(pathout+'Snow_depth_PIOMAS_%s.png' %(month))
                #plt.show()
                '''

            else:
                
                for Itype in ['MYI','FYI','ALL']:
                    for Istat in ['mean','std','rmsd','dmean','R']:
                        SD_dict['ASD'][Itype][Istat].append(np.nan)
                
                

            """
            #------------------------------
            # Get Laku
            #-----------------------------
            
            print("\n#  LaKu \n#############")
            
            #sd_Laku = list()
            sd_Laku_track = list()
            pixsize = 50
            #months = [date.strftime('%Y%m') for date in found_dates]
            lat_grid,lon_grid,sd_grid,sit_grid,roughn_grid,icetype = cf.get_Laku(month)
            if lat_grid is not None:
                sd_grid = np.squeeze(sd_grid)


                for ndate,date in enumerate(date_list):

                    #lon_grid,lat_grid = np.meshgrid(lon_ist,lat_ist)
                    print(date_list[ndate])
                    sd_al = cf.grid_to_track(sd_grid,lon_grid,lat_grid,ref_lon[ndate],ref_lat[ndate])
                    #print(sd_al.shape)
                    sd_Laku_track.append(sd_al)
                sd_Laku_full = np.ma.concatenate(sd_Laku_track,axis=0)
                print(sd_Laku_full.shape)
                sd_Laku_all.append(sd_Laku_full)
                
                # Delta fb vs W99 snow depth
                x_data =  sd_Laku_full
                x_label = 'Snow depth Laku [m]'
                #y_label= r'$\Delta fb$'
                y_data = snow_depth
                y_label= 'snow depth LaKu [m]'
                x_data = ma.masked_where(np.isnan(x_data),x_data,copy=True)
                y_data = ma.masked_where(np.isnan(y_data),y_data,copy=True)
                mask_data = np.logical_and(~x_data.mask,~y_data.mask)
                
                #print(np.sum(mask_data))

                
                # find smoothing radius
                #----------------------
                
                '''
                print("measuring smoothing radius\n")
                f16, ax = plt.subplots(1, 1,figsize=(8,9))
                f16.suptitle('Determination of smoothing radius Laku', fontsize=12)

                R_list,RMSD_list,smoothmin = cf.find_smoothing_radius(x_data,y_data,mean_dist_btw_data,max_smoothing,True)
                
                plt.show()
                '''

                # scatter plot
                #--------------------
                nkm = 75 #km
                snow_depth_smooth = st.rolling_stats(snow_depth,int(nkm/mean_dist_btw_data), stats=['mean'])[0]
                snow_depth_smooth_Laku =  ma.masked_where(~mask_data,snow_depth_smooth,copy=True)
                y_data = ma.masked_where(~mask_data,y_data,copy=True)

                
                f15, ax = plt.subplots(1, 1,figsize=(8,9))
                f15.suptitle('comparison with snow depth Laku', fontsize=12)
                R,RMSD,slope,dmean,ndata = st.plot_scatter(ax,xylim,'Laku','m',x_data,x_label,snow_depth_smooth_Laku,y_label,None)

                # statistics
                #--------------------
                SD_dict['LaKu']['ALL']['rmsd'].append(RMSD)
                SD_dict['LaKu']['ALL']['dmean'].append(dmean)
                SD_dict['LaKu']['ALL']['R'].append(R)
                SD_dict['LaKu']['ALL']['ndata'].append(ndata)

                R,RMSD,slope,dmean,ndata = st.plot_scatter(ax,xylim,'Laku','m',x_data[mask_FYI],x_label,snow_depth_smooth_Laku[mask_FYI],y_label,None)
                 
                SD_dict['LaKu']['FYI']['rmsd'].append(RMSD)
                SD_dict['LaKu']['FYI']['dmean'].append(dmean)
                SD_dict['LaKu']['FYI']['R'].append(R)
                SD_dict['LaKu']['FYI']['ndata'].append(ndata)

                R,RMSD,slope,dmean,ndata = st.plot_scatter(ax,xylim,'Laku','m',x_data[mask_MYI],x_label,snow_depth_smooth_Laku[mask_MYI],y_label,None)
                
                SD_dict['LaKu']['MYI']['rmsd'].append(RMSD)
                SD_dict['LaKu']['MYI']['dmean'].append(dmean)
                SD_dict['LaKu']['MYI']['R'].append(R)
                SD_dict['LaKu']['MYI']['ndata'].append(ndata)
                
                
                
                # map
                #--------------------
                
                f1, ax = plt.subplots(1, 1,figsize=(8,9))
                bmap,cmap = st.plot_track_map(f1,ax,lon_grid,lat_grid,sd_grid,'Snow depth',[0.,0.5],mid_date,'m',False,alpha,size=sizepixmap)
                st.add_data_track(bmap,cmap,lon_full,lat_full,snow_depth_smooth,maplim)
                #plot_track_map(f1,ax,lon_full,lat_full,snow_depth,'Snow depth',maplim,None,'m',False)
                plt.savefig(pathout+'Snow_depth_Laku_%s.png' %(month))
                #plt.show()
                

            else:
                
                for Itype in ['MYI','FYI','ALL']:
                    for Istat in ['mean','std','rmsd','dmean','R']:
                        SD_dict['ASD'][Itype][Istat].append(np.nan)

                      

            
            #-------------------------
            # Get SD AMSR
            #-------------------------
            """
            print("\n#  AMSR \n#############\n")
            
            SD_AMSR_al_full = list()
            for ndate,date in enumerate(date_list):
                lat_grid,lon_grid,SD_AMSR = cf.get_SD_AMSR(date)
                SD_AMSR = SD_AMSR/100
                SD_AMSR_al = cf.grid_to_track(SD_AMSR,lon_grid,lat_grid,ref_lon[ndate],ref_lat[ndate])
                SD_AMSR_al_full.append(SD_AMSR_al)

                if ndate==mid_data_idx:
                   SD_AMSR_middate = SD_AMSR
                    
                
            SD_AMSR_full = np.ma.concatenate(SD_AMSR_al_full,axis=0)
            sd_AMSR_all.append(SD_AMSR_full)
            
            # Delta fb vs AMSR snow depth
            x_data =  SD_AMSR_full
            x_label = 'Snow depth AMSR [m]'
            #y_label= r'$\Delta fb$'
            y_data = snow_depth
            y_label= 'snow depth LaKu [m]'
            x_data = ma.masked_where(np.isnan(x_data),x_data,copy=True)
            y_data = ma.masked_where(np.isnan(y_data),y_data,copy=True)
            mask_data = np.logical_and(~x_data.mask,~y_data.mask)
            

            # scatter plot
            #--------------------
            #nkm = 50 #km
            
            snow_depth_smooth = st.rolling_stats(snow_depth,int(nkm/mean_dist_btw_data), stats=['mean'])[0]
            snow_depth_smooth_AMSR =  ma.masked_where(~mask_data,snow_depth_smooth,copy=True)
            y_data = ma.masked_where(~mask_data,y_data,copy=True)

            
            f15, ax = plt.subplots(1, 1,figsize=(6,6))
            f15.suptitle('comparison with snow depth AMSR', fontsize=12)
            R,RMSD,slope,dmean,ndata = st.plot_scatter(ax,xylim,'AMSR','m',x_data,x_label,snow_depth_smooth_AMSR,y_label,None)
            

            # statistics
            #--------------------
            SD_dict['AMSR']['ALL']['rmsd'].append(RMSD)
            SD_dict['AMSR']['ALL']['dmean'].append(dmean)
            SD_dict['AMSR']['ALL']['R'].append(R)
            SD_dict['AMSR']['ALL']['ndata'].append(ndata)

            R,RMSD,slope,dmean,ndata = st.plot_scatter(ax,xylim,'AMSR','m',x_data[mask_FYI],x_label,snow_depth_smooth_AMSR[mask_FYI],y_label,None)
                 
            SD_dict['AMSR']['FYI']['rmsd'].append(RMSD)
            SD_dict['AMSR']['FYI']['dmean'].append(dmean)
            SD_dict['AMSR']['FYI']['R'].append(R)
            SD_dict['AMSR']['FYI']['ndata'].append(ndata)

            R,RMSD,slope,dmean,ndata = st.plot_scatter(ax,xylim,'AMSR','m',x_data[mask_MYI],x_label,snow_depth_smooth_AMSR[mask_MYI],y_label,None)
                
            SD_dict['AMSR']['MYI']['rmsd'].append(RMSD)
            SD_dict['AMSR']['MYI']['dmean'].append(dmean)
            SD_dict['AMSR']['MYI']['R'].append(R)
            SD_dict['AMSR']['MYI']['ndata'].append(ndata)

            
            # map
            #--------------------
            '''
            f1, ax = plt.subplots(1, 1,figsize=(10,10))
            bmap,cmap = st.plot_track_map(f1,ax,lon_grid,lat_grid,SD_AMSR_middate,'Snow depth',maplim,mid_date,'m',False,alpha=alpha,size=sizepixmap)
            st.add_data_track(bmap,cmap,lon_full,lat_full,snow_depth_smooth,maplim)
            #plot_track_map(f1,ax,lon_full,lat_full,snow_depth,'Snow depth',maplim,None,'m',False)
            plt.savefig(pathout+'Snow_depth_AMSR_%s.png' %(month))
            #plt.show()
            '''
            
            """
            #-------------------------
            # Get Warren climatology
            #-------------------------
            
            """
            print("/n#  W99 \n#############\n")

            #from W99 import W99
            sd_w99 = list()
            #month = date.month
            
            datestr= mid_date.strftime('%Y%m')
            lat_grid,lon_grid,sd_grid = cf.get_W99(str(month))
            sd_grid = ma.masked_invalid(sd_grid)/100

            # mid date grid
            sd_grid_w99_mid = sd_grid
            sd_grid_w99_mid[icetype[idx[mid_data_idx]]==2] = 0.5*sd_grid_w99_mid[icetype[idx[mid_data_idx]]==2]
            sd_grid_w99_mid = ma.masked_where(icetype[idx[mid_data_idx]].mask,sd_grid_w99_mid,copy=True)
            
            sd_w99_full = list()
            for ndate,date in enumerate(date_list):
                
                lon1 = ref_lon[ndate]
                if any(np.abs(np.diff(lon1)) > 20): lon1[lon1 > 180] = lon1[lon1 > 180] - 360
                SD_W99 = cf.grid_to_track(sd_grid,lon_grid,lat_grid,lon1,ref_lat[ndate])
                
                icetype_alongtrack = cf.grid_to_track(icetype[idx[ndate]],lons_icetype[idx[ndate]],lats_icetype[idx[ndate]],lon1,ref_lat[ndate])
                #icetype_al.append(icetype_alongtrack)
                SD_W99[icetype_alongtrack==2]= 0.5*SD_W99[icetype_alongtrack==2]
                sd_w99.append(SD_W99)
                
            sd_w99_full = np.ma.concatenate(sd_w99,axis=0)
            sd_W99_all.append(sd_w99_full)
                                                         
                                                         
            # Delta fb vs W99 snow depth
            #----------------------
            x_data =  sd_w99_full
            x_label = 'Snow depth Warren99 modified [m]'
            y_label= 'snow depth LaKu [m]'
            y_data = snow_depth
            x_data = ma.masked_where(np.isnan(x_data),x_data,copy=True)
            y_data = ma.masked_where(np.isnan(y_data),y_data,copy=True)
            mask_data = np.logical_and(~x_data.mask,~y_data.mask)
            
            
            # scatter plot
            #--------------------
            #nkm = 50 #km
            snow_depth_smooth = st.rolling_stats(snow_depth,int(nkm/mean_dist_btw_data), stats=['mean'])[0]
            snow_depth_smooth_W99 =  ma.masked_where(~mask_data,snow_depth_smooth,copy=True)
            y_data = ma.masked_where(~mask_data,y_data,copy=True)

            
            f15, ax = plt.subplots(1, 1,figsize=(6,6))
            f15.suptitle('comparison with snow depth W99', fontsize=12)
            R,RMSD,slope,dmean,ndata = st.plot_scatter(ax,xylim,'W99','m',x_data,x_label,snow_depth_smooth_W99,y_label,None)
            #plt.show()
            
            
            # statistics
            #--------------------
            SD_dict['W99']['ALL']['rmsd'].append(RMSD)
            SD_dict['W99']['ALL']['dmean'].append(dmean)
            SD_dict['W99']['ALL']['R'].append(R)
            SD_dict['W99']['ALL']['ndata'].append(ndata)

            R,RMSD,slope,dmean,ndata = st.plot_scatter(ax,xylim,'W99','m',x_data[mask_FYI],x_label,snow_depth_smooth_W99[mask_FYI],y_label,None)
                 
            SD_dict['W99']['FYI']['rmsd'].append(RMSD)
            SD_dict['W99']['FYI']['dmean'].append(dmean)
            SD_dict['W99']['FYI']['R'].append(R)
            SD_dict['W99']['FYI']['ndata'].append(ndata)

            R,RMSD,slope,dmean,ndata = st.plot_scatter(ax,xylim,'W99','m',x_data[mask_MYI],x_label,snow_depth_smooth_W99[mask_MYI],y_label,None)
                
            SD_dict['W99']['MYI']['rmsd'].append(RMSD)
            SD_dict['W99']['MYI']['dmean'].append(dmean)
            SD_dict['W99']['MYI']['R'].append(R)
            SD_dict['W99']['MYI']['ndata'].append(ndata)
            
            
            # map
            #--------------------
            '''
            f1, ax = plt.subplots(1, 1,figsize=(10,10))
            bmap,cmap = st.plot_track_map(f1,ax,lon_grid,lat_grid,sd_grid_w99_mid,'Snow depth',maplim,mid_date,'m',False,alpha=alpha,size=sizepixmap)
            st.add_data_track(bmap,cmap,lon_full,lat_full,snow_depth_smooth_W99,maplim)
            #plot_track_map(f1,ax,lon_full,lat_full,snow_depth,'Snow depth',maplim,None,'m',False)

            plt.savefig(pathout+'Snow_depth_W99_%s.png' %(month))
            #plt.show()
            '''
           

            """
            for prod in SD_dict.keys():
                print("\n %s ####\n" %(prod));
                for ntype,dtype in enumerate(['ALL','MYI','FYI']):
                    print("\n%s ####\n" %(prod));
                    print("dmean[%s]=%.2f" %(dtype,np.nanmean(SD_dict[prod][dtype]['dmean'])));
                    print("rmsd[%s]=%.2f" %(dtype,np.nanmean(SD_dict[prod][dtype]['rmsd'])));
                    print("R[%s]=%.2f" %(dtype,np.nanmean(SD_dict[prod][dtype]['R'])));
            

        # Temporal series
        #---------------------------
        plt.style.use('seaborn-darkgrid')
        palette = plt.get_cmap('Set1')

        # Laku
        #---------------------------
        
        f1, ax = plt.subplots(1, 1,figsize=(12,5))
        plt.plot(monthstr_list,SD_dict['LaKu']['ALL']['mean'],label='ALL',color=palette(0),marker='*')
        plt.fill_between(monthstr_list,np.array(SD_dict['LaKu']['ALL']['mean'])-np.array(SD_dict['LaKu']['ALL']['std']),np.array(SD_dict['LaKu']['ALL']['mean'])+np.array(SD_dict['LaKu']['ALL']['std']),color=palette(0),alpha=0.1)
        plt.plot(monthstr_list,SD_dict['LaKu']['MYI']['mean'],label='MYI',color=palette(2),marker='*')
        plt.fill_between(monthstr_list,np.array(SD_dict['LaKu']['MYI']['mean'])-np.array(SD_dict['LaKu']['MYI']['std']),np.array(SD_dict['LaKu']['MYI']['mean'])+np.array(SD_dict['LaKu']['MYI']['std']),color=palette(2),alpha=0.1)
        plt.plot(monthstr_list,SD_dict['LaKu']['FYI']['mean'],label='FYI',color=palette(3),marker='*')
        plt.fill_between(monthstr_list,np.array(SD_dict['LaKu']['FYI']['mean'])-np.array(SD_dict['LaKu']['FYI']['std']),np.array(SD_dict['LaKu']['FYI']['mean'])+np.array(SD_dict['LaKu']['FYI']['std']),color=palette(3),alpha=0.1)
        plt.legend()
        plt.ylabel("Snow depth (La-Ku) [m]")
        plt.show()
        


       
        
        # ASD
        #---------------------------
        palette = plt.get_cmap('Set1')
        f1, ax = plt.subplots(1, 1,figsize=(3,4))
        for ntype,dtype in enumerate(['ALL','MYI','FYI']):
            
            plt.plot(monthstr_list,SD_dict['ASD'][dtype]['rmsd'],label='RMSD %s' %(dtype),color=palette(ntype),linestyle='-');
            plt.plot(monthstr_list,SD_dict['ASD'][dtype]['R'],label='Rpearson %s' %(dtype),color=palette(ntype),linestyle='--');
            plt.plot(monthstr_list,SD_dict['ASD'][dtype]['dmean'],label='Dmean %s' %(dtype),color=palette(ntype),linestyle=':');
        #plt.legend()
        plt.show()
        


        # PIOMAS
        #--------------------------
        palette = plt.get_cmap('Set1')
        f1, ax = plt.subplots(1, 1,figsize=(7,4))
        for ntype,dtype in enumerate(['ALL','MYI','FYI']):
            
            plt.plot(monthstr_list,SD_dict['PIOMAS'][dtype]['rmsd'],label='RMSD %s' %(dtype),color=palette(ntype),linestyle='-');
            plt.plot(monthstr_list,SD_dict['PIOMAS'][dtype]['R'],label='Rpearson %s' %(dtype),color=palette(ntype),linestyle='--');
            plt.plot(monthstr_list,SD_dict['PIOMAS'][dtype]['dmean'],label='Dmean %s' %(dtype),color=palette(ntype),linestyle=':');
        plt.legend()
        plt.show()

        # AMSR
        #--------------------------
        palette = plt.get_cmap('Set1')
        f1, ax = plt.subplots(1, 1,figsize=(12,5))
        for ntype,dtype in enumerate(['ALL','MYI','FYI']):
            
            plt.plot(monthstr_list,SD_dict['AMSR'][dtype]['rmsd'],label='RMSD %s' %(dtype),color=palette(ntype),linestyle='-');
            plt.plot(monthstr_list,SD_dict['AMSR'][dtype]['R'],label='Rpearson %s' %(dtype),color=palette(ntype),linestyle='--');
            plt.plot(monthstr_list,SD_dict['AMSR'][dtype]['dmean'],label='Dmean %s' %(dtype),color=palette(ntype),linestyle=':');
        plt.legend()
        plt.show()


        # W99
        #--------------------------
        palette = plt.get_cmap('Set1')
        f1, ax = plt.subplots(1, 1,figsize=(12,5))
        for ntype,dtype in enumerate(['ALL','MYI','FYI']):
            
            plt.plot(monthstr_list,SD_dict['W99'][dtype]['rmsd'],label='RMSD %s' %(dtype),color=palette(ntype),linestyle='-');
            plt.plot(monthstr_list,SD_dict['W99'][dtype]['R'],label='Rpearson %s' %(dtype),color=palette(ntype),linestyle='--');
            plt.plot(monthstr_list,SD_dict['W99'][dtype]['dmean'],label='Dmean %s' %(dtype),color=palette(ntype),linestyle=':');
        plt.legend()
        plt.show()

        
            
            
        # SD map
        #--------------------

        # Full period statistics
        laser_fb_full = np.ma.concatenate(laser_fb_full,axis=0)
        radar_fb_full = np.ma.concatenate(radar_fb_full,axis=0)
        #snow_depth_full = np.ma.concatenate(snow_depth_full,axis=0)

        sd_laku = (laser_fb_full - radar_fb_full)*ns
        sd_laku = ma.masked_where(np.isnan(sd_laku),sd_laku,copy=True)
        sd_mask = sd_laku.mask
        sd_laku_smooth = st.rolling_stats(sd_laku,int(nkm/mean_dist_btw_data), stats=['mean'])[0]
        sd_laku_smooth = ma.masked_where(sd_mask,sd_laku_smooth,copy=True)
        
        # snow depth products
        sd_AMSR = np.ma.concatenate(sd_AMSR_all,axis=0)
        sd_W99 =  np.ma.concatenate(sd_W99_all,axis=0)
        sd_PIOMAS =  np.ma.concatenate(sd_PIOMAS_all,axis=0)
        sd_ASD =  np.ma.concatenate(sd_ASD_all,axis=0)

        data_list = [sd_ASD] #sd_AMSR,sd_W99,sd_PIOMAS,sd_ASD] #sd_AMSR_all,sd_W99_all]
        data_names = ['ASD'] #['W99','PIOMAS','ASD'] #'AMSR','W99','PIOMAS',

        """
        #y_data = sd_laku_smooth
        for n,data in enumerate(data_list):
            f0, ax = plt.subplots(1, 1, sharey=True)
            f0.suptitle('LaKu SD (m) vs auxiliary SD products (m)', fontsize=12)
            y_label='SD LaKu (m)'
            y_data = sd_laku_smooth
           
            x_data = data
            x_label = "%s (m)" %(data_names[n])

            y_data = ma.masked_where(np.isnan(y_data),y_data,copy=True)
            x_data = ma.masked_where(np.isnan(x_data),x_data,copy=True)
            
            mask_data = np.logical_and(~x_data.mask,~y_data.mask)
            y_data = ma.masked_where(~mask_data,y_data,copy=True)
            x_data = ma.masked_where(~mask_data,x_data,copy=True)
            
            print(np.sum(mask_data))
            
            st.plot_scatter(ax,xylim,data_names[n],'m',x_data,x_label,y_data,y_label,None)
            plt.show()
            """

        for n,data in enumerate(data_list): 
            f1, axn = plt.subplots(1, 1,figsize=(6,6))
            f1.suptitle('Determination of smoothing radius %s' %(data_names[n]), fontsize=12)
            R_list,RMSD_list,smoothmin = cf.find_smoothing_radius(data,sd_laku,mean_dist_btw_data,200,True)
    
            plt.show()
        
    if param=='simba':

        # show buoys
        #----------------------------------------

        
        """
        f1, ax = plt.subplots(1, 1,figsize=(6,6))
        bmap,cmap = st.plot_track_map(f1,ax,np.ma.ones(1),np.ma.ones(1),np.ma.zeros(1),'#buoys',None,mid_date,'m',False,alpha=1)
        lat_simba = {}; lon_simba= {};
        color = ['red','blue']
        for n,id_simba in enumerate(['608','607']):
            lat_simba,lon_simba = cf.get_SIMBA_traj(id_simba)
            #id_array[id_simba] = np.ma.ones(lat_simba[id_simba].shape)*(int(id_simba)-607)
            
            x,y = bmap(lon_simba,lat_simba)
            bmap.plot(x,y, linewidth=1.5, color=color[n],linestyle='-',zorder=2)
        plt.show()
        """
            
        
        # find intersections
        #---------------------------------------
        lat = np.concatenate(ref_seg_lat,axis=0)
        lon = np.concatenate(ref_seg_lon,axis=0)
        time = np.concatenate(ref_seg_time,axis=0)
        delay=5 #days
        max_dist=30 #km

        # get SIMBA cross-overs

        # 607
        lat_simba,lon_simba,time_simba,sd_simba7,sit_simba7,year_simba,month_simba,day_simba,date_simba = cf.get_SIMBA_data('607')
        idx_colloc,lon_colloc,lat_colloc,delay_colloc,day_colloc,sit_colloc7,sd_colloc7 = cf.get_CRYO2ICE_xings(lat_simba,lon_simba,time_simba,sd_simba7,sit_simba7,year_simba,month_simba,day_simba,lon,lat,time,date_simba,delay,max_dist)

        # 608
        #lat_simba8,lon_simba8,time_simba8,sd_simba8,sit_simba8 = cf.get_CRYO2ICE_xings('608')
        #idx_colloc8,lon_colloc8,lat_colloc8,delay_colloc8,day_colloc8,sit_colloc8,sd_colloc8 = cf.get_xings_SIMBA(lat_simba8,lon_simba8,time_simba8,sd_simba8,sit_simba8,lat,lon,time,delay,max_dist)

        
        """
        month0 = day_colloc[0]
        id_month = list()
        for days in day_colloc:
            factor = ((days.year - month0.year) * 12 + days.month - month0.month)
            id_month.append(factor)
        id_month = np.ma.array(id_month)
        
        
        
        # show map
        
        id_date = np.ma.array([mdates.date2num(i) for i in day_colloc])
        f1, ax = plt.subplots(1, 1,figsize=(10,6))
        f1.suptitle('Month ids from October with ESA_BD from %s-%s \n delay=+-%i days/dist=%i km' %(date_period_str[0],date_period_str[-1],delay,max_dist), fontsize=12)
        bmap,cmap = st.plot_track_map(f1,ax,lon_colloc,lat_colloc,id_date,'months',None,mid_date,'m',False,alpha=1)
       

        x,y = bmap(lon_simba,lat_simba)
        bmap.plot(x,y, linewidth=1.5, color='red',linestyle='-',zorder=2)
        plt.show()
        """


        # Get snow depth:idx_dates
        radar_fb = np.ma.concatenate(list(np.array(data_dict['CS2'][REF_GDR]['radar_fb'],dtype=object)[idx_dates]),axis=0)
        laser_fb =  np.ma.concatenate(list(np.array(data_dict['IS2']['ATL10']['laser_fb_mean'],dtype=object)[idx_dates]),axis=0)

        # Get icetype
        latref = list()
        lonref = list()
        icetyperef = list()
        for month,idx in idx_dates_monthly.items():
            latref.append(np.ma.concatenate(list(np.array(data_dict['CS2'][REF_GDR]['latref'],dtype=object)[idx_dates[idx]]),axis=0))
            lonref.append(np.ma.concatenate(list(np.array(data_dict['CS2'][REF_GDR]['lonref'],dtype=object)[idx_dates[idx]]),axis=0))
            icetyperef.append(np.ma.concatenate(np.array(icetype_al)[idx],axis=0))
        icetype = np.ma.concatenate(icetyperef,axis=0)

        # Get W99
        sd_w99 = list()
        for ndate,date in enumerate(month_list):
            
            lat_grid,lon_grid,sd_grid = cf.get_W99(date)
            lon1 = lonref[ndate]
            if any(np.abs(np.diff(lon1)) > 20): lon1[lon1 > 180] = lon1[lon1 > 180] - 360
            SD_W99 = cf.grid_to_track(sd_grid,lon_grid,lat_grid,lon1,latref[ndate])
            SD_W99[icetyperef[ndate]==2]= 0.5*SD_W99[icetyperef[ndate]==2]
            SD_W99 = SD_W99/100
            sd_w99.append(SD_W99)
        sd_w99_full = np.ma.concatenate(sd_w99,axis=0)
        
        
        ds = 0.300
        ns = (1 + 0.51*ds)**(-1.5)
        nkm = 75 #km

        sd_laku = (laser_fb - radar_fb)*ns

        # Get only coincident measurements with SIMBA
        sd_laku_sim = sd_laku[idx_colloc]
        SD_W99_sim = sd_w99_full[idx_colloc]
        radar_fb_sim =  radar_fb[idx_colloc]
        laser_fb_sim = laser_fb[idx_colloc]
        icetype_sim = icetype[idx_colloc]

        # snow density (from Mallett)                                                              
        #------------------------
        """
        if dateTime.month in np.arange(6):
            nbMonthFromOct = 12 + (dateTime.month - 10)
        else:
            nbMonthFromOct = (dateTime.month - 10)

        #dw = 1024                                                                                       
        ds = (6.50*nbMonthFromOct + 274.51)/1000
        ds_grid = np.ma.ones(lat.shape)*ds
        ns = (1 + 0.51*ds)**(-1.5)
        """
        ds=300

        
        sit_radar_w99m = cf.fbr2sit(radar_fb_sim,SD_W99_sim,icetype_sim,ds)

        sit_laser_w99m = cf.fbt2sit(laser_fb_sim,SD_W99_sim,icetype_sim,ds)

        sit_cryo2ice = cf.fbt2sit(laser_fb_sim,sd_laku_sim,icetype_sim,ds)

        #sit_radar_sdsimba = cf.fbr2sit(radar_fb_sim,SD_W99_sim,icetype_sim,day_colloc)

        # get mean and std value
        #--------------------------
        sit_radar_w99m_mean = list()
        sit_radar_w99m_std = list()
        sit_laser_w99m_mean = list()
        sit_laser_w99m_std = list()
        sit_cryo2ice_mean = list()
        sit_cryo2ice_std = list()
        sit_simba7_mean = list()
        sit_simba7_std = list()
        
        sd_cryo2ice_mean = list()
        sd_cryo2ice_std = list()
        sd_simba7_mean = list()
        sd_simba7_std = list()
        sd_simba8_mean = list()
        sd_simba8_std = list()
        sd_w99m_mean = list()
        sd_w99m_std = list()
        list_days8 = list()
        
        for day in np.unique(day_colloc):
            idx = np.argwhere(day_colloc==day)

            #if idx.size>10:
            if np.sum(~sit_radar_w99m[idx].mask) > 5:
                sit_radar_w99m_mean.append(np.ma.mean(sit_radar_w99m[idx]))
                sit_radar_w99m_std.append(np.ma.std(sit_radar_w99m[idx]))
            else:
                sit_radar_w99m_mean.append(np.nan)
                sit_radar_w99m_std.append(np.nan)

            if np.sum(~sit_laser_w99m[idx].mask) > 5:
                sit_laser_w99m_mean.append(np.ma.mean(sit_laser_w99m[idx]))
                sit_laser_w99m_std.append(np.ma.std(sit_laser_w99m[idx]))
            else:
                sit_laser_w99m_mean.append(np.nan)
                sit_laser_w99m_std.append(np.nan)

            if np.sum(~sit_cryo2ice[idx].mask) > 5:
                sit_cryo2ice_mean.append(np.ma.mean(sit_cryo2ice[idx]))
                sit_cryo2ice_std.append(np.ma.std(sit_cryo2ice[idx]))
            else:
                sit_cryo2ice_mean.append(np.nan)
                sit_cryo2ice_std.append(np.nan)

            if np.array(sit_colloc7)[idx].size > 5:
                sit_simba7_mean.append(np.ma.mean(np.array(sit_colloc7)[idx])/100)
                sit_simba7_std.append(np.ma.std(np.array(sit_colloc7)[idx])/100)
            else:
                sit_simba7_mean.append(np.nan)
                sit_simba7_std.append(np.nan)

            
            if np.sum(~sd_laku_sim[idx].mask) > 5:
                sd_cryo2ice_mean.append(np.ma.mean(sd_laku_sim[idx]))
                sd_cryo2ice_std.append(np.ma.std(sd_laku_sim[idx]))
            else:
                sd_cryo2ice_mean.append(np.nan)
                sd_cryo2ice_std.append(np.nan)
            
            if np.array(sd_colloc7)[idx].size > 5:
                sd_simba7_mean.append(np.ma.mean(np.array(sd_colloc7)[idx])/100)
                sd_simba7_std.append(np.ma.std(np.array(sd_colloc7)[idx])/100)
            else:
                sd_simba7_mean.append(np.nan)
                sd_simba7_std.append(np.nan)
            
            if SD_W99_sim[idx].size > 5:
                sd_w99m_mean.append(np.ma.mean(SD_W99_sim[idx]))
                sd_w99m_std.append(np.ma.std(SD_W99_sim[idx]))
            else:
                sd_w99m_mean.append(np.nan)
                sd_w99m_std.append(np.nan)
            

            """
            idx8 = np.argwhere(day_colloc8==day)
            if idx8.size>0:
                list_days8.append(day)
                sd_simba8_mean.append(np.ma.mean(np.ma.array(sd_colloc8)[idx8])/100)
                sd_simba8_std.append(np.ma.std(np.ma.array(sd_colloc8)[idx8])/100)
            """
                
                
            
        list_days =  np.unique(day_colloc)

        plt.style.use('seaborn-darkgrid')
        palette = plt.get_cmap('Set1')

        # plot various data
        #-----------------------

        # sea-ice thickness
        #-------------------------
        f1, ax = plt.subplots(1, 1,figsize=(12,5))
        palette = plt.get_cmap('Set1')
        plt.plot(list_days,sit_radar_w99m_mean,label='sit_radar_w99m',color=palette(1))
        plt.fill_between(list_days,np.array(sit_radar_w99m_mean)-np.array(sit_radar_w99m_std),np.array(sit_radar_w99m_mean)+np.array(sit_radar_w99m_std),color=palette(1),alpha=0.1)
        
        plt.plot(list_days,sit_laser_w99m_mean,label='sit_laser_w99m',color=palette(2))
        plt.fill_between(list_days,np.array(sit_laser_w99m_mean)-np.array(sit_laser_w99m_std),np.array(sit_laser_w99m_mean)+np.array(sit_laser_w99m_std),color=palette(2),alpha=0.1)

        plt.plot(list_days,sit_cryo2ice_mean,label='sit_cryo2ice',color=palette(0))
        plt.fill_between(list_days,np.array(sit_cryo2ice_mean)-np.array(sit_cryo2ice_std),np.array(sit_cryo2ice_mean)+np.array(sit_cryo2ice_std),color=palette(0),alpha=0.1)

        plt.plot(list_days,sit_simba7_mean,label='sit_simba #607',color=palette(3))
        plt.fill_between(list_days,np.array(sit_simba7_mean)-np.array(sit_simba7_std),np.array(sit_simba7_mean)+np.array(sit_simba7_std),color=palette(3),alpha=0.1)
        
        #plt.plot(list_days,np.array(sit_colloc)/100,label='sit_colloc',marker='.',color=palette(3))
        plt.legend()
        plt.xlabel("date")
        plt.ylabel("sea-ice thickness [m]")
        plt.show()
        

        
        """
        plt.plot(day_colloc,sit_radar_w99m,label='sit_radar_w99m',marker='*')
        plt.plot(day_colloc,sit_laser_w99m,label='sit_radar_w99m')
        plt.plot(day_colloc,sit_cryo2ice,label='sit_cryo2ice')
        plt.plot(day_colloc,np.array(sit_colloc)/100,label='sit_colloc',marker='.')
        plt.legend()
        #plt.xlabel("")
        #plt.ylabel("")
        plt.show()
        """


        # snow depth
        #-------------------------
        f2, ax = plt.subplots(1, 1,figsize=(12,5))
        plt.plot(list_days,sd_cryo2ice_mean,label='sd_cryo2ice',color=palette(0))
        plt.fill_between(list_days,np.array(sd_cryo2ice_mean)-np.array(sd_cryo2ice_std),np.array(sd_cryo2ice_mean)+np.array(sd_cryo2ice_std),color=palette(0),alpha=0.1)
        
        plt.plot(list_days,sd_w99m_mean,label='sd_w99m',color=palette(1))
        plt.fill_between(list_days,np.array(sd_w99m_mean)-np.array(sd_w99m_std),np.array(sd_w99m_mean)+np.array(sd_w99m_std),color=palette(1),alpha=0.1)
        
        plt.plot(list_days,sd_simba7_mean,label='sd_simba #607',color=palette(3))
        plt.fill_between(list_days,np.array(sd_simba7_mean)-np.array(sd_simba7_std),np.array(sd_simba7_mean)+np.array(sd_simba7_std),color=palette(3),alpha=0.1)

        #plt.plot(np.ma.array(list_days8),sd_simba8_mean,label='sd_simba #608',color=palette(4))
        #plt.fill_between(list_days,np.array(sd_simba8_mean)-np.array(sd_simba8_std),np.array(sd_simba8_mean)+np.array(sd_simba8_std),color=palette(4),alpha=0.1)
       
        plt.legend()
        plt.xlabel("date")
        plt.ylabel("snow depth [m]")
        plt.show()


        # compare snow depth
        #-------------------------
        plt.plot(SD_W99_sim,label='sd_w99m')
        plt.plot(sd_laku_sim,label='sd_laku')
        plt.plot(sd_colloc,label='sd_simba')
        plt.legend()
        plt.show()


    if param=='simba_grid':


      

       
        
        delay=5 #days
        max_dist=30 #km

        # 607
        lat_simba,lon_simba,time_simba,sd_simba,sit_simba,year_simba,month_simba,day_simba,date_simba = cf.get_SIMBA_data('607')

        Laku_monthly_sd =list()
        Laku_monthly_sit =list()
        w99_monthly_sd =list()
        w99_monthly_sd_rms =list()
        Laku_monthly_sd_rms =list()
        Laku_monthly_sit_rms =list()

        icetype_monthly =list()

        simba_monthly_sd =list()
        simba_monthly_sit =list()
        simba_monthly_sd_rms =list()
        simba_monthly_sit_rms =list()
        
        for month,idx in idx_dates_monthly.items():

            mid_date = datetime.strptime(month,'%Y%m') + timedelta(days=20)
            yearLK = int(month[:4])
            monthLK = int(month[4:6])
            flag_year = (year_simba==yearLK)
            flag_month = (month_simba==monthLK)

            flag = np.logical_and(flag_year,flag_month)
            
            # get W99m
            lat_w99_grid,lon_w99_grid,sd_w99_grid = cf.get_W99(month)

            # get Laku
            lat_grid,lon_grid,sd_grid,sit_grid,roughness_grid,ice_type = cf.get_Laku(month)

          

            
            # show on map
            """
            f1, ax = plt.subplots(1, 1,figsize=(9,8))
            bmap,cmap = st.plot_track_map(f1,ax,lon_grid,lat_grid,sit_grid,'SIT LaKu',[0,2.],mid_date,'m',False,alpha=1)
            for n,id_simba in enumerate(['607']):
                lat_simba,lon_simba = cf.get_SIMBA_traj(id_simba)
                x,y = bmap(lon_simba,lat_simba)
                bmap.plot(x,y, linewidth=1.5, color='red',linestyle='-',zorder=2)
            plt.show()
            """

            lat_month = lat_simba[flag]
            lon_month = lon_simba[flag]

            simba_monthly_sd.append(np.mean(sd_simba[flag])/100)
            simba_monthly_sit.append(np.mean(sit_simba[flag])/100)
            simba_monthly_sd_rms.append(np.std(sd_simba[flag])/100)
            simba_monthly_sit_rms.append(np.std(sit_simba[flag])/100)

            sd_list = list()
            sit_list = list()
            sd_w99_list = list()
            icetype_list = list()
            
            if sit_grid is not None:
                sit_grid = sit_grid.flatten()
            if sd_grid is not None:
                sd_grid = sd_grid.flatten()
            if sd_w99_grid is not None:
                sd_w99_grid = sd_w99_grid.flatten()/100
            if ice_type is not None:
                ice_type = ice_type.flatten()
            
            for coord_simba in zip(lat_month,lon_month):

                # Min dist Laku
                ###############
                dist_sdelay = np.sqrt((coord_simba[0]-lat_grid)**2 + (coord_simba[1]-lon_grid)**2).flatten()
                idx = np.argmin(dist_sdelay)
                min_dist= dist_sdelay[idx]

                #print(min_dist)

                if dist_sdelay[idx]< 1:
                    sd_list.append(sd_grid[idx])
                    sit_list.append(sit_grid[idx])
                    icetype_list.append(ice_type[idx])
                else:
                    sd_list.append(np.nan) 
                    sit_list.append(np.nan)
                    icetype_list.append(np.nan)

                # Min dist W99
                ###########
                dist_sdelay = np.sqrt((coord_simba[0]-lat_w99_grid)**2 + (coord_simba[1]-lon_w99_grid)**2).flatten()
                idx = np.argmin(dist_sdelay)
                min_dist= dist_sdelay[idx]

                #print(min_dist)

                if dist_sdelay[idx]< 1:
                    sd_w99_list.append(sd_w99_grid[idx])
                else:
                    sd_w99_list.append(np.nan)
                   

            Laku_monthly_sd.append(np.nanmean(sd_list))
            Laku_monthly_sit.append(np.nanmean(sit_list))
            Laku_monthly_sd_rms.append(np.nanstd(sd_list))
            Laku_monthly_sit_rms.append(np.nanstd(sit_list))
            icetype_monthly.append(np.nanmedian(icetype_list))
            w99_monthly_sd.append(np.nanmean(sd_w99_list))
            w99_monthly_sd_rms.append(np.nanstd(sd_w99_list))

        
        plt.style.use('seaborn-darkgrid')
        palette = plt.get_cmap('Set1')

        # plot various data
        #-----------------------

        # sea-ice thickness
        #-------------------------
        f1, ax = plt.subplots(1, 1,figsize=(12,5))
        plt.plot(monthstr_list,Laku_monthly_sit,label='sit_LaKu',color=palette(0),marker='*')
        plt.fill_between(monthstr_list,np.array(Laku_monthly_sit)-np.array(Laku_monthly_sit_rms),np.array(Laku_monthly_sit)+np.array(Laku_monthly_sit_rms),color=palette(0),alpha=0.1)
        
        plt.plot(monthstr_list,simba_monthly_sit,label='sit_SIMBA',color=palette(3),marker='*')
        plt.fill_between(monthstr_list,np.array(simba_monthly_sit)-np.array(simba_monthly_sit_rms),np.array(simba_monthly_sit)+np.array(simba_monthly_sit_rms),color=palette(3),alpha=0.1)
        
        plt.legend()
        plt.xlabel("date")
        plt.ylabel("sea-ice thickness [m]")
        plt.show()

        # snow depth
        #-------------------------
        f1, ax = plt.subplots(1, 1,figsize=(12,5))
        plt.plot(monthstr_list,Laku_monthly_sd,label='sd_LaKu',color=palette(0),marker='*')
        plt.fill_between(monthstr_list,np.array(Laku_monthly_sd)-np.array(Laku_monthly_sd_rms),np.array(Laku_monthly_sd)+np.array(Laku_monthly_sd_rms),color=palette(0),alpha=0.1)

        plt.plot(monthstr_list,w99_monthly_sd,label='sd_w99m',color=palette(1),marker='*')
        plt.fill_between(monthstr_list,np.array(w99_monthly_sd)-np.array(w99_monthly_sd_rms),np.array(w99_monthly_sd)+np.array(w99_monthly_sd_rms),color=palette(1),alpha=0.1)
        
        plt.plot(monthstr_list,simba_monthly_sd,label='sd_SIMBA',color=palette(3),marker='*')
        plt.fill_between(monthstr_list,np.array(simba_monthly_sd)-np.array(simba_monthly_sd_rms),np.array(simba_monthly_sd)+np.array(simba_monthly_sd_rms),color=palette(3),alpha=0.1)
        
        plt.legend()
        plt.xlabel("date")
        plt.ylabel("snow depth [m]")
        plt.show()
               
                

            
            
             



           
        

        

        
        

    if param=='find_regions':


        # delimitate regions
        #--------------------------------------
        region_dict= {
            'FYI1' : {'lat':[78,78,82,82], 'lon':[-160,-170,-170,-160]},
            'FYI2' : {'lat':[76,76,80,80], 'lon':[-180,170,170,-180]},
            'MYI1' : {'lat':[82,82,86,86], 'lon':[-70,-120,-120,-70]},
            'MYI2' : {'lat':[86,86,88,88], 'lon':[0,20,20,0]},
        }


        # get data
        #--------------------------------------
        radar_fb = np.ma.concatenate(list(np.array(data_dict['CS2'][REF_GDR]['radar_fb'],dtype=object)[idx_dates]),axis=0)

        laser_fb =  np.ma.concatenate(list(np.array(data_dict['IS2']['ATL10']['laser_fb_mean'],dtype=object)[idx_dates]),axis=0)
        
        delta_fb_LaKu = laser_fb - radar_fb
        mask_sd = delta_fb_LaKu.mask

        month_id_list = list()
        #month_list = list()
        month0 = found_dates[0]
        for idx in idx_dates:
            factor = ((found_dates[idx].year - month0.year) * 12 + found_dates[idx].month - month0.month)
            id_month = np.ma.ones(data_dict['CS2'][REF_GDR]['latref'][idx].size)*factor
            month_id_list.append(id_month)
            #month_list.append(found_dates[idx].strftime('%Y%m'))

        id_month = np.ma.concatenate(month_id_list,axis=0)
        id_month = ma.masked_where(mask_sd,id_month,copy=True)
        nmonth = np.unique(id_month).size

        #------------------------------------------
        # plot monthly tracks, polygons and buoys
        #-------------------------------------------
        """
        f1, ax = plt.subplots(1, 1,figsize=(12,12))
        f1.suptitle('Month ids with ESA_BD from %s-%s' %(date_period_str[0],date_period_str[-1]), fontsize=12)
        lat = np.ma.concatenate(ref_seg_lat,axis=0)
        lon = np.ma.concatenate(ref_seg_lon,axis=0)
        bmap,cmap = st.plot_track_map(f1,ax,lon,lat,id_month,'Month id',[0.,nmonth-2],mid_date,'',False,alpha=1)

        # plot 82deg limit
        lon_82 = np.linspace(-180,180)
        lat_82 = np.linspace(82,82)
        x,y = bmap(lon_82,lat_82)
        bmap.plot(x,y, linewidth=2, color='black',linestyle='--')
        #plt.show()
        
        
        lat_simba,lon_simba = get_SIMBA_traj()
        x,y = bmap(lon_simba,lat_simba)
        bmap.plot(x,y, linewidth=2, color='black',linestyle='-')

        # FYI 1
        st.draw_polygon(region_dict['FYI1']['lat'], region_dict['FYI1']['lon'],bmap)

        # FYI2
        st.draw_polygon(region_dict['FYI2']['lat'], region_dict['FYI2']['lon'],bmap)

        # MYI1
        st.draw_polygon(region_dict['MYI1']['lat'], region_dict['MYI1']['lon'],bmap)
         
        # MYI2
        st.draw_polygon(region_dict['MYI2']['lat'], region_dict['MYI2']['lon'],bmap)

        plt.show()
        """
        
        #------------------------------------------
        # regional values
        #-------------------------------------------
        """
        product_list =  ['AMSR','ASD','W99','PIOMAS']
        data_list = product_list +['LaKu']
        
        sd = dict()
        for reg in region_dict.keys():
            sd[reg] = {}
            for dataName in data_list: 
                sd[reg][dataName] = {}

        sd['FYI1']['flag_FYI'] = True
        sd['FYI2']['flag_FYI'] = True
        sd['MYI1']['flag_FYI'] = False
        sd['MYI2']['flag_FYI'] = False


        # Get regional SD data
        for reg in region_dict.keys():
            for dataName in product_list:
                sd[reg][dataName]['mean'],sd[reg][dataName]['std'] = cf.get_regional_sd_mean(dataName,region_dict[reg],month_list,sd[reg]['flag_FYI'])


        # Get Laku SD data
        for reg in region_dict.keys():
            
            sd[reg]['LaKu']['mean'] = list()
            sd[reg]['LaKu']['std'] = list()
            sd[reg]['LaKu']['ndata'] = list()
            
            for month,idx in idx_dates_monthly.items():

                ref_lat = np.ma.concatenate(list(np.array(data_dict['CS2'][REF_GDR]['latref'],dtype=object)[idx_dates[idx]]))
                ref_lon = np.ma.concatenate(list(np.array(data_dict['CS2'][REF_GDR]['lonref'],dtype=object)[idx_dates[idx]]))
                region = cf.get_data_polygon(ref_lat,ref_lon,region_dict[reg])

                radar_fb = np.ma.concatenate(list(np.array(data_dict['CS2'][REF_GDR]['radar_fb'],dtype=object)[idx_dates[idx]]),axis=0)

                laser_fb =  np.ma.concatenate(list(np.array(data_dict['IS2']['ATL10']['laser_fb_mean'],dtype=object)[idx_dates[idx]]),axis=0)

                delta_fb_LaKu = laser_fb - radar_fb
                #mask_sd = delta_fb_LaKu.mask

                ds = 0.300
                ns = (1 + 0.51*ds)**(-1.5)
                nkm = 75 #km

                sd_laku = (laser_fb - radar_fb)*ns
                sd_laku = ma.masked_where(np.isnan(sd_laku),sd_laku,copy=True)
                sd_mask = sd_laku.mask
                sd_laku_smooth = st.rolling_stats(sd_laku,int(nkm/mean_dist_btw_data), stats=['mean'])[0]
                sd_laku_smooth = ma.masked_where(sd_mask,sd_laku_smooth,copy=True)

                snow_region = ma.masked_where(~region,sd_laku_smooth,copy=True)*100

                sd[reg]['LaKu']['mean'].append(np.ma.mean(snow_region))
                sd[reg]['LaKu']['std'].append(np.ma.std(snow_region))
                sd[reg]['LaKu']['ndata'].append(np.sum(~snow_region.mask))
    

        import seaborn as sns
        dates = [datetime.strptime(month,'%Y%m').strftime("%b") for month in month_list]
        sns.set_style("whitegrid")
        palette = plt.get_cmap('Set1')
        
        for reg in region_dict.keys():
            
            f1, ax = plt.subplots(1, 1,figsize=(4,4))
            ax.set_title(reg)
            for ndata,dataName in enumerate(data_list):

                mean = np.array(sd[reg][dataName]['mean'])
                std = np.array(sd[reg][dataName]['std'])
                ax.plot(dates, mean,label=dataName,color=palette(ndata))
                ax.fill_between(dates, mean - std, mean + std,color=palette(ndata),alpha=0.1)
                ax.set_ylabel('snow depth [cm]')
                #ax.legend()
        """
        #plt.show()
        
        #------------------------------------------
        # Plot full map
        #-------------------------------------------
        lat = np.concatenate(ref_seg_lat,axis=0)
        lon = np.concatenate(ref_seg_lon,axis=0)
        radar_fb = np.ma.concatenate(list(np.array(data_dict['CS2'][REF_GDR]['radar_fb'],dtype=object)[idx_dates]),axis=0)
        # Get IS2 freeboard
        laser_fb =  np.ma.concatenate(list(np.array(data_dict['IS2']['ATL10']['laser_fb_mean'],dtype=object)[idx_dates]),axis=0)

        data= {}
        data['fb'] = radar_fb

        
        # Slow down factor
        ds = 0.300
        ns = (1 + 0.51*ds)**(-1.5)
        nkm = 75 #km

        sd_laku = (laser_fb - radar_fb)*ns
        sd_laku = ma.masked_where(np.isnan(sd_laku),sd_laku,copy=True)
        sd_mask = sd_laku.mask
        sd_laku_smooth = st.rolling_stats(sd_laku,int(nkm/mean_dist_btw_data), stats=['mean'])[0]
        sd_laku_smooth = ma.masked_where(sd_mask,sd_laku_smooth,copy=True)

        # grid data
        import grid_data as grid
        
        data= {}
        sd_laku_smooth = ma.masked_invalid(sd_laku_smooth,copy=True)
        data['sd'] = sd_laku_smooth.data
        data['sd'][sd_laku_smooth.mask] = np.nan
        label ='SD LaKu'
        units = 'm'
        
        f2, ax = plt.subplots(1, 1,figsize=(9,8))
        m = Basemap(projection='npstere', llcrnrlat=0,urcrnrlat=90,llcrnrlon=-180,urcrnrlon=180,boundinglat=60,lon_0=0, resolution='l',round=True,ax=ax)
        x_grid, y_grid, lat_grid_mesh, lon_grid_mesh, x_grid_mesh, y_grid_mesh, data_results = st.grid_and_filter_wrt_distance(lon, lat, data, m, pixel_size=12500,mode='filter_mean',range_filter=25000, verbose=0)
        
        sd = np.ma.array(data_results['sd'])
        bmap,cmap = st.plot_track_map(f2,ax,lon_grid_mesh,lat_grid_mesh, sd,'snow depth',[0,0.4],mid_date,'m',False,alpha=1,size=3)

        # show regions
        st.draw_polygon(region_dict['FYI1']['lat'], region_dict['FYI1']['lon'],bmap,'black')
        st.draw_polygon(region_dict['FYI2']['lat'], region_dict['FYI2']['lon'],bmap,'black')
        st.draw_polygon(region_dict['MYI1']['lat'], region_dict['MYI1']['lon'],bmap,'black')
        st.draw_polygon(region_dict['MYI2']['lat'], region_dict['MYI2']['lon'],bmap,'black')
        
        
        plt.show()

        
    if param=='roughness':
        print("\nComparing roughness")
        nkm = 25

        # with ESA_BD only
        """
        correlation_list = list()
        for month,idx in idx_dates_monthly.items():

            laser_fb = np.ma.concatenate(list(np.array(data_dict['IS2']['ATL10']['laser_fb_mean'],dtype=object)[idx_dates[idx]]),axis=0)
            radar_fb = np.ma.concatenate(list(np.array(data_dict['CS2']['ESA_BD_GDR']['radar_fb'],dtype=object)[idx_dates[idx]]),axis=0)
            lat = np.ma.concatenate(list(np.array(data_dict['CS2']['ESA_BD_GDR']['lat'],dtype=object)[idx_dates[idx]]),axis=0)
            lon = np.ma.concatenate(list(np.array(data_dict['CS2']['ESA_BD_GDR']['lon'],dtype=object)[idx_dates[idx]]),axis=0)
            
            delta_laku = laser_fb - radar_fb
            mask_data = delta_laku.mask
            delta_laku_smooth = st.rolling_stats(delta_laku,int(nkm/mean_dist_btw_data), stats=['mean'])[0]
            delta_laku_smooth =  ma.masked_where(~mask_data,delta_laku_smooth,copy=True)
            

            lat_grid,lon_grid,sd_grid,sit_grid,roughness,icetype = cf.get_Laku(month)

            roughness_al = cf.grid_to_track(sd_grid,lon_grid,lat_grid,lon,lat)
            roughness_al = ma.masked_where(~mask_data,roughness_al,copy=True)
            

            #gaussian_w = np.ma.concatenate(list(np.array(data_dict['IS2']['ATL10']['gaussian_w_mean'],dtype=object)[idx_dates[idx]]),axis=0)

            
            xylim = [[-0.1,0.5],[-0.1,0.5]]
            f1, ax = plt.subplots(1, 1, sharey=True)
            f1.suptitle('Delta fb LaKu vs Gaussian width IS2', fontsize=12)
            #xylim = [[-0.1,1],[-0.1,1]]
            x_data = roughness_al
            x_label = 'Gaussian width IS2 (m)'
            y_label='Delta fb LaKu (m)'
            y_data = delta_laku_smooth
            R,RMSD,slope,dmean,ndata = st.plot_scatter(ax,xylim,'Roughness','m',x_data,x_label,y_data,y_label,None)
            plt.show()
            correlation_list.append(R)
            

        print("stop")
        """
            
        
        laser_fb = np.ma.concatenate(list(np.array(data_dict['IS2']['ATL10']['laser_fb_mean'],dtype=object)[idx_dates]),axis=0)
        gaussian_w = np.ma.concatenate(list(np.array(data_dict['IS2']['ATL10']['gaussian_w_mean'],dtype=object)[idx_dates]),axis=0)
        #roughness_log = np.ma.concatenate(list(np.array(data_dict['CS2']['UOB']['roughness'],dtype=object)[idx_dates]),axis=0)

        icetype = np.ma.concatenate(icetype_al,axis=0)

        #atIS2 = np.ma.concatenate(list(np.array(data_dict['IS2']['ATL10']['lat'],dtype=object)[idx_dates]),axis=0)
        latCS2 = np.ma.concatenate(list(np.array(data_dict['CS2']['ESA_BD_GDR']['lat'],dtype=object)[idx_dates]),axis=0)
        lonCS2 = np.ma.concatenate(list(np.array(data_dict['CS2']['ESA_BD_GDR']['lon'],dtype=object)[idx_dates]),axis=0)
        #latCS1 = np.ma.concatenate(list(np.array(data_dict['CS2']['UOB']['lat'],dtype=object)[idx_dates]),axis=0)

      
        # Get CS2 freeboard
        npts = lonCS2.shape[0]
        mask_list_all = np.zeros((npts,))
        radar_fb_list = list()
        radar_fb_matrix = ma.masked_array(np.zeros((len(prod_L2P),npts)),mask=np.ones((len(prod_L2P),npts)),dtype='float')
        for nprod,cs2_gdr in enumerate(prod_L2P):
            radar_fb = list(np.array(data_dict['CS2'][cs2_gdr]['radar_fb'],dtype=object)[idx_dates])
            fb_full=ma.masked_invalid(np.ma.concatenate(radar_fb,axis=0))
            mask_list_all = np.logical_or(mask_list_all,fb_full.mask)
            radar_fb_list.append(fb_full)
            radar_fb_matrix[nprod,:] = fb_full

        radar_fb_matrix = ma.masked_invalid(radar_fb_matrix,copy=True)
        radar_fb_mean = np.ma.mean(radar_fb_matrix,axis=0)
        radar_fb_std = np.ma.std(radar_fb_matrix,axis=0)

        
        delta_laku = laser_fb - radar_fb_mean

        dlaku = ma.masked_where(np.isnan(delta_laku),delta_laku,copy=True)
        dlaku_mask = dlaku.mask
        dlaku_smooth = st.rolling_stats(dlaku,int(nkm/mean_dist_btw_data), stats=['mean'])[0]
        dlaku_smooth = ma.masked_where(dlaku_mask,dlaku_smooth,copy=True)

        
        xylim = [0,0.4]
        f1, ax = plt.subplots(1, 1,figsize=(8,6))
        #f1.suptitle('Gaussian width', fontsize=12)
        bmap,cmap = st.plot_track_map(f1,ax,lonCS2,latCS2,dlaku_smooth,'$\Delta fb(la - ku)$',xylim,mid_date,'m',False,alpha=1)
        plt.show()



        # Histogramm freeboard
        #-------------------------------
        xylim = [-0.2,0.6]

        # ALL
        f1, axh = plt.subplots(1, 1,figsize=(8,8))
        f1.suptitle('Histogram of laser vs radar freeboard', fontsize=14)
        
        label_list = list()
        data_list = list()
        legend_list = list()
        
        label_IS2 = 'laser fb IS2 (m)'
        legend_list.append(label_IS2)
        data_list.append(laser_fb)
        xlabel= 'freeboard (m)'
        
        for nprod,cs2_gdr in enumerate(prod_L2P):

            legend_list.append('radar freeboard [%s] (m)' %(cs2_gdr))
            data_list.append(radar_fb_list[nprod])
        
        st.plot_histo(axh,xylim,'m',xlabel,legend_list,data_list,True)
        plt.show()
       

        # FYI
        f2, axh = plt.subplots(1, 1,figsize=(8,8))
        flag_FYI = icetype==2
        f2.suptitle('Histogram of laser vs radar freeboard FYI', fontsize=14)
        
        label_list = list()
        data_list = list()
        legend_list = list()
        
        label_IS2 = 'laser fb IS2 (m)'
        legend_list.append(label_IS2)
        data_list.append(laser_fb[flag_FYI])
        xlabel= 'freeboard (m)'
        
        for nprod,cs2_gdr in enumerate(prod_L2P):

            legend_list.append('radar freeboard [%s] (m)' %(cs2_gdr))
            data_list.append(radar_fb_list[nprod][flag_FYI])
        
        st.plot_histo(axh,xylim,'m',xlabel,legend_list,data_list,True)
        plt.show()


        # MYI
        f3, axh = plt.subplots(1, 1,figsize=(8,8))
        flag_MYI = icetype==4
        f3.suptitle('Histogram of laser vs radar freeboard MYI', fontsize=14)
        
        label_list = list()
        data_list = list()
        legend_list = list()
        
        label_IS2 = 'laser fb IS2 (m)'
        legend_list.append(label_IS2)
        data_list.append(laser_fb[flag_MYI])
        xlabel= 'freeboard (m)'
        
        for nprod,cs2_gdr in enumerate(prod_L2P):

            legend_list.append('radar freeboard [%s] (m)' %(cs2_gdr))
            data_list.append(radar_fb_list[nprod][flag_MYI])
        
        st.plot_histo(axh,xylim,'m',xlabel,legend_list,data_list,True)
        plt.show()

       
        
        
        print("stop")

        
        # map freeboard
        #--------------------------------
        """
        mask= gaussian_w > 0.5
        gaussian_w = ma.masked_where(mask,gaussian_w,copy=True)
        xylim = [0,0.25]
        f1, ax = plt.subplots(1, 1,figsize=(8,6))
        f1.suptitle('Gaussian width', fontsize=12)
        bmap,cmap = st.plot_track_map(f1,ax,lonCS2,latCS2,gaussian_w,'Gaussian width',xylim,enddate,'m',False,alpha=1)
        plt.show()
        """

        # map freeboard
        #--------------------------------
        """
        xylim = [0,0.4]
        f1, ax = plt.subplots(1, 1,figsize=(8,6))
        f1.suptitle('Delta fb', fontsize=12)
        bmap,cmap = st.plot_track_map(f1,ax,lonCS2,latCS2,delta_laku,'Delta fb',xylim,mid_date,'m',False,alpha=1)
        plt.show()
        """


        # Roughness lognormal vs roughness
        #---------------------------------------------
        
        xylim = [[0.05,0.5],[0,0.5]]
        f1, ax = plt.subplots(1, 1, sharey=True)
        f1.suptitle('Lognormal roughness CS2 vs Gaussian width IS2', fontsize=12)
        x_data = gaussian_w
        x_label = 'Gaussian width IS2 (m)'
        y_label='Roughness lognormal CS2 (m)'
        y_data = roughness_log
        st.plot_scatter(ax,xylim,'','m',x_data,x_label,y_data,y_label,None)
        plt.show()
        


        # Scatter plot LakU
        #-----------------------------------------------
        xylim = [[0.05,0.3],[0.05,0.5]]
        
        f1, ax = plt.subplots(1, 1, sharey=True)
        f1.suptitle('Delta fb LaKu vs Gaussian width IS2', fontsize=12)
        #xylim = [[-0.1,1],[-0.1,1]]
        x_data = gaussian_w
        x_label = 'Gaussian width IS2 (m)'
        y_label='Delta fb LaKu (m)'
        y_data = delta_laku
        st.plot_scatter(ax,xylim,'all','m',x_data,x_label,y_data,y_label,None)
        
        Rpearson = list()
        RMSDev = list()
        slope = list()
        for nprod,cs2_gdr in enumerate(prod_L2P):
            
           
            
            print(cs2_gdr)
            f1, ax = plt.subplots(1, 1, sharey=True)
            f1.suptitle('Delta fb LaKu vs Gaussian width IS2', fontsize=12)
            x_data = gaussian_w
            GW = ma.masked_where(np.isnan(x_data),x_data,copy=True)
            GW_mask = GW.mask
            GW_smooth = st.rolling_stats(GW,int(nkm/mean_dist_btw_data), stats=['mean'])[0]
            GW_smooth = ma.masked_where(GW_mask,GW_smooth,copy=True)
            GW_smooth = ma.masked_where(mask_list_all,GW_smooth,copy=True)
            x_label = 'Gaussian width IS2 (m)'
            y_label='Delta fb LaKu (m)'
            y_data =  laser_fb - radar_fb_list[nprod]
            dlaku = ma.masked_where(np.isnan(y_data),y_data,copy=True)
            dlaku_mask = dlaku.mask
            dlaku_smooth = st.rolling_stats(dlaku,int(nkm/mean_dist_btw_data), stats=['mean'])[0]
            dlaku_smooth = ma.masked_where(dlaku_mask,dlaku_smooth,copy=True)
            dlaku_smooth = ma.masked_where(mask_list_all,dlaku_smooth,copy=True)
            R,RMSD,slp = st.plot_scatter(ax,xylim,cs2_gdr,'m',GW_smooth,x_label,dlaku_smooth,y_label,None)
            Rpearson.append(R)
            RMSDev.append(RMSD)
            slope.append(slp)

        #plt.show()
        columns = ['products','Rpearson','slope']
        data = [(prod,Rpearson[nprod],slope[nprod]) for nprod,prod in enumerate(prod_L2P)]
        df = pd.DataFrame.from_records(data,columns=columns)

        df_melted = pd.melt(df, id_vars=columns[0],var_name="prods", value_name="value_numbers")

        #mask = df_melted.products.isin(['slope'])
        #scale = df_melted[~mask].value_numbers.mean()/df_melted[mask].value_numbers.mean()
        #df_melted.loc[mask, 'value_numbers'] = df_melted.loc[mask, 'value_numbers']*scale

        # Plotpalette='mako', 
        fig, ax1 = plt.subplots()
        g = sns.barplot(x=columns[0], y="value_numbers", hue="prods",data=df_melted,palette='mako', ax=ax1)
        
        # Create a second y-axis with the scaled ticks
        ax1.set_ylabel('slope')
        ax2 = ax1.twinx()

        # Ensure ticks occur at the same positions, then modify labels
        ax2.set_ylim(ax1.get_ylim())
        #ax2.set_yticklabels(np.round(ax1.get_yticks()/scale,1))
        ax2.set_ylabel('Rpearson')

        plt.show()

        
        df = pd.DataFrame({'Products': prod_L2P,
                'Rpearson': Rpearson,
                'RMSD':RMSDev,
                'slope':slope,})

        

            
       
        tips = sns.load_dataset("tips")
        
        fig, ax1 = plt.subplots(figsize=(10, 10))
        tidy = df.melt(id_vars='Products').rename(columns=str.title)
        sns.barplot(x='Products', y='RMSD', data=df, ax=ax1)
        sns.despine(fig)

        data = {'Products': prod_L2P,
                'Rpearson': Rpearson,
                'RMSD':RMSDev,
                'slope':slope,
        }
        

        """
        df = pd.DataFrame(data, columns = ['Products','Rpearson','RMSD','slope'])

        print (df)

        import seaborn as sns

        
        sns.set_theme(style="whitegrid")
        tips = sns.load_dataset("tips")
        ax = sns.barplot(x='Products', y="total_bill", hue="sex", data=tips)

        f1, ax = plt.subplots(1, 1, sharey=True)
        ax.bar(prod_L2P, Rpearson, width=0.8) 
        ax.bar(prod_L2P,  RMSDev, width=0.8) 
        """
        
        

        xylim = [[0.05,0.25],[-0.20,0.2]]
        f1, ax = plt.subplots(1, 1, sharey=True)
        f1.suptitle('Delta fb SAM -T50 vs Gaussian width IS2', fontsize=12)
        x_data = gaussian_w
        GW = ma.masked_where(np.isnan(x_data),x_data,copy=True)
        GW_mask = GW.mask
        GW_smooth = st.rolling_stats(GW,int(nkm/mean_dist_btw_data), stats=['mean'])[0]
        GW_smooth = ma.masked_where(GW_mask,GW_smooth,copy=True)
        GW_smooth = ma.masked_where(mask_list_all,GW_smooth,copy=True)
        x_label = 'Gaussian width IS2 (m)'
        y_label='Delta fb LaKu (m)'
        y_data =  radar_fb_list[2] - radar_fb_list[1]
        dlaku = ma.masked_where(np.isnan(y_data),y_data,copy=True)
        dlaku_mask = dlaku.mask
        dlaku_smooth = st.rolling_stats(dlaku,int(nkm/mean_dist_btw_data), stats=['mean'])[0]
        dlaku_smooth = ma.masked_where(dlaku_mask,dlaku_smooth,copy=True)
        dlaku_smooth = ma.masked_where(mask_list_all,dlaku_smooth,copy=True)
        st.plot_scatter(ax,xylim,cs2_gdr,'m',GW_smooth,x_label,dlaku_smooth,y_label,None)

        plt.show()

        
       


        #
        # Histogram
        #-------------

        xylim = [-0.3,0.5]
        f2, axh = plt.subplots(1, 1,figsize=(8,8))
        f2.suptitle('Histogram of laser vs radar freeboard', fontsize=14)
        
        label_list = list()
        data_list = list()
        legend_list = list()
        
        label_IS2 = 'laser fb IS2 (m)'
        legend_list.append(label_IS2)
        data_list.append(laser_fb)
        xlabel= 'freeboard (m)'
        
        for nprod,cs2_gdr in enumerate(prod_L2P):

            legend_list.append('radar freeboard [%s] (m)' %(cs2_gdr))
            data_list.append(radar_fb_list[nprod])
        
        st.plot_histo(axh,xylim,'m',xlabel,legend_list,data_list,True)
        plt.show()
        


    if param=='mean_grid':

        
        lat = np.concatenate(ref_seg_lat,axis=0)
        lon = np.concatenate(ref_seg_lon,axis=0)
        radar_fb = np.ma.concatenate(list(np.array(data_dict['CS2'][REF_GDR]['radar_fb'],dtype=object)[idx_dates]),axis=0)
        # Get IS2 freeboard
        laser_fb =  np.ma.concatenate(list(np.array(data_dict['IS2']['ATL10']['laser_fb_mean'],dtype=object)[idx_dates]),axis=0)

        data= {}
        data['fb'] = radar_fb

        
        # Slow down factor
        ds = 0.300
        ns = (1 + 0.51*ds)**(-1.5)
        nkm = 75 #km

        sd_laku = (laser_fb - radar_fb)*ns
        sd_laku = ma.masked_where(np.isnan(sd_laku),sd_laku,copy=True)
        sd_mask = sd_laku.mask
        sd_laku_smooth = st.rolling_stats(sd_laku,int(nkm/mean_dist_btw_data), stats=['mean'])[0]
        sd_laku_smooth = ma.masked_where(sd_mask,sd_laku_smooth,copy=True)

        # plot along-track
        
        f1, ax = plt.subplots(1, 1,figsize=(9,8))
        bmap,cmap = st.plot_track_map(f1,ax,lon,lat,sd_laku_smooth,'Snow depth',[0,0.4],mid_date,'m',False,alpha=1,size=5)
        
        plt.show()
        
        
        # grid data
        import grid_and_filter as grid
        
        
        data= {}
        sd_laku_smooth = ma.masked_invalid(sd_laku_smooth,copy=True)
        data['sd'] = sd_laku_smooth.data
        data['sd'][sd_laku_smooth.mask] = np.nan
        label ='SD LaKu'
        units = 'm'
        
        f2, ax = plt.subplots(1, 1,figsize=(9,8))
        m = Basemap(projection='npstere', llcrnrlat=0,urcrnrlat=90,llcrnrlon=-180,urcrnrlon=180,boundinglat=60,lon_0=0, resolution='l',round=True,ax=ax)
        x_grid, y_grid, lat_grid_mesh, lon_grid_mesh, x_grid_mesh, y_grid_mesh, data_results = st.grid_and_filter_wrt_distance(lon, lat, data, m, pixel_size=12500,mode='filter_mean',range_filter=25000, verbose=0)
        
        sd = np.ma.array(data_results['sd'])
        bmap,cmap = st.plot_track_map(f2,ax,lon_grid_mesh,lat_grid_mesh, sd,'snow depth',[0,0.4],mid_date,'m',False,alpha=1,size=3)
        
        plt.show()
        

        print("stop")
        


    if param=='comp_grid':


        # Slow down factor
        ds = 0.300
        ns = (1 + 0.51*ds)**(-1.5)
        nkm = 25 #km

        maplim = [0,0.4]
        sizepixmap = 1
        
        # open gridded products
        sd_laku_grid_list = list()
        lat_list = list()
        lon_list = list()
        sd_laku_grid_al_list = list()
        
        for month,idx in idx_dates_monthly.items():

            datestr = month #.strftime('%Y%m')
            print("%s \n" %(datestr))

            mid_idx = int((idx_dates[idx][-1] -idx_dates[idx][0])/2)
            mid_date = available_dates[mid_idx]
            
            #--------------------------------------
            # Get gridded data
            #-------------------------------------

            # Get CryoSat-2 grid
            filepattern =PATH_GRID +'CS2/%s/*%s_%s_*500.nc' %(REF_GDR,REF_GDR,datestr)
            filename = glob.glob(filepattern)
            if len(filename)==0: sys.exit("\n%s: No found" %(filepattern))
            else:
                filename = filename[0]
                print("\nReading CS2 grid file %s" %(filename))
        
            f = nc.Dataset(filename)
            lat_grid = f.variables['latitude'][:]
            lon_grid = f.variables['longitude'][:]
            radar_fb = f.variables['radar_fb'][:]

            # flag canadian archipelagos
            #----------------------------
            flag_lat = lat_grid < 78
            flag_lon = np.logical_and(lon_grid>-120,lon_grid<-30)
            flag_CA = np.logical_and(flag_lat,flag_lon)
            #--------------------------

            # Get IceSat-2 grid
            filepattern =PATH_GRID +'IS2/ATL10/*ATL10_%s_*500.nc' %(datestr)
            filename = glob.glob(filepattern)
            if len(filename)==0: sys.exit("\n%s: No found" %(filepattern))
            else:
                filename = filename[0]
                print("\nReading IS2 grid file %s" %(filename))
        
            f = nc.Dataset(filename)
            #lat = f.variables['lat'][:]
            #lon = f.variables['lon'][:]
            laser_fb = f.variables['laser_fb'][:]
            gaussian_w = f.variables['gaussian_w'][:]

            delta_laku = laser_fb - radar_fb
            sd_laku_grid = delta_laku*ns
            sd_laku_grid = ma.masked_where(flag_CA,sd_laku_grid,copy=True)
            sd_laku_grid_list.append(sd_laku_grid)
            lat_list.append(lat_grid)
            lon_list.append(lon_grid)

            # Get Icetype in right grid
            #--------------------------
           
            icetype_mid = icetype[mid_idx]
            lon_icetype = lons_icetype[mid_idx]
            lat_icetype = lats_icetype[mid_idx]            
            import scipy.interpolate
            icetype_500= scipy.interpolate.griddata((lon_icetype.flatten(),lat_icetype.flatten()),icetype_mid.flatten() , (lon_grid,lat_grid),method='nearest')
            
            print("stop")

            # Get AMSR
            #-----------------------------------------
            lat_AMSR_grid,lon_AMSR_grid,SD_AMSR = cf.get_SD_AMSR(mid_date)
            SD_AMSR = SD_AMSR
            #SD_AMSR[SD_AMSR.mask] = np.nan
            SD_AMSR_500= scipy.interpolate.griddata((lon_AMSR_grid.flatten(),lat_AMSR_grid.flatten()),SD_AMSR.flatten() , (lon_grid,lat_grid),method='nearest')
            SD_AMSR_500 = ma.masked_where(SD_AMSR_500==110,SD_AMSR_500,copy=True)
            SD_AMSR_500 = ma.masked_where(SD_AMSR_500==120,SD_AMSR_500,copy=True)
            SD_AMSR_500 = ma.masked_where(SD_AMSR_500==130,SD_AMSR_500,copy=True)
            SD_AMSR_500 = ma.masked_where(SD_AMSR_500==140,SD_AMSR_500,copy=True)
            SD_AMSR_500 = ma.masked_where(SD_AMSR_500==150,SD_AMSR_500,copy=True)
            SD_AMSR_500 = ma.masked_where(SD_AMSR_500==160,SD_AMSR_500,copy=True)
            SD_AMSR_500 = SD_AMSR_500/100
            #SD_AMSR_500 = ma.masked_invalid(SD_AMSR_500,copy=True)

            # Get ASD
            #-------------------------------------
            pixsize = 25
            lat_ASD_grid,lon_ASD_grid,sd_ASD_grid,sd_unc_grid = cf.get_ASD(pixsize,month)
            sd_ASD_grid = ma.masked_where(flag_CA,sd_ASD_grid,copy=True)


            pathout = "/home/antlafe/Documents/work/projet_cryo2ice/figure/roughness_vs_penetration/"
            #--------------------------------------
            # Plot Gridded data
            #-------------------------------------

            # Maps
            #-----------

            
            # ASD
            xylim = [0,0.4]
            """
            xylim = [0,0.4]
            f1, ax = plt.subplots(1, 1,figsize=(8,6))
            f1.suptitle('Snow depth ASD %s' %(month), fontsize=12)
            bmap,cmap = st.plot_track_map(f1,ax,lon_ASD_grid,lat_ASD_grid,sd_ASD_grid,'Snow depth ASD',xylim,mid_date,'m',False,alpha=1)
            plt.savefig(pathout+'Snow_depth_ASD_%s.png' %(month))
            """

            
            # AMSR
            f1, ax = plt.subplots(1, 1,figsize=(8,6))
            f1.suptitle('Snow depth AMSR %s' %(month), fontsize=12)
            bmap,cmap = st.plot_track_map(f1,ax,lon_grid,lat_grid,SD_AMSR_500,'Snow depth AMSR',xylim,mid_date,'m',False,alpha=1)
            plt.savefig(pathout+'Snow_depth_AMSR_%s.png' %(month))
            plt.show()

            # LaKu
            f2, ax = plt.subplots(1, 1,figsize=(8,6))
            f2.suptitle('Snow depth LaKu %s' %(month), fontsize=12)
            bmap,cmap = st.plot_track_map(f2,ax,lon_grid,lat_grid,sd_laku_grid,'Snow depth LaKu',xylim,mid_date,'m',False,alpha=1)
            plt.savefig(pathout+'Snow_depth_Laku_%s.png' %(month))

            # LaKu - AMSR
            delta_sd =  sd_laku_grid - SD_AMSR_500
            f2, ax = plt.subplots(1, 1,figsize=(8,6))
            f2.suptitle('Snow depth LaKu - ASD %s' %(month), fontsize=12)
            bmap,cmap = st.plot_track_map(f2,ax,lon_grid,lat_grid,delta_sd,'Delta SD LaKu - AMSR',[-0.1,0.1],mid_date,'m',True,alpha=1)
            plt.savefig(pathout+'delta_sd_Laku_AMSR_%s.png' %(month))

            # LaKu - ASD
            delta_sd =  sd_laku_grid - sd_ASD_grid
            f2, ax = plt.subplots(1, 1,figsize=(8,6))
            f2.suptitle('Snow depth LaKu - ASD %s' %(month), fontsize=12)
            bmap,cmap = st.plot_track_map(f2,ax,lon_grid,lat_grid,delta_sd,'Delta SD LaKu - ASD',[-0.1,0.1],mid_date,'m',True,alpha=1)
            plt.savefig(pathout+'delta_sd_Laku_ASD_%s.png' %(month))
            
            plt.show()
            
            # GW
            f3, ax = plt.subplots(1, 1,figsize=(8,6))
            f3.suptitle('Gaussian Width %s' %(month), fontsize=12)
            bmap,cmap = st.plot_track_map(f3,ax,lon_grid,lat_grid,gaussian_w,'Gaussian width',[0,0.2],mid_date,'m',False,alpha=1)
            plt.savefig(pathout+'Snow_depth_GW_%s.png' %(month))
            #plt.savefig(pathout+'scat_sd_%s.png' %(datestr))


            # Scatters
            #----------            
            xylim =  [[-0.1,0.4],[-0.1,0.4]]
            f1, ax = plt.subplots(1, 1,figsize=(6,6))
            f1.suptitle('ASD vs Laku', fontsize=12)
            st.plot_scatter(ax,xylim,'','m',sd_ASD_grid,"SD ASD",sd_laku_grid,"SD LaKu",None)
            plt.savefig(pathout+'Laku_vs_ASD_%s.png' %(month))

            f2, ax = plt.subplots(1, 1,figsize=(6,6))
            f2.suptitle('ASD vs GW', fontsize=12)
            st.plot_scatter(ax,xylim,'','m',gaussian_w,"GW",sd_ASD_grid,"SD ASD",icetype_500)
            plt.savefig(pathout+'ASD_vs_GW_%s.png' %(month))

            f3, ax = plt.subplots(1, 1,figsize=(6,6))
            f3.suptitle('LaKu vs GW', fontsize=12)
            st.plot_scatter(ax,xylim,'','m',gaussian_w,"GW",sd_laku_grid,"SD Laku",icetype_500)
            plt.savefig(pathout+'Laku_vs_GW_%s.png' %(month))
            
            

            #plt.show()
           

            print("stop")

            #--------------------------------------
            # Get gridded data
            #-------------------------------------

            # Get LaKu
            #-----------------

            ref_lat = np.ma.concatenate(list(np.array(data_dict['CS2'][REF_GDR]['lat'],dtype=object)[idx_dates[idx]]),axis=0)
            ref_lon = np.ma.concatenate(list(np.array(data_dict['CS2'][REF_GDR]['lon'],dtype=object)[idx_dates[idx]]),axis=0)
            
            # Get CS2 freeboard
            npts = ref_lat.shape[0]
            radar_fb_list = list()
            radar_fb_matrix = ma.masked_array(np.zeros((len(prod_L2P),npts)),mask=np.ones((len(prod_L2P),npts)),dtype='float')
            for nprod,cs2_gdr in enumerate(prod_L2P):
                radar_fb = np.ma.concatenate(list(np.array(data_dict['CS2'][cs2_gdr]['radar_fb'],dtype=object)[idx_dates[idx]]),axis=0)
                fb_full=ma.masked_invalid(radar_fb)
                radar_fb_list.append(fb_full)
                radar_fb_matrix[nprod,:] = fb_full

            radar_fb_matrix = ma.masked_invalid(radar_fb_matrix,copy=True)
            radar_fb_mean = np.ma.mean(radar_fb_matrix,axis=0)
            radar_fb_std = np.ma.std(radar_fb_matrix,axis=0)
            #radar_fb = np.ma.concatenate(list(np.array(data_dict['CS2'][REF_GDR]['radar_fb'],dtype=object)[idx_dates[idx]]),axis=0)

            # Get IS2 freeboard
            laser_fb =  np.ma.concatenate(list(np.array(data_dict['IS2']['ATL10']['laser_fb_mean'],dtype=object)[idx_dates[idx]]),axis=0)
            
            
            sd_laku_al = (laser_fb - radar_fb_mean)*ns
            sd_laku_al = ma.masked_where(np.isnan(sd_laku_al),sd_laku_al,copy=True)
            sd_mask = sd_laku_al.mask
            sd_laku_smooth = st.rolling_stats(sd_laku_al,int(nkm/mean_dist_btw_data), stats=['mean'])[0]
            sd_laku_smooth = ma.masked_where(sd_mask,sd_laku_smooth,copy=True)

            
            # get Laku from grid
            #------------------
            sd_laku_grid_al = cf.grid_to_track(sd_laku_grid,lon_grid,lat_grid,ref_lon,ref_lat)

            # Get ASD
            #-----------------
            sd_ASD_al = cf.grid_to_track(sd_ASD_grid,lon_grid,lat_grid,ref_lon,ref_lat)


            # Roughness from logNorm
            #-------------
            rough_lognorm = np.ma.concatenate(list(np.array(data_dict['CS2']["UOB"]['roughness'],dtype=object)[idx_dates[idx]]),axis=0)


            # Roughness from GW
            #-------------
            rough_GW = np.ma.concatenate(list(np.array(data_dict['IS2']['ATL10']['gaussian_w_mean'],dtype=object)[idx_dates[idx]]),axis=0)

            # icetype
            #------------
            icetype_alongtrack = cf.grid_to_track(icetype_mid,lon_icetype,lat_icetype,ref_lon,ref_lat)

            # Grid to track
            #sd_laku_grid_al = cf.grid_to_track(sd_ASD_grid,lon_grid,lat_grid,ref_lon,ref_lat)
            #sd_laku_grid_al_list.append(sd_laku_grid_al)

            #pathout = '/home/antlafe/Documents/work/projet_cryo2ice/figure/sd_grid/'

            # Figures
            #---------------------------------


            # Scatters
            #----------            
            xylim =  [[-0.1,0.4],[-0.1,0.4]]
            f1, ax = plt.subplots(1, 1,figsize=(6,6))
            f1.suptitle('ASD vs Laku AL', fontsize=12)
            st.plot_scatter(ax,xylim,'','m',sd_ASD_al,"SD ASD",sd_laku_smooth,"SD LaKu",None)
            plt.savefig(pathout+'ASD_vs_Laku_AL_%s.png' %(month))

            f2, ax = plt.subplots(1, 1,figsize=(6,6))
            f1.suptitle('ASD vs Laku AL from grid', fontsize=12)
            st.plot_scatter(ax,xylim,'','m',sd_ASD_al,"SD ASD",sd_laku_grid_al,"SD LaKu",None)
            plt.savefig(pathout+'ASD_vs_Laku_AL_from_grid_%s.png' %(month))
            #plt.show()
            
            f3, ax = plt.subplots(1, 1,figsize=(6,6))
            f3.suptitle('ASD vs GW AL', fontsize=12)
            st.plot_scatter(ax,xylim,'','m',rough_GW,"GW",sd_ASD_al,"SD ASD",icetype_alongtrack)
            plt.savefig(pathout+'ASD_vs_GW_AL_%s.png' %(month))

            f3, ax = plt.subplots(1, 1,figsize=(6,6))
            f3.suptitle('ASD vs LogN Rough AL', fontsize=12)
            st.plot_scatter(ax,xylim,'','m',rough_lognorm,"LogNorm rough",sd_ASD_al,"SD ASD",icetype_alongtrack)
            plt.savefig(pathout+'ASD_vs_LogN_Rough_AL_%s.png' %(month))

            #plt.show()
            
            f3, ax = plt.subplots(1, 1,figsize=(6,6))
            f3.suptitle('SD LaKu vs GW AL', fontsize=12)
            st.plot_scatter(ax,xylim,'','m',rough_GW,"GW",sd_laku_smooth,"SD LaKu",icetype_alongtrack)
            plt.savefig(pathout+'ASD_LaKu_vs_GW_AL_%s.png' %(month))

            f3, ax = plt.subplots(1, 1,figsize=(6,6))
            f3.suptitle('SD LaKu vs LogN Rough AL', fontsize=12)
            st.plot_scatter(ax,xylim,'','m',rough_lognorm,"LogN rough",sd_laku_smooth,"SD LaKu",icetype_alongtrack)
            plt.savefig(pathout+'SD_LaKu_vs_LogN_Rough_AL_%s.png' %(month))

            #plt.show()

            f4, ax = plt.subplots(1, 1,figsize=(6,6))
            f4.suptitle('GW vs LogN Rough AL', fontsize=12)
            st.plot_scatter(ax,xylim,'','m',rough_GW,"SD ASD",rough_lognorm,"LogN rough",None)
            plt.savefig(pathout+'GW_vs_LogN_Rough_AL_%s.png' %(month))
            #plt.show()
            
            """
            x_data =  sd_laku_grid_al
            x_label = 'Snow depth LaKu month [m]'
            y_data = sd_laku_smooth
            y_label= 'snow depth LaKu CRYO2ICE [m]'
            x_data = ma.masked_where(np.isnan(x_data),x_data,copy=True)
            y_data = ma.masked_where(np.isnan(y_data),y_data,copy=True)
            mask_data = np.logical_and(~x_data.mask,~y_data.mask)
            

            # scatter map
            #-------------------
            xylim = [maplim,maplim]
            f1, ax = plt.subplots(1, 1,figsize=(6,6))
            f1.suptitle('comparison with monthly gridded map', fontsize=12)
            st.plot_scatter(ax,xylim,'Laku','m',x_data,x_label,y_data,y_label,None)

            #plt.savefig(pathout+'scat_sd_%s.png' %(datestr))
            
            # map
            #--------------------
            f2, ax = plt.subplots(1, 1,figsize=(10,10))
            bmap,cmap = st.plot_track_map(f2,ax,lon_grid,lat_grid,sd_laku_grid,'Snow depth',maplim,mid_date,'m',False,size=sizepixmap)
            st.add_data_track(bmap,cmap,ref_lon,ref_lat,sd_laku_smooth,maplim)

            plt.savefig(pathout+'map_sd_%s.png' %(datestr))
            #plt.savefig('books_read.png')
            plt.show()
            """

            """
            x_label = "Gaussian width [m]"
            y_label = "delta fb LaKu monthly [m]"

            xylim = [maplim,maplim]
            f1, ax = plt.subplots(1, 1,figsize=(8,6))
            f1.suptitle('comparison between Laku and surface roughness', fontsize=12)
            st.plot_scatter(ax,xylim,'Laku','m',gaussian_w,x_label,delta_laku,y_label,None)

            plt.show()
            #plt.savefig(pathout+'map_sd_%s.png' %(datestr))
            #plt.savefig(pathout+'scat_rough_%s.png' %(datestr))

            # Gaussian width gridded
            f2, ax = plt.subplots(1, 1,figsize=(8,6))
            bmap,cmap = st.plot_track_map(f2,ax,lon_grid,lat_grid,gaussian_w,'Gaussian width',[0.07,0.2],mid_date,'m',False,size=sizepixmap)
            plt.show()
            #plt.savefig(pathout+'map_rough_%s.png' %(datestr))

            

            # comparison with ASD
            #------------------------------
            
            pixsize = 50
            #months = [date.strftime('%Y%m') for date in found_dates]
            lat_grid,lon_grid,sd_grid,sd_unc_grid = cf.get_ASD(pixsize,month)
            
            
            x_data =  sd_laku_grid
            x_label = 'Snow depth LaKu month [m]'
            y_data = sd_grid
            y_label= 'snow depth ASD [m]'
            x_data = ma.masked_where(np.isnan(x_data),x_data,copy=True)
            y_data = ma.masked_where(np.isnan(y_data),y_data,copy=True)
            mask_data = np.logical_and(~x_data.mask,~y_data.mask)
            
            
            xylim = [maplim,maplim]
            f1, ax = plt.subplots(1, 1,figsize=(6,6))
            f1.suptitle('comparison with monthly gridded map', fontsize=12)
            st.plot_scatter(ax,xylim,'Laku','m',x_data,x_label,y_data,y_label,None)

            plt.show()

            # Gaussian width vs ASD
            f2, ax = plt.subplots(1, 1,figsize=(8,6))
            f2.suptitle('ASD vs Gaussian width', fontsize=12)
            st.plot_scatter(ax,xylim,'Laku','m',x_data,x_label,y_data,y_label,None)
            

            plt.savefig(pathout+'scat_sd_%s.png' %(datestr))
            """
            
            
    
            
    if param=='comp_surf':

        flag_lead_list = list()                                  
        for nprod,cs2_gdr in enumerate(prod_L2P):
            flag_leads = np.ma.concatenate(list(np.array(data_dict['CS2'][cs2_gdr]['flag_leads'],dtype=object)[idx_dates]),axis=0)
            
                
        flag_leads_is2 = np.ma.concatenate(list(np.array(data_dict['IS2'][cs2_gdr]['flag_leads'],dtype=object)[idx_dates]),axis=0)
        flag_leads_is2 = np.ma.concatenate(list(np.array(data_dict['IS2'][cs2_gdr]['flag_leads'],dtype=object)[idx_dates]),axis=0)


        
        
        
        
                                          

        
            
