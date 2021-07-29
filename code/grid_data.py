#! /home/antlafe/anaconda3/bin/python

#
# grid_data.py
#
# ----------------------------------------------------------------------
# Copyright (c) 2020 LEGOS/SERCO
# All rights reserved.
#



"""
DESCRIPTION:

     Program to grid data

USAGE:

     grid_data.py [options]

optional arguments:


EXAMPLES:

    python grid_data.py -s CS2 -g  -f path/file_names -p sla -opt 500,...ect
    python -m pdb grid_data.py -s IS2 -g ATL10 -d 202011 -p laser_fb,surface_h -hp 01

    # nease 500 lat0=61.6
    # npstere 451 lat0=60 

COMMENTS:

    - Only one product at once  

"""
import os
import sys
import h5py
import netCDF4 as nc
import numpy as np
from numpy import ma 
import matplotlib.pyplot as plt
import glob
from datetime import date, timedelta, datetime
import argparse
from scipy import signal
import cs2_dict
#import cryosat2_dict as cs2_dict
import is2_dict
import common_functions as cf
import time
from mpl_toolkits.basemap import Basemap
import stats_tools as st
from netCDF4 import Dataset
import path_dict

# Global attributs
###########################################

#PATH_INPUT = "/home/antlafe/Documents/work/data/"

# on HAL
#PATH_INPUT = "/work/ALT/odatis/seaice/users/laforga/projet_cryo2ice/data/all/"
PATH_INPUT = path_dict.PATH_DICT['PATH_DATA']
#PATH_GRID = path_dict.PATH_DICT['PATH_GRID']

# change if looking at Antarctica
global LAT_BOUND
LAT_BOUND = 49.08  #1.8772498200144 #65 # deg north [61.6: 500] [49.08: 712]
#LAT_BOUND = 61.6  #1.8772498200144 #65 # deg north [61.6: 500]

show_figure = False

###########################################
#
#              Functions
#
###########################################



# ----------------------------------------------------------------------
# 
# init_grid
# 
def init_grid(lon, lat, all_data_in, map_frame,weight=None,
              pixel_size=10000,verbose=0):

    # in case in old fashion uniq data
    if not isinstance(all_data_in,dict):
        all_data = {}
        all_data['fb'] = all_data_in
    else:
        all_data = all_data_in

    # INITIALIZATION    
    # --------------------------------

    print('\t---> Build up grid and project track points')
    start_time_init = time.time()

    # Remove all points with a NaN data
    nan_data_flag = None
    for p_name,data in all_data.items():
        if nan_data_flag is None:
            nan_data_flag = np.isnan(data)
        else:
            nan_data_flag = nan_data_flag + np.isnan(data)
    for p_name in all_data.keys():
        all_data[p_name] = all_data[p_name][~nan_data_flag]
    lon = lon[~nan_data_flag]
    lat = lat[~nan_data_flag]
    if weight is not None: weight = weight[~nan_data_flag]

    # Projections of the longitudes and latitudes of the satellite ground-track that will be filtered to fit the grid 
    x_track,y_track = map_frame(lon,lat)

    # Mask the invalid data
    x_track = np.array(x_track)
    y_track = np.array(y_track)

    # Grid coordinates array build up
    x_max = map_frame.xmax
    x_min = map_frame.xmin

    y_max = map_frame.ymax
    y_min = map_frame.ymin

    nb_pixels_1D = int(np.floor((y_max-y_min)/pixel_size))
    pixel_size = (y_max-y_min)/nb_pixels_1D

    x_min_grid = x_min + pixel_size/2  
    x_max_grid = x_max - pixel_size/2
    
    y_min_grid = y_min + pixel_size/2
    y_max_grid = y_max - pixel_size/2 # same reason

    x_grid = np.linspace(x_min_grid, x_max_grid, nb_pixels_1D)
    y_grid = np.linspace(y_max_grid, y_min_grid, nb_pixels_1D)

    # Meshgrid
    x_grid_mesh, y_grid_mesh = np.meshgrid(x_grid, y_grid)
    
    lon_grid_mesh, lat_grid_mesh = map_frame(x_grid_mesh, y_grid_mesh, inverse=True)

    # Convert x,y (projections of the along-track coordinates) in pixel subscripts 
    # Beware: x values translate into column subscripts and y valus into row subscripts, and subscripts of pixels can be out of the grid (i.e subscripts < 0 or > maximum size)
    # round to the nearest int so the x,y are centered around the grid pixels coordinates
    pixels_track = np.zeros((2, x_track.size), dtype=int)
    pixels_track[0,:] = np.around((y_max_grid- y_track)/pixel_size).astype(int)
    pixels_track[1,:] = np.around((x_track-x_min_grid)/pixel_size).astype(int)

    return lon, lat, all_data, x_grid, y_grid, x_track, y_track, lon_grid_mesh, lat_grid_mesh,  x_grid_mesh, y_grid_mesh, pixels_track, nb_pixels_1D, weight



# ----------------------------------------------------------------------
# 
# grid_and_filter_wrt_distance
# 
def grid_and_filter_wrt_distance(lon, lat, all_data_in, map_frame, pixel_size=10000,mode='filter_gauss',range_filter=50000,weight=None, verbose=0):
    
    print('\n\tGrid and filter with a filter range greater than the pixel size')
    start_time = time.time()

    # init grid
    lon, lat, all_data, x_grid, y_grid, x_track, y_track, lon_grid_mesh, lat_grid_mesh,  x_grid_mesh, y_grid_mesh, pixels_track, nb_pixels_1D, weight  = init_grid(lon, lat, all_data_in, map_frame,weight,pixel_size, verbose)

    # init weight factor
    if weight is None:
        weight_coef = np.ones(lon.shape)
    else:
        weight_coef = weight

 
    # SORT TRACK POINTS    
    # 
    # --------------------------------
    print('\t---> Assign track points to the grid pixels')

    start_time_sort = time.time()

    # For each track point, choose the grid pixels that will use this point to compute the pixel averaged data
    range_filter_in_nb_of_pixels = np.ceil(range_filter/pixel_size)
    # Allocate the 2D jagged list
    grid_data_considered = [[[] for j in range(nb_pixels_1D)] for i in range(nb_pixels_1D)] # empty matrix of lists (shape=gridCoord shape)
    min_dist_to_closest_track = np.zeros((nb_pixels_1D,nb_pixels_1D)) + range_filter_in_nb_of_pixels
    weight_min_dist_to_closest_track = np.zeros((nb_pixels_1D,nb_pixels_1D))
    grid_number_of_data_considered = np.zeros((nb_pixels_1D,nb_pixels_1D))

    progress_prev = -1
    nb_track_data = pixels_track.shape[1]
    processed_pixels = []
    for k in range(nb_track_data):

        if verbose>0:
            progress = np.floor(100*k/nb_track_data)
            if(progress != progress_prev): print("sorting progress: %.0f %%" %progress)
            progress_prev = progress

        pixel_covering_data = pixels_track[:,k] # subscripts of the pixel including the current projected track point
        x_i = pixel_covering_data[0]
        y_i = pixel_covering_data[1]

        """
        if (x_i,y_i) in processed_pixels:
            continue"""
        
        is_pixel_in_the_grid = x_i>=0 and x_i<nb_pixels_1D and y_i>=0 and y_i < nb_pixels_1D
        if not is_pixel_in_the_grid: continue

        # To only fill pixels containing a sat track
        """
        interval_in_x = ((pixels_track[0,:] > x_i-range_filter_in_nb_of_pixels) & (pixels_track[0,:] < x_i+range_filter_in_nb_of_pixels))
        interval_in_y = ((pixels_track[1,:] > y_i-range_filter_in_nb_of_pixels) & (pixels_track[1,:] < y_i+range_filter_in_nb_of_pixels))
        index = np.squeeze(np.argwhere(interval_in_x & interval_in_y))
        grid_data_considered[x_i][y_i] = index

        processed_pixels.append((x_i,y_i))"""

        # Pixel_covering_data can be in the grid but the grid pixels that are in range for the filtering can be out of the grid.
        min_row_subGrid = max(0, x_i-range_filter_in_nb_of_pixels)
        max_row_subGrid = min(nb_pixels_1D-1, x_i+range_filter_in_nb_of_pixels)
        min_col_subGrid = max(0, y_i-range_filter_in_nb_of_pixels)
        max_col_subGrid = min(nb_pixels_1D-1, y_i+range_filter_in_nb_of_pixels)

        #grid_data_considered[int(min_row_subGrid):int(max_row_subGrid)+1][int(min_col_subGrid):int(max_col_subGrid)+1].append(k)
        #grid_data_considered[int(min_row_subGrid):int(max_row_subGrid)+1,int(min_col_subGrid):int(max_col_subGrid)+1] = k
        grid_number_of_data_considered[int(min_row_subGrid):int(max_row_subGrid)+1,int(min_col_subGrid):int(max_col_subGrid)+1] = grid_number_of_data_considered[int(min_row_subGrid):int(max_row_subGrid)+1,int(min_col_subGrid):int(max_col_subGrid)+1] + 1   
        
        for i in range(int(min_row_subGrid), int(max_row_subGrid)+1):
            for j in range(int(min_col_subGrid), int(max_col_subGrid)+1):
                # Check if a satellite tracks goes over pixel
                #f np.any(np.all(pixels_track-np.array([i,j])[:, np.newaxis] == 0, axis=0)):
                grid_data_considered[i][j].append(k)

    if verbose>0:
        print("sort duration: %.1f minutes " %((time.time()-start_time_sort)/60))
    
    # FILTER           
    # --------------------------------
    print('\t---> Spatial filtering\n')
    
    start_time_filter = time.time()

    # Distance between the grid points and the (x,y) data points within range_filter
    squared_range_filter = range_filter**2 # to avoid to compute the squared root when comparing the distances to the range filter

    # Allocate the array that will contain the filtered and gridded data
    #data_results=ma.masked_all((nb_pixels_1D,nb_pixels_1D))
    data_results = {}
    for p_name in all_data.keys():
        all_data[p_name] = np.array(all_data[p_name])
        data_results[p_name] = np.empty((nb_pixels_1D,nb_pixels_1D))
        data_results[p_name][:] = np.nan

    # Adding Rms for each params
    for p_name in all_data.keys():
        rms_p_name = p_name + '_rms'
        data_results[rms_p_name] = np.empty((nb_pixels_1D,nb_pixels_1D))
        data_results[rms_p_name][:] = np.nan

    # Loop on the grid 
    progress_prev = -1
    for i in range(nb_pixels_1D):
        if verbose>0:
            progress = np.floor(100*i/nb_pixels_1D)
            if(progress != progress_prev): print("progress: %.0f %%" %progress)
            progress_prev = progress

        for j in range(nb_pixels_1D):
            numbers_of_data_considered = grid_data_considered[i][j]
            
            # If there are no data available to filter on the current pixel i,j, filter the next pixel
            #if not numbers_of_data_considered: continue
            if len(numbers_of_data_considered)==0: continue

            # XXX empirique: no color unconsistent pixels (not enough tracks near)
            if grid_number_of_data_considered[i][j] < range_filter_in_nb_of_pixels*10: continue

            # Compute squared distance of all the points that have good chance of being in range of the pixel center
            squared_dist = (x_track[numbers_of_data_considered]-x_grid[j])**2 + (y_track[numbers_of_data_considered]-y_grid[i])**2
            dist_in_range_bool_array = squared_dist < squared_range_filter
            squared_dist = squared_dist[dist_in_range_bool_array]

            # Get the numbers of the data that can be used to compute the filtering
            numbers_of_data_used = np.array(numbers_of_data_considered)[dist_in_range_bool_array]
            
            # If there are no track points that are within the filter range, filter the next pixel
            if(numbers_of_data_used.size == 0): continue

            # Compute the spatial filter on the current pixel i,j
            #mode = 'gaussian_radius'
            #weighting = weight
            if mode=='filter_gauss':
                weighting = np.exp(-squared_dist/squared_range_filter)*weight_coef[numbers_of_data_used]
                
                for p_name,data in all_data.items():
                    data_results[p_name][i,j] = np.nansum(data[numbers_of_data_used]*weighting)/np.sum(weighting)
                    data_results[rms_p_name][i,j] = np.ma.sqrt(np.ma.sum(weighting*(data[numbers_of_data_used]-data_results[p_name][i,j])**2) / np.ma.sum(weighting) )
                    #data_results[rms_p_name][i,j] = np.std(data[numbers_of_data_used])
            
            """
            elif mode=='filter_median':
                
                for p_name,data in all_data.items():
                    data_results[p_name][i,j] = np.nanmedian(data[numbers_of_data_used]*weighting)/np.sum(weighting)
                    data_results[rms_p_name][i,j] =  np.std(data[numbers_of_data_used])

            elif mode=='filter_mean':
                for p_name,data in all_data.items():
                    data_results[p_name][i,j] = np.nanmean(data[numbers_of_data_used]*weighting)/np.sum(weighting)
                    data_results[rms_p_name][i,j] = np.std(data[numbers_of_data_used])
            """

        
    print("grid_and_filter_wrt_distance mode %s range %f pixel %f" % (mode, range_filter, pixel_size))
    if verbose>0:       
        print('filtering duration: %.1f minutes' %((time.time()-start_time_filter)/60))
        print('total duration of the grid_and_filter function: %.1f minutes' %((time.time()-start_time)/60))

    return x_grid, y_grid, lat_grid_mesh, lon_grid_mesh, x_grid_mesh, y_grid_mesh, data_results


def get_strong_beams(filename,is2Beams):

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



###########################################
#
#              Main
#
###########################################

if __name__ == '__main__':

    # Define programme description
    description ='Program to plot welch spectrale analysis'

    # Initiate the parser with a description
    parser = argparse.ArgumentParser(description=description)

    
    # Add long and short arguments
    parser.add_argument("-f","--inputfile",help="provide input data files")

    parser.add_argument("-o","--outputfile",default=None,help="provide input data files")

    parser.add_argument("-p","--parameter",default='list',help="provide parameter to be tested")

    parser.add_argument("-s","--satellite",help="provide input satellite name",required=True)

    parser.add_argument("-g","--gdr",help="provide input gdr")

    parser.add_argument("-b","--is2Beams",required=True,help="provide IS2 strongs beams to plot")

    parser.add_argument("-d","--date",help="provide input date (month)")

    parser.add_argument("-hp","--hemisphere",required=True,help="provide hemisphere code (N=01/S=02)")

    # Read arguments
    # ----------------------------------------------------------
    args = parser.parse_args()

    hemispherecode = args.hemisphere

    # 
    sat = args.satellite
    gdr = args.gdr
    date = args.date

    # 
    outfilename = args.outputfile

    #
    is2Beams = [b for b in args.is2Beams.split(',')]

    #
    if args.inputfile is None: inputfile=None
    else:
        inputfile = args.inputfile #[fname for fname in args.inputfile.split(',')]

    if (sat is None and gdr is None and date is None) and inputFileName is None:
        print("Provide either options -s; -g; -d or -f to locate files \n")
        sys.exit()       

   

    plist = [pname for pname in args.parameter.split(',')]

    # Open file
    # ---------------------------------------------------------
    filelist = list()

    if sat=='CS2':
        extension='.nc'
    elif sat=='IS2':
        extension='.h5'
    else:
        print("Add mission extension \n")
        sys.exit()
     
    if inputfile is not None:

        #for infile in inputfile:
        inputfilepattern = inputfile +"*"+ extension
        print("\nread file %s" %(inputfilepattern))
        filename = glob.glob(inputfilepattern)
        if len(filename)==0:
            sys.exit("\n%s: No found" %(inputfilepattern))
        elif len(filename)==1:
            filelist.append(filename[0])
        else:
            filelist.extend(filename)

    elif sat is not None and gdr is not None and date is not None:
        
        inputfilepattern = PATH_INPUT + "%s/%s/%s/*%s" %(sat,gdr,date,extension)
        print("\nread file %s" %(inputfilepattern))
        filename = glob.glob(inputfilepattern)
        if len(filename)==0: sys.exit("\n%s: No found" %(inputfilepattern))
        elif len(filename)==1: filelist.append(filename[0])
        else:
            filelist.extend(filename)

    else:
        print("Provide either options -s; -g; -d or -f to locate files \n")
        sys.exit()  
        

    print("file found:",filelist)
    
    lat_list = list()
    lon_list = list()
    data_track_list = dict()
    for pname in plist:
        data_track_list[pname] = list()

    # add segment length as weighting factor for gridding
    if sat=='IS2':
        weight = list()
    else:
        weight = None  

    
    # read data
    for filename in filelist:

        print("reading: %s" %(filename.split('/')[-1]))
        if sat=='CS2':
            flag_1hz = False
            data_desc_cs2 = cs2_dict.init_dict(gdr,flag_1hz)
            lat,lon,timeCS2,x_dist,selected_idx = cf.get_coord_from_netcdf(filename,data_desc_cs2,hemispherecode,LAT_BOUND)

            if lat is None: continue
            lat_list.append(lat)
            lon_list.append(lon)
            
            for pname in plist:
                param,units,param_is_flag = cf.get_param_from_netcdf(filename,data_desc_cs2,pname,hemispherecode,LAT_BOUND)

                if param is None:
                    lat_list.pop()
                    lon_list.pop()
                    break
                if param.size==0:
                    lat_list.pop()
                    lon_list.pop()
                    break
                data_track_list[pname].append(param)


        elif sat=='IS2':
            beamName = get_strong_beams(filename,is2Beams)
            
            for beam in beamName.keys():
                
                # data type
                # if SWATH data
                if 'sla' in plist:
                    flag_swath=True
                    data_desc_is2 = is2_dict.init_dict(gdr,beamName[beam],'swath')
                    weight = None  
                else:
                    flag_swath=False
                    data_desc_is2 = is2_dict.init_dict(gdr,beamName[beam],'granule')
                    Lseg,units,param_is_flag = cf.get_param_from_hf5(filename,data_desc_is2,'Lseg',hemispherecode,LAT_BOUND)
                    weight.append(Lseg)

                print(filename)
                lat,lon,timeIS2,x_dist,selected_idx = cf.get_coord_from_hf5(filename,data_desc_is2,hemispherecode,LAT_BOUND)
                if lat is None: continue
                
                lat_list.append(lat)
                lon_list.append(lon)

                
                for pname in plist:
                    param,units,param_is_flag = cf.get_param_from_hf5(filename,data_desc_is2,pname,hemispherecode,LAT_BOUND)
                    # make it for each beam XXX
                    data_track_list[pname].append(param)
        else:
            print("Unknown satname: %s" %(sat))

    
    # Creating data array
    data_array = dict()
    data2grid = dict()

    lat_array = np.ma.concatenate(lat_list,axis=0)
    lon_array = np.ma.concatenate(lon_list,axis=0)

    if sat=='IS2' and not flag_swath: weight = np.ma.concatenate(weight,axis=0)
    
    data_results = dict()
    lat_grid = dict()
    lon_grid = dict()
    
    for pname in plist:
    
        data_array[pname] = np.ma.concatenate(data_track_list[pname],axis=0)
        data_array[pname] = ma.masked_invalid(data_array[pname],copy=True)
        data2grid[pname] = data_array[pname].data
        data2grid[pname][data_array[pname].mask] = np.nan
        data2grid[pname][data_array[pname]==0.0] = np.nan

    if hemispherecode=='01':
        llcrnrlat=0; urcrnrlat=90
    else:
        llcrnrlat=-90; urcrnrlat=0
         
        
    # Gridding the data
    f2, ax = plt.subplots(1, 1,figsize=(9,8)) #'nplaea'
    m = Basemap(projection='nplaea', llcrnrlat=llcrnrlat,urcrnrlat=urcrnrlat,llcrnrlon=-180,urcrnrlon=180,boundinglat=LAT_BOUND,lon_0=0, resolution='l',round=True,ax=ax)
    x_grid, y_grid, lat_grid_mesh, lon_grid_mesh, x_grid_mesh, y_grid_mesh,data_grid = grid_and_filter_wrt_distance(lon_array, lat_array, data2grid, m, pixel_size=12500,mode='filter_gauss',range_filter=25000,weight=weight,verbose=0)
    
    
    if show_figure:
        data = np.ma.array(data_grid[plist[0]])
        bmap,cmap = st.plot_track_map(f2,ax,lon_grid_mesh,lat_grid_mesh, data,plist[0],[0,0.4],None,'m',False,alpha=1,size=3)
        plt.show()



    # Saving data in NETCDF
    #-----------------------------------------------
    
    if outfilename:
        pathout = '/'.join(outfilename.split('/')[:-1])+'/'
        fileoutname = outfilename.split('/')[-1]
        file_out = outfilename
        
    else:
        pathout = PATH_INPUT+ "grid/%s/%s/" %(sat,gdr)
        fileoutname = '%s_%s_%s.nc' %(sat,gdr,date)
        file_out = pathout + fileoutname

    

    if not os.path.exists(pathout):
        print("creating:",pathout)
        os.makedirs(pathout)
    
    
    size_grid = lon_grid_mesh.shape[0]
    
    # save into Netcdf
    dataset = Dataset(file_out, 'w',format='NETCDF4_CLASSIC')

    u = dataset.createDimension('u', size_grid)
    v = dataset.createDimension('v', size_grid)
    
    #LATITUDE                                                                                    
    latitude = dataset.createVariable('latitude', np.float32, ('u','v'))
    latitude[:]=lat_grid_mesh
    latitude.units = 'degree_north'
    latitude.long_name='latitude'
    latitude.description='ease grid'

    #LONGITUDE                                                                                   
    longitude = dataset.createVariable('longitude', np.float32, ('u','v'))
    longitude[:]=lon_grid_mesh
    longitude.units = 'degree_east'
    longitude.long_name='longitude'
    longitude.long_name='ease grid'

    outparams = {}
    for pname in data_grid.keys():
        
        outparams[pname] = dataset.createVariable(pname, np.float32, ('u','v'))
        outparams[pname][:,:]=data_grid[pname]
        outparams[pname].units = 'm'
        #snow_depth.long_name='Ka-Ku snow_depth'
        #snow_depth.description='snow_depth calculated from %s and %s' %(sat_ku,sat_ka)


    print("files %s saved in %s" %(fileoutname,pathout))
    dataset.close()




    

    


    
