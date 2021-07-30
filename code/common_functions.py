
#
# common_function.py
#
# ----------------------------------------------------------------------
# Copyright (c) 2020 LEGOS/SERCO
# All rights reserved.
#


"""
DESCRIPTION:

     Script of common functions

LIST OF FUNCTIONS:

     get_osisaf_ice_type: get osisaf ice type

"""

from ftplib import FTP
import os
import sys
import netCDF4 as nc
import glob
import numpy as np
from matplotlib import pyplot as plt
from numpy import ma
import h5py
from mpl_toolkits.basemap import Basemap
import scipy.spatial
from scipy.interpolate import interp1d
import datetime
#from pyhdf.SD import SD, SDC
import re
import pyproj
from scipy.stats import pearsonr, gaussian_kde,linregress
import requests
from calendar import monthrange
import path_dict

def get_osisaf_ice_type(year,month,day,hemispherecode):

    """
    Download or read OSISAF gridded netCDF data

    Args:
       mapProj (basemap) : map projection
       year (str)        : year
       month (str)       : month
       day (str)         : day

    """

    # converting date from int to str
    year_str = str(year)
    month_str='%02d' % month
    day_str = '%02d' % day
    hemisphere = 'nh' if hemispherecode=='01' else 'sh'
    
    # local attributs
    filepattern = 'ice_type_%s_polstere-100_multi_%s%s%s1200.nc' %(hemisphere,year_str,month_str,day_str)
    #http_address= 'https://thredds/fileServer/osisaf/met.no'
    #file_path = '/ice/type/%s/%s/' %(year_str,month_str)
    ftp_address = 'ftp.osisaf.met.no'
    #file_path = './archive/ice/type/%s/%s' %(year_str,month_str)
    file_path = './archive/ice/type/%s/%s' %(year_str,month_str)
    local_dir = path_dict.PATH_DICT['PATH_DATA']+'OSISAF/'

    # Test if file already exists in local repertory
    filename = glob.glob(local_dir+filepattern)
    if len(filename) == 0: 
        download_ftp_file(ftp_address,file_path,local_dir,filepattern)
        #download_http_file(http_address,file_path,local_dir,filepattern)
        filename = glob.glob(local_dir+filepattern)

    # read file
    f = nc.Dataset(filename[0])

    ice_type = np.squeeze(f.variables['ice_type'][:])
    lats = f.variables['lat'][:]
    lons = f.variables['lon'][:]

    print('Reading ' + f.variables['ice_type'].long_name +' from OSISAF')
    print(f.variables['ice_type'].flag_descriptions)

    return lons,lats,ice_type



def download_http_file(http_address,file_path,local_dir,filepattern):


    print('downloading %s from %s' %(filepattern,http_address))
    
    url = http_address+file_path+filepattern
    outfolder = local_dir+filepattern
    r = requests.get(url, allow_redirects=True)
    open(outfolder, 'wb').write(r.content)
    print('NetCDF stored in ' + local_dir)
    

def grid_to_track(data_grid,lon_grid,lat_grid,lon,lat):

    
    """
    retreive along_track data from grid

    Args:
      data_grid (nxm array) : data gridded
      lon_grid  (nxm array) : corresponding longitude
      lat_grid  (nxm array) : corresponding latitude
      lon       (p array)   : track lon
      lat       (p array)   : track lat

    """
    # Dist computed in cartesien diff with dist in lat,lon
    if np.any(lon_grid<0) and not np.any(lon<0):
        lon[lon>180] = lon[lon>180] - 360
    elif np.any(lon_grid>180) and not np.any(lon>180):
        lon[lon<0] = lon[lon<0] + 360
    else:
        lon = lon

    x,y,z = lon_lat_to_cartesian(lon, lat)
    points = list(np.vstack((x,y,z)).T)

    x_grid,y_grid,z_grid = lon_lat_to_cartesian(lon_grid.ravel(), lat_grid.ravel())
    coords_grid_flat = np.dstack([x_grid,y_grid,z_grid])[0]
    data_flat = data_grid.ravel()
    
    mytree = scipy.spatial.cKDTree(coords_grid_flat)
    dist, indexes = mytree.query(points)

    data_al = data_flat[indexes]

    if any(np.abs(np.diff(lon)) > 300) or any(lon < 0): lon[lon < 0] = lon[lon < 0] + 360
    
    return data_al

def get_sst_metoffice(date):

    """
    retreive a map of Arctic sea-ice temperature

    Args:
      date (str) : datetime
    
    """

    # Surface temperature
    path_data = path_dict.PATH_DICT['PATH_DATA'] +'SST/'
    filepattern =path_data +'*.nc'
    filename = glob.glob(filepattern)
    if len(filename)==0: sys.exit("\n%s: No found" %(filepattern))
    else:
        filename = filename[0]
        print("\nReading SST file %s" %(filename))

    # read file
    try:
        f = nc.Dataset(filename)
    except:
        sys.exit("Cannot open file %s" % filename)

    # get date
    time = ma.ravel(f.variables['time'][:])
    datelist = np.ma.array([datetime.timedelta(seconds=int(tm)) + datetime.datetime(1981,1,1) for tm in time ])
    dateliststr = np.array([dates.strftime("%d/%m/%Y") for dates in datelist])
    datestr = date.strftime("%d/%m/%Y")

    #datestr = date.strftime("%m/%Y")
    index = np.argwhere(datestr==dateliststr)
    if index.size==0:
        print('\n Date %s not found in available dates' %(datestr,dateliststr))
        sys.exit()
    else:
        index = int(index)

    # Get data
    lat = f.variables['lat'][:]
    lon = f.variables['lon'][:]
    sst = f.variables['analysed_sst'][index,:,:]

    return lat,lon,sst

    
    

    

    


def download_ftp_file(ftp_address,file_path,local_dir,filename):

    """
    Download ftp netCDF files from given ftp_address

    Args:
      ftp_address (str) : ftp address
      file_path (str)   : file path within ftp address
      local_dir (str)   : local dir to store data
      filename  (str)   : filename

    """

    ftp = FTP(ftp_address)
    ftp.login() # login anonymously
    ftp.cwd(file_path) # navigate to exact FTP folder where data is located

    files = ftp.nlst() # collect file names into vector
    
    # Creates a local GOES repository in the current directory
    if os.path.isdir(local_dir)==False:
        os.mkdir(local_dir)

    print('downloading %s from %s' %(filename,ftp_address))
    
    with open(local_dir+'/'+filename, 'wb') as localfile:
    #localfile = open(local_dir+'/'+filename, 'wb') # local download
        ftp.retrbinary('RETR ' + filename, localfile.write, 1024) # FTP saver

    print('NetCDF stored in ' + local_dir)

    ftp.quit()

    
def distance_from_first_trk_pts(latDeg,lonDeg,n):

    """
    Computes cumulated distance from first data point

    Args:
        lat (np.array) : along-track lat
        lon (np.array) : along-track lon
        n (int): start from idx n

    """
    lat1 = latDeg[n:-1]; lat2=latDeg[n+1:]
    lon1 = lonDeg[n:-1]; lon2=lonDeg[n+1:]
    dist_btw_coord = dist_btw_two_coords(lat1,lat2,lon1,lon2)
    #nanvalues = ~np.isnan(dist_btw_coord)
    #valid_idx = np.argwhere(nanvalues.reshape((nanvalues.size,)))
    #valid_idx = valid_idx.reshape((valid_idx.shape[0],))
    dist_from_first_trk_pt = np.nancumsum(dist_btw_coord) #[valid_idx]
    dist_from_first_trk_pt =  np.concatenate((np.zeros(1),dist_from_first_trk_pt))
    return dist_from_first_trk_pt #,valid_idx
    #return dist_from_first_trk_pt




def dist_btw_two_coords(latDeg1,latDeg2,lonDeg1,lonDeg2):
    """
    Computes distance between two coodinates

    Args:
       (latDeg1,lonDeg1) : Coordinates pts 1
       (latDeg2,lonDeg2) : Coordinates pts 2

    """
    
    dist = earthradius_fct_of_lat(latDeg1)*np.arccos( np.sin(deg_to_rad(latDeg1))*np.sin(deg_to_rad(latDeg2)) + np.cos(deg_to_rad(latDeg1))*np.cos(deg_to_rad(latDeg2))*np.cos(deg_to_rad(lonDeg2 - lonDeg1)) )
    
    return dist


def lon_lat_to_cartesian(lonDeg, latDeg):
    """
    calculates lon, lat coordinates of a point on a sphere with
    radius R
    """
    lon_r = np.radians(lonDeg)
    lat_r = np.radians(latDeg)
    R = earthradius_fct_of_lat(latDeg)

    x =  R * np.cos(lat_r) * np.cos(lon_r)
    y = R * np.cos(lat_r) * np.sin(lon_r)
    z = R * np.sin(lat_r)

    return x,y,z

def cartesian_to_polar(x,y,z):
    """
    
    """
    R = np.sqrt(x**2 + y**2 + z**2)
    lat = rad_to_deg(np.arctan(z/np.sqrt(x**2 + y**2)))
    lon = rad_to_deg(np.arccos(x/np.sqrt(x**2 + y**2)))
    return R,lat,lon


def earthradius_fct_of_lat(latDeg):
    """
    Computes earth radius at provided latitude
    """

    earth_radius_pole = 6356.752
    earth_radius_eq = 6378.137 #km
    lat = deg_to_rad(latDeg)
    earth_radius = np.sqrt( ( (earth_radius_eq**2 * np.cos(lat))**2 +  (earth_radius_pole**2 * np.sin(lat))**2 ) / ( (earth_radius_eq * np.cos(lat))**2 +  (earth_radius_pole * np.sin(lat))**2 ))

    return earth_radius


def deg_to_rad(angDeg):

    return angDeg*np.pi/180


def rad_to_deg(angRad):

    return angRad*180/np.pi


#def grid2track(mapProj,datagrid,lat,lon):


def statistics(p_name,param,units,is_flag):
    """
    Computes and print statistics

    Args:
    p_name  : param name (str)
    param   : param (np.array)
    units   : param units (str)
    is_flag : True if param is a flag (bool)
    
    """
    if param.mask.size>1:
        param[param.mask] = np.nan
    print('\n',p_name)
    print("-------------------------")
    if is_flag:
        flag_values = np.unique(param)
        ndata = np.sum(~np.isnan(param))
        stats_str = "Nd=%.i,\n" %(ndata)
        list_perc_flag = list()
        for n,fv in enumerate(flag_values):
            list_perc_flag.append(100*np.sum(param==fv)/ndata)
            stats_str += "prop[%s]=%.1f%%\n" %(fv,list_perc_flag[n])
    else:
        mean = np.nanmean(param)
        std =  np.nanstd(param)
        ndata = np.sum(~np.isnan(param))
        pmin = np.nanmin(param)
        pmax = np.nanmax(param)
        stats_str = '\n'.join((
            'Nd=%i' % (ndata,),
            'Mean=%.2f %s' %(mean,units),
            'Std=%.2f %s' %(std,units),
            'Max=%.2f %s' %(pmax,units),
            'Min=%.2f %s' %(pmin,units)))
            #r'$\mu=%.2f$' % (mean, ),
            #r'$\sigma=%.2f$' % (std, )))
        #stats_str = "ndata=%.i,\n mean=%.2f,\n std=%.2f,\n" %(mean,std,ndata)
    print(stats_str)
    
    return stats_str


def find_smoothing_radius(x_data,y_data,mean_dist_btw_data,max_smooth,flag_plot):

    R_list = list()
    RMSD_list = list()
    smooth_radius = np.arange(0,max_smooth,5)

    #x_data = ma.masked_where(np.isnan(x_data),x_data,copy=True)
    #y_data = ma.masked_where(np.isnan(y_data),y_data,copy=True)
    mask_data = np.logical_and(~x_data.mask,~y_data.mask)
     
    for nsmooth in np.arange(0,max_smooth,5):

        x = ma.masked_where(~mask_data,x_data,copy=True)
        x = x[mask_data]

        # smoothing y
        y = rolling_stats(y_data,int(nsmooth/mean_dist_btw_data), stats=['mean'])[0]
        y =  ma.masked_where(~mask_data,y,copy=True)
        y = y[mask_data]

        
        #plt.plot(y_data[mask_data])
        #plt.plot(x)
        #plt.plot(y)
        #plt.show()
        
        nb_data = np.sum(mask_data)
        R_list.append(pearsonr(x,y)[0])
        RMSD_list.append(np.sqrt((1/(nb_data-1))*np.sum((y - x)**2))) #*100

    #idx_min_corr = np.argmax(np.array(R_list))
    idx_min_corr = np.argmax(np.diff(np.array(R_list)) < 0.001)
    smoothmin = smooth_radius[idx_min_corr]

    if flag_plot:
        ytext = np.mean(np.array(RMSD_list))
        plt.plot(smooth_radius,R_list,label='pearson')
        plt.plot(smooth_radius,RMSD_list,label='RMSD')
        textstr = '\n'.join(('Ndata= %i' % (nb_data),'Rsmooth= %i km' % (smoothmin)))
        props = dict(boxstyle='round', facecolor='white', alpha=0.6)
        plt.text(0,0.05, textstr, fontsize=12,verticalalignment='bottom',horizontalalignment ='left', bbox=props)
        #plt.axvline(x=smoothmin)
        plt.xlabel("Smoothing radius [km]")
        plt.ylabel("RMSD [m]/Pearson")
        plt.grid()
        plt.legend()
        #plt.show()
        

    return R_list,RMSD_list,smoothmin


def allDaysMonth(y, m):
    return [datetime.datetime(y,m,d) for d in range(1, monthrange(y, m)[1] + 1)]


# Rolling statistics for masked data
#
def rolling_stats(data, window_size, stats=['median']):
    """
    Get the statistics parameters (mean, std etc) for each data point using window_size points around the data point
    Use broadcasting properties of Python as well as the ma.median, ma.mean etc functions to vectorize the code
    - Inputs: 
          data: 1D numpy masked array
          stats radius: in number of pts
          stats: which stat parameters to compute -> can contain ['mean', 'median', 'std', 'min', 'max']
    - Output:
          list of rolling_stats: list of 1D numpy masked array with the same mask and the same size as data
    Drawback of this function: needs to have sufficient RAM to save data.size*window_size elements (probably floats) 
    """
 
    if(window_size%2==0): window_size+=1 # be sure to have an odd number for the window_size
    pts_to_filter_idx = np.arange((window_size-1)/2,(window_size-1)/2+data.size)
    # Add masked values at the edges to filter the pts near the edge the same way than the others
    data = ma.hstack((ma.array(np.zeros(int((window_size-1)/2)), mask=True),
                      data,
                      ma.array(np.zeros(int((window_size-1)/2)), mask=True)))
    # Build an array of shape (data,window_size): for each data point, contains the subscripts of the data that will be used to compute the median
    pts_to_filter_idx_arr = (np.ones((pts_to_filter_idx.size, window_size), dtype=int)*
                             np.arange(-(window_size-1)/2,(window_size+1)/2, dtype=int))
    pts_to_filter_idx_arr = (pts_to_filter_idx_arr + pts_to_filter_idx[:,np.newaxis]).astype(int)
    pts_to_filter_idx_arr = np.ravel(pts_to_filter_idx_arr)
    # Array of shape (data,window_size): for each data, contains the neighbours data used to compute the median
    data_to_filter_arr = data[pts_to_filter_idx_arr]
    data_to_filter_arr = data_to_filter_arr.reshape(-1,window_size)
    # Compute the rolling statistics
    rolling_stats_list = []
    if 'median' in stats: 
        rolling_median = ma.median(data_to_filter_arr, axis=1)
        rolling_stats_list.append(rolling_median)
    if 'mean' in stats: 
        rolling_mean = ma.mean(data_to_filter_arr, axis=1)
        rolling_stats_list.append(rolling_mean)
    if 'std' in stats: 
        rolling_std = ma.std(data_to_filter_arr, axis=1)
        rolling_stats_list.append(rolling_std)
    if 'min' in stats: 
        rolling_min = np.ma.min(data_to_filter_arr, axis=1)
        rolling_stats_list.append(rolling_min)
    if 'max' in stats: 
        rolling_max = ma.max(data_to_filter_arr, axis=1)
        rolling_stats_list.append(rolling_max)
    if 'numb' in stats:  #only bool masked array
        rolling_numb = ma.sum(data_to_filter_arr, axis=1) # count true
        rolling_stats_list.append(rolling_numb)
    if 'nb_masked' in stats: # only for masked array
        rolling_nbmask = ma.count_masked(data_to_filter_arr, axis=1) # count true
        rolling_stats_list.append(rolling_nbmask)
    return rolling_stats_list


def get_coord_from_uob(filename,data_desc,hemispherecode,LAT_BOUND):


    if bool(data_desc) is False:
        return None,None,None,None,None
    
    #init lists
    lon = list()
    lat = list()
    year = list()
    month = list()
    day = list()
    hour = list()
    minute = list()
    second = list()
    
    #for p in pnames:
    #    if p not in data_desc.keys():
    #        print("%s not found in dictionnary:" %(p),[key for key in data_desc.keys()])
    #        sys.exit()

    with open(filename, "r") as f:
        lines = f.readlines()
        for nline,line in enumerate(lines):
            if nline==0:
                param_list=line.split(',')
            else:
                if line.split(',')[0]=='': continue
                year.append(int(line.split(',')[1]))
                month.append(int(line.split(',')[2]))
                day.append(int(line.split(',')[3]))
                hour.append(int(line.split(',')[4]))
                minute.append(int(line.split(',')[5]))
                second.append(float(line.split(',')[6]))
                lon.append(float(line.split(',')[8]))
                lat.append(float(line.split(',')[7]))

    time = list()
    for n in range(len(year)):

        milliseconds = int((second[n] - int(second[n]))*1000)
        t = datetime.datetime(year[n], month[n], day[n], hour[n], minute[n],int(second[n]),milliseconds)
        time_sec = (t-datetime.datetime(2000,1,1)).total_seconds()
        time.append(time_sec)

    # converting into arrays
    time = np.array(time)
    lat = np.array(lat)
    lon = np.array(lon)

    # Select coord above desired region
    if hemispherecode=='01':
        select_zone, = np.where(lat > LAT_BOUND)
    else:
        select_zone, = np.where(lat < -LAT_BOUND)
         
    
    if select_zone.size==0:
        print("\n\nNo data over %i N for file: %s\n" %(LAT_BOUND,filename))
        return None,None,None,None,None

    lat =lat[select_zone]
    lon =lon[select_zone]
    if any(np.abs(np.diff(lon)) > 300) or any(lon < 0): lon[lon < 0] = lon[lon < 0] + 360
    time= time[select_zone]

    
    
    # Calculate distance from first point
    x_dist = distance_from_first_trk_pts(lat,lon,0)
    selected_idx = select_zone #[valid_idx]

    return lat,lon,time,x_dist,selected_idx



def get_coord_from_cpom(filename,data_desc,hemispherecode,LAT_BOUND):
    """
    Extract coord (lat,lon) from .dat file

    Args:
    filename   : filename (str)
    
    """

    if bool(data_desc) is False:
        return None,None,None,None,None
    
    
    data = np.loadtxt(filename)
    time = data[:,data_desc['time']]
    lat = data[:,data_desc['lat']]
    lon = data[:,data_desc['lon']]
    
    if lat is None or lon is None or time is None:
        print("%s not found in dictionnary:" %(pname),[key for key in data_desc.keys()])
        sys.exit()
        
    # Select coord above desired region
    if hemispherecode=='01':
        select_zone, = np.where(lat > LAT_BOUND)
    else:
        select_zone, = np.where(lat < -LAT_BOUND)

    lat =lat[select_zone]
    lon =lon[select_zone]
    if any(np.abs(np.diff(lon)) > 300) or any(lon < 0): lon[lon < 0] = lon[lon < 0] + 360
    time= time[select_zone]

    x_dist = distance_from_first_trk_pts(lat,lon,0)
    #lat = lat[valid_idx]; lon = lon[valid_idx]; time=time[valid_idx]

    selected_idx = select_zone #[valid_idx]

    return lat,lon,time,x_dist,selected_idx


#b common_functions.py:353
def get_param_from_cpom(filename,data_desc,pname,hemispherecode,LAT_BOUND):
    """
    Extract param from .dat file

    Args:
    p_name  : param name (str)
    file   : file name (str)
    
    """
    data = np.loadtxt(filename)
    lat = data[:,data_desc['lat']]

    # Select coord above desired region
    if hemispherecode=='01':
        select_zone, = np.where(lat > LAT_BOUND)
    else:
        select_zone, = np.where(lat < -LAT_BOUND)

    param = data[:,data_desc[pname]]
    param = param[select_zone]

    if 'lon' in pname:
        if any(np.abs(np.diff(param)) > 300): param[param < 0] = param[param < 0] + 360

    units = ''

    param_is_flag=False
    unique, counts = np.unique(param, return_counts=True)
    if counts.size<6: param_is_flag=True
    
    if param_is_flag: units = ''
    else: units = 'm'
    
    return param,units,param_is_flag
    
    
    

def get_param_from_uob(filename,data_desc,pname,hemispherecode,LAT_BOUND):

    
    lat = list()
    param = list()
    
    if pname not in data_desc.keys():
        print("%s not found in dictionnary:" %(pname),[key for key in data_desc.keys()])
        sys.exit()

    with open(filename, "r") as f:
        lines = f.readlines()
        for nline,line in enumerate(lines):
            if nline==0:
                param_list = [p.lstrip().rstrip("\n") for p in line.split(',')]
                idx_p = param_list.index(data_desc[pname])
            else:
                if line.split(',')[0]=='': continue
                lat.append(float(line.split(',')[7]))
                param.append(float(line.split(',')[idx_p]))
    
    # converting into arrays
    lat = ma.masked_invalid(np.ma.array(lat), copy=True)
    param = ma.masked_invalid(np.ma.array(param), copy=True)

    # if data does not exist
    if lat is None:
        print("\n%s:%s not found in %s" %(pname,data_desc[pname],filename))
        return None,None,None

    # Select coord above desired region
    if hemispherecode=='01':
        select_zone, = np.where(lat > LAT_BOUND)
    else:
        select_zone, = np.where(lat < -LAT_BOUND)

    if select_zone.size==0:
        print("\n\nNo data over %i N for file: %s\n" %(LAT_BOUND,filename))
        return None,None,None
    
    param = param[select_zone]

    if param is None: print("\nNo param %s:%s in %s" %(pname,data_desc[pname],filename))
    
    units=''
    
    if np.all(param.mask):
        print("No data in %s" %(pname))
        return param,units,False

    if 'lon' in pname:
        if any(np.abs(np.diff(param)) > 300): param[param < 0] = param[param < 0] + 360
    
    param_is_flag=False
    unique, counts = np.unique(param, return_counts=True)
    if counts.size<6: param_is_flag=True
            
    return param,units,param_is_flag




def get_coord_from_netcdf(filename,data_desc,hemispherecode,LAT_BOUND):
    """
    Extract coord (lat,lon) from NetCDF file

    Args:
    filename   : filename (str)
    
    """

    if bool(data_desc) is False:
        return None,None,None,None,None
    
    try:
        f = nc.Dataset(filename)
    except:
        sys.exit("Cannot open file %s" % filename)

    pnames = ['lat','lon','time']
    for p in pnames:
        if p not in data_desc.keys():
            print("%s not found in dictionnary:" %(p),[key for key in data_desc.keys()])
            sys.exit()
   
    # Read params
    lat = ma.ravel(f.variables[data_desc[pnames[0]]][:])
    lon = ma.ravel(f.variables[data_desc[pnames[1]]][:])
    time =  ma.ravel(f.variables[data_desc[pnames[2]]][:])

    if lat is None or lon is None or time is None:
        sys.exit("\nMissing \n%s,\n%s,\n%s in \n%s" %(data_desc[pnames[0]],data_desc[pnames[1]],data_desc[pnames[2]],filename))
    
    # Select coord above desired region
    if hemispherecode=='01':
        select_zone, = np.where(lat > LAT_BOUND)
    else:
        select_zone, = np.where(lat < -LAT_BOUND)
         
    
    if select_zone.size==0:
        print("\n\nNo data over %i N for file: %s\n" %(LAT_BOUND,filename))
        return None,None,None,None,None

    lat =lat[select_zone]
    lon =lon[select_zone]
    if any(np.abs(np.diff(lon)) > 300) or any(lon < 0): lon[lon < 0] = lon[lon < 0] + 360
    time= time[select_zone]
    # Calculate distance from first point
    #x_dist,valid_idx = distance_from_first_trk_pts(lat,lon)
    x_dist = distance_from_first_trk_pts(lat,lon,0)
    #lat = lat[valid_idx]; lon = lon[valid_idx]; time=time[valid_idx]

    selected_idx = select_zone #[valid_idx]

    return lat,lon,time,x_dist,selected_idx


def get_param_from_netcdf(filename,data_desc,p_name,hemispherecode,LAT_BOUND):
    """
    Extract param from NetCDF file

    Args:
    p_name  : param name (str)
    file   : file name (str)
    
    """
    try:
        f = nc.Dataset(filename)
    except:
        sys.exit("Cannot open file %s" % filename)

    # check if 1Hz data
    if '01' in p_name:
        latname='lat01'
    else:
        latname='lat'
    # get lattitude
    lat = ma.ravel(f.variables[data_desc[latname][:]])

    # if data does not exist
    if f.variables[data_desc[p_name]] is None or lat is None:
        print("\n%s:%s not found in %s" %(p_name,data_desc[p_name],filename))
        return None,None,None

    # Select coord above desired region
    if hemispherecode=='01':
        select_zone, = np.where(lat > LAT_BOUND)
    else:
        select_zone, = np.where(lat < -LAT_BOUND)
    # Select coord above desired region
    #select_north, = np.where(lat > LAT_MIN)
    
    param = ma.ravel(f.variables[data_desc[p_name]][:])

    # Applying quality flag for baseline-D
    if 'ESA_BD' in filename.split('/')[-3]:
        if p_name=='radar_fb':
            quality_flag = ma.ravel(f.variables[data_desc['quality_flag']][:])
            flag = get_valid_freeboard_flag(quality_flag).astype(bool)
            param = ma.masked_where(~flag,param,copy=True)
        elif p_name=='sla':
            quality_flag = ma.ravel(f.variables[data_desc['quality_flag']][:])
            flag = get_valid_sla_flag(quality_flag).astype(bool)
            param = ma.masked_where(~flag,param,copy=True)
        else:
            param = ma.masked_where(param==0.0,param,copy=True)
    
    
    param = param[select_zone]

    if param is None: print("\nNo param %s:%s in %s" %(p_name,data_desc[p_name],filename))
    
    if p_name not in ['coherence','ph_diff','wvf']:
        param = ma.ravel(param)
    if len(param.shape)==3:
        param = param.reshape((param.shape[0]*param.shape[1],param.shape[2]))

    # Get units
    if 'units' in f.variables[data_desc[p_name]].ncattrs():
        units =  f.variables[data_desc[p_name]].units
    else:
        units = ''

    if param.size - np.sum(param.mask) <= 3 or np.sum(param.mask)==param.size:
        print("No data in %s" %(p_name))
        return None,None,False

    if 'lon' in p_name:
        if any(np.abs(np.diff(param)) > 300): param[param < 0] = param[param < 0] + 360
        #if any(np.abs(np.diff(param)) > 20): param[param < 0] = param[param < 0] + 360
    
    param_is_flag=False 
    if 'flag_values' in f.variables[data_desc[p_name]].ncattrs(): param_is_flag=True
            
    return param,units,param_is_flag


def get_coord_from_hf5(filename,data_desc,hemispherecode,LAT_BOUND):
    """
    Extract coord from hdf5 file

    Args:
    filename    : file name (str)
    """
    if bool(data_desc) is False:
        return None,None,None,None,None
    
    
    # open file
    try:
        f=h5py.File(filename,'r')
    except:
        #return None
        print("Cannot open file %s" %filename)
        return None,None,None,None,None
        #sys.exit("Cannot open file %s" %filename)

    # Test if data exists
    if f.get(data_desc['lat']) is None or f.get(data_desc['lon']) is None or f.get(data_desc['time']) is None:
        sys.exit("\nMissing \n%s,\n%s,\n%s \nin \n%s" %(data_desc['lat'],data_desc['lon'],data_desc['time'],filename))

    # Read params from beams
    lat = np.array(f.get(data_desc['lat']))
    lon = np.array(f.get(data_desc['lon']))
    if any(np.abs(np.diff(lon)) > 300) or any(lon < 0): lon[lon < 0] = lon[lon < 0] + 360
    time =  ma.ravel(f.get(data_desc['time']))

    # Select coord above desired region
    if hemispherecode=='01':
        select_zone, = np.where(lat > LAT_BOUND)
    else:
        select_zone, = np.where(lat < -LAT_BOUND)

    if np.sum(select_zone)==0:
        print("\n\nNo data over %i N for file: %s\n" %(LAT_BOUND,filename))
        return None,None,None,None,None
    
    # Select coord above desired region
    lat =lat[select_zone]
    lon =lon[select_zone]
    time= time[select_zone]
    
    #x_dist,valid_idx = distance_from_first_trk_pts(lat,lon)
    x_dist = distance_from_first_trk_pts(lat,lon,0)
    #lat = lat[valid_idx]; lon = lon[valid_idx]; time = time[valid_idx]
    if any(np.abs(np.diff(lon)) > 300) or any(lon < 0): lon[lon < 0] = lon[lon < 0] + 360
    selected_idx = select_zone#[valid_idx]

    return lat,lon,time,x_dist,selected_idx

    

def get_param_from_hf5(filename,data_desc,p_name,hemispherecode,LAT_BOUND):
    """
    Extract param from hdf5 file

    Args:
    p_name    : param name (str)
    filename    : file name (str)
    data_desc : data name dict (dict)
    """
    
    # open file
    try:
        f=h5py.File(filename,'r')
    except:
        return None,None,None
        sys.exit("Cannot open file %s" % filename)

    # get lattitude
    lat = np.array(f.get(data_desc['lat']))

    # if data does not exist
    if f.get(data_desc[p_name]) is None or lat is None:
        print("\n%s:%s not found in %s" %(p_name,data_desc[p_name],filename))
        return None,None,None

    # Select coord above desired region
    if hemispherecode=='01':
        select_zone, = np.where(lat > LAT_BOUND)
    else:
        select_zone, = np.where(lat < -LAT_BOUND)

    if select_zone.size==0:
        print("\n\nNo data over %i N for file: %s\n" %(LAT_BOUND,filename))
        return None,None,None
        
    param = np.array(f.get(data_desc[p_name]))
    param = param[select_zone]
    units = f.get(data_desc[p_name]).attrs['units'].decode('UTF-8')

    if '_FillValue' in f.get(data_desc[p_name]).attrs.keys():  
        fill_value = f.get(data_desc[p_name]).attrs['_FillValue']
        param[param==fill_value] = np.nan

    param_is_flag=False 
    if 'flag_values' in f.get(data_desc[p_name]).attrs.keys(): param_is_flag=True

    return param,units,param_is_flag


def plot_tracks_map(coord_list,name_list):

    colors_track = ['seagreen','cornflowerblue','blue','black']

    # define map
    fig1 = plt.figure(1,figsize=(6,6))
    fig1.suptitle('Satellite tracks')

    m = Basemap(projection='npstere',boundinglat=60,lon_0=0, resolution='l' , round=False)
    m.drawcoastlines(linewidth=0.25, zorder=5)
    m.drawparallels(np.arange(90,-90,-5), linewidth = 0.25, zorder=10)
    m.drawmeridians(np.arange(-180.,180.,30.), latmax=85, linewidth = 0.25, zorder=10)
    m.fillcontinents(color='0.9',lake_color='grey', zorder=3)

    for ite,(coord,name) in enumerate(zip(coord_list,name_list)):

        lon = coord[:,0]
        lat = coord[:,1]
        xpts, ypts= m(lon,lat)
        trk = m.plot(xpts, ypts,color=colors_track[ite],label=name,marker='.')
        trk[0].axes.annotate(name,xy=(xpts[0], ypts[0]),size=10 )
        start_ind1 = np.argmin(np.absolute(xpts - xpts.mean()))
        end_ind1 = start_ind1 + 1
        plt.legend()
    
        trk[0].axes.annotate('',xytext=(xpts[start_ind1], ypts[start_ind1]),xy=(xpts[end_ind1], ypts[end_ind1]),arrowprops=dict(arrowstyle="->", color='seagreen'),size=25 )

    plt.show()



def off_nadir_range_corr(offnadir_dist,altitude,latDeg):

    """
    Calculate off-nadir range correction to apply if considering a scatter located at an offnadir location on the radar footprint
    
    Args:
    offnadir_dist : off-nadir distance (m)
    altitude      : altitude  (m)
    latDeg        : lattitude in degrees (dict)
    """

    earth_curve_coeff = 1 + (altitude/(1000*earthradius_fct_of_lat(latDeg)))
    ofrc = earth_curve_coeff*(offnadir_dist**2)/(2*altitude)
    return ofrc



def off_nadir_distance(ofrc,altitude,latDeg):

    """
    Calculate off-nadir distance of a scatter from nadir over earth flat surface when a off-nadir range correction is considered
    
    Args:
    ofrc          : off-nadir range correction (m)
    altitude      : altitude  (m)
    latDeg        : lattitude in degrees (dict)
    """

    earth_curve_coeff = 1 + (altitude/(1000*earthradius_fct_of_lat(latDeg)))
    offnadir_dist = np.sqrt(2*altitude*ofrc/earth_curve_coeff)
    return offnadir_dist


def interp_coord_1hz_to_20hz(lon01,lat01,time01,time20):

    """
    Interpolatate 1Hz coordinates (lon01,lat01) into 20hwz coordinates based on time 20hz
    
    """

    time01 = (time01 - time01[0])/(time01[-1]-time01[0])   
    time20 = (time20 -time20[0]) /(time20[-1]-time20[0])

    # set longitude continuous
    if any(np.abs(np.diff(lon01)) > 20): lon01[lon01 > 180] = lon01[lon01 > 180] - 360        
    points =  np.array([lat01.tolist(),lon01.tolist()]).T
    interpolator =  interp1d(time01, points, kind='quadratic', axis=0,fill_value="extrapolate")
    interp_pts = interpolator(time20.data)
    lat20 = interp_pts[:,0]
    lon20 = interp_pts[:,1]

    # reset longitude into original spatial reference [0,360] 
    if any(np.abs(np.diff(lon20)) > 300) or any(lon20 < 0): lon20[lon20 < 0] = lon20[lon20 < 0] + 360
    return lat20,lon20

def interp_1hz_to_20hz(param01,time01,time20):

    """
    Interpolatate 1Hz coordinates param01 into 20hz sampling based on time 20hz
    
    """

    time01 = (time01 - time01[0])/(time01[-1]-time01[0])   
    time20 = (time20 -time20[0]) /(time20[-1]-time20[0])

    maskdata = param01.mask
    if maskdata.size>1:
        f =  interp1d(time01, maskdata, kind='linear', axis=0,fill_value="extrapolate")
        maskdata20 = f(time20.data)

        # eliminate masked values
        param01 = param01[~maskdata]
        time01 = time01[~maskdata]
        
    # set longitude continuous
    interpolator =  interp1d(time01, param01, kind='quadratic', axis=0,fill_value="extrapolate")
    param20 = interpolator(time20.data)

    if maskdata.size>1:
        param20 = ma.masked_where(maskdata20.astype(bool),param20,copy=True)

    
    
    return param20




def plot_scatter(ax,xylim,x_data,x_label,y_data,y_label,icetype):

    # Scatter    
    data = np.hstack((x_data.reshape(x_data.size,1),y_data.reshape(y_data.size,1)))
    #z = gaussian_kde(np.transpose(data))(np.transpose(data))
    if icetype is None:
        ax.scatter(x_data,y_data, s=50, edgecolor='',marker='.',cmap='jet')
    else:
        ax.scatter(x_data[icetype==4],y_data[icetype==4], s=50, edgecolor='',marker='^',cmap='jet',label='MYI')
        ax.scatter(x_data[icetype==2],y_data[icetype==2], s=50, edgecolor='',marker='s',cmap='jet',label='FYI')

    ax.plot([-1, 1], [-1, 1], color = 'black', linestyle = 'dashed')
    ax.grid()
    axes = plt.gca()
    props = dict(boxstyle='round', facecolor='white', alpha=0.8)
    #color_map  =  plt.cm.jet
    ax.set_xlim((xylim[0],xylim[1]))
    ax.set_ylim((xylim[0],xylim[1]))

    ax.set_ylabel(y_label, fontsize=12)
    ax.set_xlabel(x_label, fontsize=12)
    ax.legend()
    #ax.rcParams.update({'font.size': 20})

    # statistics
    x = ma.masked_where(np.isnan(x_data),x_data,copy=True)
    y = ma.masked_where(np.isnan(y_data),y_data,copy=True)
    mask_data = np.logical_and(~x.mask,~y.mask)
    nb_data = np.sum(mask_data)
    mean_x = np.nanmean(x)
    mean_y = np.ma.mean(y)
    mean_bias = np.ma.mean(y-x)
    R = pearsonr(x_data[mask_data],y_data[mask_data])[0]
    RMSD = np.sqrt((1/(nb_data-1))*np.sum((y - x)**2))*100

    textstr = '\n'.join(('Bias= %.2f cm' % (mean_bias),'RMSD= %.2f cm' % (RMSD),'R= %.2f' % (R),'Npts = %i' %(nb_data)))
    props = dict(boxstyle='round', facecolor='white', alpha=0.8)
    ax.text(xylim[1]-0.22,xylim[0]+0.02 , textstr, fontsize=12,verticalalignment='bottom',horizontalalignment ='left', bbox=props) 
    
    ax.set_aspect('equal', adjustable='box')

    #plt.show()


def plot_histo(ax,xylim,units,xlabel,legend_list,data_list,commun_mask=None):

    #nite = 0
    """
    if flag_commun_data:
        print("Histo show only common pts")
        common_mask = ~ma.masked_where(np.isnan(data_list[0]),data_list[0],copy=True).mask
        for data in data_list:
            data = ma.masked_where(np.isnan(data),data,copy=True)
            common_mask = np.logical_and(common_mask,~data.mask)
    else:
        common_mask = np.ones((data_list[0].size,))"""
        
    if np.any(commun_mask):
        print("Histo show only common pts")
        for nd,data in enumerate(data_list):
            data = ma.masked_where(~commun_mask,data,copy=True)
            data_list[nd] = data
    
    for n,(data,label) in enumerate(zip(data_list,legend_list)):
        ax.hist(data[~data.mask], 25, range=[xylim[0],xylim[1]], histtype='step',alpha=0.4,fill=True,label=label,color=colors_histo[n])
        #data = ma.masked_where(np.isnan(data),data,copy=True) #[common_mask]
        print("Npts(%s) = %i" %(label,np.sum(~data.mask)))
        mean_data = np.ma.mean(data)
        #print(mean_data)
        ax.axvline(x=mean_data,color=color_line_histo[n])
        ypos = xylim[1] - xylim[0]
        ax.annotate('%.3f%s' %(mean_data,units),xy=(mean_data, ypos))
        
    
    ax.set_xlabel(xlabel,fontsize=12)
    ax.set_ylabel('density',fontsize=12)
    ax.legend()


def get_emissivity_SSMIS(date):

    datestr = date.strftime('%Y%m%d')
    path_data = path_dict.PATH_DICT['PATH_DATA']
    filepattern =path_data +'SSMIS/*_%s*.nc' %(datestr)
    filename = glob.glob(filepattern)
    if len(filename)==0: sys.exit("\n%s: No found" %(filepattern))
    else:
        filename = filename[0]
        print("\nReading SSMIS file %s" %(filename))
        
    f = nc.Dataset(filename)
    lat = f.variables['lat'][:]
    lon = f.variables['lon'][:]
    #teff= f.variables['teff'][:]
    e_SSMIS= f.variables['ev'][:] #surface emissivity at 50GHz ev (SSMIS)
    #e_AMSU= f.variables['ev'][:] # surface emissivity at 50GHz e (AMSU)
    
    return lat,lon,e_SSMIS


def get_SD_AMSR(date):

    datestr = date.strftime('%Y%m%d')
    path_data = path_dict.PATH_DICT['PATH_DATA']+'AMSR/'
    filepattern =path_data +'*_%s.he5' %(datestr)
    filename = glob.glob(filepattern)
    if len(filename)==0: sys.exit("\n%s: No found" %(filepattern))
    else:
        filename = filename[0]
        print("\nReading AMSR file %s" %(filename))
        
    import h5py
    # open file
    try:
        f=h5py.File(filename,'r')
    except:
        sys.exit("Cannot open file %s" % filename)

    # get data
    groupname = 'HDFEOS/GRIDS/NpPolarGrid12km/'
    lat = np.array(f.get(groupname+'lat'))
    lon = np.array(f.get(groupname+'lon'))
    SD = np.ma.array(f.get(groupname+'Data Fields/SI_12km_NH_SNOWDEPTH_5DAY'))
    #comment = "110 -- missing/not calculated, 120 -- Land, 130 -- Open water, 140 -- multiyear sea ice, 150 -- variability in snow depth, 160 -- snow melt";
    SD = ma.masked_where(SD==110,SD,copy=True)
    SD = ma.masked_where(SD==120,SD,copy=True)
    SD = ma.masked_where(SD==130,SD,copy=True)
    SD = ma.masked_where(SD==140,SD,copy=True)
    SD = ma.masked_where(SD==150,SD,copy=True)
    SD = ma.masked_where(SD==160,SD,copy=True)

    return lat,lon,SD


"""
def get_MODIS_IST(date,LAT_MIN):

    DATAFIELD_NAME = 'Ice_Surface_Temperature_NP'
    scale_factor = 0.01

    # Get file
    year = date.year
    day_of_year = date.timetuple().tm_yday
    path_data = '/home/antlafe/Documents/work/data/MODIS/'
    filepattern = path_data +'MYD29E1D.A%s%s*.hdf' %(year,day_of_year)
    filename = glob.glob(filepattern)
    if len(filename)==0: sys.exit("\n%s: No found" %(filepattern))
    else:
        filename = filename[0]
        print("\nReading MODIS IST file %s" %(filename))

   
    hdf = SD(filename, SDC.READ)

    # Read dataset
    data2D = hdf.select(DATAFIELD_NAME)
    data = data2D[:,:].astype(np.float64)*scale_factor

    # Read global attribute.
    fattrs = hdf.attributes(full=1)
    ga = fattrs["StructMetadata.0"]
    gridmeta = ga[0]

    ul_regex = re.compile(r'''UpperLeftPointMtrs=\(
                          (?P<upper_left_x>[+-]?\d+\.\d+)
                          ,
                          (?P<upper_left_y>[+-]?\d+\.\d+)
                          \)''', re.VERBOSE)
    match = ul_regex.search(gridmeta)
    x0 = np.float(match.group('upper_left_x'))
    y0 = np.float(match.group('upper_left_y'))

    lr_regex = re.compile(r'''LowerRightMtrs=\(
                          (?P<lower_right_x>[+-]?\d+\.\d+)
                          ,
                          (?P<lower_right_y>[+-]?\d+\.\d+)
                          \)''', re.VERBOSE)
    match = lr_regex.search(gridmeta)
    x1 = np.float(match.group('lower_right_x'))
    y1 = np.float(match.group('lower_right_y'))

    ny, nx = data.shape
    x = np.linspace(x0, x1, nx)
    y = np.linspace(y0, y1, ny)
    xv, yv = np.meshgrid(x, y)

    # Reproject the coordinates out of lamaz into lat/lon.
    lamaz = pyproj.Proj("+proj=laea +a=6371228 +lat_0=90 +lon_0=0 +units=m")
    wgs84 = pyproj.Proj("+init=EPSG:4326") 
    lon, lat= pyproj.transform(lamaz, wgs84, xv, yv)

   
    
    idx_mid = int(lat.shape[0]/2)+1
    idx_min = np.argmax(lat[idx_mid,:]>LAT_MIN)
    idx_dist = idx_mid - idx_min
    
    lat = lat[idx_mid-idx_dist:idx_mid+idx_dist,idx_mid-idx_dist:idx_mid+idx_dist]
    lon = lon[idx_mid-idx_dist:idx_mid+idx_dist,idx_mid-idx_dist:idx_mid+idx_dist]
    data = data[idx_mid-idx_dist:idx_mid+idx_dist,idx_mid-idx_dist:idx_mid+idx_dist]

     #Key = "0.0=missing data, 1.0=no decision, 5.0=non-production mask, 7.0=tile fill, 8.0=no input tile expected, 25.0=land, 37.0=inland water, 50.0=cloud";
    mask_values=[0.0,1.0,5.0,7.0,8.0,25.0,37.0,50.0]
    
    data = ma.masked_where(data==0.0,data,copy=True)
    data = ma.masked_where(data==1.0,data,copy=True)
    data = ma.masked_where(data==5.0,data,copy=True)
    data = ma.masked_where(data==7.0,data,copy=True)
    data = ma.masked_where(data==8.0,data,copy=True)
    data = ma.masked_where(data==25.0,data,copy=True)
    data = ma.masked_where(data==37.0,data,copy=True)
    data = ma.masked_where(data==50.0,data,copy=True)
    data = ma.masked_invalid(data,copy=True)

    # Reduce array size
    lat = ma.masked_invalid(lat,copy=True)
    lon = ma.masked_invalid(lon,copy=True)
    

    return lat,lon,data
"""


def get_PIOMAS_SD(date):

    # Get file
    year = date.year
    month = date.month

    path_data = path_dict.PATH_DICT['PATH_DATA']+'PIOMAS/'
    filepattern = path_data +'snow.H%s' %(year)
    filename = glob.glob(filepattern)
    if len(filename)==0:
        print("\n%s: No found" %(filepattern))
        return None,None,None
    else:
        filename = filename[0]
        print("\nReading PIOMAS snow file %s" %(filename))

    f = open(filename,'rb')
    data = np.fromfile(f, dtype='float32')
    data = np.double(data.reshape((12,120,360)))
    data[data==0]=np.NaN
    
    snow=np.double(np.squeeze(data[month-1,:,:]))
    snow=snow.reshape((360*120,1))
    
    lon=[]
    lat=[]

    file=path_data + 'lon.txt'
    f = open(file, 'r')
    x = f.readlines()
    for i in range(4320):
        for j in range(10):
            l=x[i]
            L=np.double((l[2+j*8:2+j*8+6]))
            #print(L)                                                    
            lon=np.concatenate((lon,[L]))


    file=path_data +'lat.txt'
    f = open(file, 'r')
    x = f.readlines()
    for i in range(4320):
        for j in range(10):
            l=x[i]
            L=np.double((l[3+j*8:2+j*8+6]))
            lat=np.concatenate((lat,[L]))


    snow=np.ma.squeeze(ma.masked_invalid(snow,copy=True))

    ####### LON/LAT to polar                                                     
    #x=(90-lat)*111000*np.cos(lon/180*np.pi)
    #y=(90-lat)*111000*np.sin(lon/180*np.pi)
    #x2=(90-latitude)*111000*np.cos(longitude/180*np.pi)
    #y2=(90-latitude)*111000*np.sin(longitude/180*np.pi)

    ####### Interpolation                                                        
    #x2=np.squeeze(np.reshape(x2, (500*500, 1)))
    #y2=np.squeeze(np.reshape(y2, (500*500, 1)))

    return lat,lon,snow



def get_W99(datestr):

    #datestr = date.strftime('%Y%m%d')                                                               
    path_data = path_dict.PATH_DICT['PATH_DATA']+'W99/'
    filepattern =path_data +'snow_w99_*%s.mat' %(datestr[-2:])
    filename = glob.glob(filepattern)
    if len(filename)==0: sys.exit("\n%s: No found" %(filepattern))
    else:
        filename = filename[0]
        print("\nReading W99 file %s" %(filename))

    import scipy.io
    mat = scipy.io.loadmat(filename)

    # Read params                                                                                    
    lat = mat['lat'][:]
    lon = mat['lon'][:]

    snow_w99 = mat['w99'][:]

    # get Warren modified from years                                                                 
    #snow_m = np.squeeze(f.variables['snow_depth'][:])                                               
    #ice_type_old = np.squeeze(f.variables['sea_ice_type_osisaf'][:])                                
    #snow_m[ice_type_old==2] = 2*snow_m[ice_type_old==2]                                             

    return lat,lon,snow_w99



def get_ASD(pixsize,datestr):
    
    #datestr = date.strftime('%Y%m%d')
    path_data = path_dict.PATH_DICT['PATH_DATA']+'ASD/'
    filepattern =path_data +'*w%i*_%s.nc' %(pixsize,datestr)
    filename = glob.glob(filepattern)
    if len(filename)==0:
        print("\n%s: No found" %(filepattern))
        return None,None,None,None
    else:
        filename = filename[0]
        print("\nReading ASD file %s" %(filename))

    try:
        f = nc.Dataset(filename)
    except:
        sys.exit("Cannot open file %s" % filename)

    # Read params
    lat = f.variables['latitude'][:]
    lon = f.variables['longitude'][:]
    snow = np.squeeze(f.variables['snow_depth'][:])
    snow_unc = np.squeeze(f.variables['snow_depth_unc'][:])

    return lat,lon,snow,snow_unc




def get_Laku(datestr):
    
    #datestr = date.strftime('%Y%m%d')
    path_data = path_dict.PATH_DICT['PATH_DATA']+'Laku/'
    filepattern =path_data +'LaKu_%s.nc' %(datestr)
    filename = glob.glob(filepattern)
    if len(filename)==0:
        print("\n%s: No found" %(filepattern))
        return None,None,None,None
    else:
        filename = filename[0]
        print("\nReading Laku file %s" %(filename))

    try:
        f = nc.Dataset(filename)
    except:
        sys.exit("Cannot open file %s" % filename)

    # Read params
    lat = f.variables['latitude'][:]
    lon = f.variables['longitude'][:]
    snow = np.squeeze(f.variables['snow_depth_laku_sam'][:])
    #snow_unc = np.squeeze(f.variables['snow_depth_unc'][:])

    return lat,lon,snow




def get_SIMBA_traj(id_simba):

    # plot buoys data SIMBA
    filepattern = path_dict.PATH_DICT['PATH_DATA']+'SIMBA/FMI*%sGPS*.dat' %(id_simba)
    filename = glob.glob(filepattern)
    if len(filename)==0: sys.exit("\n%s: No found" %(filepattern))
    else:
        filename = filename[0]
        print("\nReading SIMBA file %s" %(filename))

    # Get data
    data = np.loadtxt(filename)
    lat_simba = data[:,5]
    lon_simba = data[:,6]

    return lat_simba,lon_simba



def get_xings_SIMBA(id_simba,lat_cs2,lon_cs2,time_cs2,delay=3,max_dist=100):

    # Get trajectories and time
    #----------------------
    
    filepattern =path_dict.PATH_DICT['PATH_DATA'] +'SIMBA/FMI*%sGPS*.dat' %(id_simba)
    filename = glob.glob(filepattern)
    if len(filename)==0: sys.exit("\n%s: No found" %(filepattern))
    else:
        filename = filename[0]
        print("\nReading SIMBA file %s" %(filename))

    # Get data
    data_traj = np.loadtxt(filename)
    lat_simba = data_traj[:,5]
    lon_simba = data_traj[:,6]

    time_simba = list()
    year = data_traj[:,2].astype(int)
    month = data_traj[:,1].astype(int)
    day = data_traj[:,0].astype(int)

    start_date = datetime.datetime(year[0],month[0],day[0])
    end_date = datetime.datetime(year[-1],month[-1],day[-1])

    mid_date_simba = start_date + (end_date - start_date)/2
    date_simba = list()
    for n in range(lon_simba.size):
        t = datetime.datetime(year[n],month[n],day[n],data_traj[:,3].astype(int)[n],data_traj[:,4].astype(int)[n])
        date_simba.append(t)
        time_simba.append((t-datetime.datetime(2000,1,1)).total_seconds())
    time_simba = np.array(time_simba)
    #date_simba = np.array(date_simba)

    # Get thickness data
    #-----------------------
    
    filepattern =path_dict.PATH_DICT['PATH_DATA'] +'SIMBA/Hsi*%s*.dat' %(id_simba)
    filename = glob.glob(filepattern)
    if len(filename)==0: sys.exit("\n%s: No found" %(filepattern))
    else:
        filename = filename[0]
        print("\nReading SIMBA file %s" %(filename))

    # Get data
    data = np.loadtxt(filename)
    date_sit = [datenum_to_datetime(datenum) for datenum in data[:,0].astype(int)]

    time_sit = np.array([(t-datetime.datetime(2000,1,1)).total_seconds() for t in date_sit])
    #date_interp = [date_simba[0] + datetime.timedelta(days=d) for d in np.arange(1,(date_simba[-1] - date_simba[0]).days,2)]

    
    #timstamps_sit = [d.timestamp() for d in date_simba]
    #day_sec = 60*60*24
    #timestamps_interp = [d for d in np.arange(timstamps_simba[0],timstamps_simba[-1],day_sec*2)]
    
    
    if id_simba=='607':
        sd_simba = data[:,1]
        sit_simba =  data[:,2]
        sit_simba[sit_simba<0] = -sit_simba[sit_simba<0] 
    else: # case 608
        sd_simba = data[:,2]
        #sd_simba = np.interp(timestamps_interp,timstamps_simba,sd_simba)
        sit_simba = np.zeros(sd_simba.shape)
        #date_simba = [datetime.datetime.fromtimestamp(tm) for tm in timestamps_interp]
    sd_simba = np.interp(time_simba,time_sit,sd_simba)
    sit_simba = np.interp(time_simba,time_sit,sit_simba)
        

    #time_simba = np.array([(t-datetime.datetime(2000,1,1)).total_seconds() for t in date_simba])
    

    from scipy.spatial import distance
    x,y,z = lon_lat_to_cartesian(lon_simba, lat_simba)
    coord_simba = np.vstack((x,y,z)).T
    idx_simba = np.arange(lon_simba.size)

    x,y,z = lon_lat_to_cartesian(lon_cs2, lat_cs2) # in km
    coord_cs2 = np.vstack((x,y,z)).T
   

    idx_buoys = list()
    deltaD_list = list()
    deltaT_list = list()
    idx_colloc = list()
    day_colloc = list()
    sit_colloc = list()
    sd_colloc = list()

    outF = open("/home/antlafe/Documents/work/data/SIMBA/CRYO2ICE_xings.txt", "w")
    
    for icre,(coord,time) in enumerate(zip(coord_cs2,time_cs2)):

        # find data with limited delay
        delta_time = time_simba - time
        flag_time = np.abs(delta_time) < delay*24*60*60

        xyz_sdelay = coord_simba[flag_time]
        if not np.any(flag_time): continue

        # compute distance
        dist_sdelay = np.sqrt((xyz_sdelay[:,0]-coord[0])**2 + (xyz_sdelay[:,1]-coord[1])**2 + (xyz_sdelay[:,2]-coord[2])**2)
        flag_delay = dist_sdelay < max_dist
            
        if np.any(flag_delay):
                
            idx_min = np.argmin(dist_sdelay)
            
            idx = idx_simba[flag_time][idx_min]
            idx_buoys.append(idx_simba[flag_time][idx_min])
            deltaD_list.append(dist_sdelay[idx_min])
            deltaT_list.append(delta_time[flag_time][idx_min])
            idx_colloc.append(icre)

            collocDate= datetime.datetime(year[idx],month[idx],day[idx])
            day_colloc.append(collocDate)
            if collocDate in date_simba:
                idx = date_simba.index(collocDate)
                print("found data for this date")
                sit_colloc.append(sit_simba[idx])
                sd_colloc.append(sd_simba[idx])
            else:
                sit_colloc.append(np.nan)
                sd_colloc.append(np.nan)
                    

            #if day_colloc[icre]==day_colloc[icre-1]:  

            found_match_str = "Found match: %i/%i/%i (%.2f,%.2f): Delay=%i h, Dist=%i km " %(day[idx],month[idx],year[idx],lat_simba[idx],lon_simba[idx],deltaT_list[-1]/(60*60),deltaD_list[-1])
            print(found_match_str)

            outF.write(found_match_str)
            outF.write("\n")
    
    outF.close()
            

    # transform lists into arrays
    idx_buoys = np.ma.array(idx_buoys)
    deltaD_list =  np.ma.array(deltaD_list)
    deltaT_list =  np.ma.array(deltaT_list)
    idx_colloc =  np.ma.array(idx_colloc)
    day_colloc = np.ma.array(day_colloc)

    lon_colloc = lon_cs2[idx_colloc]
    lat_colloc = lat_cs2[idx_colloc]
    delay_colloc = deltaT_list/(60*60)

    return idx_colloc,lon_colloc,lat_colloc,delay_colloc,day_colloc,lon_simba,lat_simba,sit_colloc,sd_colloc


def get_data_polygon(lat,lon,polygon):

    """
    provided region corresponding to polygon in lat,lon format
    """
    lon[lon > 180] = lon[lon > 180] - 360
    lat_selected = np.logical_and(lat< max(polygon['lat']),lat> min(polygon['lat']))
    lon_selected = np.logical_and(lon< max(polygon['lon']),lon> min(polygon['lon']))
    region = np.logical_and(lat_selected,lon_selected)
    
    return region





def get_regional_sd_mean(datasetName,polygon,monthsList,flag_FYI):

    if datasetName not in ['ASD','W99','PIOMAS','AMSR']:
        print("Unknown dataset: %s" %(datasetName))
        sys.exit()

    snow_list = list()
    snow_unc_list = list()

    for months in monthsList:
        
        if datasetName=='ASD':
            
            lat,lon,snow,snow_unc = get_ASD(50,months)
            
            # if no data found for this date
            if lat is None:
                snow_list.append(np.nan)
                snow_unc_list.append(np.nan)
                continue

            snow = ma.masked_invalid(snow,copy=True)

            region = get_data_polygon(lat,lon,polygon)
            snow_region = ma.masked_where(~region,snow,copy=True)
            snow_unc_region = ma.masked_where(~region,snow,copy=True)

            snow_list.append(np.ma.mean(snow_region)*100)
            snow_unc_list.append(np.ma.std(snow_region)*100)

        elif datasetName=='W99':
                
            lat,lon,snow = get_W99(months)
            snow = ma.masked_invalid(snow,copy=True)

            region = get_data_polygon(lat,lon,polygon)
            snow_region = ma.masked_where(~region,snow,copy=True)
            snow_unc_region = ma.masked_where(~region,snow,copy=True)

            factor=0.5 if flag_FYI else 1
            snow_list.append(np.ma.mean(factor*snow_region))
            snow_unc_list.append(np.ma.std(factor*snow_region))

        elif datasetName=='AMSR':

            y = int(months[:4])
            m = int(months[4:6])
            daylist = allDaysMonth(y, m)
            lat,lon,snow = get_SD_AMSR(daylist[0])
            snow = ma.masked_invalid(snow,copy=True)

            region = get_data_polygon(lat,lon,polygon)

            snow_day_list = list()
            for day in daylist:
                lat,lon,snow = get_SD_AMSR(day)
                snow_region = ma.masked_where(~region,snow,copy=True)
                snow_day_list.append(snow_region)

            snow = np.ma.concatenate(snow_day_list)
            
            snow_list.append(np.ma.mean(snow))
            snow_unc_list.append(np.ma.std(snow))

            
        elif datasetName=='PIOMAS':

            date = datetime.datetime.strptime(months,'%Y%m')
            lat,lon,snow = get_PIOMAS_SD(date)
            
            # if no data found for this date
            if lat is None:
                snow_list.append(np.nan)
                snow_unc_list.append(np.nan)
                continue

            snow = ma.masked_invalid(snow,copy=True)
            region = get_data_polygon(lat,lon,polygon)
        
            snow_region = ma.masked_where(~region,snow,copy=True)
            snow_unc_region = ma.masked_where(~region,snow,copy=True)

            snow_list.append(np.ma.mean(snow_region)*100)
            snow_unc_list.append(np.ma.std(snow_region)*100)

            
    return snow_list,snow_unc_list

def get_valid_sla_flag(flag):

    dict_error= {
       
        #'sarin_bad_velocity'       : 2,
        #'sarin_out_of_range'       : 4,
        'sarin_bad_baseline'       : 8,
        'delta_time_error'         :32,
        'mispointing_error'        :64,
        'sarin_height_ambiguous'   :2048,
        'surf_type_class_ocean'    :32768,
        #'peakiness_error'          :131072,
        'ssha_interp_error'        :262144,
        'orbit_error'              :67108864,
        #'height_sea_ice_error'     :268435456,
        }

    all_flag= {

        'calibration_warning'      : 1,
        'sarin_bad_velocity'       : 2,
        'sarin_out_of_range'       : 4,
        'sarin_bad_baseline'       : 8,
        'lrm_slope_model_invalid'  :16,
        'delta_time_error'         :32,
        'mispointing_error'        :64,
        'surface_model_unavailable':128,
        'sarin_side_redundant'     :256,
        'sarin_rx_2_error'         :512,
        'sarin_rx_1_error'         :1024,
        'sarin_height_ambiguous'   :2048,
        'surf_type_class_undefined':4096,
        'surf_type_class_sea_ice'  :8192,
        'surf_type_class_lead'     :16384,
        'surf_type_class_ocean'    :32768,
        'freeboard_error'          :65536,
        'peakiness_error'          :131072,
        'ssha_interp_error'        :262144,
        'sig0_3_error'             :524288,
        'sig0_2_error'             :1048576,
        'sig0_1_error'             :2097152,
        'height_3_error'           :4194304,
        'height_2_error'           :8388608,
        'height_1_error'           :16777216,
        'orbit_discontinuity'      :33554432,
        'orbit_error'              :67108864,
        'block_degraded'           :134217728,
        'height_sea_ice_error'     :268435456,
        }

    # sea ice class
    flag_seaice = np.bitwise_and(flag, all_flag['surf_type_class_sea_ice'])/all_flag['surf_type_class_sea_ice']

    # errors
    flag_error = np.zeros(flag.size)
    for key in dict_error.keys():
        flag0_error = np.bitwise_and(flag,dict_error[key])/dict_error[key]
        flag_error = np.logical_or(flag0_error,flag_error)

    flag_valid_sla = flag_seaice - flag_error
    flag_valid_sla[flag_valid_sla<0] =0

    return flag_valid_sla


def get_valid_freeboard_flag(flag):

    """

    flag_masks = 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576, 2097152, 4194304, 8388608, 16777216, 33554432, 67108864, 134217728, 268435456; // int
    """

    dict_error= {
       
        'sarin_bad_velocity'       : 2,
        'sarin_out_of_range'       : 4,
        'sarin_bad_baseline'       : 8,
        'delta_time_error'         :32,
        'mispointing_error'        :64,
        'sarin_height_ambiguous'   :2048,
        'surf_type_class_ocean'    :32768,
        'freeboard_error'          :65536,
        'peakiness_error'          :131072,
        'ssha_interp_error'        :262144,
        'orbit_error'              :67108864,
        'height_sea_ice_error'     :268435456,
        }
       

    all_flag= {

        'calibration_warning'      : 1,
        'sarin_bad_velocity'       : 2,
        'sarin_out_of_range'       : 4,
        'sarin_bad_baseline'       : 8,
        'lrm_slope_model_invalid'  :16,
        'delta_time_error'         :32,
        'mispointing_error'        :64,
        'surface_model_unavailable':128,
        'sarin_side_redundant'     :256,
        'sarin_rx_2_error'         :512,
        'sarin_rx_1_error'         :1024,
        'sarin_height_ambiguous'   :2048,
        'surf_type_class_undefined':4096,
        'surf_type_class_sea_ice'  :8192,
        'surf_type_class_lead'     :16384,
        'surf_type_class_ocean'    :32768,
        'freeboard_error'          :65536,
        'peakiness_error'          :131072,
        'ssha_interp_error'        :262144,
        'sig0_3_error'             :524288,
        'sig0_2_error'             :1048576,
        'sig0_1_error'             :2097152,
        'height_3_error'           :4194304,
        'height_2_error'           :8388608,
        'height_1_error'           :16777216,
        'orbit_discontinuity'      :33554432,
        'orbit_error'              :67108864,
        'block_degraded'           :134217728,
        'height_sea_ice_error'     :268435456,
        }

    # sea ice class
    flag_seaice = np.bitwise_and(flag, all_flag['surf_type_class_sea_ice'])/all_flag['surf_type_class_sea_ice']

    # errors
    flag_error = np.zeros(flag.size)
    for key in dict_error.keys():
        flag0_error = np.bitwise_and(flag,dict_error[key])/dict_error[key]
        flag_error = np.logical_or(flag0_error,flag_error)

    flag_valid_fb = flag_seaice - flag_error
    flag_valid_fb[flag_valid_fb<0] =0

    return flag_valid_fb



def datenum_to_datetime(datenum):
    """
    Convert Matlab datenum into Python datetime.
    :param datenum: Date in datenum format
    :return:        Datetime object corresponding to datenum.
    """
    days = datenum % 1
    hours = days % 1 * 24
    minutes = hours % 1 * 60
    seconds = minutes % 1 * 60
    return datetime.datetime.fromordinal(int(datenum)) \
           + datetime.timedelta(days=int(days)) \
           + datetime.timedelta(hours=int(hours)) \
           + datetime.timedelta(minutes=int(minutes)) \
           + datetime.timedelta(seconds=round(seconds)) \
           - datetime.timedelta(days=366)



def fbr2sit(fb_radar,snow_depth,ice_type,ds=300,d_w=1024):

    # Get ice density
    d_i = np.ma.ones(fb_radar.shape)
    d_i[ice_type==2] = 917
    d_i[ice_type==3] = 882

    # Get snow density
    #t = ((datetime - datetime.datetime(2020,10,1))/30).days
    #if t > 8: sys.exit()
    #d_s =6.50t+274.51
    #d_s = 300

    snow_density = d_s/1000
    speed_of_light_ratio = np.power(1 + 0.51*snow_density,1.5) # speed of light in snow over speed of light in vacuum
    height_snow_penetration_corr = snow_depth*(speed_of_light_ratio-1)
    freeboard = fb_radar + height_snow_penetration_corr
    sit = (d_w*freeboard + d_s*snow_depth)/(d_w-d_i)
    #sit2 = (d_w/(d_w-d_i))*fb_radar + (((speed_of_light_ratio-1)*d_w + d_s)/(d_w - d_i))*snow_depth

    return sit



def fbt2sit(fb_total,snow_depth,ice_type,ds=300,d_w=1024):

    # Get ice density
    d_i = np.ma.ones(fb_total.shape)
    d_i[ice_type==2] = 917
    d_i[ice_type==3] = 882

    # Get snow density
    #t = ((datetime - datetime.datetime(2020,10,1))/30).days
    #if t > 8: sys.exit()
    #d_s =6.50t+274.51
    #d_s = 300

    
    sit = (d_w*(fb_total-snow_depth) + d_s*snow_depth)/(d_w-d_i)

    return sit
