#! /home/antlafe/anaconda3/bin/python

#
# comp_al_IS2_CS2.py
#
# ----------------------------------------------------------------------
# Copyright (c) 2020 LEGOS/SERCO
# All rights reserved.
#



"""
DESCRIPTION:

     Animated plot to show comparison of IS2 and CS2 data

USAGE:

     animation_cs2_is2.py [options]

optional arguments:

EXAMPLES:

    python -m ipdb animation_cs2_is2.py -g ESA_BD,AWI -p radar_fb -g ATL07 -b gt1r,gt2r,gt3r -p surface_h -d20200301,20200303 -f 202003

    python -m pdb animation_cs2_is2_scatters.py -g ESA_BD_GDR -p sla -g ATL10 -b b1,b2,b3 -p sla -sw -d20201101,20201103 -f NovJan_ESA

PRINCIPLE:

- interpolating a reference track
- indexing each IS2 beams and CS2 product data on this reference beam
- indenting these indexes on IS2 ref track (for the display)
- producing data arrays indented on this ref track
- developping data frame based on ref track


# ERRORS: possible errors on ref track can spread to the rest of the data  


"""

import sys
import h5py
import netCDF4 as nc
import numpy as np
from numpy import ma 
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
from matplotlib import cm
import glob
from datetime import date, timedelta, datetime
import argparse
import cs2_dict
import is2_dict
import common_functions as cf
import stats_tools as stats
import warnings
import scipy.spatial
from scipy.stats import gaussian_kde
import pickle
import matplotlib.animation as animation
from matplotlib import gridspec
from parserObjects import ParentAction,ChildAction
import matplotlib as mpl
import os
from scipy.interpolate import interp1d
from operator import itemgetter
import seaborn as sns
import path_dict

# Global attributs
###########################################

PATH_DATA= path_dict.PATH_DICT['PATH_DATA']
PATH_INPUT =  path_dict.PATH_DICT['PATH_OUT']
PATH_OUT = path_dict.PATH_DICT['PATH_FIG']

# info: possibility to use lat01 to avoid POCA coord over SARin mask areas
REF_GDR = 'ESA_BD_GDR'

is2_gdrs = ['ATL10']

colors_plot_cs2 = ['deepskyblue','dodgerblue','turquoise','royalblue','palegreen']
colors_plot_is2 =['palegreen','mediumseagreen','green']
cs2_color = 'dodgerblue'
is2_color = 'seagreen'
common_color ='black'
colors_lineplot = [is2_color,cs2_color]

beamList=['gt1r','gt2r','gt3r','gt1l','gt2l','gt3l']

list_midnight_dates = {
    'ATL07 ': ['20201005','20201018','20201022','20201108','20201125','20201129','20201216','20210102','20210119','20210123','20210209','20210226','20210315','20210319','20210401','20210422'],
    'ATL10': ['20201005','20201018','20201022','20201108','20201125','20201129','20201216','20210102','20210119','20210123','20210209','20210226','20210315','20210319','20210401','20210422'],
    'ATL12': ['20201014','20201031'], #not up to date
    }


#list_midnight_dates = ['20201018','20201108'] 

dist_frame = 100#km
interval = 10 #ms
show_plot = True
outfilename= 'colloc_nov_jan20_full_esa'

# mean density to show data
MIN_IS2_DATA_DENSITY = 1 #00 # pts/km
MIN_CS2_DATA_DENSITY = 0.3 # pts/km

# limits 
xylim = [-0.3, 0.6] # scatter limits
snowlim = [0,0.4]


arr_step_is2 = 1 # for swath # 15
arr_step_cs2 = 1

###########################################
#
#              Functions
#
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

#---------------------------------
# Get CS2 dates
#--------------------------------
"""
def is2date_2_cs2date(date,is2gdr):
    '''
    this function deals with cases when IS2/CS2 collocated tracks are one day apart
    the function simplys add one day to CS2 date for these specific cases
    '''

    
    if date in [datetime.strptime(d,'%Y%m%d') for d in list_midnight_dates[is2gdr]]:
        date_1 =  date + timedelta(days=1)
        print("IS2 date: %s correspond to CS2 date: %s" %(date.strftime('%d/%m/%Y'),date_1.strftime('%d/%m/%Y')))
        #date_1 = date_1.strftime('%Y%m%d')
    else:
        date_1 = date

    return date_1
"""


# function to interpolate coordinates where data are missing
def interp_coordinates(lat,lon,delta_d,arr_step):

    """
    Function to interpolate over a set of coordinates with a distance of delta_d

    lat       : (1D array) latitude
    lon       : (1D array) longitude
    delta_d   : (int) along-track length interval  
    arr_step  : (int) only consider data every arr_step in lat,lon arrays
    ADVICE: increase this number if double number error in interpolation

    """
    if any(np.abs(np.diff(lon)) > 20): lon[lon > 180] = lon[lon > 180] - 360   
    lat = lat[::arr_step] # warning using only every 5 data to avoid having same data in distance param
    lon = lon[::arr_step] # if bug increase that value
    #lon[lon <0] = lon[lon <0] + 360
    distance = np.nancumsum(cf.dist_btw_two_coords(lat[1:],lat[:-1],lon[1:],lon[:-1]))
    
    max_distance = distance[-1]   
    distance = np.insert(distance, 0, 0)/max_distance
    alpha = np.linspace(0, 1, int(max_distance/delta_d))
    points =  np.array([lat.tolist(),lon.tolist()]).T
    points[points==None] = np.nan
    interpolator =  interp1d(distance, points, kind='quadratic', axis=0)
    interp_pts = interpolator(alpha)
    lat = interp_pts[:,0]; lon = interp_pts[:,1]
    if any(np.abs(np.diff(lon)) > 300) or any(lon < 0): lon[lon < 0] = lon[lon < 0] + 360
    return lat,lon


def interp_coordinate_swath(lat,lon):

    distbtwpts= 10 #km
    if any(np.abs(np.diff(lon)) > 20): lon[lon > 180] = lon[lon > 180] - 360   
    distance=np.nancumsum(np.ones(lat.shape)[:-1]*distbtwpts)


    mask_coords = lat.mask
    
    max_distance = distance[-1]   
    distance = np.insert(distance, 0, 0)/max_distance
    alpha = np.linspace(0, 1, int(max_distance/delta_d))
    points =  np.array([lat.tolist(),lon.tolist()]).T
    points[points==None] = np.nan
    interpolator =  interp1d(distance, points, kind='quadratic', axis=0)
    interp_pts = interpolator(alpha)
    lat = interp_pts[:,0]; lon = interp_pts[:,1]
    if any(np.abs(np.diff(lon)) > 300) or any(lon < 0): lon[lon < 0] = lon[lon < 0] + 360
    return lat,lon




def onClick(event):
    global pause
    pause ^= True


import matplotlib.lines
from matplotlib.transforms import Bbox, TransformedBbox
from matplotlib.legend_handler import HandlerBase
from matplotlib.image import BboxImage

class HandlerLineImage(HandlerBase):

    def __init__(self, path, space=15, offset = 10):
        self.space=space
        self.offset=offset
        #self.width = width
        self.image_data = plt.imread(path)        
        super(HandlerLineImage, self).__init__()

    def create_artists(self, legend, orig_handle,
                       xdescent, ydescent, width, height, fontsize, trans):

        l = matplotlib.lines.Line2D([xdescent+self.offset,xdescent+(width-self.space)/3.+self.offset],
                                     [ydescent+height/2., ydescent+height/2.])
        l.update_from(orig_handle)
        l.set_clip_on(False)
        l.set_transform(trans)

        bb = Bbox.from_bounds(xdescent +(width+self.space)/3.+self.offset,
                              ydescent,
                              height*self.image_data.shape[1]/self.image_data.shape[0],
                              height)

        tbb = TransformedBbox(bb, trans)
        image = BboxImage(tbb)
        image.set_data(self.image_data)

        self.update_prop(image, orig_handle, legend)
        return [l,image]


###########################################
#
#              Main
#
###########################################


if __name__ == '__main__':

    # Define programme description
    description ='Animated plot to show comparison of IS2 and CS2 data'

    # Initiate the parser with a description
    parser = argparse.ArgumentParser(description=description)

    # Add long and short arguments
    # -------------------------------------------------------
    parent = parser.add_argument('-g', '--gdr', action=ParentAction)

    parser.add_argument("-b","--beamN",default=None,help="set desired IS2 beam",action=ChildAction, parent=parent)
    
    parser.add_argument("-p","--parameters",required=True,help="provide parameters to plot. If absent, list all available parameters",action=ChildAction, parent=parent)
    
    parser.add_argument("-f","--inputfolder",help="provide input pickle data file",required=True)

    #parser.add_argument("-g","--gdrs",help="set desired CS2 products to best tested",required=True)

    parser.add_argument("-d","--date",required=True,help="provide CS2 track date")

    #parser.add_argument("-p","--parameters",required=True,help="provide CS2 parameter to plot")

    parser.add_argument("-o","--outfilename",default=outfilename,help="[optionnal] provide outpath")
    
    parser.add_argument("-sw","--swath",action="store_true",help="option to add swath aligned data") 

    
    # Read arguments
    # ----------------------------------------------------------
    args = parser.parse_args()

    # Open pikle file
    #-----------------------------------------------------------
    inputfolder = args.inputfolder +'/'
    inputfilepattern = PATH_INPUT + inputfolder + 'data_dict.pkl'
    filename = glob.glob(inputfilepattern)
    if len(filename)==0: sys.exit("\n%s: No found" %(inputfilepattern))

    pkl_file = open(filename[0], 'rb')
    data_dict = pickle.load(pkl_file)

    # Outpath
    #----------------------------------------------------------
    outfilename = args.outfilename
    
    # Open info params file
    #-----------------------------------------------------------
    inputfilepattern = PATH_INPUT + inputfolder + 'info_params.pkl'
    filename = glob.glob(inputfilepattern)
    if len(filename)==0: sys.exit("\n%s: Not found" %(inputfilepattern))
    pkl_file = open(filename[0], 'rb')
    info_params = pickle.load(pkl_file)
            
    # Get required dates
    #-----------------------------------------------------------
    date = [d for d in args.date.split(',')]
    if len(date) != 2: print('Provide date -d as YYYYMMDD,YYYYMMDD in arguments');sys.exit()

    start_date = datetime.strptime(date[0], '%Y%m%d')
    end_date = datetime.strptime(date[1], '%Y%m%d')

    requested_dates =  [start_date + timedelta(n) for n in range(int((end_date - start_date).days)+1)]
    available_dates = data_dict['dates']

    found_dates = [d for d in requested_dates if d in available_dates]
    if len(found_dates)==0: print('Unavailable dates',date,'\nAvailable dates are:',[d.strftime("%Y%m%d") for d in available_dates]);sys.exit()
    ndates = len(found_dates)
    idx_dates = np.array([available_dates.index(date) for date in found_dates])

    # Unwrap GDRs:
    #-----------------------------------------------------------
    gdrs = [gdr for gdr in args.gdr.items()]
    if len(gdrs)>2: print("To many GDR provided: please provide -g IS2-gdr ... -g CS2-gdr");sys.exit()
    # find indexes of IS2 and CS2
    idx_is2 = int(np.argwhere(['ATL' in gdr[0] for gdr in gdrs]))
    idx_cs2 = int(np.argwhere(['ATL' not in gdr[0] for gdr in gdrs]))
    
    # Find CS2 L2P product
    #-----------------------------------------------------------
    gdrs_cs2 = [g for g in gdrs[idx_cs2][0].split(',')]

    global avail_gdrs_cs2
    avail_gdrs_cs2 = data_dict['CS2'].keys()

    gdrs_cs2_checking = gdrs_cs2.copy()
    for gdr in gdrs_cs2:
        if gdr not in avail_gdrs_cs2:
            print("%s not found in:" %(gdr),avail_gdrs_cs2)
            sys.exit("Add %s in %s" %(gdr,pkl_file))

    #Get parameter: XXX one for the moment
    pname_cs2 = gdrs[idx_cs2][1].parameters

   
    # Test if param is available:
    print("\nChecking %s for CS2 \n#----------------" %(pname_cs2))
    for gdr in gdrs_cs2:
        print("\n %s:" %(gdr))
        for d in found_dates:
            d_cs2 = is2date_2_cs2date(d,'ATL10')
            date_str_cs2 = d_cs2.strftime('%Y%m%d')
            print("%s:" %(date_str_cs2),end='')
            if pname_cs2 in info_params[gdr][date_str_cs2].keys():
                print(info_params[gdr][date_str_cs2][pname_cs2]["status"])
                units_cs2=info_params[gdr][date_str_cs2][pname_cs2]["units"]
            else:
                print("%s not available in data dict, chose param in" %(pname_cs2),[p for p in info_params[gdr][date_str_cs2].keys()])

    # Find IS2 data
    #------------------------------------------------------------
    gdr_is2 = [g for g in gdrs[idx_is2][0].split(',')]
    if len(gdr_is2)>1: print("To many GDR provided for IS2 please choose between ATL07 & ATL10");sys.exit()
    else: gdr_is2=gdr_is2[0]

    #Get parameter: XXX one for the moment
    pname_is2 = gdrs[idx_is2][1].parameters
    
    print("\nChecking %s for IS2 \n#----------------" %(pname_is2))
    
    # Get requested beams
    if 'beamN' not in vars(gdrs[idx_is2][1]).keys():
        beam_is2 =  beamList
    else:
        beam_is2 = [b for b in gdrs[idx_is2][1].beamN.split(',')]

    flag_beam = [b in info_params[gdr_is2][d.strftime("%Y%m%d")].keys() for b in beam_is2]

    found_beams = list(np.array(beam_is2)[np.array(flag_beam)])
    if np.all(flag_beam): print("\nAll beams %s were found" %(beam_is2))
    else: print("\nOnly",found_beams,"were found")
    beam_is2 = found_beams.copy()

     
    # SWATH option
    #----------------------------------------------------
    flag_swath = args.swath
    if flag_swath and not 'swath' in data_dict['IS2'].keys():
            print("No SWATH data in data_dict: %s \n" %(inputfolder))
            sys.exit()
   
    # Test if param is available:
    print("\n %s:" %(gdr_is2))
    for d in found_dates:
        print("%s:" %(d.strftime("%Y%m%d")))
        for b in found_beams:
            print("%s:" %(b),end='')
            # check params for SWATH DATA
            if flag_swath:
                if pname_is2 in data_dict['IS2']['swath'][b].keys():
                    units_is2='m' # to be changed
                else:
                    print("%s not available in data dict, chose param in" %(pname_is2),[p for p in data_dict['IS2']['swath'][b].keys()])
                    sys.exit() 
            else:        
                if pname_is2 in info_params[gdr_is2][d.strftime("%Y%m%d")][b].keys():
                    print(info_params[gdr_is2][d.strftime("%Y%m%d")][b][pname_is2]["status"])
                    units_is2=info_params[gdr_is2][d.strftime("%Y%m%d")][b][pname_is2]["units"]
                else:
                    print("%s not available in data dict, chose param in" %(pname_is2),[p for p in info_params[gdr_is2][d.strftime("%Y%m%d")][b].keys()])
                    sys.exit()
                
   
    # Check units
    #----------------------------------------------
    units = units_is2
        
    ################################################################
    #
    #
    #                      ANIMATION
    #
    #
    ###############################################################

    #--------------------------------------------------
    #
    #               get trajectory data
    #
    #---------------------------------------------------
    beam_to_show = found_beams[0] # warning if not required !

    if flag_swath:
        
        REF_GDR = gdr_is2='swath'
        is2_full_lats = list(np.array(data_dict['IS2']['swath'][beam_to_show]['lat'],dtype=object)[idx_dates])                    
        is2_full_lons = list(np.array(data_dict['IS2']['swath'][beam_to_show]['lon'],dtype=object)[idx_dates])

        # Get full coordinates
        lat_swath = list(np.array(data_dict['CS2']['swath'][beam_to_show]['lat'],dtype=object)[idx_dates])
        lon_swath = list(np.array(data_dict['CS2']['swath'][beam_to_show]['lat'],dtype=object)[idx_dates])
        
        ref_seg_lats = list()
        ref_seg_lons = list()
        mask_list = list()
        for n in range(ndates):
            ref_seg_lats.append(lat_swath[n][~lat_swath[n].mask])
            ref_seg_lons.append(lon_swath[n][~lat_swath[n].mask])
            mask_list.append(~lat_swath[n].mask)
            
        cs2_full_lats = ref_seg_lats
        cs2_full_lons = ref_seg_lons
        print("str")
        
    else:
        # Get full lat/lon
        is2_full_lats = list(np.array(data_dict['IS2'][gdr_is2][beam_to_show]['latfull'],dtype=object)[idx_dates])                    
        is2_full_lons = list(np.array(data_dict['IS2'][gdr_is2][beam_to_show]['lonfull'],dtype=object)[idx_dates])

        # Get full coordinates
        cs2_full_lats = list(np.array(data_dict['CS2'][REF_GDR]['latref_full'],dtype=object)[idx_dates])
        cs2_full_lons = list(np.array(data_dict['CS2'][REF_GDR]['lonref_full'],dtype=object)[idx_dates])
        # Get ref lat/lon
        ref_seg_lats = list(np.array(data_dict['CS2'][REF_GDR]['latref'],dtype=object)[idx_dates])
        ref_seg_lons = list(np.array(data_dict['CS2'][REF_GDR]['lonref'],dtype=object)[idx_dates])
   
    
    
    #--------------------------------------------------
    #
    #               get ICESAT-2 data
    #
    #---------------------------------------------------

    # get data to display
    delay = dict()
    dist = dict()
    data_is2 = dict()
    data_cs2_b = dict()
    x_dist_seg_is2 = dict()
    lat_is2 = dict()
    lon_is2 = dict()
    #for b in beam_is2: x_dist_seg_is2[b]=list() 

    # Retreive data
    for b in beam_is2:
        lat_is2[b] = list(np.array(data_dict['IS2'][gdr_is2][b]['lat'],dtype=object)[idx_dates]) 
        lon_is2[b] = list(np.array(data_dict['IS2'][gdr_is2][b]['lon'],dtype=object)[idx_dates])
        data_is2[b] = list(np.array(data_dict['IS2'][gdr_is2][b][pname_is2],dtype=object)[idx_dates])       
        delay[b] = list(np.array(data_dict['IS2'][gdr_is2][b]['delay'],dtype=object)[idx_dates])
        dist[b] = list(np.array(data_dict['IS2'][gdr_is2][b]['dist'],dtype=object)[idx_dates])

        if flag_swath:
            data_cs2_b[b] = list(np.array(data_dict['CS2']['swath'][b][pname_cs2],dtype=object)[idx_dates])

            
    # interpolate to get missing data (over the land)
    is2_full_lats_interp = list() # coordinates to display
    is2_full_lons_interp = list()
    x_dist_is2 = list()
    for n in range(ndates):
            # eliminates duplicates (To do)
            #lat = is2_full_lats[n]
            #lon = is2_full_lons[n]
            #unique,idx,num_occ = np.unique(lat,return_index=True,return_counts=True) 
            #flag_occurence = num_occ>1
            lat_interp,lon_interp = interp_coordinates(is2_full_lats[n],is2_full_lons[n],dist_frame,arr_step_is2)
            is2_full_lats_interp.append(lat_interp)
            is2_full_lons_interp.append(lon_interp)
    
    # compute mean distance
    #mean_delta_dist = [np.mean(d) for d in dist[b]]

    # Compute mean delay
    mean_delay = list()
    for n in range(ndates):
        mean_delay.append(np.ma.mean(np.ma.array([np.mean(delay[b][n]) for b in beam_is2])))
                          
    mean_delay_sign = [np.sign(md) for md in mean_delay]
    mean_delay_str = [str(timedelta(minutes=np.abs(mins)))[:7] for mins in mean_delay]
    

    #--------------------------------------------------
    #
    #               get CRYOSAT-2 data
    #
    #---------------------------------------------------

    # check if REF GDR in data_file
    if not REF_GDR in data_dict['CS2'].keys():
        print("CS2 REF GDR: %s is not in data_dict" %(REF_GDR))
        print("Relaunch sortnsave algorithm with REF_GDR")
        sys.exit()

   
    # get data to display
    data_is2_2d = list()
    data_cs2 = dict()
    for cs2_prod in gdrs_cs2: data_cs2[cs2_prod]= list()

    # XXXX prboleme
    
    for n in idx_dates:

        # get 2-D data from IS2
        data_mat = data_dict['IS2'][gdr_is2][beam_to_show][pname_is2][n]
        data_mat[data_mat.mask] = np.nan
        data_is2_2d.append(data_mat.data)

        for cs2_prod in gdrs_cs2:
            if flag_swath:
                data_cs2[cs2_prod] = data_cs2_b[beam_to_show]
            else:
               data_array_cs2 = ma.masked_invalid(data_dict['CS2'][cs2_prod][pname_cs2][n])
               data_cs2[cs2_prod].append(data_array_cs2) # [valid_idx]) 
               

    """
    for n in idx_dates:
        
        # get 2-D data from IS2
        data_mat = data_is2[n]
        data_mat[data_mat.mask] = np.nan
        data_is2_2d.append(data_mat.data)
        for cs2_prod in gdrs_cs2:
            # convert to masked array
            if flag_swath:
                data_array_cs2 = data_cs2[n]
            else:
                data_array_cs2 = ma.masked_invalid(data_dict['CS2'][cs2_prod][pname_cs2][n])
            data_cs2[cs2_prod].append(data_array_cs2) # [valid_idx])
            #data_cs2[cs2_prod].append(np.mean(data_dict['IS2']['ATL10']['laser_fb'][n],axis=0))
     """   
            
   
    # interpolate to get missing data (over the land)
    cs2_full_lats_interp = list()
    cs2_full_lons_interp = list()
    x_dist_cs2 = list()
    #if flag_swath:
        #if not flag_swath:
    for n in range(ndates):
        lat_interp,lon_interp = interp_coordinates(is2_full_lats[n],is2_full_lons[n],dist_frame,arr_step_cs2)
        cs2_full_lats_interp.append(lat_interp)
        cs2_full_lons_interp.append(lon_interp)
    #else:
    #    cs2_full_lats_interp = cs2_full_lats
    #    cs2_full_lons_interp = cs2_full_lons
         
    
    #--------------------------------------------------
    #
    #         Define common interpolated track
    #
    #---------------------------------------------------

    # Min lat to show in basemap (depends on MIN_LAT chosen in sortnsave algo)
    min_lat = min(np.min(np.concatenate(is2_full_lats,axis=0)), np.min(np.concatenate(cs2_full_lats,axis=0)))

    # Rq: Track used to plot the data

     # Get ref lat/lon
    ref_lats_interp = list()
    ref_lons_interp = list()
    x_dist = list()
    for n in range(ndates):
        if not flag_swath:
            lat_ref_interp,lon_ref_interp = interp_coordinates(ref_seg_lats[n],ref_seg_lons[n],dist_frame,2)
            ref_lats_interp.append(lat_ref_interp)
            ref_lons_interp.append(lon_ref_interp)
            x_dist.append(cf.distance_from_first_trk_pts(ref_lats_interp[n],ref_lons_interp[n],0))
        else:
            ref_lats_interp.append(ref_seg_lats[n])
            ref_lons_interp.append(ref_seg_lons[n])
            x_dist.append(cf.distance_from_first_trk_pts(ref_lats_interp[n],ref_lons_interp[n],0))
    #ref_lons_interp[0]
    #from scipy.signal import savgol_filter
    #yhat = savgol_filter(ref_lons_interp[0],11, 2) # window size 51, polynomial order 3
    
    #--------------------------------------------------
    #
    #               get OSISAF data
    #
    #---------------------------------------------------


    
    # Draw osisaf ice type with first data
    lons_icetype = list()
    lats_icetype = list()
    icetype = list()
    icetype_al = list()
    
    for n in range(ndates):

        date = found_dates[n]
        lons,lats,OSISAF_ice_type = cf.get_osisaf_ice_type(date.year,date.month,date.day,'01')
        OSISAF_ice_type[OSISAF_ice_type==1] = ma.masked # masked ocean
        OSISAF_ice_type[OSISAF_ice_type==3] = 4 # ambigous becomes multi-year ice
        OSISAF_ice_type[560,380] = 2 # to keep colorbar 
    
        lons_icetype.append(lons)
        lats_icetype.append(lats)
        icetype.append(OSISAF_ice_type)
        
        #icetype_alongtrack = cf.grid_to_track(OSISAF_ice_type,lons,lats,lon_cs2[n],lat_cs2[n])
        lon = ref_lons_interp[n]
        if any(np.abs(np.diff(lon)) > 20): lon[lon > 180] = lon[lon > 180] - 360   
        icetype_alongtrack = cf.grid_to_track(OSISAF_ice_type,lons,lats,lon,ref_lats_interp[n])
        icetype_al.append(icetype_alongtrack)

    icetype_al_full = np.ma.concatenate(icetype_al,axis=0)
        
            

    #---------------------------------------------------------------------------------------------


    #--------------------------------------------------
    #
    #               Define sequence
    #
    #---------------------------------------------------


    # IS2 params

    # frame lists
    is2_start_index = list()
    is2_interval =list()
    is2_interval_seg = list()
    is2_seg_frame = list()
    
   
    # seg lists
    is2_idx_common_data_seg = list()
    cs2_idx_common_data_seg = list()
    
    is2_frame_seg_b = dict()
    is2_frame_seg_uniq_b = dict()
    for b in beam_is2:
        is2_frame_seg_b[b] = list()
        is2_frame_seg_uniq_b[b] = list()

    # CS2 params
    cs2_start_index = list()
    cs2_interval =list()
    cs2_interval_seg = list()
    cs2_seg_frame = list()
    
    cs2_frame_seg = list()
    cs2_frame_seg_uniq = list()
    
    last_frame= 0
    full_frame = list()
    ntrack_idx = [0]
    
    
    for ntrack in range(ndates):


        # Create KDTree for reference track
        # In order to associate data points to this reference track
        
        x_interp,y_interp,z_interp = cf.lon_lat_to_cartesian(ref_lons_interp[ntrack],ref_lats_interp[ntrack])
        coord_interp = np.vstack((x_interp,y_interp,z_interp)).T
        tree_seg = scipy.spatial.cKDTree(coord_interp)
        coord_start = np.array([x_interp[0],y_interp[0],z_interp[0]])

        ##########################
        # IS2 time frame
        ############################

        # Create KDTree from interpolated tracks
        
        
        x_interp_i,y_interp_i,z_interp_i = cf.lon_lat_to_cartesian(is2_full_lons_interp[ntrack],is2_full_lats_interp[ntrack])
        coord_interp_is2 = np.vstack((x_interp_i,y_interp_i,z_interp_i)).T
        
        tree_display_is2 = scipy.spatial.cKDTree(coord_interp_is2)

        # find indexes conversion from CS2 ref track interp to IS2 ref track interp
        distance,ref_cs2_into_ref_is2 = tree_display_is2.query(coord_interp,1)
        
        
        for nb,b in enumerate(beam_is2):
            
            # Get segments coordinates
            is2_lats_b = lat_is2[b][ntrack]
            is2_lons_b = lon_is2[b][ntrack]
            x_b,y_b,z_b = cf.lon_lat_to_cartesian(is2_lons_b, is2_lats_b)

            # Find indexes of seg in frame
            coord_seg_is2 = np.vstack((x_b,y_b,z_b)).T
            distance,is2_frame_seg = tree_seg.query(coord_seg_is2,1)
            is2_frame_seg_uniq = np.unique(is2_frame_seg) #,return_index=True)

            # indexes associated to each data frame points w.r.t CS2 ref track
            is2_frame_seg_b[b].append(is2_frame_seg)
            is2_frame_seg_uniq_b[b].append(is2_frame_seg_uniq)

            
            # find starting points of segment
            distance,is2_frame_seg_start = tree_display_is2.query(coord_start,1)
            
            # get longest common section of all beams
            if nb==0: min_val = is2_frame_seg[0]; max_val = is2_frame_seg[-1]
            min_val = min(is2_frame_seg[0],min_val)
            max_val = max(is2_frame_seg[-1]+1,max_val)


        # indexes where data are found for each beams wrt to CS2 ref track
        idx_in_ref_cs2 = np.arange(min_val,max_val)
        is2_idx_common_data_seg.append(idx_in_ref_cs2)

        # indexes where data are found converted wrt to IS2 ref track
        # usefull to display the data at the right frame
        idx_in_ref_is2 = ref_cs2_into_ref_is2[idx_in_ref_cs2]
        
        
        
        # create sequence IS2
        #uncomment this section if IS2 track is better defined
        """
        # when using IS2 interp: need enough data
        is2_start_index.append(last_frame)
        full_frame.extend(last_frame+np.arange(1,is2_full_lons_interp[ntrack].shape[0]))
        is2_interval.extend(last_frame+np.arange(1,is2_full_lons_interp[ntrack].shape[0]))
        is2_interval_seg.extend(last_frame+is2_frame_seg_start+idx_in_ref_is2)
        ntrack_idx.extend(ntrack*np.ones(is2_full_lons_interp[ntrack].shape[0]-1,dtype=int))
        """
        is2_start_index.append(last_frame+1)
        full_frame.extend(last_frame+np.arange(ref_lons_interp[ntrack].shape[0])+1)
        is2_interval.extend(last_frame+np.arange(ref_lons_interp[ntrack].shape[0])+1)
        is2_interval_seg.extend(last_frame+np.arange(ref_lons_interp[ntrack].shape[0])+1)
        ntrack_idx.extend(ntrack*np.ones(ref_lons_interp[ntrack].shape[0],dtype=int))
        
        
        last_frame = is2_interval[-1]
        
        ###########################
        # CS2 time frame
        ############################

        # Full coord track 
        x_interp_c,y_interp_c,z_interp_c = cf.lon_lat_to_cartesian(cs2_full_lons_interp[ntrack],cs2_full_lats_interp[ntrack])
        coord_interp_cs2 = np.vstack((x_interp_c,y_interp_c,z_interp_c)).T
        tree_display_cs2 = scipy.spatial.cKDTree(coord_interp_cs2)

        # find starting points of segment
        distance,cs2_frame_seg_start = tree_display_cs2.query(coord_start,1)

        
        # Get segments coordinates
        cs2_lats = ref_seg_lats[ntrack]
        cs2_lons = ref_seg_lons[ntrack]
        x_c,y_c,z_c = cf.lon_lat_to_cartesian(cs2_lons, cs2_lats)
        
        # Find indexes of seg in frame
        coord_cs2 = np.vstack((x_c,y_c,z_c)).T
        #coord_cs2 = np.vstack((lat_cs2[ntrack],lon_cs2[ntrack])).T
        distance,frame_seg = tree_seg.query(coord_cs2,1)
        frame_seg_uniq = np.unique(frame_seg) #,return_index=True)
        
        # indexes associated to each data frame points
        cs2_frame_seg.append(frame_seg)
        cs2_frame_seg_uniq.append(frame_seg_uniq)
        cs2_idx_common_data_seg.append(np.arange(frame_seg[0],frame_seg[-1]+1))
        

        # create sequence CS2
        cs2_start_index.append(last_frame+1)
        full_frame.extend(last_frame+np.arange(cs2_full_lats_interp[ntrack].shape[0])+1)
        cs2_interval.extend(last_frame+np.arange(cs2_full_lats_interp[ntrack].shape[0])+1)
        cs2_interval_seg.extend(last_frame+cs2_frame_seg_start+cs2_idx_common_data_seg[ntrack]+1)
        ntrack_idx.extend(ntrack*np.ones(cs2_full_lats_interp[ntrack].shape[0],dtype=int))
        last_frame = cs2_interval[-1]
    # for the last frame
    ntrack_idx.append(ntrack)
    full_frame.append(len(full_frame)+1)

    #-------------------------------------------------------------------------------------------


    #--------------------------------------------------
    #
    #               Retreive data
    #
    #---------------------------------------------------

    

    is2_data_pts = dict()
    is2_mean_dist = dict()
    is2_ndata_pts = dict()
    idx_pts_is2 = dict()
    dist_save = dict()
    is2_mean_data_line = list()
    is2_mean_dist_line = list()
    is2_mean_data_pts = list()
    for b in beam_is2:
        idx_pts_is2[b] = list()
        is2_data_pts[b] = list()
        is2_mean_dist[b] = list()
        is2_ndata_pts[b] = list()
        dist_save[b] = list()

    cs2_data_pts = dict()
    cs2_mean_dist = dict()
    cs2_ndata_pts = dict()
    idx_pts_cs2 = dict()
    cs2_mean_data_pts = list()
    for gdr in gdrs_cs2:
        idx_pts_cs2[gdr] = list()
        cs2_data_pts[gdr] = list()
        cs2_mean_dist[gdr] = list()
        cs2_ndata_pts[gdr] = list()

    for ntrack in range(ndates):
        
        ##########################
        # define IS2 data array
        ##########################
        #dist_start_seg = x_dist_is2[ntrack][is2_idx_common_data_seg[ntrack][0]]
        for b in beam_is2:

            for idx_frame in is2_idx_common_data_seg[ntrack]:

                # Case data found around this point
                idx_pts_is2[b].append(idx_frame)
                if idx_frame in is2_frame_seg_uniq_b[b][ntrack]:

                    idx = is2_frame_seg_b[b][ntrack]==idx_frame
                    ndata = np.sum(~np.isnan(data_is2[b][ntrack][idx]))
                    data_density = ndata/dist_frame
                    is2_ndata_pts[b].append(ndata)

                    # if data density sufficient
                    if data_density > MIN_IS2_DATA_DENSITY:
                        is2_mean_dist[b].append(x_dist[ntrack][idx_frame])
                        is2_data_pts[b].append(np.nanmean(data_is2[b][ntrack][idx]))

                    else:
                        is2_data_pts[b].append(np.nan)
                        is2_mean_dist[b].append(np.nan)
                    #print("%s: density: %s, ndata: %s, dist: %s, data: %s" %(b,data_density,ndata,np.nanmean(data_is2[b][ntrack][idx]),x_dist[ntrack][idx_frame]))

                # case no data found around this point
                else:
                    is2_data_pts[b].append(np.nan)
                    is2_mean_dist[b].append(np.nan)
                    is2_ndata_pts[b].append(0)

        
       
        
        
        ##########################
        # define CS2 data array
        ##########################
        for ngdr,gdr in enumerate(gdrs_cs2):

            for idx_frame in cs2_idx_common_data_seg[ntrack]:

                # Case data found around this point
                if idx_frame in cs2_frame_seg_uniq[ntrack]:

                    idx = cs2_frame_seg[ntrack]==idx_frame
                    
                    ndata = np.ma.sum(~np.isnan(data_cs2[gdr][ntrack][idx]))
                    if np.ma.is_masked(ndata): ndata=np.sum(~data_cs2[gdr][ntrack][idx].mask)
                    data_density = ndata/dist_frame
                    cs2_ndata_pts[gdr].append(ndata)

                    # if data density sufficient
                    if data_density > MIN_CS2_DATA_DENSITY:
                        cs2_data_pts[gdr].append(np.ma.mean(data_cs2[gdr][ntrack][idx]))
                        cs2_mean_dist[gdr].append(x_dist[ntrack][idx_frame])
                        
                        #Save mean data for IS2 sorted in the CS2 beams
                        if ngdr==0:
                            is2_mean_data_pts.append(np.nanmean(data_is2_2d[ntrack][idx]))
                            #is2_mean_data_pts.append(np.nanmean(data_is2_2d[ntrack][:,idx]))
                       
                    else:
                        cs2_data_pts[gdr].append(np.nan)
                        cs2_mean_dist[gdr].append(np.nan)
                        if ngdr==0: is2_mean_data_pts.append(np.nan)
                    #print("%s: density: %s, ndata: %s, dist: %s, data: %s" %(gdr,data_density,ndata,np.ma.mean(data_cs2[gdr][ntrack][idx]),x_dist[ntrack][idx_frame]))

                # case no data found around this point
                else:
                    cs2_data_pts[gdr].append(np.nan)
                    cs2_mean_dist[gdr].append(np.nan)
                    cs2_ndata_pts[gdr].append(0)
                    if ngdr==0: is2_mean_data_pts.append(np.nan)


    # define mean values CS2 & IS2
    #-------------------------------------
    data_list = list()
    data_dist_list = list()
    for b in beam_is2:
        data_list.append(is2_data_pts[b])
        data_dist_list.append(is2_mean_dist[b])
    is2_data_matrix = np.array(data_list)
    is2_data_dist_matrix = np.array(data_dist_list)
    is2_mean_data_line.extend(np.nanmean(is2_data_matrix,axis=0))
    is2_mean_dist_line.extend(np.nanmedian(is2_data_dist_matrix,axis=0))

    # define rolling median of CS2
    data_array = np.ma.masked_invalid(np.array(cs2_data_pts[gdr]),copy=True)
    cs2_mean_data = stats.rolling_stats(data_array, 4, stats=['mean'])[0]
    # temporary XXX
    cs2_mean_data[np.isnan(cs2_data_pts[gdr])] = ma.masked
    cs2_mean_data_pts.extend(cs2_mean_data)

    
    print("stop")
        
    """
    plt.plot(is2_data_matrix[0,:],'*')
    plt.plot(is2_data_matrix[1,:],'*')
    plt.plot(is2_data_matrix[2,:],'*')
    plt.plot(is2_mean_data_line,'.-')
    plt.show()
    """

    
    
    """
    # XXX TEST scatter
    xylim = [-0.3,0.5]
    f11, ax = plt.subplots(1, 1, sharey=True)
    x_data = np.array(is2_mean_data_pts)
    x_label = 'IS2 laser fb (m)'
    y_data = np.array(cs2_data_pts['ESA_BD_GDR'])
    y_label = 'CS2 radar fb (m)'
    commun_mask = np.logical_and(~np.isnan(x_data),~np.isnan(y_data))

    # SCATTER
    #cf.statistics('freeboard',param,'m',is_flag)
    icetype = np.ma.concatenate(icetype_al,axis=0)
    f11, ax = plt.subplots(1, 1, sharey=True)
    stats.plot_scatter(ax,xylim,y_data,y_label,x_data,x_label,icetype)
   
    #tips = sns.load_dataset("tips")
    #ax = sns.regplot(x="total_bill", y="tip", data=tips)
    p = sns.regplot(x=y_data, y=x_data, color="darkgray",fit_reg=True,scatter=False,ax=ax)
    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(x=p.get_lines()[0].get_xdata(),y=p.get_lines()[0].get_ydata())
    
    ax.legend()

    # HISTO
    f12, ax = plt.subplots(1, 1, sharey=True)
    legend_list = [x_label,y_label]
    data_list = [x_data,y_data]
    stats.plot_histo(ax,xylim,'m','freeboard [m]',legend_list,data_list,commun_mask)
    plt.show()
    """

    
    #print("stop")
    """
    x_cs2_myi = np.array(cs2_mean_data_pts)[icetype_al_full.data==4]
    y_is2_myi = np.array(is2_mean_data_line)[icetype_al_full.data==4]
    common_mask = np.logical_and(~np.isnan(x_cs2_myi),~np.isnan(y_is2_myi))
    res = scipy.stats.linregress(x_cs2_myi[common_mask],y_is2_myi[common_mask])
    x_abs_myi = np.arange(np.round(np.nanmin(x_cs2_myi),2),np.round(np.nanmax(x_cs2_myi),2),0.02)
    y_reg = res.intercept + res.slope*x_abs_myi
    plt.plot(x_abs_myi,x_abs_myi)
    plt.fill_between(x_abs_myi, x_abs_myi, y_reg, color='lightgrey' )
    plt.show()
    """
                    
    #--------------------------------------------------
    #
    #               Define figures
    #
    #---------------------------------------------------


    #fig3 = plt.figure(constrained_layout=True)

    
    """
    gs = fig3.add_gridspec(3, 3)
    f3_ax1 = fig3.add_subplot(gs[0, :])
    f3_ax1.set_title('gs[0, :]')
    f3_ax2 = fig3.add_subplot(gs[1, :-1])
    f3_ax2.set_title('gs[1, :-1]')
    f3_ax3 = fig3.add_subplot(gs[1:, -1])
    f3_ax3.set_title('gs[1:, -1]')
    f3_ax4 = fig3.add_subplot(gs[-1, 0])
    f3_ax4.set_title('gs[-1, 0]')
    f3_ax5 = fig3.add_subplot(gs[-1, -2])
    f3_ax5.set_title('gs[-1, -2]')
    """
    
    # Plot parameter as animation
    #-----------------------------------------------------------
    fig = plt.figure(1,figsize=(20,7)) #,constrained_layout=True)
    #fig = plt.figure(1,figsize=(14,5))
    #gs = fig.add_gridspec(3, 3)
    spec = gridspec.GridSpec(ncols=11, nrows=10) #,width_ratios=[1.2, 2])
    
    spec_ESAlogo = spec[:,0]
    spec_map = spec[:,1:5]
    spec_STplot = spec[:4,6:11]
    spec_myiscat = spec[6:10,6:8]
    spec_fyiscat = spec[6:10,9:11]

    # Add ESA logo
    #----------------------------------------------------------
    #import matplotlib.image as image
    from PIL import Image
    ax1 = fig.add_subplot(spec_ESAlogo)
    im = Image.open('../images/esa.png')
    #im = im.rotate(90)
    
    
    #im = image.imread('../images/esa.png')
    #im = im.reshape((im.shape[1],im.shape[0],im.shape[2]))
    ax1.imshow(im)
    ax1.axis('off')
    
    # Define Arctic map
    # ---------------------------------------------------------
    ax2 = fig.add_subplot(spec_map)
    
    #m = Basemap(projection='npstere',boundinglat=min_lat,lon_0=0, resolution='l' , round=False)
    llrx = 2.47e+06
    llry = 6.0e+06
    urrx = 9.26e+06
    urry = 1.17e+07
    m = Basemap(projection='ortho',lat_0=70,lon_0=0,resolution='l',ax=ax2) #,llcrnrx=llrx,llcrnry=llry,urcrnrx=urrx,urcrnry=urry)
    #m = Basemap(projection='npstere',boundinglat=min_lat,lon_0=0, resolution='l' , round=False)
    m.drawcoastlines(linewidth=0.25, zorder=1)
    m.drawparallels(np.arange(90,-90,-5), linewidth = 0.25, zorder=3)
    #m.drawmeridians(np.arange(-180.,180.,30.), latmax=85, linewidth = 0.25, zorder=1)
    m.drawmeridians(np.arange(0,360,45),labels=[1,1,1,1],linewidth=0.5, fontsize=14, dashes=[1,5],zorder=3)
    m.fillcontinents(color='0.9',lake_color='grey', zorder=1)
    m.bluemarble(scale=1, zorder=1)
    stats.draw_round_frame(m,ax2)

    
    xptsT, yptsT = m(lons_icetype[0], lats_icetype[0])
    
    cmap = cmap = mpl.colors.ListedColormap(["white", "lightgrey"])
    im = m.contourf(xptsT , yptsT, icetype[0],linewidths=0.5,cmap=cmap, alpha=1,zorder=2)
    norm = mpl.colors.BoundaryNorm(np.arange(2,4), cmap.N)
    cbar = fig.colorbar(im,ticks=[2.5,3.5],orientation='horizontal',fraction=0.046, pad=0.04,extend='both')
    cbar.ax.set_xticklabels(['First Year Ice','Multi-Year Ice'])
    cbar.set_label('OSISAF daily ice type', labelpad=3)

   
    
   
    # IS2 track
    #-----------------------------------------------------------
    x,y = m(is2_full_lons[0],is2_full_lats[0])
    is2_full = list()
    is2_seg = list()
    for n in range(ndates):
        is2_full.append(m.plot(0, 0, linewidth=2,color=is2_color,zorder=3)[0])
        is2_seg.append(m.plot(0, 0, linewidth=2,color=is2_color,zorder=3)[0])
    is2_sat = m.plot(x[0], y[0], markersize=4,marker='8',color=is2_color,zorder=5)[0]

    
    # CS2 track
    #------------------------------------------------------------
    x,y = m(cs2_full_lons[0],cs2_full_lats[0])
    cs2_full = list()
    cs2_seg = list()
    for n in range(ndates):
        cs2_full.append(m.plot(0, 0, linewidth=2,color=cs2_color,zorder=3)[0])
        cs2_seg.append(m.plot(0, 0, linewidth=2 ,color=common_color,zorder=3)[0])
    cs2_sat = m.plot(0, 0, markersize=4,marker='8',color=cs2_color,zorder=5)[0]

    ax2.legend([is2_full[0],cs2_full[0]], ["", ""],
               handler_map={ is2_full[0]: HandlerLineImage('../images/icesat21.png'), cs2_full[0]: HandlerLineImage('../images/cryosat21.png')}, 
               handlelength=2, labelspacing=0.1, fontsize=30, borderpad=0.2, loc="lower right", 
               handletextpad=1.2, borderaxespad=0.15) #,loc="lower right")
    #ax2.legend(loc="lower right")

    # common data track
    #-----------------------------------------------------------
    data_seg = list()
    for n in range(ndates):
        data_seg.append(m.scatter(0,0,c=0,s=3,cmap='magma',vmin=snowlim[0],vmax=snowlim[1],zorder=4,alpha=0.8))
    cb = fig.colorbar(data_seg[0], ax=ax2,extend='both',fraction=0.046, pad=0.04,shrink=0.80)
    cb.set_label(r'$\Delta$fb(La-Ku) [m]',fontsize=12)
    
    

    # set label:
    #m.plot(0, 0, linewidth=2,color=is2_color,label='IceSat-2 tracks')
    #m.plot(0, 0, linewidth=2,color=cs2_color,label='CryoSat-2 tracks')
    #m.plot(0, 0, linewidth=2,color='black',label='Collocated section')
    
    
    

    # add plots
    # ---------------------------------------------------------
    
    ax3 = fig.add_subplot(spec_STplot)
    ax3.title.set_text('Measured freeboard between IceSat-2 and CryoSat-2')
    
    is2_data_plot = list()
    for nb,b in enumerate(beam_is2):
        is2_data_plot.append(ax3.plot(-1,0,marker='*',linestyle = 'None',markersize=4,color=colors_plot_is2[nb])[0])
    is2_data_plot_line = ax3.plot(is2_mean_dist_line[0],is2_mean_data_line[0],linestyle = '-',color=is2_color,label="IS2: laser freeboard")
    

    cs2_data_plot = list()
    for ng,cs2_prod in enumerate(gdrs_cs2):
        cs2_data_plot.append(ax3.plot(-1,0,marker='*',linestyle = 'None', markersize=4,color=colors_plot_cs2[ng])[0])
    cs2_data_plot_line = ax3.plot(cs2_mean_dist[cs2_prod][0],cs2_mean_data_pts[0],linestyle = '-',color=cs2_color,label="CS2: radar freeboard")

    #radius_str = "Aver radius= %i km" %(dist_frame) 
    #ax3.annotate(radius_str, xy=(0.02, 0.95), xycoords='axes fraction', fontsize=12)
    
    
    
    #ax3.legend()
    ax3.grid()
    ax3.set_xlim([x_dist[0][0],x_dist[0][-1]]) #limites x_dist
    ax3.set_ylim(xylim) # limite param
    #ax3.set_ylim([min_y-1, max_y-1]) # limite param
    ax3.set_xlabel("Along-track distance (km)",fontsize=10)
    ax3.set_ylabel("Freeboard (m)",fontsize=10)
    ax3.legend(loc="lower right")
    #ax3.set_ylabel("Freeboard (%s)" %(units),fontsize=10)
    #ax3.set_title("CS2:%s / IS2:%s (%s)" %(pname_cs2,pname_is2,units),fontsize=12)

    # ----------------------------------------------------------


    # define annotations
    # ---------------------------------------------------------
    bbox_props = dict(boxstyle="round", fc="white", ec="black",alpha=0.8, lw=1)
    # annotations
    an_is2 = ax2.annotate('', xy=(0, 0), xycoords='data', xytext=(-500, -500), textcoords='data',fontsize=14,color='black',zorder=4,bbox=bbox_props)
    an_cs2 = ax2.annotate('', xy=(0, 0), xycoords='data', xytext=(-500, -500), textcoords='data',fontsize=14,color='black',zorder=4,bbox=bbox_props)
    an_delay = ax2.annotate("", xy=(0.02, 0.90), xycoords='axes fraction', fontsize=11,bbox=bbox_props)
    an_dist = ax2.annotate("", xy=(0.02, 0.85), xycoords='axes fraction', fontsize=11,bbox=bbox_props)
    an_date = ax2.annotate("", xy=(0.02, 0.95), xycoords='axes fraction', fontsize=11,bbox=bbox_props)


    # add scatter plots
    # ---------------------------------------------------------
    # MYI scatter plot
    
    ax4 = fig.add_subplot(spec_myiscat)
    ax4.set_xlim(xylim) #limites x_dist
    ax4.set_ylim(xylim)
    ax4.set_aspect('equal', adjustable='box')
    ax4.plot([-1, 1], [-1, 1], color = 'black', linestyle = 'dashed')
    ax4.title.set_text('Multi-year ice')
    ax4.set_xlabel("cryoSat-2 freeboard [m]")
    ax4.set_ylabel("iceSat-2 freeboard [m]")
    scat_myi = ax4.scatter([np.nan],[np.nan], s=15,marker='o',edgecolors='black',c='lightgrey',alpha=0.8)
    ax4.grid()

    # FYI scatter plot
    
    ax5 = fig.add_subplot(spec_fyiscat)
    ax5.set_xlim(xylim) #limites x_dist
    ax5.set_ylim(xylim)
    ax5.set_aspect('equal', adjustable='box')
    ax5.plot([-1, 1], [-1, 1], color = 'black', linestyle = 'dashed')
    ax5.title.set_text('First-year ice')
    ax5.set_xlabel("cryoSat-2 freeboard [m]")
    ax5.set_ylabel("iceSat-2 freeboard [m]")
    scat_fyi = ax5.scatter([np.nan],[np.nan], s=15, marker='o',edgecolors='black',c='whitesmoke',alpha=0.8)
    ax5.grid()
                        
    #plt.show()
    
    # init data lists
    # ---------------------------------------------------------
    
    
    # IS2 data lists
    x_data_is2 = list()
    y_data_is2 = list()
    xseg_data_is2 = list()
    yseg_data_is2 = list()
    x_dist_data_is2_line = list()
    is2_data_line = list()
    x_dist_data_is2 = dict()
    is2_data = dict()
    for b in beam_is2:
        is2_data[b] = list()
        x_dist_data_is2[b] = list()

    # CS2 data lists
    x_data_cs2 = list()
    y_data_cs2 = list()
    xseg_data_cs2 = list()
    yseg_data_cs2 = list()
    #x_dist_data_cs2 = list()
    x_dist_data_cs2_line = list()
    cs2_data_line = list()
    cs2_data = dict()
    x_dist_data_cs2= dict()
    for gdr in gdrs_cs2:
        cs2_data[gdr] = list()
        x_dist_data_cs2[gdr] = list()

    # snow depth list
    snow_depth = list()

    #plt.show()
    
    # animation function
    # ---------------------------------------------------------
    # animation function.  This is called sequentially

    
    """
    def run_animation():
        anim_running = True

        def onClick(event):
            nonlocal anim_running
            if anim_running:
                anim.event_source.stop()
                anim_running = False
            else:
                anim.event_source.start()
                anim_running = True
    """
    
    #plt.show()
     
    def animate(i):


        print('start: %i/%i' %(i,len(full_frame)))
        # Get track number i.e date
        #print(i)
        if i==0: ntrack=0;ntrack_prev=0;
        else:
            ntrack = ntrack_idx[i]
            ntrack_prev = ntrack_idx[i-1]



        # get date
        str_date = "Date: %s" %(found_dates[ntrack].strftime("%d/%m/%y"))
        an_date.set_text(str_date)


        # For each new track
        #-------------------------------------------------------
        if np.abs(ntrack - ntrack_prev)==1 or i==1:

            print('new track: %s - %i' %(ntrack,i))

            # Clear all lists
            x_data_is2.clear()
            y_data_is2.clear()
            xseg_data_cs2.clear()
            yseg_data_cs2.clear()
            xseg_data_is2.clear()
            yseg_data_is2.clear()
            x_data_cs2.clear()
            y_data_cs2.clear()
            snow_depth.clear()
            #an_is2.set_text('')
            #an_cs2.set_text('')

            # clear curves CS2
            for ng,gdr in enumerate(gdrs_cs2):
                x_dist_data_cs2[gdr].clear()
                cs2_data[gdr].clear()
                cs2_data_plot[ng].set_data(x_dist_data_cs2[gdr],cs2_data[gdr])
            # full line
            x_dist_data_cs2_line.clear()
            cs2_data_line.clear()
            cs2_data_plot_line[0].set_data(x_dist_data_cs2[gdr],cs2_data[gdr])

            # clear curves IS2
            for nb,b in enumerate(beam_is2):
                is2_data[b].clear()
                x_dist_data_is2[b].clear()
                is2_data_plot[nb].set_data(x_dist_data_is2[b],is2_data[b])
            # full line
            x_dist_data_is2_line.clear()
            is2_data_line.clear()
            is2_data_plot_line[0].set_data(x_dist_data_is2,is2_data)

            ax2 = fig.add_subplot(spec_map)
            xptsT, yptsT = m(lons_icetype[ntrack], lats_icetype[ntrack])
            im = m.contourf(xptsT , yptsT, icetype[ntrack],cmap=cmap, alpha=1,zorder=2)

            ax3.collections.clear()
            ax3.set_xlim([x_dist[ntrack][0],x_dist[ntrack][-1]])
            ax3.fill_between(x_dist[ntrack], 0, 1, where=icetype_al[ntrack] == 4, facecolor='lightgrey', alpha=0.5, transform=ax3.get_xaxis_transform())
            ax3.fill_between(x_dist[ntrack], 0, 1, where=icetype_al[ntrack] < 2, facecolor='white', alpha=0.5, transform=ax3.get_xaxis_transform())
            ax3.fill_between(x_dist[ntrack], 0, 1, where=icetype_al[ntrack].mask,facecolor='white',hatch='//' , alpha=0.5, transform=ax3.get_xaxis_transform())


        # IS2 track animation
        #---------------------------------------------------------
        if i in is2_interval:

            print('IS2',ntrack,i)
            n_is2 = i -  is2_start_index[ntrack]
            #x,y = m(ref_lons_interp[ntrack][n_is2],ref_lats_interp[ntrack][n_is2])
            x, y = m(is2_full_lons_interp[ntrack][n_is2],is2_full_lats_interp[ntrack][n_is2])
            x,y = round(x, 0),round(y, 0)
            x_data_is2.append(x)
            y_data_is2.append(y)

            # Full track data
            is2_full[ntrack].set_data(x_data_is2,y_data_is2)
            is2_sat.set_data(x, y)

            an_is2.set_position((x, y))
            an_is2.set_text('IS2')
            an_delay.set_text("")
            an_dist.set_text("")

            # Draw common data section

            if i in is2_interval_seg:

                #print('IS2 seg',ntrack,i)
                xseg_data_is2.append(x)
                yseg_data_is2.append(y)
                is2_seg[ntrack].set_data(xseg_data_is2,yseg_data_is2)

                for nb,b in enumerate(beam_is2):

                    # index of data
                    N= is2_interval_seg.index(i)

                    # data points
                    #---------------

                    # get data
                    x_dist_data_is2[b].append(is2_mean_dist[b][N])
                    is2_data[b].append(is2_data_pts[b][N])

                    # plot data
                    is2_data_plot[nb].set_data(x_dist_data_is2[b],is2_data[b])
                    #print("%s: dist: %.1f ndata: %i" %(b,is2_mean_dist[b][N],is2_ndata_pts[b][N]))
                # data curve
                #---------------

                # get data
                x_dist_data_is2_line.append(is2_mean_dist_line[N])
                is2_data_line.append(is2_mean_data_line[N])

                # plot data
                is2_data_plot_line[0].set_data(x_dist_data_is2_line,is2_data_line)






        # CS2 track animation
        #---------------------------------------------------------------
        if i in cs2_interval:

            print('CS2',ntrack,i)
            n_cs2 = i -  cs2_start_index[ntrack]

            x, y = m(cs2_full_lons_interp[ntrack][n_cs2],cs2_full_lats_interp[ntrack][n_cs2])
            x,y = round(x, 0),round(y, 0)
            x_data_cs2.append(x)
            y_data_cs2.append(y)

            # Full track data
            cs2_full[ntrack].set_data(x_data_cs2,y_data_cs2)
            cs2_sat.set_data(x, y)

            an_cs2.set_position((x,y))
            an_cs2.set_text("CS2")
            delay_str = r'Mean $\Delta$t: %s' %(mean_delay_str[ntrack])
            an_delay.set_text(delay_str)
            mean_dist_str = r'Mean $\Delta$d: %.1f km' %(mean_delta_dist[ntrack])
            an_dist.set_text(mean_dist_str)

            # Draw continuously black line:
            if i in cs2_interval_seg:

                xseg_data_cs2.append(x)
                yseg_data_cs2.append(y)
                cs2_seg[ntrack].set_data(xseg_data_cs2,yseg_data_cs2)

                # indexof data
                N= cs2_interval_seg.index(i)

                #x_dist_data_cs2.append(cs2_mean_dist[cs2_prod][N])

                for ng,cs2_prod in enumerate(gdrs_cs2):

                    # data points
                    #---------------
                    # get data
                    x_dist_data_cs2[cs2_prod].append(cs2_mean_dist[cs2_prod][N])
                    cs2_data[cs2_prod].append(cs2_data_pts[cs2_prod][N])

                    # plot data
                    cs2_data_plot[ng].set_data(x_dist_data_cs2[cs2_prod],cs2_data[cs2_prod])


                    # update common data
                    SD_n = np.array(is2_mean_data_line)[N] - np.array(cs2_mean_data_pts)[N]
                    snow_depth.append(SD_n)
                    #print('sd',snow_depth)
                    #print('xdist',x_dist_data_cs2[cs2_prod])
                    #print('ice_type',icetype_al_full[N])

                    data_seg[ntrack].set_offsets(np.c_[xseg_data_cs2,yseg_data_cs2])
                    data_seg[ntrack].set_array(np.array(snow_depth))


                    # update scatters
                    flag_myi = icetype_al_full[:N+1].data==4
                    dcs2_myi = np.array(cs2_mean_data_pts)[:N+1][flag_myi]
                    dis2_myi = np.array(is2_mean_data_line)[:N+1][flag_myi]
                    scat_myi.set_offsets(np.c_[dcs2_myi,dis2_myi])

                    flag_fyi = icetype_al_full[:N+1].data==2
                    dcs2_fyi = np.array(cs2_mean_data_pts)[:N+1][flag_fyi]
                    dis2_fyi = np.array(is2_mean_data_line)[:N+1][flag_fyi]
                    scat_fyi.set_offsets(np.c_[dcs2_fyi,dis2_fyi])
                    #scat_myi.set_offsets(scatter_data_fyi)

                # data curve
                #---------------

                # get data
                x_dist_data_cs2_line.append(cs2_mean_dist[cs2_prod][N])
                cs2_data_line.append(cs2_mean_data_pts[N])

                # plot data
                cs2_data_plot_line[0].set_data(x_dist_data_cs2_line,cs2_data_line)


            # show snow depth on scatters
        if i==len(full_frame):

            print("last frame")
            fig.savefig("tracks.png")
            
            # Regression curve MYI
            x_cs2_myi = np.array(cs2_mean_data_pts)[icetype_al_full.data==4]
            y_is2_myi = np.array(is2_mean_data_line)[icetype_al_full.data==4]
            common_mask = np.logical_and(~np.isnan(x_cs2_myi),~np.isnan(y_is2_myi))
            x_cs2_myi = np.array(x_cs2_myi)[common_mask]
            y_is2_myi = np.array(y_is2_myi)[common_mask]
            res = scipy.stats.linregress(x_cs2_myi,y_is2_myi)

            x_abs_myi = np.arange(np.round(np.nanmin(x_cs2_myi),2),np.round(np.nanmax(x_cs2_myi),2),0.02)
            y_reg = res.intercept + res.slope*x_abs_myi
            ax4.plot(x_abs_myi, y_reg, color='darkgray')
            delta_mean = np.nanmean(y_is2_myi- x_cs2_myi)
            props = dict(boxstyle='round', facecolor='white', alpha=0.8)
            textstr = u'Δfb(MYI)= %.1fcm' %(delta_mean*100)
            ax4.text(0,-0.1, textstr, fontsize=10,verticalalignment='bottom',horizontalalignment ='left', bbox=props)
            x_abs_myi = np.arange(np.round(np.nanmin(x_cs2_myi),2),np.round(np.nanmax(x_cs2_myi),2),0.02)
            y_reg = res.intercept + res.slope*x_abs_myi
           
            ax4.fill_between(x_abs_myi, x_abs_myi, y_reg, color='lightgrey', alpha=0.6 )

            # Regression curve FYI
            x_cs2_fyi = np.array(cs2_mean_data_pts)[icetype_al_full.data==2]
            y_is2_fyi = np.array(is2_mean_data_line)[icetype_al_full.data==2]
            common_mask = np.logical_and(~np.isnan(x_cs2_fyi),~np.isnan(y_is2_fyi))
            x_cs2_fyi = np.array(x_cs2_fyi)[common_mask]
            y_is2_fyi = np.array(y_is2_fyi)[common_mask]
            res = scipy.stats.linregress(x_cs2_fyi,y_is2_fyi)

            x_abs_fyi = np.arange(np.round(np.nanmin(x_cs2_fyi),2),np.round(np.nanmax(x_cs2_fyi),2),0.02)
            y_reg = res.intercept + res.slope*x_abs_fyi

            ax5.plot(x_abs_fyi, y_reg, color='darkgray')
            delta_mean = np.nanmean(y_is2_fyi- x_cs2_fyi)
            props = dict(boxstyle='round', facecolor='white', alpha=0.8)
            textstr = u'Δfb(FYI)= %.1fcm' %(delta_mean*100)
            ax5.text(0, -0.1, textstr, fontsize=10,verticalalignment='bottom',horizontalalignment ='left', bbox=props)
            ax5.fill_between(x_abs_fyi, x_abs_fyi, y_reg, color='lightgrey' , alpha=0.6)
            

            
            #plt.ion()
            
            x_data_is2.clear()
            y_data_is2.clear()
            xseg_data_cs2.clear()
            yseg_data_cs2.clear()
            xseg_data_is2.clear()
            yseg_data_is2.clear()
            x_data_cs2.clear()
            y_data_cs2.clear()
            an_is2.set_text('')
            an_cs2.set_text('')

            # warning if ends in cs2_seg
            for n in range(len(cs2_seg)):
                cs2_seg[n].set_data(0,0)
                cs2_full[n].set_data(0,0)
                is2_seg[n].set_data(0,0)
                is2_full[n].set_data(0,0)

            cs2_seg.clear()
            cs2_full.clear()
            is2_seg.clear()
            is2_full.clear()
            cs2_sat.set_data(0,0)
            is2_sat.set_data(0,0)
            fig.canvas.draw()

            
            #def init_sat_track(ndates):
            
            for n in range(ndates):
                is2_full.append(m.plot(0, 0, linewidth=2,color=is2_color,zorder=3)[0])
                is2_seg.append(m.plot(0, 0, linewidth=2,color=is2_color,zorder=3)[0])
                cs2_full.append(m.plot(0, 0, linewidth=2,color=cs2_color,zorder=3)[0])
                cs2_seg.append(m.plot(0, 0, linewidth=2 ,color=common_color,zorder=3)[0])
            #is2_sat = m.plot(0, 0, markersize=4,marker='8',color=is2_color)[0]
            #cs2_sat = m.plot(0, 0, markersize=4,marker='8',color=cs2_color)[0]
            #return is2_full,is2_seg,cs2_full,cs2_seg,is2_sat,cs2_sat

            
            #is2_full,is2_seg,cs2_full,cs2_seg,is2_sat,cs2_sat = init_sat_track(ntrack)
            #fig.canvas.blit() # or draw()
            #fig.canvas.start_event_loop(20) #1 ms seems enough


            #print("start")
            #plt.pause(10)
            #print("end")

        #return None

        # INDENTc
        """
        fig.canvas.mpl_connect('button_press_event', onClick)
        anim = animation.FuncAnimation(fig, animate, frames=full_frame, interval=interval,blit=False,repeat=False)
        return anim
        """

    #anim = run_animation()
    
    
        
            
    # call the animator.  blit=True means only re-draw the parts that have changed.
    #fig.canvas.mpl_connect('button_press_event', onClick)
    anim = animation.FuncAnimation(fig, animate, frames=full_frame, interval=interval,blit=False,repeat=False)
    
    if show_plot: plt.show()
    
    # Set up formatting for the movie files
    outfolder = PATH_OUT + inputfolder
    if not os.path.exists(outfolder):
        os.makedirs(outfolder)
    outfile = outfolder + outfilename+'.mp4'
    outimage = outfolder + outfilename+'.png'
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=60, metadata=dict(artist='Me'), bitrate=1800)
    anim.save(outfile, writer=writer)
    fig.savefig(outimage)
    print("\n Writing file: %s \n" %(outfile))

    #anim.close()
    

    
