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

COMMENTS:

    - Only one product at once  

"""

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
import common_functions as cf
import warnings
import scipy.spatial
import pickle
from scipy.stats import pearsonr, gaussian_kde,linregress
import matplotlib as mpl
import os
import stats_tools as st

# Global attributs
###########################################

varHome = os.environ['HOME']

PATH_DATA = path_dict.PATH_DICT[varHome]['PATH_DATA']
PATH_INPUT = path_dict.PATH_DICT[varHome]['PATH_OUT']


param_opts = ['surface_classif','sla','ish','freeboard','roughness','xings','penetration','wvf','corrs','snow_depth','ocean','regions','sd_map','tracks_map']

beamList=['gt1r','gt2r','gt3r'] #,'gt1l','gt2l','gt3l']
MAX_ACROSS_DIST = 1.5 #KM
MAX_NADIR_DIST = 0.1 #KM
colors_scatter = ['mediumseagreen','cornflowerblue','red']
colors_histo = ['mediumseagreen','cornflowerblue','royalblue','dodgerblue','navy','turquoise']
color_line_histo =  ['mediumseagreen','cornflowerblue','royalblue','dodgerblue','navy','turquoise','dodgerblue']
colors_plot_cs2 = ['deepskyblue','dodgerblue','turquoise','royalblue','palegreen','cornflowerblue','royalblue']


###########################################
#
#              Functions
#
###########################################

def plot_scatter(ax,xylim,sat,units,x_data,x_label,y_data,y_label,icetype,color='cornflowerblue'):

    # Scatter
    x = ma.masked_where(np.isnan(x_data),x_data,copy=True)
    y = ma.masked_where(np.isnan(y_data),y_data,copy=True)
    mask_data = np.logical_and(~x.mask,~y.mask)
    
    data = np.hstack((x[mask_data].reshape(x[mask_data].size,1),y[mask_data].reshape(y[mask_data].size,1)))
    z = gaussian_kde(np.transpose(data))(np.transpose(data))
    if icetype is None:
        #ax.scatter(x_data,y_data, s=50, edgecolor='',marker='.',edgecolors='black',)
        ax.scatter(x[mask_data],y[mask_data], s=8,c=z,marker='.')
        #ax.scatter(x_data,y_data, s=8,c=color,marker='.')
    else:
        ax.scatter(x_data[icetype==4],y_data[icetype==4], s=5,marker='o',c='black',label='MYI')
        ax.scatter(x_data[icetype==2],y_data[icetype==2], s=5,marker='o',c='gray',label='FYI')

    if xylim==None:
        xylim = list()
        xylim.append([np.ma.min(x_data),np.ma.max(x_data)])
        xylim.append([np.ma.min(y_data),np.ma.max(y_data)])
        

    start_biss = min(xylim[0][0],xylim[1][0])
    end_biss = max(xylim[0][1],xylim[1][1])
    ax.plot([start_biss,end_biss], [start_biss,end_biss], color = 'black', linestyle = 'dashed')
    ax.grid()
    axes = plt.gca()
    props = dict(boxstyle='round', facecolor='white', alpha=0.8)
    #color_map  =  plt.cm.jet
    ax.set_xlim((xylim[0][0],xylim[0][1]))
    ax.set_ylim((xylim[1][0],xylim[1][1]))
    unit_ord = xylim[1][1]-xylim[1][0]

    ax.set_ylabel(y_label, fontsize=12)
    ax.set_xlabel(x_label, fontsize=12)
    

    # slope
    #------------------------
    x = ma.masked_where(np.isnan(x_data),x_data,copy=True)
    y = ma.masked_where(np.isnan(y_data),y_data,copy=True)
    mask_data = np.logical_and(~x.mask,~y.mask)
    x = x_data[mask_data]
    y = y_data[mask_data]
    slope, intercept, r_value, p_value, std_err = linregress(x,y)
    ax.plot(x,slope*x + intercept, label="Slope %.2f" %(slope),color = 'black', linestyle = 'dashed')
    ax.legend(loc="lower right",fontsize=12)

    #ax.rcParams.update({'font.size': 20})

    # statistics
    
    
    nb_data = np.sum(mask_data)
    mean_x = np.ma.mean(x)
    mean_y = np.ma.mean(y)
    min_x =  np.ma.min(x); max_x =  np.ma.max(x)
    min_y =  np.ma.min(y); max_y =  np.ma.max(y)
    mean_bias = np.ma.mean(y-x)
    R = pearsonr(x,y)[0]
    RMSD = np.sqrt((1/(nb_data-1))*np.sum((y - x)**2)) #*100
    
    textstr = '\n'.join(('%s' %(sat),'Bias= %.2f %s' % (mean_bias,units),'RMSD= %.2f %s' % (RMSD,units),'R= %.2f' % (R),'Npts = %i' %(nb_data), 'Min/Max px = %.1f/%.1f' %(min_x,max_x), 'Min/Max py = %.1f/%.1f' %(min_y,max_y)))
    props = dict(boxstyle='round', facecolor='white', alpha=0.8)
    ax.text(xylim[0][1]-0.005,xylim[1][1] , textstr, fontsize=10,verticalalignment='top',horizontalalignment ='right', bbox=props)
    
    #else:
    if icetype is not None:
        for n,type_str in enumerate(['FYI','MYI']):
            # statistics
            flag_ice_type = icetype==2*(n+1) # [2,4]
            x = ma.masked_where(np.isnan(x_data[flag_ice_type]),x_data[flag_ice_type],copy=True)
            y = ma.masked_where(np.isnan(y_data[flag_ice_type]),y_data[flag_ice_type],copy=True)
            mask_data = np.logical_and(~x.mask,~y.mask)
            nb_data = np.sum(mask_data)
            if nb_data==0: continue
            mean_x = np.ma.mean(x)
            mean_y = np.ma.mean(y)
            mean_bias = np.ma.mean(y-x)
            R = pearsonr(x[mask_data],y[mask_data])[0]
            RMSD = np.sqrt((1/(nb_data-1))*np.ma.sum((y - x)**2))*100

            textstr = '\n'.join((type_str,'Bias= %.2f %s' % (mean_bias,units),'RMSD= %.2f %s' % (RMSD,units),'R = %.2f' %(R),'Npts = %i' %(nb_data)))
            props = dict(boxstyle='round', facecolor='white', alpha=0.7)
            ax.text(xylim[0][1]-0.005,xylim[1][0]+(n)*0.25*unit_ord , textstr, fontsize=9,verticalalignment='bottom',horizontalalignment ='right', bbox=props)
    
    #ax.set_aspect('equal') #, adjustable='box')
    #set_axes_equal(ax)

    #plt.show()


def plot_histo(ax,xylim,units,xlabel,legend_list,data_list,flag_commun_mask=False):

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

    if xylim is None:xylim = [0,0]
    
    if flag_commun_mask==True:
        commun_mask = ~data_list[0].mask
        print("Histo show only common pts")
        
        # defining common mask
        for nd,data in enumerate(data_list):
            commun_mask = np.logical_and(commun_mask,~data_list[nd].mask)
            if xylim is None:
                xylim = [min(xylim[0],np.ma.min(data)),max(xylim[1],np.ma.max(x_data))]
            
        # applying common mask
        for nd,data in enumerate(data_list):
            data = ma.masked_where(~commun_mask,data,copy=True)
            data_list[nd] = data
    else:
        common_mask = np.ones(data_list[0].size)

    
    
    
    for n,(data,label) in enumerate(zip(data_list,legend_list)):
        #ax.hist(data[~data.mask], 100, range=[xylim[0],xylim[1]], histtype='step',alpha=0.1,linewidth=1.5,fill=True,label=label,color=colors_histo[n],edgecolor='black')
        h = ax.hist(data[~data.mask], 50, range=[xylim[0],xylim[1]],lw=1,histtype=u'step', facecolor="None",edgecolor=colors_histo[n],label=label)
        if n==0: max_h = np.max(h[0])
        #data = ma.masked_where(np.isnan(data),data,copy=True) #[common_mask].
        Npts = np.sum(~data.mask)
        print("Npts(%s) = %i" %(label,Npts))
        mean_data = np.ma.mean(data)
        std_data = np.ma.std(data)
        print(mean_data)
        ax.axvline(x=mean_data,color=color_line_histo[n],lw=1)
        ypos = (n+1)/(2*(len(data_list)+1))*max_h
        ax.annotate('Mean = %.3f%s, std=  %.3f%s' %(mean_data,units,std_data,units),xy=(mean_data, ypos),weight='bold',fontsize=10,color=colors_histo[n])

    textstr = "Npts = %i" %(Npts)
    props = dict(boxstyle='round', facecolor='white', alpha=0.8)
    ax.text(xylim[0]+0.005,1 , textstr, fontsize=12,verticalalignment='bottom',horizontalalignment ='left', bbox=props)
    ax.set_xlabel(xlabel,fontsize=12)
    ax.set_ylabel('density',fontsize=12)
    ax.legend()



    
def plot_track_map(fig,axm,lon,lat,data,label,xylim,date_icetype,units,flag_comp=True,alpha=1):


    """
    ax: axis
    lon_list: list of longitude to plot
    lat_list: list of latitude to plot
    data: list of corresponding data
    label: str of label of the data
    xylim: limits of color bar
    date_icetype: datetime object for ice type

    """

    boundinglat = min(np.ma.min(lat),70)
    m = Basemap(projection='npstere',boundinglat=boundinglat,lon_0=0, resolution='l' , round=False,ax=axm)
    m.drawcoastlines(linewidth=0.25, zorder=0)
    m.drawparallels(np.arange(90,-90,-5), linewidth = 0.25, zorder=0)
    m.drawmeridians(np.arange(-180.,180.,30.), latmax=85, linewidth = 0.25, zorder=0)
    m.fillcontinents(color='0.9',lake_color='grey', zorder=-2)
    m.bluemarble(scale=1, zorder=-1)

    # defining color map
    if flag_comp:
        cmap='bwr'
        if xylim is None:
            #xylim = [-np.max(np.abs(data)),np.max(np.abs(data))]            
            xylim = [-np.std(np.abs(data)),np.std(np.abs(data))]            
    else:
        cmap='jet'
        if xylim is None:
            xylim = [np.min(data),np.max(data)]

    # defining boundaries
    if xylim is None:
        xylim

    # show ice_type
    if date_icetype:
        print("\n Show OSISAF for %s" %(date_icetype))
        lons,lats,OSISAF_ice_type = cf.get_osisaf_ice_type(date_icetype.year,date_icetype.month,date_icetype.day,'01')
        OSISAF_ice_type[OSISAF_ice_type==1] = ma.masked # masked ocean
        OSISAF_ice_type[OSISAF_ice_type==3] = 4 # ambigous becomes multi-year ice
        OSISAF_ice_type[560,381] = 4 # to display colormap
        OSISAF_ice_type[560,380] = 2
        #lons_icetype, lats_icetype = coord_icetype
        xptsT, yptsT = m(lons, lats)
    
        cmap_ice = mpl.colors.ListedColormap(["white", "lightgrey"])
        im = m.contourf(xptsT , yptsT, OSISAF_ice_type,linewidths=0.5,cmap=cmap_ice, alpha=1,zorder=0)
        norm = mpl.colors.BoundaryNorm(np.arange(2,4), cmap_ice.N)
        cbar = fig.colorbar(im,ax=axm,ticks=[2.5,3.5],orientation='horizontal',fraction=0.046, pad=0.04,extend='both')
        cbar.ax.set_xticklabels(['First Year Ice','Multi-Year Ice'])
        cbar.set_label('OSISAF daily ice type', labelpad=3)

    x,y = m(lon,lat)
    # show coordinates
    #for data in data:
    ndata = data.size - np.sum(data.mask)
    
    scat= m.scatter(x,y,c=data,s=3,cmap=cmap,vmin=xylim[0],vmax=xylim[1],zorder=2,alpha=alpha)

    cb = fig.colorbar(scat, ax=axm,extend='both',fraction=0.046, pad=0.04,shrink=0.80)
    cb.set_label("%s [%s]" %(label,units),fontsize=12)

    # text box
    textstr ='%5s=%.2f %s %5s=%.2f %s %5s=%i' %('mean',np.ma.mean(data),units,r'$\sigma$',np.ma.std(data),units,'Npts',ndata)
    #axm.text(1.5E6,5E5,textstr,fontsize=12,bbox=dict(boxstyle="round",ec=(0., 0., 0.),fc=(1., 1., 1.)))
    #axm.text(0.05, 0.05,textstr, transform=axm.transAxes,fontsize=12,bbox=dict(boxstyle="round",ec=(0., 0., 0.),fc=(1., 1., 1.)))

    return m,cmap


def add_data_track(bmap,cmap,lon,lat,data,xylim):

    x,y = bmap(lon,lat)
    if data is not None:
        scat= bmap.scatter(x,y,c=data,s=5,marker='o',cmap=cmap,vmin=xylim[0],vmax=xylim[1],zorder=2,alpha=1)
        #cb = fig.colorbar(scat, ax=axm,extend='both',fraction=0.046, pad=0.04,shrink=0.80)
        #cb.set_label("%s [%s]" %(label,units),fontsize=12)
    else:
        bmap.plot(x,y, linewidth=1.5, color='yellow',linestyle='--')


# ----------------------------------------------------------------------
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


def signif_psd(probability,dof):

   from scipy.stats import chi2

   alpha = 1.-probability
   v = 2.*dof
   c = chi2.ppf([1. - alpha / 2., alpha / 2.], v)
   c = v/c
   return c




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

    parser.add_argument("-o","--outpath",default=PATH_OUT,help="[optionnal] provide outpath")

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
    found_dates = [d for d in requested_dates if d in available_dates]
    if len(found_dates)==0: print('Unavailable dates',date,'\nAvailable dates are:',[d.strftime("%Y%m%d") for d in available_dates]);sys.exit()
    ndates = len(found_dates)
    idx_dates = np.array([available_dates.index(date) for date in found_dates])
    month_list = list(np.unique([d.strftime('%Y%m') for d in found_dates]))

    idx_dates_monthly = dict()
    for month in month_list:
        idx_dates_monthly[month] = np.array([available_dates.index(date) for date in found_dates if date.strftime('%Y%m')==month])

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


    # Get Osisaf Ice Type
    #--------------------------------------------------------
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
        
        


    # Summarizing required analysis
    # --------------------------------------------------------
    print("\nStatistical analysis of %s between %s-%s for" %(prod_L2P,start_date,end_date),param)

    # Statistical analyses
    #---------------------------------------------------------
    if param=='surface_classif':
        print("\nSurface classification comparison")

        #offnadir_dist = cf.off_nadir_distance(np.arange(0,0.5,0.01),730000,70)
        offnadir_dist = cf.off_nadir_distance(0.02,730000,70)

        
        print("stop")
        """
        dist = data_dict['IS2']['ATL07']['dist'][1]
        fig, (ax1,ax2)= plt.subplots(2,1) #4sharex=True)
        ax1.plot(dist[0,:])
        im = ax2.imshow(dist)
        cbar = plt.colorbar(im,orientation='horizontal',fraction=0.046, pad=0.04,extend='both',cax=ax2)
        cbar.set_label('distance to CS2 [km]')
        plt.show()

        plt.plot(dist[0,:])
        plt.ylabel("distance to CS2 [km]")
        plt.show()

        plt.imshow(dist)
        cbar = plt.colorbar(orientation='horizontal',fraction=0.046, pad=0.04,extend='both')
        cbar.set_label('distance to CS2 [km]')
        plt.show()
        

        is2_surf_type = data_dict['IS2']['ATL07']['surface_type']
        is2_flag_lead = data_dict['IS2']['ATL07']['flag_leads']
        plt.imshow(is2_surf_type)
        cbar = plt.colorbar(orientation='horizontal',fraction=0.046, pad=0.04,extend='both')
        cbar.set_label('Surface type')
        #plt.show()
        """

        #fig = plt.figure()
        is2_beam = np.ma.concatenate(data_dict['IS2']['ATL07']['beam'],axis=1)
        plt.imshow(is2_beam)
        cbar = plt.colorbar(orientation='horizontal',ticks=[11,21,31],fraction=0.046, pad=0.04,extend='both')
        cbar.set_label('Beams')
        plt.show()
        
        dist = data_dict['IS2']['ATL07']['dist']

        cs2_surf_type = np.concatenate(data_dict['CS2']['ESA_BD']['surface_type'],axis=0)
        cs2_wvf = np.concatenate(data_dict['CS2']['ESA_BD_1B']['wvf'],axis=0)


        # show offnadir on wvf vs surf_type is2
        # figure over Arctic
        

        #def comp_surface_classif()

    if param=='xings':

        # global param
        show_colloc=True
        delay_path = 24 #hours
        to_sec = 60*60
        delay_inc = delay_path*to_sec

        
        # Get IS2 laser fb xings
        #-------------------------------
        laser_fb_xings = [obj for nobj,obj in enumerate(data_dict['IS2']['xings']['ATL10']['laser_fb']) if nobj in idx_dates]
        laser_fb_full_xings = np.ma.concatenate(laser_fb_xings,axis=1)
        npts = laser_fb_full_xings.shape[1]

        lat_is2xings =  np.ma.concatenate([obj for nobj,obj in enumerate(data_dict['IS2']['xings']['ATL10']['lat']) if nobj in idx_dates],axis=1)
        lon_is2xings =  np.ma.concatenate([obj for nobj,obj in enumerate(data_dict['IS2']['xings']['ATL10']['lon']) if nobj in idx_dates],axis=1)
        #lat_xings = np.ma.concatenate(lat_xings,axis=1)
        
        # Get IS2 laser fb colloc
        #-------------------------------
        #laser_fb = [obj for nobj,obj in enumerate(data_dict['IS2']['ATL10']['laser_fb']) if nobj in idx_dates]
        #laser_fb_full = np.ma.concatenate(laser_fb,axis=1)
        #laser_fb_colloc = np.ma.mean(laser_fb_full,axis=0)

        #delay = [obj for nobj,obj in enumerate(data_dict['IS2']['ATL07']['delay']) if nobj in idx_dates]
        #delay = np.ma.concatenate(delay,axis=1)
        #delay_colloc = np.abs(np.ma.mean(delay)*60)

        # with mean values
        laser_fb_colloc = np.ma.concatenate(list(np.array(data_dict['IS2']['ATL10']['laser_fb_mean'],dtype=object)[idx_dates]),axis=0)
        delay_colloc = 3.5*60*60
        delayC = np.ma.ones(laser_fb_colloc.shape)*delay_colloc

        

        
        

        print("Collocated data:")
        #print("Delay: Mean %i mn, Max %i mn, Min %i mn" %(np.ma.mean(delay),np.ma.min(delay),np.ma.max(delay)))

        
        # Get CS2 freeboard
        #---------------------------------
        lat_colloc_CS2 = np.ma.concatenate(list(np.array(data_dict['CS2'][prod_L2P[0]]['lat'],dtype=object)[idx_dates]),axis=0)
        lon_colloc_CS2 = np.ma.concatenate(list(np.array(data_dict['CS2'][prod_L2P[0]]['lon'],dtype=object)[idx_dates]),axis=0)
        radar_fb_colloc = np.ma.concatenate(list(np.array(data_dict['CS2'][prod_L2P[0]]['radar_fb'],dtype=object)[idx_dates]),axis=0)
        radar_fb_colloc = ma.masked_where(radar_fb_colloc==0.0,radar_fb_colloc,copy=True)
        
        """
        radar_fb_list = list()
        radar_fb_full = list()
        radar_fb_matrix = ma.masked_array(np.zeros((len(prod_L2P),npts)),mask=np.ones((len(prod_L2P),npts)),dtype='float')
        for nprod,cs2_gdr in enumerate(prod_L2P):
            radar_fb = list(np.array(data_dict['CS2'][cs2_gdr]['radar_fb'],dtype=object)[idx_dates])
            
            radar_fb_list.append(radar_fb)
            fb_full = ma.masked_invalid(np.ma.concatenate(radar_fb,axis=0))
            radar_fb_full.append(fb_full)
            radar_fb_matrix[nprod,:] = fb_full
        
        radar_fb_mean = np.ma.mean(radar_fb_matrix,axis=0)
        radar_fb_std = np.ma.std(radar_fb_matrix,axis=0)
        """

        # onlys for ESA_BD_GDR
        #----------------------
        radar_fb_xings = [obj for nobj,obj in enumerate(data_dict['CS2']['xings']['ESA_BD_GDR']['radar_fb']) if nobj in idx_dates]
        radar_fb_full_xings = np.ma.concatenate(radar_fb_xings,axis=1)
        
        lat_cs2xings = np.ma.concatenate([obj for nobj,obj in enumerate(data_dict['CS2']['xings']['ESA_BD_GDR']['lat']) if nobj in idx_dates],axis=1)
        lon_cs2xings = np.ma.concatenate([obj for nobj,obj in enumerate(data_dict['CS2']['xings']['ESA_BD_GDR']['lon']) if nobj in idx_dates],axis=1)

        
        #npts = laser_fb_full_xings.

        # delta freeboard for collocated tracks
        delta_fb_colloc = laser_fb_colloc- radar_fb_colloc


        dtime_is2 = [obj for nobj,obj in enumerate(data_dict['IS2']['xings']['ATL10']['delay']) if nobj in idx_dates]
        dtime = np.ma.concatenate(dtime_is2,axis=1)

        std_delta_fb = list()
        mean_delta_fb = list()
        dt = list()
        ndata = list()

        ndata_colloc = list()
        std_delta_fb_colloc = list()
        mean_delta_fb_colloc = list()
        
        delay_0 = 0
        delay=delay_inc
        
        while delay<np.ma.max(dtime):

            idx_time = np.abs(dtime.data)>delay
            #print(delay_0)
            #print(delay)
            #idx_time = np.logical_or(delay_0>np.abs(dtime),np.abs(dtime)>delay)
            #idx_time[dtime.mask] = False

            dt.append(delay/(60*60*24)) #in days
            print("Dtime %.1f days" %(delay/(60*60*24)))

            
            # Xings from IS2
            #----------------------------------------------
            laser_fb_xings = np.ma.masked_where(idx_time,laser_fb_full_xings,copy=True)
            ndata_is2_xings = np.sum(~laser_fb_xings.mask)
            print('ndata_is2_xings',ndata_is2_xings)

             # Get mean value per beams
            laser_fb = np.ma.mean(laser_fb_xings,axis=0)

            # delta freeboard
            delta_fb_is2xings = laser_fb - radar_fb_colloc
            #delta_fb = laser_fb - radar_fb_mean

            # Xings from CS2
            #--------------------------------------------
            
            radar_fb_xings = np.ma.masked_where(idx_time,radar_fb_full_xings,copy=True)
            ndata_cs2_xings = np.sum(~radar_fb_xings.mask)
            print('ndata_cs2_xings',ndata_cs2_xings)

            radar_fb = np.ma.mean(radar_fb_xings,axis=0)

            delta_fb_cs2xings = laser_fb_colloc - radar_fb

            delta_fb = np.ma.concatenate((delta_fb_cs2xings,delta_fb_is2xings))
            
            
            #lat_is2 =  np.ma.masked_where(idx_time,lat_xings,copy=True)
            
            #latis2 = np.ma.mean(lat_is2,axis=0)
            #plt.plot(latis2)
            #plt.plot(lat_CS2)
            #plt.show()

            # n IS2 data 
            
            #ndata_xings = np.sum(~delta_fb.mask)
            #print('ndata_xings',ndata_xings)

            if delay > delay_colloc:
                delta_fb_colloc_all = np.ma.concatenate((delta_fb,delta_fb_colloc))
                print('ndata_colloc',np.sum(~delta_fb_colloc_all.mask))
                ndata_colloc.append(np.sum(~delta_fb_colloc_all.mask))
                std_delta_fb_colloc.append(np.ma.std(delta_fb_colloc_all))
                mean_delta_fb_colloc.append(np.ma.mean(delta_fb_colloc_all))
                
            
            
            #if show_colloc==True and delay > delay_colloc:
            #if show_colloc==True and delay > delay_colloc and delay_0 < delay_colloc:
            #    delta_fb = np.ma.concatenate((delta_fb,delta_fb_colloc))

            # n CS2 beam has cross-over data
            #flag_MYI = delta_fb>0.5
            #delta_fb = ma.masked_where(flag_MYI,delta_fb,copy=True)
            
            nbeams = np.sum(~delta_fb.mask)
            print('ndata_xings',nbeams)
            ndata.append(nbeams)
            
            if nbeams==0:
                std_delta_fb.append(np.nan)
                mean_delta_fb.append(np.nan)
            else:
                std_delta_fb.append(np.ma.std(delta_fb))
                mean_delta_fb.append(np.ma.mean(delta_fb))

            # show XO
            #------------------------
            #dataones = np.ones(lon_cs2xings.shape)
            #dataones = np.ma.masked_where(idx_time,dataones,copy=True)
            #dataones = np.ma.mean(dataones,axis=0)

            lon_cs2x = np.ma.mean(np.ma.masked_where(idx_time,lon_cs2xings,copy=True),axis=0)
            lat_cs2x = np.ma.mean(np.ma.masked_where(idx_time,lat_cs2xings,copy=True),axis=0)

            lon_is2x = np.ma.mean(np.ma.masked_where(idx_time,lon_is2xings,copy=True),axis=0)
            lat_is2x = np.ma.mean(np.ma.masked_where(idx_time,lat_is2xings,copy=True),axis=0)

            lat_xings = np.ma.concatenate((lat_cs2x,lat_is2x))
            lon_xings = np.ma.concatenate((lon_cs2x,lon_is2x))
             

            """
            f1, ax = plt.subplots(1, 1,figsize=(9,8))
            bmap,cmap = st.plot_track_map(f1,ax,lon_xings,lat_xings,delta_fb,'',[0,0.5],mid_date,'m',False,alpha=1,size=5)
        
            plt.show()
            """
                
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
        bmap,cmap = st.plot_track_map(f2,ax,lon_colloc_CS2,lat_colloc_CS2,delta_fb_colloc,'',[0,0.5],mid_date,'m',False,alpha=1,size=5)
        
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
        ax2.plot(dt,ndata,label="ndata",linestyle='--',color=palette(1))
        ax2.plot(dt,ndata_colloc,label="ndata[colloc]",linestyle='--',color=palette(2))
        ax2.set_ylabel('ndata')
        ax.legend()

        plt.show()
        
    
        

        
    if param=='snow_depth':

        # Get parameters
        #------------------------------------------

        # ref coordinates
        ref_seg_lat = list(np.array(data_dict['CS2'][REF_GDR]['latref'],dtype=object)[idx_dates])
        ref_seg_lon = list(np.array(data_dict['CS2'][REF_GDR]['lonref'],dtype=object)[idx_dates])

        ref_seg_time = list(np.array(data_dict['CS2'][REF_GDR]['time'],dtype=object)[idx_dates])


        lat_cs2 = np.ma.concatenate(ref_seg_lat,axis=0)
        lon_cs2 = np.ma.concatenate(ref_seg_lon,axis=0)
        time_cs2 = np.ma.concatenate(ref_seg_time,axis=0)

        # get distance step
        lat1 = np.ma.concatenate(ref_seg_lat,axis=0)[:-1]; lat2=np.ma.concatenate(ref_seg_lat,axis=0)[1:]
        lon1 = np.ma.concatenate(ref_seg_lon,axis=0)[:-1]; lon2=np.ma.concatenate(ref_seg_lon,axis=0)[1:]
        mean_dist_btw_data = np.median(cf.dist_btw_two_coords(lat1,lat2,lon1,lon2))
        window_size = int(mean_dist_btw_data*25)

        # x_dist ref
        x_dist = list()
        for lat,lon in zip(ref_seg_lat,ref_seg_lon):
            x_dist.append(cf.distance_from_first_trk_pts(lat,lon,0))
        x_dist_full = np.ma.concatenate(x_dist,axis=0)
        x_dist_sum = np.cumsum(x_dist_full)
        npts = x_dist_sum.shape[0]
        
        # Get icetype along-track
        icetype_al = list()
        for n in range(ndates):
            lon = ref_seg_lon[n]
            if any(np.abs(np.diff(lon)) > 20): lon[lon > 180] = lon[lon > 180] - 360   
            icetype_alongtrack = cf.grid_to_track(icetype[n],lons_icetype[n],lats_icetype[n],lon,ref_seg_lat[n])
            icetype_al.append(icetype_alongtrack)
        icetype_al_full = np.ma.concatenate(icetype_al,axis=0)

        # Slow down factor
        ds = 0.300
        ns = (1 + 0.51*ds)**(-1.5)

        # Get mean CS2 freeboard
        #radar_fb_list = list()
        radar_fb_full = list()
        radar_fb_matrix = ma.masked_array(np.zeros((len(prod_L2P),npts)),mask=np.ones((len(prod_L2P),npts)),dtype='float')
        for nprod,cs2_gdr in enumerate(prod_L2P):
            radar_fb = list(np.array(data_dict['CS2'][cs2_gdr]['radar_fb'],dtype=object)[idx_dates])
            #radar_fb_list.append(radar_fb)
            fb_full = ma.masked_invalid(np.ma.concatenate(radar_fb,axis=0))
            radar_fb_full.append(fb_full)
            radar_fb_matrix[nprod,:] = fb_full
        
        radar_fb_mean = np.ma.mean(radar_fb_matrix,axis=0)
        radar_fb_std = np.ma.std(radar_fb_matrix,axis=0)
        #radar_fb_spline =  rolling_stats(radar_fb_mean, 100, stats=['mean'])[0]

        # Get IS2 freeboard
        laser_fb = [obj for nobj,obj in enumerate(data_dict['IS2']['ATL10']['laser_fb']) if nobj in idx_dates]
        laser_fb = np.ma.concatenate(laser_fb,axis=1)
        laser_fb_full_ave = np.ma.mean(laser_fb,axis=0)
        laser_fb_full_std = np.ma.std(laser_fb,axis=0)
        #laser_fb_spline = rolling_stats(laser_fb_full_ave, 100, stats=['mean'])[0]
            
        #delta_fb_LaSar = laser_fb_full_ave - np.ma.concatenate(radar_fb_list[1],axis=0)
        delta_fb_LaSar = laser_fb_full_ave - radar_fb_mean
        snow_depth = delta_fb_LaSar*ns
        snow_depth = ma.masked_invalid(snow_depth)

        # dates
        datelist = list(np.array(data_dict['dates'],dtype=object)[idx_dates])
        date_period_str = [datelist[0].strftime('%d/%m/%Y'),datelist[-1].strftime('%d/%m/%Y')]


        
        #-------------------------------
        # Map snow depth
        #-------------------------------
        
        f1, ax = plt.subplots(1, 1,figsize=(6,6))
        f1.suptitle('Snow depth LaKu from ESA_BD from %s-%s' %(date_period_str[0],date_period_str[-1]), fontsize=12)
        lat = np.ma.concatenate(ref_seg_lat,axis=0)
        lon = np.ma.concatenate(ref_seg_lon,axis=0)
        plot_track_map(f1,ax,lon,lat,snow_depth,'Snow depth LaKu',[0.,0.3],mid_date,'m',False)
        plt.show()

        
        """
        month_list = list()
        month0 = found_dates[0].month
        for idx in idx_dates:
            factor = found_dates[idx].month - month0
            id_month = np.ma.ones(data_dict['CS2'][REF_GDR]['latref'][idx].size)*factor
            month_list.append(id_month)

        id_month = np.ma.concatenate(month_list,axis=0)
        id_month = ma.masked_where(snow_depth.mask,id_month,copy=True)

        f1, ax = plt.subplots(1, 1,figsize=(6,6))
        f1.suptitle('Month ids from October with ESA_BD from %s-%s' %(date_period_str[0],date_period_str[-1]), fontsize=12)
        lat = np.ma.concatenate(ref_seg_lat,axis=0)
        lon = np.ma.concatenate(ref_seg_lon,axis=0)
        bmap,cmap = plot_track_map(f1,ax,lon,lat,id_month,'Month id',[0.,1],mid_date,'m',False,alpha=1)
        
        lon_82 = np.linspace(-180,180)
        lat_82 = np.linspace(82,82)
        x,y = bmap(lon_82,lat_82)
        bmap.plot(x,y, linewidth=1.5, color='black',linestyle='--')
        plt.show()
        """

        
        #-------------------------------
        # Xings with SIMBA
        #-------------------------------
        """
        #xidx,xcoords = xings_SIMBA(ref_seg_lat,ref_seg_lon,time)

        path_data = '/home/antlafe/Documents/work/data/SIMBA/'
        filepattern =path_data +'FMI*.dat'
        filename = glob.glob(filepattern)
        if len(filename)==0: sys.exit("\n%s: No found" %(filepattern))
        else:
            filename = filename[0]
            print("\nReading SIMBA file %s" %(filename))

        # Get data
        data = np.loadtxt(filename)
        lat_simba = data[:,5]
        lon_simba = data[:,6]

        time_simba = list()
        year = data[:,2].astype(int)
        month = data[:,1].astype(int)
        day = data[:,0].astype(int)

        start_date = datetime(year[0],month[0],day[0])
        end_date = datetime(year[-1],month[-1],day[-1])
    
        mid_date_simba = start_date + (end_date - start_date)/2
        
        for n in range(lat_simba.size):

            t = datetime(year[n],month[n],day[n],data[:,3].astype(int)[n],data[:,4].astype(int)[n])
            
            time_simba.append((t-datetime(2000,1,1)).total_seconds())
        time_simba = np.array(time_simba)

        from scipy.spatial import distance
        x,y,z = cf.lon_lat_to_cartesian(lon_simba, lat_simba)
        coord_simba = np.vstack((x,y,z)).T
        idx_simba = np.arange(lon_simba.size)

        x,y,z = cf.lon_lat_to_cartesian(lon_cs2, lat_cs2) # in km
        coord_cs2 = np.vstack((x,y,z)).T
        delay = 3 # days
        max_dist = 100 #km

        idx_buoys = list()
        deltaD_list = list()
        deltaT_list = list()
        idx_colloc = list()
        
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
                
                print("Found match: %i/%i/%i (%.2f,%.2f): Delay=%i sec, Dist=%i km " %(day[idx],month[idx],year[idx],lat_simba[idx],lon_simba[idx],deltaT_list[-1],deltaD_list[-1]))

        # transform lists into arrays
        idx_buoys = np.ma.array(idx_buoys)
        deltaD_list =  np.ma.array(deltaD_list)
        deltaT_list =  np.ma.array(deltaT_list)
        idx_colloc =  np.ma.array(idx_colloc)

        lon_colloc = lon_cs2[idx_colloc]
        lat_colloc = lat_cs2[idx_colloc]
        delay_colloc = deltaT_list/(60*60)
        
        id_month = np.ma.ones(lat_simba.size)
        # show map
        f1, ax = plt.subplots(1, 1,figsize=(6,6))
        f1.suptitle('Month ids from October with ESA_BD from %s-%s' %(date_period_str[0],date_period_str[-1]), fontsize=12)
        bmap,cmap = plot_track_map(f1,ax,lon_colloc,lat_colloc,delay_colloc,'delay [hours]',None,mid_date,'m',False,alpha=1)
        
        x,y = bmap(lon_simba,lat_simba)
        bmap.plot(x,y, linewidth=1.5, color='black',linestyle='-')
        plt.show()
        """
            

        print("stop")
            
        
        
        
        
        
        
        #---------------------------
        # Get Warren climatology
        #-------------------------
        
        from W99 import W99
        sd_w99 = list()
        for ndate,date in enumerate(found_dates):
            month = date.month

            # Option 1 
            year = date.year
            
            SD_W99,d_s = W99(month,year,ref_seg_lon[ndate],ref_seg_lat[ndate],0)
            SD_W99 = SD_W99/100 # cm to m
            SD_W99[icetype_al[ndate]==2]= 0.5*SD_W99[icetype_al[ndate]==2]

            # Option 2
            datestr= date.strftime('%Y%m')
            lat_grid,lon_grid,sd_grid = cf.get_W99(str(month))
            sd_grid = np.squeeze(sd_grid)
            
            #lon_grid,lat_grid = np.meshgrid(lon_ist,lat_ist)
        
            #sd_al = cf.grid_to_track(sd_grid,lon_grid,lat_grid,ref_seg_lon[ndate],ref_seg_lat[ndate])
            #sd_al[icetype_al[ndate]==2]= 0.5*sd_al[icetype_al[ndate]==2]
            #sd_al = ma.masked_where(icetype_al[ndate].mask,sd_al,copy=True)
            
            sd_w99.append(SD_W99)

        sd_w99_full = np.ma.concatenate(sd_w99,axis=0)
        
        
        # Delta fb vs W99 snow depth

        xylim = [[-0.05,0.4],[-0.05,0.4]]
        x_data =  sd_w99_full
        x_label = 'Snow depth Warren99 modified [m]'
        y_label= 'snow depth LaKu [m]'
        y_data = snow_depth
        x_data = ma.masked_where(np.isnan(x_data),x_data,copy=True)
        y_data = ma.masked_where(np.isnan(y_data),y_data,copy=True)
        mask_data = np.logical_and(~x_data.mask,~y_data.mask)


        # find smoothing radius
        #----------------------
        f16, ax = plt.subplots(1, 1,figsize=(6,6))
        f16.suptitle('Determination of smoothing radius W99', fontsize=12)

        R_list,RMSD_list,smoothmin = cf.find_smoothing_radius(x_data,y_data,mean_dist_btw_data,200,True)
        
        # scatter plot
        #--------------------
        nkm = smoothmin #km
        snow_depth_smooth = rolling_stats(snow_depth,int(nkm/mean_dist_btw_data), stats=['mean'])[0]
        snow_depth_smooth =  ma.masked_where(~mask_data,snow_depth_smooth,copy=True)
        
        f15, ax = plt.subplots(1, 1,figsize=(6,6))
        f15.suptitle('comparison with snow depth W99', fontsize=12)
        plot_scatter(ax,xylim,'ASD','m',x_data,x_label,snow_depth_smooth,y_label,None)

        # map
        #--------------------
        f1, ax = plt.subplots(1, 1,figsize=(10,10))
        bmap,cmap = plot_track_map(f1,ax,lon_grid,lat_grid,sd_grid,'Snow depth',[0.,0.5],mid_date,'m',False)
        lat = np.ma.concatenate(ref_seg_lat,axis=0)
        lon = np.ma.concatenate(ref_seg_lon,axis=0)
        add_data_track(bmap,cmap,lon,lat,snow_depth_smooth,[0.,0.5])
        #plot_track_map(f1,ax,lon,lat,snow_depth,'Snow depth',[0.,0.5],None,'m',False)
        plt.show()
        
        
        #------------------------------
        # Get ASD
        #-----------------------------
        
        sd_ASD = list()
        pixsize = 25
        months = [date.strftime('%Y%m') for date in found_dates]
        for ndate,date in enumerate(found_dates):

            lat_grid,lon_grid,sd_grid = cf.get_ASD(pixsize,months[ndate])
            sd_grid = np.squeeze(sd_grid)
            
            #lon_grid,lat_grid = np.meshgrid(lon_ist,lat_ist)
            sd_al = cf.grid_to_track(sd_grid,lon_grid,lat_grid,ref_seg_lon[ndate],ref_seg_lat[ndate])
            sd_ASD.append(sd_al)
        sd_ASD_full = np.ma.concatenate(sd_ASD,axis=0)

        
        # Delta fb vs W99 snow depth

        xylim = [[-0.05,0.4],[-0.05,0.4]]
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
        f16, ax = plt.subplots(1, 1,figsize=(6,6))
        f16.suptitle('Determination of smoothing radius ASD', fontsize=12)

        R_list,RMSD_list,smoothmin = cf.find_smoothing_radius(x_data,y_data,mean_dist_btw_data,500,True)
        #plt.show()
       
        # scatter plot
        #--------------------
        nkm = 100 #km
        snow_depth_smooth = rolling_stats(snow_depth,int(nkm/mean_dist_btw_data), stats=['mean'])[0]
        snow_depth_smooth =  ma.masked_where(~mask_data,snow_depth_smooth,copy=True)
        
        f15, ax = plt.subplots(1, 1,figsize=(6,6))
        f15.suptitle('comparison with snow depth ASD', fontsize=12)
        plot_scatter(ax,xylim,'ASD','m',x_data,x_label,snow_depth_smooth,y_label,None)

        
        # map
        #--------------------
        f1, ax = plt.subplots(1, 1,figsize=(10,10))
        bmap,cmap = plot_track_map(f1,ax,lon_grid,lat_grid,sd_grid,'Snow depth',[0.,0.5],mid_date,'m',False)
        lat = np.ma.concatenate(ref_seg_lat,axis=0)
        lon = np.ma.concatenate(ref_seg_lon,axis=0)
        add_data_track(bmap,cmap,lon,lat,snow_depth_smooth,[0.,0.5])
        #plot_track_map(f1,ax,lon,lat,snow_depth,'Snow depth',[0.,0.5],None,'m',False)
        plt.show()

        #-------------------------
        # Get SD PIOMAS
        #-------------------------
        
        PIOMAS_SD_track = list()
        for ndate,date in enumerate(found_dates):
            lat_grid,lon_grid,sd_grid = cf.get_PIOMAS_SD(date)
            
            #lon_grid,lat_grid = np.meshgrid(lon_ist,lat_ist)
            sd_al = cf.grid_to_track(sd_grid,lon_grid,lat_grid,ref_seg_lon[ndate],ref_seg_lat[ndate])
            PIOMAS_SD_track.append(sd_al)
        PIOMAS_SD_full = np.ma.concatenate(PIOMAS_SD_track,axis=0)


        # Delta fb vs W99 snow depth
        xylim = [[-0.05,0.4],[-0.05,0.4]]
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
        f16, ax = plt.subplots(1, 1,figsize=(6,6))
        f16.suptitle('Determination of smoothing radius PIOMAS', fontsize=12)

        R_list,RMSD_list,smoothmin = cf.find_smoothing_radius(x_data,y_data,mean_dist_btw_data,500,True)
        #plt.show()
       
        # scatter plot
        #--------------------
        nkm = 70 #km
        snow_depth_smooth = rolling_stats(snow_depth,int(nkm/mean_dist_btw_data), stats=['mean'])[0]
        snow_depth_smooth =  ma.masked_where(~mask_data,snow_depth_smooth,copy=True)
        
        f15, ax = plt.subplots(1, 1,figsize=(6,6))
        f15.suptitle('comparison with snow depth PIOMAS', fontsize=12)
        plot_scatter(ax,xylim,'PIOMAS','m',x_data,x_label,snow_depth_smooth,y_label,None)

        
        # map
        #--------------------
        f1, ax = plt.subplots(1, 1,figsize=(10,10))
        bmap,cmap = plot_track_map(f1,ax,lon_grid,lat_grid,sd_grid,'Snow depth',[0.,0.5],mid_date,'m',False)
        lat = np.ma.concatenate(ref_seg_lat,axis=0)
        lon = np.ma.concatenate(ref_seg_lon,axis=0)
        add_data_track(bmap,cmap,lon,lat,snow_depth_smooth,[0.,0.5])
        #plot_track_map(f1,ax,lon,lat,snow_depth,'Snow depth',[0.,0.5],None,'m',False)
        plt.show()
        
        # Along-track
        #-------------------------------
        """
        for n,data in enumerate(data_list):
            xdist = mean_dist_btw_data*np.arange(data.size)
            plt.plot(xdist,data,label=legend_list[n])
        plt.xlabel('along-track dist [km]')
        plt.ylabel('Snow depth [m]')
        plt.grid()
        plt.legend()
        plt.show()
        """
        
        
        
        #-------------------------
        # Get SD AMSR
        #-------------------------
        
        SD_AMSR_al_full = list()
        for ndate,date in enumerate(found_dates):
            lat_grid,lon_grid,SD_AMSR = cf.get_SD_AMSR(date)
            SD_AMSR = SD_AMSR/100
            SD_AMSR_al = cf.grid_to_track(SD_AMSR,lon_grid,lat_grid,ref_seg_lon[ndate],ref_seg_lat[ndate])
            SD_AMSR_al_full.append(SD_AMSR_al)
        SD_AMSR_full = np.ma.concatenate(SD_AMSR_al_full,axis=0)

        # Delta fb vs AMSR snow depth
        xylim = [[-0.05,0.4],[-0.05,0.4]]
        x_data =  SD_AMSR_full
        x_label = 'Snow depth AMSR [m]'
        #y_label= r'$\Delta fb$'
        y_data = snow_depth
        y_label= 'snow depth LaKu [m]'
        x_data = ma.masked_where(np.isnan(x_data),x_data,copy=True)
        y_data = ma.masked_where(np.isnan(y_data),y_data,copy=True)
        mask_data = np.logical_and(~x_data.mask,~y_data.mask)

        
        # find smoothing radius
        #----------------------
        f16, ax = plt.subplots(1, 1,figsize=(6,6))
        f16.suptitle('Determination of smoothing radius AMSRE', fontsize=12)

        R_list,RMSD_list,smoothmin = cf.find_smoothing_radius(x_data,y_data,mean_dist_btw_data,500,True)
        #plt.show()
       
        # scatter plot
        #--------------------
        nkm = 50 #km
        snow_depth_smooth = rolling_stats(snow_depth,int(nkm/mean_dist_btw_data), stats=['mean'])[0]
        snow_depth_smooth =  ma.masked_where(~mask_data,snow_depth_smooth,copy=True)
        
        f15, ax = plt.subplots(1, 1,figsize=(6,6))
        f15.suptitle('comparison with snow depth AMSRE', fontsize=12)
        plot_scatter(ax,xylim,'AMSRE','m',x_data,x_label,snow_depth_smooth,y_label,None)

        
        # map
        #--------------------
        f1, ax = plt.subplots(1, 1,figsize=(10,10))
        bmap,cmap = plot_track_map(f1,ax,lon_grid,lat_grid,SD_AMSR,'Snow depth',[0.,0.5],mid_date,'m',False)
        lat = np.ma.concatenate(ref_seg_lat,axis=0)
        lon = np.ma.concatenate(ref_seg_lon,axis=0)
        add_data_track(bmap,cmap,lon,lat,snow_depth_smooth,[0.,0.5])
        #plot_track_map(f1,ax,lon,lat,snow_depth,'Snow depth',[0.,0.5],None,'m',False)
        plt.show()

        
        # map
        
        """
        f1, ax = plt.subplots(1, 1,figsize=(6,6))
        bmap,cmap = plot_track_map(f1,ax,lon,lat,SD_AMSR,'Snow depth',[0.,0.5],mid_date,'m',False)
        lat = np.ma.concatenate(ref_seg_lat,axis=0)
        lon = np.ma.concatenate(ref_seg_lon,axis=0)
        add_data_track(bmap,cmap,lon,lat,snow_depth,[0.,0.5])
        #plot_track_map(f1,ax,lon,lat,snow_depth,'Snow depth',[0.,0.5],None,'m',False)
        plt.show()
        
        f16, ax = plt.subplots(1, 1,figsize=(6,6))
        f16.suptitle('comparison with Snow Depth AMSR', fontsize=12)
        xylim = [[-0.1,0.5],[-0.1,0.5]]
        x_data =  SD_AMSR_full
        x_label = 'Snow Depth AMSR [m]'
        #y_label= r'$\Delta fb$'
        nkm = 25 #km
        snow_depth_smooth = rolling_stats(snow_depth,int(nkm/mean_dist_btw_data), stats=['mean'])[0]
        y_data = snow_depth_smooth #snow_depth_smooth
        y_label= 'Snow depth LaKu [m]'
        y_data = snow_depth #delta_fb_LaSar
        plot_scatter(ax,xylim,'m',x_data,x_label,y_data,y_label,icetype_al_full)
        
        plt.show()
        """

        # snow depth histogram
        data_list = [SD_AMSR_full,PIOMAS_SD_full,sd_w99_full,sd_ASD_full,snow_depth]
        legend_list = ['AMSR','PIOMAS','W99','ASD','KuLa']
        
        """
        f12, ax = plt.subplots(1, 1)
        f12.suptitle('Histogram snow depth products over collocated tracks', fontsize=12)
        #data_list = [SD_AMSR,sd_grid,SD_W99]
       
        xylim = [0,0.5]
        xlabel = 'Snow depth [m]'
        plot_histo(ax,xylim,'m',xlabel,legend_list,data_list,True)
        plt.show()
        """

       

        
        #-------------------------------
        # Spectral analysis
        #-------------------------------

        from scipy import signal

        plt.style.use('seaborn-darkgrid')
 
        # create a color palette
        palette = plt.get_cmap('Set1')

        freq_sampling = 3 #20hz data 1/7# for 1hz
        length = 1000 #pts 
        title = "Welsh spectral analysis (Lsamples=%i km, Nsamples=%i)" %(int(length/freq_sampling),data_list[0].size/length)
        
        for ndata,data in enumerate(data_list):
            mask = ~data.mask
            if not mask.size==1:
                data = data[mask]
        
            freq, Pxx = signal.welch(data,freq_sampling,nperseg=length,window=signal.tukey(length,alpha=0.5, sym=True),detrend='linear')
        
            freq=freq[1:-2]
            Pxx_den=Pxx[1:-2]
            Pxx=Pxx_den[np.where(1/freq <= ((1/freq_sampling)*length))]
            freq=freq[np.where(1/freq <= ((1/freq_sampling)*length))]

            nsamples = int(data.size/length)
            vals=signif_psd(0.95,nsamples-1) 

            plt.loglog(freq,Pxx,label=legend_list[ndata],color=palette(ndata))
            plt.fill_between(freq, Pxx*vals[0],Pxx*vals[1],color=palette(ndata),alpha=0.1)


        plt.title(title)
        plt.grid(True, which="both", ls="-")
        plt.xlabel('wavenumber (cpkm)')
        plt.ylabel('PSD (m2/cpkm)')
        plt.legend()
        plt.show()


        
        from scipy import signal
        sampling= 10 #m-1
        fs = 1 #sampling/(mean_dist_btw_data*1000) # DopplerB-1

        for n,data in enumerate(data_list):
            f, Pxx_den = signal.periodogram(data, fs)
            wavelength = 1/(f[1:])*mean_dist_btw_data*1000 #
            plt.semilogy(wavelength, Pxx_den[1:],label=legend_list[n])
        
        #plt.ylim([1e-7, 1e2])
        plt.legend()
        plt.xlabel('wavelength [m]')
        plt.ylabel('PSD')
        plt.show()

        from scipy.fft import fft, fftfreq

        # Number of samples in normalized_tone
        N = SAMPLE_RATE * DURATION

        for n,data in enumerate(data_list):
            yf = fft(data)
            xf = fftfreq(data.size, 1)
            wf = mean_dist_btw_data*(1/xf)
            plt.semilogy(wf, np.abs(yf),label=legend_list[n])
        plt.legend()
        plt.xlabel('wavelength [km]')
        plt.show()
        
        
        """
        # spectral analysis FFT
        from scipy import fftpack
        f_s = 1
        X = fftpack.fft(sd_w99_full)
        freqs = fftpack.fftfreq(len(sd_w99_full)) * f_s
        wavelength = (1/freqs)*mean_dist_btw_data

        fig, ax = plt.subplots()
        
        ax.stem(wavelength, np.abs(X))
        #ax.plot(wavelength, np.abs(X))
        ax.set_xlabel('Wavelength [Km]')
        ax.set_ylabel('Spectrum Magnitude')
        #ax.set_xlim(-f_s / 2, f_s / 2)
        #ax.set_ylim(-5, 110)
        plt.show()
        """

        """

        
        """
        
        # Spectral analysis
        """
        from scipy import signal
        sampling= 10 #m-1
        fs = 1 #sampling/(mean_dist_btw_data*1000) # DopplerB-1
        f, Pxx_den = signal.periodogram(sd_w99_full, 1)
        wavelength = 1/(f[1:])*mean_dist_btw_data*1000 # 
        plt.semilogy(wavelength, Pxx_den[1:][::-1])
        #plt.semilogy(f, Pxx_den)
        
        #plt.ylim([1e-7, 1e2])
        plt.xlabel('wavelength [m]')
        plt.ylabel('PSD')
        plt.show()
        """

    if param=='regions':


        # Get parameters
        #------------------------------------------
        ref_seg_lat = list(np.array(data_dict['CS2'][REF_GDR]['latref'],dtype=object)[idx_dates])
        ref_seg_lon = list(np.array(data_dict['CS2'][REF_GDR]['lonref'],dtype=object)[idx_dates])

       
     

        # Get icetype along-track
        icetype_al = list()
        for n in range(ndates):
            lon = ref_seg_lon[n]
            if any(np.abs(np.diff(lon)) > 20): lon[lon > 180] = lon[lon > 180] - 360   
            icetype_alongtrack = cf.grid_to_track(ice_type[n],lons_icetype[n],lats_icetype[n],lon,ref_seg_lat[n])
            icetype_al.append(icetype_alongtrack)
        icetype_al_full = np.ma.concatenate(icetype_al,axis=0)

        

        
        f2, ax = plt.subplots(1, 1,figsize=(6,6))
        plot_track_map(f2,ax,lon,lat,ssh,'DOT',None,mid_date,'m',False)
        
        
    if param=='ocean':
        
        # Get parameters
        #------------------------------------------
        # ref coordinates
        ref_seg_lat = list(np.array(data_dict['CS2'][REF_GDR]['latref'],dtype=object)[idx_dates])
        ref_seg_lon = list(np.array(data_dict['CS2'][REF_GDR]['lonref'],dtype=object)[idx_dates])

        print("stop")
        sla_cs2 = np.ma.concatenate(data_dict['CS2']['ESA_BD_GDR']['sla'],axis=1)
        sla_cs2 = np.ma.mean(sla_cs2,axis=0)

        radar_h_cs2 = np.ma.concatenate(data_dict['CS2']['ESA_BD_GDR']['radar_h'],axis=1)
        radar_h_cs2 = np.ma.mean(radar_h_cs2,axis=0)

        mss_cs2 = np.ma.concatenate(data_dict['CS2']['ESA_BD_GDR']['mss'],axis=0)

        ssh_cs2 = sla_cs2 + mss_cs2

        sic =  np.ma.concatenate(data_dict['CS2']['ESA_BD_GDR']['sic'],axis=0)
        flag_ocean = sic==0.0

        swh = np.ma.concatenate(data_dict['CS2']['ESA_BD_GDR']['swh'],axis=0)
        
        
        ssh_is2 =  [obj for nobj,obj in enumerate(data_dict['IS2']['ATL12']['ssh']) if nobj in idx_dates]
        ssh_is2 = np.ma.concatenate(ssh_is2,axis=1)
        ssh_is2 = np.ma.mean(ssh_is2,axis=0)
        delta_ssh = ssh_is2 - ssh_cs2

        swh_is2 = [obj for nobj,obj in enumerate(data_dict['IS2']['ATL12']['swh']) if nobj in idx_dates]
        swh_is2 = np.ma.concatenate(swh_is2,axis=1)
        swh_is2 = np.ma.mean(swh_is2,axis=0)

        # Map
        """
        f1, ax = plt.subplots(1, 1,figsize=(6,6))
        lat = np.ma.concatenate(ref_seg_lat,axis=0)
        lon = np.ma.concatenate(ref_seg_lon,axis=0)
        plot_track_map(f1,ax,lon,lat,delta_ssh,'delta SSH',[-0.3,0.3],mid_date,'m',True)
        plt.show()
        """

        f2, ax = plt.subplots(1, 1,figsize=(6,6))
        lat = list(); lon =  list(); ssh = list();mss = list()
        for b in beamList:
            lat.extend(data_dict['IS2']['ATL12'][b]['lat'])
            lon.extend(data_dict['IS2']['ATL12'][b]['lon'])
            ssh.extend(data_dict['IS2']['ATL12'][b]['ssh'])
            #mss.extend(data_dict['CS2']['ESA_BD_GDR']['mss'])
        lat = np.ma.concatenate(lat,axis=0)
        lon = np.ma.concatenate(lon,axis=0)
        ssh = np.ma.concatenate(ssh,axis=0)
        #mss = np.ma.concatenate(mss,axis=0)
        plot_track_map(f2,ax,lon,lat,ssh,'DOT',None,mid_date,'m',False)

        plt.show()
        
        #data_dict['CS2']['ESA_BD_GDR']['radar_fb']

    if param=='tracks_map':

        # ref coordinates
        ref_seg_lat = np.ma.concatenate(list(np.array(data_dict['CS2'][REF_GDR]['latref'],dtype=object)[idx_dates]),axis=0)
        ref_seg_lon = np.ma.concatenate(list(np.array(data_dict['CS2'][REF_GDR]['lonref'],dtype=object)[idx_dates]),axis=0)
        id_cs2 = np.ma.ones(ref_seg_lat.shape)*0

        

        f1, ax = plt.subplots(1, 1,figsize=(10,10))
        bmap,cmap = st.plot_track_map(f1,ax,ref_seg_lon,ref_seg_lat,id_cs2,'Beam id',[0.,5],mid_date,'',False,alpha=1)
        
        # IS2 beamwise data
        lat_is2 = {}
        lon_is2 = {}
        id_is2 = {}
        for nbeam,b in enumerate(['gt1r','gt2r','gt3r']):
            nbeam = nbeam+2
            lat_is2[b] = np.ma.concatenate(list(np.array(data_dict['IS2']['ATL10'][b]['lat'],dtype=object)[idx_dates]),axis=0)
            lon_is2[b] = np.ma.concatenate(list(np.array(data_dict['IS2']['ATL10'][b]['lon'],dtype=object)[idx_dates]),axis=0)
            id_is2[b] = np.ma.ones(lat_is2[b].shape)*nbeam


            st.add_data_track(bmap,cmap,lon_is2[b],lat_is2[b],id_is2[b],[0.,5])

        plt.show()
        
            

    if param=='penetration':

        # Get parameters
        #------------------------------------------

        # ref coordinates
        ref_seg_lat = list(np.array(data_dict['CS2'][REF_GDR]['latref'],dtype=object)[idx_dates])
        ref_seg_lon = list(np.array(data_dict['CS2'][REF_GDR]['lonref'],dtype=object)[idx_dates])

        # x_dist ref
        x_dist = list()
        for lat,lon in zip(ref_seg_lat,ref_seg_lon):
            x_dist.append(cf.distance_from_first_trk_pts(lat,lon,0))
        x_dist_full = np.ma.concatenate(x_dist,axis=0)
        x_dist_sum = np.cumsum(x_dist_full)
        npts = x_dist_sum.shape[0]
        
        # Get icetype along-track
        icetype_al = list()
        for n in range(ndates):
            lon = ref_seg_lon[n]
            if any(np.abs(np.diff(lon)) > 20): lon[lon > 180] = lon[lon > 180] - 360   
            icetype_alongtrack = cf.grid_to_track(ice_type[n],lons_icetype[n],lats_icetype[n],lon,ref_seg_lat[n])
            icetype_al.append(icetype_alongtrack)
        icetype_al_full = np.ma.concatenate(icetype_al,axis=0)

        # Slow down factor
        ds = 0.300
        ns = (1 + 0.51*ds)**(-1.5)

        # Get mean CS2 freeboard
        radar_fb_list = list()
        radar_fb_full = list()
        radar_fb_matrix = ma.masked_array(np.zeros((len(prod_L2P),npts)),mask=np.ones((len(prod_L2P),npts)),dtype='float')
        for nprod,cs2_gdr in enumerate(prod_L2P):
            radar_fb = list(np.array(data_dict['CS2'][cs2_gdr]['radar_fb'],dtype=object)[idx_dates])
            radar_fb_list.append(radar_fb)
            fb_full = ma.masked_invalid(np.ma.concatenate(radar_fb,axis=0))
            radar_fb_full.append(fb_full)
            radar_fb_matrix[nprod,:] = fb_full
        
        radar_fb_mean = np.ma.mean(radar_fb_matrix,axis=0)
        radar_fb_std = np.ma.std(radar_fb_matrix,axis=0)
        radar_fb_spline =  rolling_stats(radar_fb_mean, 100, stats=['mean'])[0]

        
        lat1 = np.ma.concatenate(ref_seg_lat,axis=0)[:-1]; lat2=np.ma.concatenate(ref_seg_lat,axis=0)[1:]
        lon1 = np.ma.concatenate(ref_seg_lon,axis=0)[:-1]; lon2=np.ma.concatenate(ref_seg_lon,axis=0)[1:]
        mean_dist_btw_data = np.median(cf.dist_btw_two_coords(lat1,lat2,lon1,lon2))
        window_size = int(mean_dist_btw_data*25)
        
        """
        # Get product freeboard, ssa, isa
        radar_fb = {}
        ssa = {}
        isa = {}
        for nprod,cs2_gdr in enumerate(prod_L2P):
            radar_fb_gdr = list(np.array(data_dict['CS2'][cs2_gdr]['radar_fb'],dtype=object)[idx_dates])
            radar_fb_gdr = ma.masked_invalid(np.ma.concatenate(radar_fb_gdr,axis=0))

            ssa_gdr = list(np.array(data_dict['CS2'][cs2_gdr]['sla'],dtype=object)[idx_dates])
            ssa_gdr = ma.masked_invalid(np.ma.concatenate(ssa_gdr,axis=0))

            isa_gdr = list(np.array(data_dict['CS2'][cs2_gdr]['isa'],dtype=object)[idx_dates])
            isa_gdr = ma.masked_invalid(np.ma.concatenate(isa_gdr,axis=0))
            
            radar_fb[cs2_gdr] = radar_fb_gdr
            ssa[cs2_gdr] = ssa_gdr
            isa[cs2_gdr] = isa_gdr
        """
        
        
        # Get IS2 freeboard
        laser_fb = [obj for nobj,obj in enumerate(data_dict['IS2']['ATL10']['laser_fb']) if nobj in idx_dates]
        laser_fb_full = np.ma.concatenate(laser_fb,axis=1)
        laser_fb_full_ave = np.ma.mean(laser_fb_full,axis=0)
        laser_fb_full_std = np.ma.std(laser_fb_full,axis=0)
        laser_fb_spline = rolling_stats(laser_fb_full_ave, 100, stats=['mean'])[0]


        #delta_fb_LaSar = laser_fb_full_ave - np.ma.concatenate(radar_fb_list[1],axis=0)
        delta_fb_LaSar = laser_fb_full_ave - radar_fb_mean

        
        # Get surface roughness
        gaussian_w = [obj for nobj,obj in enumerate(data_dict['IS2']['ATL10']['gaussian_w']) if nobj in idx_dates]
        gaussian_w = np.ma.concatenate(gaussian_w,axis=1)
        is2_flag_leads =  [obj for nobj,obj in enumerate(data_dict['IS2']['ATL10']['flag_leads']) if nobj in idx_dates]
        is2_flag_leads = np.ma.concatenate(is2_flag_leads,axis=1)
        is2_flag_leads = is2_flag_leads==0.0
        
        # Get gaussian width only floes
        gaussian_w_floes = np.ma.masked_where(is2_flag_leads.data==False,gaussian_w,copy=True)
        gaussian_w_floes_mean = np.ma.mean(gaussian_w_floes,axis=0)
        gaussian_w_floes_std  = np.ma.std(gaussian_w_floes,axis=0)
        
        surf_roughness = gaussian_w_floes_mean

        """
        f2, ax = plt.subplots(1, 1,figsize=(6,6))
        xylim = [[0.,0.6],[0.,0.6]]
        x_data =  surf_roughness
        x_label = 'surface roughness [m]'
        y_label = 'Ku-band fb [m]'
        y_data  = radar_fb_full[1]
        plot_scatter(ax,xylim,'UOB','m',x_data,x_label,y_data,y_label,None,color=colors_scatter[0]) #icetype_al_full[flag_snow])
        #y_label = ' fb [m]'
        f3, ax1 = plt.subplots(1, 1,figsize=(6,6))
        y_data  = radar_fb_full[-1]
        plot_scatter(ax1,xylim,'AWI','m',x_data,x_label,y_data,y_label,None,color=colors_scatter[1]) 
        plt.show()
        

        # penetration
        f1, ax = plt.subplots(1, 1,figsize=(6,6))
        f1.suptitle('Elevation differences LaKu', fontsize=12)
        xylim = [[0.,0.6],[0.,0.6]]
        #for nb,b in enumerate(beamList):
        x_data =  radar_fb_mean
        x_label = 'Ku-radar fb [m]'
        y_label = 'laser fb [m]'
        y_data  = laser_fb_full_ave
        plot_scatter(ax,xylim,'m',x_data,x_label,y_data,y_label,None,color=colors_scatter[nb]) #icetype_al_full[flag_snow])

        plt.show()
        """

        nonmasked_data = ~radar_fb_full[0].mask
        for nprod,cs2_gdr in enumerate(prod_L2P):
            radar_fb = ma.masked_invalid(radar_fb_full[nprod],copy=True)
            nonmasked_data = np.logical_and(nonmasked_data,~radar_fb.mask)
            print(np.sum(nonmasked_data))

        x_data =  surf_roughness
        x_data = ma.masked_where(~nonmasked_data,x_data,copy=True)
        x_label = 'surface roughness [m]'
        print(np.sum(~x_data.mask))
        delta_fb_LaSar_UOB = laser_fb_full_ave - radar_fb_full[1]
        f1, ax = plt.subplots(1, 1,figsize=(6,6))
        f1.suptitle('Elevation differences LaKu', fontsize=12)
        xylim = [[0.,0.4],[-0.1,0.7]]
        #for nb,b in enumerate(beamList):
        
        y_label = r'$\Delta fb(La-ku)$ [m]'
        y_data  = delta_fb_LaSar_UOB
        y_data = ma.masked_where(~nonmasked_data,y_data,copy=True)
        print(np.sum(~y_data.mask))
        plot_scatter(ax,xylim,prod_L2P[1],'m',x_data,x_label,y_data,y_label,None,color=colors_scatter[0]) #icetype_al_full[flag_snow])
        #plt.show()

        delta_fb_LaSar_AWI = laser_fb_full_ave - radar_fb_full[-1]
        f2, ax2 = plt.subplots(1, 1,figsize=(6,6))
        f2.suptitle('Elevation differences LaKu', fontsize=12)
        xylim = [[0.,0.4],[-0.1,0.7],]
        #for nb,b in enumerate(beamList):
        y_label = r'$\Delta fb(La-ku)$ [m]'
        y_data  = delta_fb_LaSar_AWI
        y_data = ma.masked_where(~nonmasked_data,y_data,copy=True)
        print(np.sum(~y_data.mask))
        plot_scatter(ax2,xylim,prod_L2P[-1],'m',x_data,x_label,y_data,y_label,None,color=colors_scatter[0]) #icetype_al_full[flag_snow])
        #plt.show()

        delta_fb_LaSar_SAM = laser_fb_full_ave - radar_fb_full[3]
        f2, ax2 = plt.subplots(1, 1,figsize=(6,6))
        f2.suptitle('Elevation differences LaKu', fontsize=12)
        xylim = [[0.,0.4],[-0.1,0.7],]
        #for nb,b in enumerate(beamList):
        y_label = r'$\Delta fb(La-ku)$ [m]'
        y_data  = delta_fb_LaSar_SAM
        y_data = ma.masked_where(~nonmasked_data,y_data,copy=True)
        plot_scatter(ax2,xylim,prod_L2P[3],'m',x_data,x_label,y_data,y_label,None,color=colors_scatter[0]) #icetype_al_full[flag_snow])

        f2, ax = plt.subplots(1, 1,figsize=(6,6))
        f2.suptitle('Elevation differences LaKu', fontsize=12)
        xylim = [[0.,0.6],[0.,0.6]]
        #for nb,b in enumerate(beamList):
        x_data =  surf_roughness
        x_label = 'surface roughness [m]'
        y_label = 'laser fb [m]'
        y_data  = laser_fb_full_ave
        y_data = ma.masked_where(~nonmasked_data,y_data,copy=True)
        plot_scatter(ax,xylim,'IS2','m',x_data,x_label,y_data,y_label,icetype_al_full,color=colors_scatter[0]) #icetype_al_full[flag_snow])
        plt.show()
                
        

        # Laser_fb vs gaussian width
        f2, ax = plt.subplots(1, 1,figsize=(6,6))
        f2.suptitle('Elevation differences', fontsize=12)
        xylim = [[0.,0.6],[0.,0.6]]
        for nb,b in enumerate(beamList):
            x_data =  np.concatenate(data_dict['IS2']['ATL10'][b]['gaussian_w'],axis=0)
            x_label = 'surface roughness [m]'
            y_label = 'laser fb [m]'
            y_data  = np.concatenate(data_dict['IS2']['ATL10'][b]['laser_fb'],axis=0)
            plot_scatter(ax,xylim,'m',x_data,x_label,y_data,y_label,None,color=colors_scatter[nb]) #icetype_al_full[flag_snow])

        plt.show()

        # beam = beamList[1]
        flag_leads = np.concatenate(data_dict['IS2']['ATL10'][beam]['flag_leads'],axis=0)
        x_data =  np.concatenate(data_dict['IS2']['ATL10'][beam]['gaussian_w'],axis=0)
        x_label = 'surface roughness [m]'
        y_label = 'laser fb [m]'
        y_data  = np.concatenate(data_dict['IS2']['ATL10'][b]['laser_fb'],axis=0)
        plot_scatter(ax,xylim,'m',x_data,x_label,y_data,y_label,None,color=colors_scatter[nb]) #icetype_al_full[flag_snow]) 

        # Get volume roughness
        #volume_roughness = list(np.array(data_dict['CS2']['UOB']['roughness'],dtype=object)[idx_dates])
        #volume_roughness = np.ma.concatenate(volume_roughness,axis=0)


        #delta_fb_LaSar = laser_fb_full_ave - np.ma.concatenate(radar_fb_list[1],axis=0)
        delta_fb_LaSar = laser_fb_full_ave - radar_fb_mean
        snow_depth = delta_fb_LaSar*ns

        
        # Limited snow depth surfaces < 5 cm
        #-------------------------------------
        
        SD_AMSR_al_full = list()
        for ndate,date in enumerate(found_dates):
            lat,lon,SD_AMSR = cf.get_SD_AMSR(date)
            SD_AMSR_al = cf.grid_to_track(SD_AMSR,lon,lat,ref_seg_lon[ndate],ref_seg_lat[ndate])
            SD_AMSR_al = SD_AMSR_al/100 # convert to meters
            SD_AMSR_al_full.append(SD_AMSR_al)
        SD_AMSR_al = np.ma.concatenate(SD_AMSR_al_full,axis=0)

        flag_snow = SD_AMSR_al < 0.1

        # Volume vs surface roughness

        
        """
        f1, ax = plt.subplots(1, 1,figsize=(6,6))
        f1.suptitle('Roughness differences', fontsize=12)
        xylim = [[-0.1,0.6],[-0.1,0.6]]
        x_data =  surf_roughness[flag_snow]
        x_label = 'surface roughness (IS2 gauss w) [m]'
        y_label='volume roughness [m]'
        y_data = volume_roughness[flag_snow]
        plot_scatter(ax,None,'m',x_data,x_label,y_data,y_label,icetype_al_full[flag_snow])
        """

        # delta fb function of roughness
        f2, ax = plt.subplots(1, 1,figsize=(6,6))
        f2.suptitle('Elevation differences', fontsize=12)
        xylim = [[0.,0.5],[0.,0.5]]
        x_data =  surf_roughness[flag_snow]
        x_label = 'surface roughness (IS2 gauss w) [m]'
        y_label = r'$\Delta fb(La-ku)$ [m]'
        y_data  = delta_fb_LaSar[flag_snow]
        plot_scatter(ax,xylim,'m',x_data,x_label,y_data,y_label,None) #icetype_al_full[flag_snow])

        plt.show()

        
        
        

        """
        f1, ax = plt.subplots(1, 1,figsize=(6,6))
        lat = np.ma.concatenate(ref_seg_lat,axis=0)
        lon = np.ma.concatenate(ref_seg_lon,axis=0)
        
        data_desc_cs2 = cs2_dict.init_dict('UOB')
        filename = '/home/antlafe/Documents/work/projet_cryo2ice/data/CS2/UOB/ubristol_trajectory_rfb_ESA-SAR_2020_11_11_163106.txt'
        lat,lon,time,x_dist,valid_idx = cf.get_coord_from_uob(filename,data_desc_cs2,['lat','lon','time'],'01',70)
        param,units,param_is_flag = cf.get_param_from_uob(filename,data_desc_cs2,'roughness','01',70)
        plot_track_map(f1,ax,lon,lat,param,'Sea_Ice_Roughness_Lognormal',[0,0.2],mid_date)

        plt.show()
        """
        
        #plot_track_map(f1,ax,lon,lat,volume_roughness,'Sea_Ice_Roughness_Lognormal',[0,0.2],mid_date)
        
        #plt.show()

        
        """
        # Get emissivity SSMIS
        e_SSMIS_al_full = list()
        for ndate,date in enumerate(found_dates):
            lat,lon,e_SSMIS = cf.get_emissivity_SSMIS(date)
            e_SSMIS_al = cf.grid_to_track(e_SSMIS,lon,lat,ref_seg_lon[ndate],ref_seg_lat[ndate])
            e_SSMIS_al_full.append(e_SSMIS_al)
        e_SSMIS_al = np.ma.concatenate(e_SSMIS_al_full,axis=0)
        

        """

         
       

        # Get delta freeboard
       

        """
        f1, ax = plt.subplots(1, 1,figsize=(6,6))
        lat = np.ma.concatenate(ref_seg_lat,axis=0)
        lon = np.ma.concatenate(ref_seg_lon,axis=0)
        plot_track_map(f1,ax,lon,lat,delta_fb_LaSar,'Snow depth LaKu',[0,0.4],mid_date)
        plt.show()
        """
        
        # Get SST
        
        """
        sst_all_track = list()
        for ndate,date in enumerate(found_dates):
            lat_sst,lon_sst,sst_grid = cf.get_sst_metoffice(date)
            lon_grid,lat_grid = np.meshgrid(lon_sst,lat_sst)
            sst_al = cf.grid_to_track(sst_grid,lon_grid,lat_grid,ref_seg_lon[ndate],ref_seg_lat[ndate])
            sst_all_track.append(sst_al)
        sst_al_full = np.ma.concatenate(sst_all_track,axis=0)
        """

        """
        # Get IST MODIS_
        LAT_MIN = 60
        ist_MODIS_track = list()
        for ndate,date in enumerate(found_dates):
            lat_grid,lon_grid,ist_grid = cf.get_MODIS_IST(date,LAT_MIN)
            
            #lon_grid,lat_grid = np.meshgrid(lon_ist,lat_ist)
            ist_al = cf.grid_to_track(ist_grid,lon_grid,lat_grid,ref_seg_lon[ndate],ref_seg_lat[ndate])
            ist_MODIS_track.append(ist_al)
        ist_MODIS_full = np.ma.concatenate(ist_MODIS_track,axis=0)

        # Delta fb vs MODIS surface temperature
        f17, ax = plt.subplots(1, 1,figsize=(6,6))
        f17.suptitle('Penetration vs ice temperature MODIS', fontsize=12)
        xylim = [[245,270],[-0.2,0.8]]
        x_data =  ist_MODIS_full
        x_label = 'MODIS IST [K]'
        #y_label= r'$\Delta fb$'
        y_label= 'Snow depth LaKu [m]'
        y_data = snow_depth #delta_fb_LaSar
        plot_scatter(ax,xylim,'m',x_data,x_label,y_data,y_label,icetype_al_full)

        plt.show()
        """

      

        # Get scatter various roughness
        
        # PLRM
        
        #ax.title.set_text('%s' %(gdr))
        f1, ax = plt.subplots(1, 1,figsize=(6,6))
        f1.suptitle(r'$\Delta fb$ vs surface roughness', fontsize=12)
        xylim = [[0.,0.6],[0.,0.6]]
        x_data =  surf_roughness
        x_label = 'surface roughness (IS2 gauss w) [m]'
        y_label= 'laser fb [m]'
        y_data = laser_fb_full_ave
        plot_scatter(ax,xylim,'m',x_data,x_label,y_data,y_label,icetype_al_full)
        plt.show()

        LIST_GDR = ['LEGOS_SAM','LEGOS_T50','LEGOS_PLRM']
        f1, ax = plt.subplots(1, len(LIST_GDR),figsize=(6*(len(LIST_GDR)),6))
        f1.suptitle(r'$\Delta fb$ vs surface roughness', fontsize=12)
        xylim = [[0.,0.6],[-0.1,0.8]]

        for ngdr,gdr in enumerate(LIST_GDR):
            ax[ngdr].title.set_text('%s' %(gdr))
            x_data =  surf_roughness
            x_label = 'surface roughness (IS2 gauss w) [m]'
            #y_label= r'$\Delta fb$ [m]'
            y_label= 'radar fb %s [m]' %(gdr)
            #radar_fb_gdr = radar_fb[gdr]
            #radar_fb = isa[gdr]# - ssa['LEGOS_SAM']
            #delta_fb_LaSar = laser_fb_full_ave - radar_fb_gdr
            y_data = laser_fb_full_ave #radar_fb #delta_fb_LaSar
            plot_scatter(ax[ngdr],xylim,'m',x_data,x_label,y_data,y_label,icetype_al_full)
        plt.show()
        
        
        
        
        
        
        
        # Get scatter plot

        """
        # Volume vs surface roughness
        f11, ax = plt.subplots(1, 1,figsize=(6,6))
        f11.suptitle('Roughness differences', fontsize=12)
        xylim = [[-0.1,0.6],[-0.1,0.6]]
        x_data =  surf_roughness
        x_label = 'surface roughness (IS2 gauss w) [m]'
        y_label='volume roughness [m]'
        y_data = volume_roughness
        plot_scatter(ax,None,'m',x_data,x_label,y_data,y_label,icetype_al_full)

        
        # Delta fb vs surface roughness
        f12, ax = plt.subplots(1, 1,figsize=(6,6))
        f12.suptitle('Penetration vs surface roughness', fontsize=12)
        xylim = [[-0.1,0.6],[-0.1,0.6]]
        x_data =  surf_roughness
        x_label = 'surface roughness (IS2 gauss w) [m]'
        y_label= r'$\Delta fb$'
        y_data = delta_fb_LaSar
        plot_scatter(ax,None,'m',x_data,x_label,y_data,y_label,icetype_al_full)

        # Delta fb vs volume roughness
        f13, ax = plt.subplots(1, 1,figsize=(6,6))
        f13.suptitle('Penetration vs volume roughness', fontsize=12)
        xylim = [[-0.1,0.6],[-0.1,0.6]]
        x_label='volume roughness [m]'
        x_data = volume_roughness
        y_label= r'$\Delta fb$'
        y_data = delta_fb_LaSar
        plot_scatter(ax,None,'m',x_data,x_label,y_data,y_label,icetype_al_full)
        """

      

        # Delta fb vs emissivity SSMIS
        """
        f15, ax = plt.subplots(1, 1,figsize=(6,6))
        f15.suptitle('Penetration vs emissivity SSMIS', fontsize=12)
        xylim = [[0,0.4],[-0.1,0.8]]
        x_data =  e_SSMIS_al
        x_label = 'SSMIS emissivity'
        #y_label= r'$\Delta fb$'
        y_label= 'Snow depth LaKu [m]'
        y_data = snow_depth #delta_fb_LaSar
        plot_scatter(ax,None,'m',x_data,x_label,y_data,y_label,icetype_al_full)
        """
        

        """
        # Delta fb vs surface temperature
        f16, ax = plt.subplots(1, 1,figsize=(6,6))
        f16.suptitle('penetration vs surface temperature', fontsize=12)
        xylim =  [[-0.1,0.6],[-0.1,0.6]]
        x_label='surface temperature [K]'
        x_data = sst_al_full
        y_label= r'$\Delta fb$'
        y_data = delta_fb_LaSar
        plot_scatter(ax,None,'m',x_data,x_label,y_data,y_label,icetype_al_full)
        """
        
        
        plt.show()

        print("stop")


        
        
    if param=='wvf':
        print("\nwaveform comparison")

        # IS2/CS2 distance
        dist = np.ma.concatenate(data_dict['IS2']['ATL07']['dist'],axis=1)

        # IS2/CS2 delay
        delay = np.ma.concatenate(data_dict['IS2']['ATL07']['delay'],axis=1)
        delay = np.ma.mean(delay) #min

        # IS2 beams
        beams = np.ma.concatenate(data_dict['IS2']['ATL07']['beam'],axis=1)

        # IS2 flag leads
        flag_lead = np.ma.concatenate(data_dict['IS2']['ATL07']['flag_leads'],axis=1).astype('int')
        #flag_lead_true = flag_lead==1.
        dist_leads = np.ma.masked_where(flag_lead.data==0,dist,copy=True)
        dopplerB_num = np.tile(np.arange(flag_lead.shape[1]),(flag_lead.shape[0],1)).flatten()
        dist_leads = dist_leads.flatten()
        dist_leads_al = dist_leads[dist_leads.mask==0]
        dopplerB_num_al = dopplerB_num[dist_leads.mask==0]

        #offnadir_range = cf.off_nadir_range_corr(dist_leads_al,altitude,latDeg)
        

        # CS2 SARin waveforms
        wvf = np.ma.concatenate(data_dict['CS2']['ESA_BD_SIN']['wvf'],axis=1)

        # Figure
        fig, ax = plt.subplots(3,1,sharey=True)
        ax[0].imshow(wvf.T)

        ax[1].imshow(flag_lead)

        ax[2].imshow(dist)

        plt.show()

        




        
    
    elif param=='sla':
        print("\nComparing Sea Surface Height")

        # distance of each data to CS2 beam center
        dist = np.concatenate(data_dict['IS2']['ATL10']['dist'],axis=1)

        
        #------------------------------------------------        
        # Compare SSH per CS2 beams
        #------------------------------------------------
        
        # Interpolated SSH CS2 vs Mean SSH of IS2 seg pts contained in beam

        mssdtu15 = list(np.array(data_dict['CS2']['LEGOS_SAM']['mss'],dtype=object)[idx_dates])
        

        
        # Get CS2 SSH
        #-------------
        # Get CS2 freeboard
        ssh_list = list()
        ssh_full = list()
        for cs2_gdr in prod_L2P:
            sla = list(np.array(data_dict['CS2'][cs2_gdr]['sla'],dtype=object)[idx_dates])
            ssh_list.append(ssh)
            ssh_int_full = ma.masked_invalid(np.ma.concatenate(sla,axis=0))
            ssh_full.append(ssh_int_full)
            
        esa_sla = np.concatenate(data_dict['CS2']['ESA_BD_GDR']['sla'],axis=0)
        #cs2_ssh_list = list()
        #for cs2_gdr in prod_L2P:
        #    cs2_ssh_list.append(data_dict['CS2'][cs2_gdr]['ssh'])

        #ssh_esa = data_dict['CS2']['ESA_BD']['ssh'][0]

        
        
        # Get IS2 SSH
        #-------------
        is2_surface_h = [obj for nobj,obj in enumerate(data_dict['IS2']['ATL10']['surface_h']) if nobj in idx_dates]
        is2_surface_h =  np.ma.concatenate(is2_surface_h,axis=1)
        
      
        #is2_surface_h = np.concatenate(data_dict['IS2']['ATL07']['surface_h'],axis=1)
        is2_flag_leads =  [obj for nobj,obj in enumerate(data_dict['IS2']['ATL10']['flag_leads']) if nobj in idx_dates]
        is2_flag_leads = np.ma.concatenate(is2_flag_leads,axis=1)

        # Get lead mask
        is2_ssh_2d = np.ma.masked_where(is2_flag_leads.data==False,is2_surface_h,copy=True)

        # Get dist mask
        is2_ssh_dist = np.ma.masked_where(is2_flag_leads.data==False,dist,copy=True)
        mask_max_dist = is2_ssh_dist > MAX_ACROSS_DIST
        is2_ssh_2d = np.ma.masked_where(mask_max_dist,is2_ssh_2d,copy=True)

        # Retreive along-track SSH
        is2_ssh = np.mean(is2_ssh_2d,axis=0) # along-track ssh
        is2_ssh_mask = is2_ssh.mask # where IS2 data are to be found
        is2_nleads = np.sum(is2_ssh_2d.mask==False,axis=0) # number of leads per lines

        # OPTIONNAL: minimun lead number reauired
        flag_nleads_min = is2_nleads < 5
        is2_ssh1 = np.ma.masked_where(flag_nleads_min,is2_ssh,copy=True)

        # mss
        #mss_is2 = [obj for nobj,obj in enumerate(data_dict['IS2']['ATL10']['mss']) if nobj in idx_dates]
        #mss_is2 =  np.ma.concatenate(is2_surface_h,axis=1)

        # sla
        is2_sla = is2_ssh - mss

        # For each Dopplerbeams
        
        
        # Figure
        #--------------------------------
        
        # Scatter plot
        #-------------
        f11, ax = plt.subplots(1, N_prod, sharey=True,figsize=(15,6))
        f11.suptitle('Interpolated SSH (CS2) vs Mean SSH from leads (IS2)', fontsize=12)
        xylim = [[-0.3,0.5],[-0.3,0.5]]
        x_data =  is2_ssh
        x_label = 'ssh IS2 (m)'
        
        for nprod,cs2_gdr in enumerate(prod_L2P):

            y_label='ssh %s (m)' %(cs2_gdr)
            #y_data = np.concatenate(data_dict['CS2'][cs2_gdr]['ssh'],axis=0)
            y_data = ssh_full[nprod]
            plot_scatter(ax[nprod],xylim,'m',x_data,x_label,y_data,y_label,None)
            ax[nprod].title.set_text(cs2_gdr)
            
        
        # Histograms
        #-------------
        f12, ax = plt.subplots(1, 1)
        f12.suptitle('Histogram interpolated SSH (CS2) vs Mean SSH from leads (IS2)', fontsize=12)
        
        label_list = list()
        data_list = list()
        
        label_IS2 = 'ssh IS2'
        legend_list = list()
        legend_list.append(label_IS2)
        data_list.append(is2_ssh)
        xlabel= 'ssh (m)'
        
        for nprod,cs2_gdr in enumerate(prod_L2P):

            legend_list.append('ssh %s (m)' %(cs2_gdr))
            data_list.append(ssh_full[nprod])

        plot_histo(ax,xylim,'m',xlabel,legend_list,data_list,True)
        #plot_histo(ax[1,nprod],label_list,data_list)

        plt.show()


        
        #------------------------------------------------
        # Compare SSH per CS2 lead beams
        #------------------------------------------------
        # SSH in lead beams CS2 vs Mean SSH of IS2 seg pts contained in these beams

        
        # Get CS2/IS2 SSH over CS2 leads
        #-------------
        cs2_ssh_onleads_list = list()
        is2_ssh_onleads_list = list()
        
        for cs2_gdr in prod_L2P:
            cs2_ssh = np.concatenate(data_dict['CS2'][cs2_gdr]['ssh'],axis=0)
            # Get flag lead
            cs2_surface_type = np.concatenate(data_dict['CS2'][cs2_gdr]['surface_type'],axis=0)
            # flag_lead =
            cs2_ssh_leads = np.ma.masked_where(flag_leads,cs2_ssh,copy=True)
            cs2_ssh_onleads_list.append(is2_ssh_leads)

            is2_ssh_leads = np.ma.masked_where(flag_leads,is2_ssh,copy=True)
            is2_ssh_onleads_list.append(is2_ssh_leads)

        # Scatter plot
        #-------------
        
        f2, ax = plt.subplots(1, N_prod, sharey=True)
        f2.suptitle('SSH over leads (CS2) vs Mean SSH from leads (IS2)', fontsize=12)
        xylim = [[-0.05,0.05],[-0.05,0.05]]
       
        for nprod,cs2_gdr in enumerate(prod_L2P):

            x_data = is2_ssh_onleads_list[nprod]
            x_label = 'ssh IS2 (m)'
            y_label='ssh %s (m)' %(cs2_gdr)
            y_data = cs2_ssh_onleads_list[nprod]
            plot_scatter(ax[nprod],xylim,'m',x_data,x_label,y_data,y_label)

        plt.show()


        
        #------------------------------------------------        
        # Compare SSH when IS2 finds leads at CS2 beam nadir
        #------------------------------------------------
        # SSH in lead beams CS2 vs Mean SSH of IS2 seg pts contained in these beams

        mask_nadir_dist = is2_ssh_dist > MAX_NADIR_DIST
        is2_ssh_2d_nadir = np.ma.masked_where(mask_nadir_dist,is2_ssh_2d,copy=True)
        
        # Retreive along-track SSH
        is2_ssh_nadir = np.mean(is2_ssh_2d_nadir,axis=0) # along-track ssh

        # Figures
        f3, ax = plt.subplots(2, N_prod, sharey=True)
        f3.suptitle('SSH over leads (CS2) when IS2 finds nadir lead', fontsize=16)
        xylim = [[-0.05,0.05],[-0.05,0.05]]
        
        # Scatter plot
        #-------------
        
        x_data = is2_ssh_nadir
        x_label = 'ssh IS2 (m)'

        for nprod,cs2_gdr in enumerate(prod_L2P):
            
            y_label='ssh %s (m)' %(cs2_gdr)
            y_data = cs2_ssh_onleads_list[nprod]
            plot_scatter(ax[0,nprod],xylim,'m',x_data,x_label,y_data,y_label)


        # Histograms
        #------------

        

        plt.show()

    elif param=='corrs':
        print("\nCompare corrections")

        # Get parameters
        #------------------------------------------
        beams = [obj for nobj,obj in enumerate(data_dict['IS2']['ATL07']['beam']) if nobj in idx_dates]
        beams = np.ma.concatenate(beams,axis=1)
        #plt.imshow(beams)
        #plt.show()
            
        # ref coordinates
        ref_seg_lat = list(np.array(data_dict['CS2'][REF_GDR]['latref'],dtype=object)[idx_dates])
        ref_seg_lon = list(np.array(data_dict['CS2'][REF_GDR]['lonref'],dtype=object)[idx_dates])

        # Get CS2 corrections
        #---------------------------------
        N_prod = len(prod_L2P)
        param_list_cs2 = ['ocean_tide01','pole_tide01','dac01','ocean_eq_tide01','load_tide01']
        #param_list_cs2 = ['ocean_tide01','pole_tide01','dac01','solid_earth_tide01','ocean_eq_tide01','load_tide01']

        corr_dict_cs2 = dict()
        for p in param_list_cs2:
            corr_dict_cs2[p] = dict()
            for cs2_gdr in prod_L2P:
                parameter = list(np.array(data_dict['CS2'][cs2_gdr][p],dtype=object)[idx_dates])
                corr_dict_cs2[p][cs2_gdr] = np.concatenate(parameter,axis=0).astype('float64')            
        
        #radar_fb_full.append(fb_full)


        # IS2 corrections
        #---------------------------------
        is2_gdr = 'ATL07'
        param_list_is2 = ['ocean_tide','pole_tide','dac','long_period_tide','load_tide']
        #param_list_is2 = ['ocean_tide','pole_tide','dac','solid_earth_tide','long_period_tide','load_tide']
        corr_dict_is2 = dict()
        for p in param_list_is2:
            data_is2 = [obj for nobj,obj in enumerate(data_dict['IS2'][is2_gdr][p]) if nobj in idx_dates]            
            corr_dict_is2[p] = np.ma.mean(np.ma.concatenate(data_is2,axis=1),axis=0)
            std =  np.ma.std(np.ma.concatenate(data_is2,axis=1),axis=0)

            """
            plt.plot(std,label='std:%s' %(p))
            plt.plot(corr_dict_is2[p],label='mean:%s' %(p))
            plt.legend()
            plt.show()
            """
            #print("std(%s)=%.2fm" %(p,std))


        # Scatters
        #-------------------------------
        
        """
        for p_cs2,p_is2 in zip(param_list_cs2,param_list_is2):

            f1, ax = plt.subplots(1, N_prod, sharey=True,sharex=True,figsize=(6*N_prod,5))
            f1.suptitle('%s differences' %(p_is2), fontsize=14)
            # ocean tide
            x_data = corr_dict_is2[p_is2]*100
            xlim = [np.min(x_data),np.max(x_data)]
            x_label = '%s IS2 (cm)' %(p_is2)
            for nprod,cs2_gdr in enumerate(prod_L2P):
            
                y_label='%s CS2[%s] (cm)' %(p_cs2,cs2_gdr)
                y_data = corr_dict_cs2[p_cs2][cs2_gdr]*100
                xylim = [min(np.min(y_data),xlim[0]),max(np.max(y_data),xlim[1])]
                xylim = [xylim,xylim]
                
                if N_prod==1:
                    plot_scatter(ax,xylim,'cm',x_data,x_label,y_data,y_label,None)
                    ax.title.set_text(cs2_gdr)
                else:
                    plot_scatter(ax[nprod],xylim,'cm',x_data,x_label,y_data,y_label,None)
                    ax[nprod].title.set_text(cs2_gdr)
        """

            #plt.show()

        # Total corrections
        #-------------------------------
        # IS2
        """
        total_corrs_is2 = np.zeros(corr_dict_is2[param_list_is2[0]].shape)
        for p in param_list_is2:
            total_corrs_is2 = total_corrs_is2 + corr_dict_is2[p]

        # CS2
        total_corrs_cs2 = dict()
        for nprod,cs2_gdr in enumerate(prod_L2P):
            total_corrs_cs2[cs2_gdr] = np.zeros(corr_dict_cs2[param_list_cs2[0]][prod_L2P[0]].shape)
            for p in param_list_cs2:
                total_corrs_cs2[cs2_gdr] = total_corrs_cs2[cs2_gdr] + corr_dict_cs2[p][cs2_gdr]

        
        f2, ax = plt.subplots(1, N_prod, sharey=True,sharex=True,figsize=(6*N_prod,5))
        x_data = total_corrs_is2*100
        x_lim = [np.min(x_data),np.max(x_data)]
        x_label = 'All corrs IS2 (cm)'
        for nprod,cs2_gdr in enumerate(prod_L2P):
            ax.title.set_text('All corrs: %s' %(cs2_gdr))        
            y_label='All corrs CS2 (cm)'
            y_data = total_corrs_cs2[cs2_gdr]*100
            xylim = [min(np.min(y_data),xlim[0]),max(np.max(y_data),xlim[1])]
            xylim = [xylim,xylim]
            plot_scatter(ax,xylim,'cm',x_data,x_label,y_data,y_label,None)
            
        plt.show()
        """
        
        # Show maps
        #--------------
        
        pname = 'ocean_tide'
        
        f3, axm = plt.subplots(1, N_prod,figsize=(6*N_prod,5))
        f3.suptitle('Distribution of %s difference %s-%s' %(pname,start_date,end_date), fontsize=14)
        
        label_list = list()
        data_list = list()
        legend_list = list()
        
        label_IS2 = '%s fb IS2 (m)' %(pname)
        legend_list.append(label_IS2)
        #data_list.append(is2_ssh.compressed())
        #data_list.append(laser_fb_full_ave)
        label= r'$\Delta$%s [m]' %(pname)
        lon = np.ma.concatenate(ref_seg_lon,axis=0)
        lat = np.ma.concatenate(ref_seg_lat,axis=0)
        data_list = list()
        for nprod,cs2_gdr in enumerate(prod_L2P): 
            data = (corr_dict_is2[pname] - corr_dict_cs2[pname+'01'][cs2_gdr])*100
            if N_prod==1:
                xylim =[300,250]
                axm.set_title(cs2_gdr)
                plot_track_map(f3,axm,lon,lat,data,label,xylim,mid_date,'m',False)
            else:
                axm[nprod].set_title(cs2_gdr)
                plot_track_map(f3,axm[nprod],lon,lat,data,label,xylim,mid_date,'m',False)
                
        # diff maps
        plt.show()
        
            
        
 
        
        
    elif param=='ish':
        print("\nComparing Ice Surface Height")
        
        

    elif param=='freeboard':
        print("\nComparing freeboard")

        # Get parameters
        #------------------------------------------

        # ref coordinates
        ref_seg_lat = list(np.array(data_dict['CS2'][REF_GDR]['latref'],dtype=object)[idx_dates])
        ref_seg_lon = list(np.array(data_dict['CS2'][REF_GDR]['lonref'],dtype=object)[idx_dates])

        # x_dist ref
        x_dist = list()
        for lat,lon in zip(ref_seg_lat,ref_seg_lon):
            x_dist.append(cf.distance_from_first_trk_pts(lat,lon,0))
        x_dist_full = np.ma.concatenate(x_dist,axis=0)
        x_dist_sum = np.cumsum(x_dist_full)
        npts = x_dist_sum.shape[0]
        
        # Get icetype along-track
        icetype_al = list()
        for n in range(ndates):
            lon = ref_seg_lon[n]
            if any(np.abs(np.diff(lon)) > 20): lon[lon > 180] = lon[lon > 180] - 360   
            icetype_alongtrack = cf.grid_to_track(ice_type[n],lons_icetype[n],lats_icetype[n],lon,ref_seg_lat[n])
            icetype_al.append(icetype_alongtrack)
        icetype_al_full = np.ma.concatenate(icetype_al,axis=0)

        # Get CS2 freeboard
        radar_fb_list = list()
        radar_fb_full = list()
        radar_fb_matrix = ma.masked_array(np.zeros((len(prod_L2P),npts)),mask=np.ones((len(prod_L2P),npts)),dtype='float')
        for nprod,cs2_gdr in enumerate(prod_L2P):
            radar_fb = list(np.array(data_dict['CS2'][cs2_gdr]['radar_fb'],dtype=object)[idx_dates])
            radar_fb_list.append(radar_fb)
            fb_full = ma.masked_invalid(np.ma.concatenate(radar_fb,axis=0))
            radar_fb_full.append(fb_full)
            radar_fb_matrix[nprod,:] = fb_full

        radar_fb_matrix = ma.masked_invalid(radar_fb_matrix,copy=True)
        radar_fb_mean = np.ma.mean(radar_fb_matrix,axis=0)
        radar_fb_std = np.ma.std(radar_fb_matrix,axis=0)
        radar_fb_spline =  rolling_stats(radar_fb_mean, 100, stats=['mean'])[0]
        
        # Get IS2 freeboard
        laser_fb = [obj for nobj,obj in enumerate(data_dict['IS2']['ATL10']['laser_fb']) if nobj in idx_dates]
        laser_fb_full = np.ma.concatenate(laser_fb,axis=1)
        laser_fb_full = ma.masked_invalid(laser_fb_full,copy=True)
        laser_fb_full_ave = np.ma.mean(laser_fb_full,axis=0)
        laser_fb_spline = rolling_stats(laser_fb_full_ave, 100, stats=['mean'])[0]
        laser_fb_full_std = np.ma.std(laser_fb_full,axis=0)

        # Get SARAL freeboard
        #radar_fb_saral = list(np.array(data_dict['SARAL']['LEGOS_T50']['radar_fb_mean'],dtype=object)[idx_dates])
        #radar_fb_sa = [obj for nobj,obj in enumerate(data_dict['SARAL']['LEGOS_T50']['radar_fb']) if nobj in idx_dates]
        #radar_fb_sa = np.ma.concatenate(radar_fb_sa,axis=1)
        #radar_fb_saral = ma.masked_invalid(np.ma.concatenate(radar_fb_saral,axis=0))

        # make figures
        #------------------------------------------
        
        nplots = len(prod_L2P)
        
        # Scatter plot
        #-------------
        f1, ax = plt.subplots(1, N_prod, sharey=True,sharex=True,figsize=(4*nplots,4))
        f1.suptitle('freeboard difference', fontsize=14)

        nonmasked_data = ~radar_fb_full[0].mask
        for nprod,cs2_gdr in enumerate(prod_L2P):
            radar_fb = ma.masked_invalid(radar_fb_full[nprod],copy=True)
            nonmasked_data = np.logical_and(nonmasked_data,~radar_fb.mask)
            print(np.sum(nonmasked_data))
        
        y_data = laser_fb_full_ave
        y_data = ma.masked_where(~nonmasked_data,y_data,copy=True)
        y_label = 'laser fb IS2 (m)'
        print(np.sum(~x_data.mask))

        for nprod,cs2_gdr in enumerate(prod_L2P):

            xylim = [[-0.3,0.5],[0.,0.7]]
            x_label='rada2r fb CS2[%s] (m)' %(cs2_gdr)
            x_data = radar_fb_full[nprod]
            x_data = ma.masked_where(~nonmasked_data,x_data,copy=True)
            print(np.sum(~x_data.mask))
            
            if nplots==1:
                plot_scatter(ax,xylim,cs2_gdr,'m',x_data,x_label,y_data,y_label,icetype_al_full)
                ax.title.set_text(cs2_gdr)
            else:
               plot_scatter(ax[nprod],xylim,cs2_gdr,'m',x_data,x_label,y_data,y_label,None)
               ax[nprod].title.set_text(cs2_gdr)

        plt.show()
               
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
        #data_list.append(is2_ssh.compressed())
        data_list.append(laser_fb_full_ave)
        xlabel= 'freeboard (m)'
        
        for nprod,cs2_gdr in enumerate(prod_L2P):

            legend_list.append('radar freeboard [%s] (m)' %(cs2_gdr))
            data_list.append(radar_fb_full[nprod])
        
        plot_histo(axh,xylim,'m',xlabel,legend_list,data_list,True)
        plt.show()
       
        """
        # Show maps
        #--------------
        
        f3, axm = plt.subplots(1, nplots,figsize=(6*nplots,5))
        f3.suptitle('Distribution of freeboard difference %s-%s' %(start_date,end_date), fontsize=14)
        
        label_list = list()
        data_list = list()
        legend_list = list()
        
        label_IS2 = 'laser fb IS2 (m)'
        legend_list.append(label_IS2)
        #data_list.append(is2_ssh.compressed())
        #data_list.append(laser_fb_full_ave)
        label= r'$\Delta$freeboard [m]'
        lon = np.ma.concatenate(ref_seg_lon,axis=0)
        lat = np.ma.concatenate(ref_seg_lat,axis=0)
        data_list = list()
        for nprod,cs2_gdr in enumerate(prod_L2P): 
            data = laser_fb_full_ave - radar_fb_full[nprod]
            if nplots==1:
                axm.set_title(cs2_gdr)
                plot_track_map(f3,axm,lon,lat,data,label,xylim,mid_date)
            else:
                axm[nprod].set_title(cs2_gdr)
                plot_track_map(f3,axm[nprod],lon,lat,data,label,xylim,mid_date) 

        # diff maps
        plt.show()
        """

        
        # Show along-track
        #-------------------

        import seaborn as sns
        f4, axp = plt.subplots(1, 1,figsize=(15,8))
        f4.suptitle('Along-track freeboard %s-%s' %(start_date,end_date), fontsize=14)
        #fig, ax = plt.subplots()
        """

        clrs = sns.color_palette("husl", 5)
        with sns.axes_style("darkgrid"):
            axp.plot(radar_fb_full[nprod][mask], label=means.ix[i]["label"], c=clrs[i])
            axp.fill_between(radar_fb_full[nprod][mask], meanst+sdt ,alpha=0.3, facecolor=clrs[i])
        ax.legend()
        """
        
       
        xylim = [-0.3,1]
        #x_dist_acc =
        mask_is2 = ~laser_fb_full_ave.mask
        mask_cs2 = np.sum(radar_fb_matrix.mask==0,axis=0)==5

        mask = np.logical_and(mask_is2,mask_cs2)
        axp.fill_between(np.arange(np.sum(mask)), xylim[0],xylim[1], where=icetype_al_full[mask] == 4, facecolor='lightgrey', alpha=0.5)
        
        epochs = list(range(np.sum(mask)))
        label_CS2 = 'radar fb CS2 (m)'
        axp.plot(epochs,radar_fb_mean[mask],label=label_CS2,color=colors_plot_cs2[0],marker='*',linestyle='-')
        label_CS2_std = 'std fb CS2 (m)'
        axp.fill_between(epochs,radar_fb_mean[mask]+(-1)*radar_fb_std[mask],radar_fb_mean[mask]+radar_fb_std[mask],label=label_CS2_std,alpha=0.4,color=colors_plot_cs2[0])

        label_IS2 = 'laser fb IS2 (m)'
        axp.plot(epochs,laser_fb_full_ave[mask],label=label_IS2,marker='*',color='green',linestyle='-')
        label_IS2_std = 'std fb IS2 (m)'
        axp.fill_between(epochs,laser_fb_full_ave[mask]-laser_fb_full_std[mask],laser_fb_full_ave[mask]+laser_fb_full_std[mask],label=label_IS2_std,alpha=0.4,color='green')
        

        
        """
        for nprod,cs2_gdr in enumerate(prod_L2P):
            label_CS2 = 'fb %s CS2 (m)' %(cs2_gdr)
            axp.plot(radar_fb_full[nprod][mask],label=label_CS2,color=colors_plot_cs2[nprod],marker='*',linestyle='-')
            axp.fill_between(radar_fb_full[nprod][mask]-radar_fb_std[nprod][mask],,label=label_CS2,alpha=0.3,color=colors_plot_cs2[nprod],marker='*',linestyle='-')

        label_IS2 = 'laser fb IS2 (m)'
        axp.plot(laser_fb_full_ave[mask],label=label_IS2,marker='*',color='palegreen',linestyle='-')
        """
        #axp.plot(radar_fb_spline[mask],label='mean CS2',color='blue')    
        #axp.plot(laser_fb_spline[mask],label='mean IS2',color='green')
        
        axp.set_ylim(xylim[0],xylim[1])
        axp.legend()

        plt.show()
        
        

    elif param=='roughness':
        print("\nComparing roughness")

        # Get parameters
        #------------------------------------------

        # ref coordinates
        ref_seg_lat = list(np.array(data_dict['CS2'][REF_GDR]['latref'],dtype=object)[idx_dates])
        ref_seg_lon = list(np.array(data_dict['CS2'][REF_GDR]['lonref'],dtype=object)[idx_dates])

        #
        lat1 = np.ma.concatenate(ref_seg_lat,axis=0)[:-1]; lat2=np.ma.concatenate(ref_seg_lat,axis=0)[1:]
        lon1 = np.ma.concatenate(ref_seg_lon,axis=0)[:-1]; lon2=np.ma.concatenate(ref_seg_lon,axis=0)[1:]
        mean_dist_btw_data = np.median(cf.dist_btw_two_coords(lat1,lat2,lon1,lon2))
        window_size = int(mean_dist_btw_data*25) #km

        # x_dist ref
        x_dist = list()
        for lat,lon in zip(ref_seg_lat,ref_seg_lon):
            x_dist.append(cf.distance_from_first_trk_pts(lat,lon,0))
        x_dist_full = np.ma.concatenate(x_dist,axis=0)
        x_dist_sum = np.cumsum(x_dist_full)
        npts = x_dist_sum.shape[0]
        
        # Get icetype along-track
        icetype_al = list()
        for n in range(ndates):
            lon = ref_seg_lon[n]
            if any(np.abs(np.diff(lon)) > 20): lon[lon > 180] = lon[lon > 180] - 360   
            icetype_alongtrack = cf.grid_to_track(ice_type[n],lons_icetype[n],lats_icetype[n],lon,ref_seg_lat[n])
            icetype_al.append(icetype_alongtrack)
        icetype_al_full = np.ma.concatenate(icetype_al,axis=0)


        # IS2 roughness
        #--------------------------

        # -> GAUSSIAN MEAN
        gaussian_w = [obj for nobj,obj in enumerate(data_dict['IS2']['ATL10']['gaussian_w']) if nobj in idx_dates]
        gaussian_w = np.ma.concatenate(gaussian_w,axis=1)
        is2_flag_leads =  [obj for nobj,obj in enumerate(data_dict['IS2']['ATL10']['flag_leads']) if nobj in idx_dates]
        is2_flag_leads = np.ma.concatenate(is2_flag_leads,axis=1)
        is2_flag_leads = is2_flag_leads==0.0
        
        # Get gaussian width only floes
        gaussian_w_ice = np.ma.masked_where(is2_flag_leads.data==False,gaussian_w,copy=True)
        gaussian_w_ice_mean = np.mean(gaussian_w_ice,axis=0)


        # -> SURFACE HEIGHT STD
        surface_h = [obj for nobj,obj in enumerate(data_dict['IS2']['ATL10']['surface_h']) if nobj in idx_dates]
        #surface_h_std = [rolling_stats(surfh, window_size, stats=['std'])[0] for surfh in surface_h]
        surface_h = np.ma.concatenate(surface_h,axis=1)
        std_surface_h = np.std(surface_h,axis=0)

      
        
        # CS2 roughness
        #--------------------------

        # ESA_BD
        
        ice_height = np.ma.concatenate(list(np.array(data_dict['CS2']['ESA_BD_GDR']['isa'],dtype=object)[idx_dates]),axis=0)
        sigma_h_bd = rolling_stats(ice_height, window_size, stats=['std'])[0]

        # LEGOS_SAM
        pp_legos = np.ma.concatenate(list(np.array(data_dict['CS2']['LEGOS_SAM']['pp'],dtype=object)[idx_dates]),axis=0)
        ice_height =  np.ma.concatenate(list(np.array(data_dict['CS2']['LEGOS_SAM']['isa'],dtype=object)[idx_dates]),axis=0)
        sigma_h_legos = rolling_stats(ice_height, window_size, stats=['std'])[0]
        flag_leads_legos = np.ma.concatenate(list(np.array(data_dict['CS2']['LEGOS_SAM']['flag_leads'],dtype=object)[idx_dates]),axis=0)
        
        # UoB
        roughness = np.ma.concatenate(list(np.array(data_dict['CS2']['UOB']['roughness'],dtype=object)[idx_dates]),axis=0)
        radar_height = np.ma.concatenate(list(np.array(data_dict['CS2']['UOB']['radar_h'],dtype=object)[idx_dates]),axis=0)
        flag_leads_uob = np.ma.concatenate(list(np.array(data_dict['CS2']['UOB']['flag_leads'],dtype=object)[idx_dates]),axis=0)
        ice_height =  np.ma.masked_where(flag_leads_uob.astype(bool)==True,radar_height,copy=True)
        sigma_h_uob = rolling_stats(ice_height, window_size, stats=['std'])[0]

        # AWI
        lew = np.ma.concatenate(list(np.array(data_dict['CS2']['AWI']['lew'],dtype=object)[idx_dates]),axis=0)
        lew = ma.masked_invalid(lew)
        
        pp_awi = np.ma.concatenate(list(np.array(data_dict['CS2']['AWI']['pp'],dtype=object)[idx_dates]),axis=0)
        radar_h =  np.ma.concatenate(list(np.array(data_dict['CS2']['AWI']['radar_h'],dtype=object)[idx_dates]),axis=0)
        flag_leads = np.ma.concatenate(list(np.array(data_dict['CS2']['AWI']['surface_type'],dtype=object)[idx_dates]),axis=0)==2.0
        flag_floes = np.ma.concatenate(list(np.array(data_dict['CS2']['AWI']['surface_type'],dtype=object)[idx_dates]),axis=0)==4.0
        ice_height = np.ma.masked_where(flag_leads==True,radar_h,copy=True)
        sigma_h_awi = rolling_stats(ice_height, window_size, stats=['std'])[0]
        

        
        # Show maps
        #---------------------------------
        """
        f1, ax = plt.subplots(1, 1,figsize=(6,6))
        gauss_w = list()
        lat_b = list()
        lon_b = list()
        for b in beamList:
            gauss_w.append(np.ma.concatenate(data_dict['IS2']['ATL10'][b]['gaussian_w'],axis=0))
            lat_b.append(np.ma.concatenate(data_dict['IS2']['ATL10'][b]['lat'],axis=0))
            lon_b.append(np.ma.concatenate(data_dict['IS2']['ATL10'][b]['lon'],axis=0))
        
        gauss_w =  np.ma.concatenate(gauss_w,axis=0)
        gauss_w = ma.masked_invalid(gauss_w,copy=True)
        lat = np.ma.concatenate(lat_b,axis=0)
        lon = np.ma.concatenate(lon_b,axis=0)
        
        plot_track_map(f1,ax,lon,lat,gauss_w,'IS2 Gaussian width',[0,0.4],mid_date)
        plt.show()
        """

        # Scatter comparison of surface roughness
        #--------------------------------

        f0, ax = plt.subplots(1, 1, sharey=True)
        f0.suptitle('Gaussian width IS2 (m) vs RMS topography IS2 (m)', fontsize=12)
        xylim = [[0.,0.4],[0.,0.8]]
        x_data = roughness
        x_label = 'Roughness lognormal CS2 (m)'
        y_label='Gaussian width IS2 (m)'
        y_data = gaussian_w_ice_mean 
        plot_scatter(ax,xylim,'m',x_data,x_label,y_data,y_label,icetype_al_full)


        # LEW vs Lognormal Landy
        f1, ax = plt.subplots(1, 1, sharey=True)
        f1.suptitle('LEW vs Lognomal roughness (Landy)', fontsize=12)
        xylim = [[0,0.7],[0,3]]
        x_data = roughness
        x_label = 'Lognormal roughness CS2 (m)'
        y_label='LeW CS2'
        y_data = lew
        plot_scatter(ax,xylim,'m',x_data,x_label,y_data,y_label,icetype_al_full)

        #plt.show()

        # sigma h vs Gaussian width
        f2, ax = plt.subplots(1, 1, sharey=True)
        f2.suptitle('Gaussian width IS2 (m) vs RMS topography IS2 (m)', fontsize=12)
        xylim = [[0.,0.4],[0.,0.8]]
        x_data = std_surface_h
        x_label = 'RMS topography IS2 (m)'
        y_label='Gaussian width IS2 (m)'
        y_data = gaussian_w_ice_mean 
        plot_scatter(ax,xylim,'m',x_data,x_label,y_data,y_label,icetype_al_full)

        #plt.show()

        # sigma H IS2 vs roughness logn
        f3, ax = plt.subplots(1, 1, sharey=True)
        f3.suptitle('sig HFloes IS2 vs roughness logN CS2', fontsize=12)
        xylim = [[0.,0.7],[0.,0.6]]
        y_data = std_surface_h
        y_label = 'RMS topography IS2 (m)'
        x_label='Lognormal roughness CS2 (m)'
        x_data = roughness 
        plot_scatter(ax,xylim,'m',x_data,x_label,y_data,y_label,icetype_al_full)

        plt.show()

        # PP vs roughness logn
        f4, ax = plt.subplots(1, 1, sharey=True)
        f4.suptitle('PP vs Lognomal roughness (Landy)', fontsize=12)
        xylim = [[0.,0.4],[0.,0.8]]
        x_data = roughness
        x_label = 'Lognormal roughness CS2 (m)'
        y_label='PP CS2'
        y_data = pp_legos
        plot_scatter(ax,None,'m',x_data,x_label,y_data,y_label,icetype_al_full)

        #plt.show()

        
        


        
        # Show along-track
        #-------------------
        xylim = [0,3]
        
        f1, axp = plt.subplots(1, 1,figsize=(15,8))
        f1.suptitle('Along-track roughness %s-%s' %(start_date,end_date), fontsize=14)

        mask = ~std_surface_h.mask
        axp.fill_between(np.arange(np.sum(mask)), xylim[0],xylim[1], where=icetype_al_full[mask] == 4, facecolor='lightgrey', alpha=0.5)

        # IS2 data
        label_IS2 = 'Gaussian w IS2 (m)'
        axp.plot(gaussian_w_ice_mean[mask],label=label_IS2)
        label_IS2 = 'std ice height IS2 (m)'
        axp.plot(std_surface_h[mask],label=label_IS2)

        # CS2 data
        # LEGOS
        label = 'pp_legos'
        axp.plot(pp_legos[mask],label=label)

        # UOB
        label = 'roughness_uob'
        axp.plot(roughness[mask],label=label)

        # AWI
        label = 'lew_awi'
        axp.plot(lew[mask],label=label)
        label = 'pp_awi'
        axp.plot(pp_awi[mask],label=label)
        
        #ESA_BD
        label = 'sigma_h_bd'
        axp.plot(sigma_h_bd[mask],label=label)
        

        axp.set_ylim(xylim[0],xylim[1])
        axp.legend()

        plt.show()


        # Figure 2
        #------------------------
        
        f2, axp = plt.subplots(1, 1,figsize=(15,8))
        f2.suptitle('Along-track roughness %s-%s' %(start_date,end_date), fontsize=14)

        xylim = [0,3]
        mask = ~std_surface_h.mask
        axp.fill_between(np.arange(np.sum(mask)), xylim[0],xylim[1], where=icetype_al_full[mask] == 4, facecolor='lightgrey', alpha=0.5)

        # IS2 data
        label_IS2 = 'std ice height IS2 (m)'
        axp.plot(std_surface_h[mask],label=label_IS2)

        # CS2 data
        # LEGOS
        label = 'sigma_h_legos'
        axp.plot(pp_legos[mask],label=label)

        # UOB
        label = 'sigma_h_uob'
        axp.plot(sigma_h_uob[mask],label=label)

        # AWI
        label = 'sigma_h_awi'
        axp.plot(sigma_h_awi[mask],label=label)

        #ESA_BD
        label = 'sigma_h_bd'
        axp.plot(sigma_h_bd[mask],label=label)
        

        axp.set_ylim(xylim[0],xylim[1])
        axp.legend()
        plt.show()

        
        
        
        
        
    else:
        print("\nUnknown parameter")
        

    

    
        
