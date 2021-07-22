#! /home/antlafe/anaconda3/bin/python

#
# stats_tools.py
#
# ----------------------------------------------------------------------
# Copyright (c) 2020 LEGOS/SERCO
# All rights reserved.
#

"""
DESCRIPTION:

     List of statistic tools/functions: Scatter, histogram, maps


"""

import netCDF4 as nc
import numpy as np
from numpy import ma 
from mpl_toolkits.basemap import Basemap
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt
from matplotlib import cm
from datetime import date, timedelta, datetime
import common_functions as cf
import warnings
import scipy.spatial
from scipy.stats import pearsonr, gaussian_kde,linregress
import matplotlib as mpl
import time
import matplotlib.dates as mdates
from matplotlib.patches import Circle, Wedge, Polygon

# Global attributs
###########################################

PATH_DATA= '/home/antlafe/Documents/work/projet_cryo2ice/data/'
PATH_INPUT = "/home/antlafe/Documents/work/projet_cryo2ice/data/Cryo2Ice/"
PATH_OUT = "/home/antlafe/Documents/work/projet_cryo2ice/outputs/"

param_opts = ['sd_month','find_regions']

colors_scatter = ['mediumseagreen','cornflowerblue','red']
colors_histo = ['mediumseagreen','cornflowerblue','royalblue','dodgerblue','navy','turquoise','blue']
colors_histo_fill = ['mediumseagreen','cornflowerblue','cornflowerblue','cornflowerblue','cornflowerblue','cornflowerblue','cornflowerblue','cornflowerblue']
color_line_histo =  ['mediumseagreen','cornflowerblue','royalblue','dodgerblue','navy','turquoise','dodgerblue']
colors_plot_cs2 = ['deepskyblue','dodgerblue','turquoise','royalblue','palegreen','cornflowerblue','royalblue','blue']

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

    return R,RMSD,slope

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
        h = ax.hist(data[~data.mask], 200, range=[xylim[0],xylim[1]],lw=1,histtype=u'step', facecolor=colors_histo_fill[n],label=label,fill=True,alpha=0.3)
        h = ax.hist(data[~data.mask], 200, range=[xylim[0],xylim[1]],lw=1,histtype=u'step', facecolor="None",edgecolor=colors_histo[n])
        if n==0: max_h = np.max(h[0])
        #data = ma.masked_where(np.isnan(data),data,copy=True) #[common_mask].
        Npts = np.sum(~data.mask)
        print("Npts(%s) = %i" %(label,Npts))
        mean_data = np.ma.mean(data)
        std_data = np.ma.std(data)
        print("%s: Mean %.2f; Std %.2f" %(label,mean_data,std_data))
        ax.axvline(x=mean_data,color=color_line_histo[n],lw=1)
        ypos = (n+1)/(2*(len(data_list)+1))*max_h
        #ax.annotate('Mean = %.3f%s, std=  %.3f%s' %(mean_data,units,std_data,units),xy=(mean_data, ypos),weight='bold',fontsize=10,color=colors_histo[n])

    textstr = "Npts = %i" %(Npts)
    props = dict(boxstyle='round', facecolor='white', alpha=0.8)
    ax.text(xylim[0]+2,1 , textstr, fontsize=12,verticalalignment='bottom',horizontalalignment ='left', bbox=props)
    ax.set_xlabel(xlabel,fontsize=12)
    ax.set_ylabel('density',fontsize=12)
    ax.legend()



    
def plot_track_map(fig,axm,lon,lat,data,label,xylim,date_icetype,units,flag_comp=True,alpha=1,size=2):


    """
    ax: axis
    lon_list: list of longitude to plot
    lat_list: list of latitude to plot
    data: list of corresponding data
    label: str of label of the data
    xylim: limits of color bar
    date_icetype: datetime object for ice type

    """

    boundinglat = 60 #max(np.ma.min(lat),70)
    #m = Basemap(projection='ortho',lat_0=70,lon_0=0,resolution='l',ax=axm)
    m = Basemap(projection='npstere', llcrnrlat=0,urcrnrlat=90,llcrnrlon=-180,urcrnrlon=180,boundinglat=boundinglat,lon_0=0, resolution='l',round=True,ax=axm)
    #m = Basemap(projection='npstere',boundinglat=boundinglat,lon_0=0, resolution='l' , round=False,ax=axm)
    m.drawcoastlines(linewidth=0.25, zorder=0)
    m.drawparallels(np.arange(90,-90,-5),  labels=[1,1,1,1],linewidth = 0.25, zorder=1)
    m.drawmeridians(np.arange(-180.,180.,30.),labels=[1,1,1,1],latmax=85, linewidth = 0.25, zorder=1)
    m.fillcontinents(color='dimgray',lake_color='grey', zorder=1)
    draw_round_frame(m,axm)
    #m.bluemarble(scale=1, zorder=-1)

    # defining color map
    if flag_comp:
        cmap='BrBG'
        if xylim is None:
            #xylim = [-np.max(np.abs(data)),np.max(np.abs(data))]            
            xylim = [-np.std(np.abs(data)),np.std(np.abs(data))]            
    else:
        cmap='tab20b' #'viridis' #'viridis' #nuplot2' # magma' #'cividis' #jet
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
    
        cmap_ice = mpl.colors.ListedColormap(["lightgrey", "grey"])
        #im = m.contour(xptsT , yptsT, OSISAF_ice_type,levels=1,colors='black',zorder=4)
        #im = m.contourf(xptsT , yptsT, OSISAF_ice_type,cmap=cmap_ice, alpha=0.8,zorder=1,antialiased=True)
        #ice_boundaries = np.ones(OSISAF_ice_type.shape)
        #ice_boundaries[np.logical_or(OSISAF_ice_type==4,OSISAF_ice_type==2)] = 2
        #axm.contour(xptsT , yptsT, ice_boundaries, linewidths=0.2, colors='grey', alpha=0.5,zorder=0)
        #im = m.contour(xptsT , yptsT, OSISAF_ice_type,linewidths=0.5,cmap=cmap_ice, alpha=1,zorder=2)

        #proxy = [plt.Rectangle((0,0),1,1,fc = pc.get_facecolor()[0]) for pc in im.collections]
        #plt.legend(proxy, ["MYI", "FYI"])

        # ice type full color
        im = m.contourf(xptsT , yptsT, OSISAF_ice_type,cmap=cmap_ice, alpha=0.8,zorder=1,antialiased=True)
        norm = mpl.colors.BoundaryNorm(np.arange(2,4), cmap_ice.N)
        cbar = fig.colorbar(im,ax=axm,ticks=[2.5,3.5],orientation='horizontal',fraction=0.046, pad=0.04,extend='both',shrink=0.70)
        cbar.ax.set_xticklabels(['First-Year Ice','Multi-Year Ice'])
        cbar.set_label('OSISAF daily ice type', labelpad=3)
        

    x,y = m(lon,lat)
    # show coordinates
    #for data in data:
    ndata = data.size - np.sum(data.mask)
    
    scat= m.scatter(x,y,c=data,s=size,cmap=cmap,vmin=xylim[0],vmax=xylim[1],zorder=3,alpha=alpha)
    cbaxes = fig.add_axes([0.85, 0.28, 0.02, 0.5]) 
    #cb = fig.colorbar(scat, ax=axm,cax = cbaxes,extend='both',fraction=0.046, pad=0.04,shrink=0.80)
    

    loc = mdates.AutoDateLocator()
    cb = fig.colorbar(scat, ticks=loc,format=mdates.AutoDateFormatter(loc),ax=axm,cax = cbaxes,extend='both',fraction=0.046, pad=0.04,shrink=0.80)
    cb.set_label("%s [%s]" %(label,units),fontsize=12)

    
    # text box
    textstr ='%5s=%.2f %s %5s=%.2f %s %5s=%i' %('mean',np.ma.mean(data),units,r'$\sigma$',np.ma.std(data),units,'Npts',ndata)
    #axm.text(1E6,4E5,textstr,fontsize=15,bbox=dict(boxstyle="round",ec=(0., 0., 0.),fc=(1., 1., 1.)))
    #axm.text(0.05, 0.05,textstr, transform=axm.transAxes,fontsize=12,bbox=dict(boxstyle="round",ec=(0., 0., 0.),fc=(1., 1., 1.)))

    return m,cmap


def add_data_track(bmap,cmap,lon,lat,data,xylim):

    x,y = bmap(lon,lat)
    if data is not None:
        x = ma.masked_where(data.mask,x,copy=True); y = ma.masked_where(data.mask,y,copy=True)
        scat= bmap.scatter(x,y,c='black',s=9,marker='o',cmap=cmap,vmin=xylim[0],vmax=xylim[1],zorder=3,alpha=1)
        scat= bmap.scatter(x,y,c=data,s=5,marker='o',cmap=cmap,vmin=xylim[0],vmax=xylim[1],zorder=3,alpha=1)
        #cb = fig.colorbar(scat, ax=axm,extend='both',fraction=0.046, pad=0.04,shrink=0.80)
        #cb.set_label("%s [%s]" %(label,units),fontsize=12)
    else:
        bmap.plot(x,y, linewidth=1.5, color='yellow',linestyle='--')



def draw_polygon(lat,lon,m,color='blue'):

    x,y = m(lon,lat)
    coord = list()
    for xs,ys in zip(x,y):
        coord.append([xs,ys])
    coord.append(coord[0])

    xs, ys = zip(*coord) #create lists of x and y values
    m.plot(xs,ys,zorder=3,color=color) 
    
        

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


def draw_round_frame(m,ax, width_percent=0.05, degree=45):
    centre_x = (ax.get_xlim()[0] + ax.get_xlim()[1]) / 2
    centre_y = (ax.get_ylim()[0] + ax.get_ylim()[1]) / 2
    width = abs(centre_x) * width_percent

    inner_radius = abs(centre_x) - width/2
    outer_radius = inner_radius + width

    angle_breaks = list(range(0, 361, degree))

    for i, (from_angle, to_angle) in enumerate(list(zip(angle_breaks[:-1], angle_breaks[1:]))):
        color='white' if i%2 == 0 else 'black'
        wedge = Wedge((centre_x, centre_y), outer_radius, from_angle, to_angle, width=outer_radius - inner_radius,
                      facecolor=color,
                      edgecolor='black',
                      clip_on=False,
                      ls='solid',
                      lw=1)
        ax.add_patch(wedge)



