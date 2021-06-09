from mpl_toolkits.basemap import Basemap
import numpy as np
import os
import sys
import pdb
import time
import numpy.ma as ma
import matplotlib.pyplot as plt

GRID_FILTERING_MODES = [
    'filter_gauss', # data weighted according to the range to the center of the pixel
    'filter_mean',  # mean of data within the pixel
    'filter_weight', # weighted mean of the data within the pixel according to err value
    'filter_median',  # median (web) of data within the pixel
    'gaussian',
    'RBF',    # memory error
    'SCIPY_L',
    'SCIPY_C',
    'kevin_filter',
]

# ----------------------------------------------------------------------
# 
# ctoh_grid_data
# 
def ctoh_grid_data(mode, lon, lat, all_data_in, data_err_or_mode, map_frame, pixel_size, range_filter,name=None,verbose=0):
    
    # in case in old fashion uniq data
    if not isinstance(all_data_in,dict):
        data = {}
        if name is None:
            data['fb'] = all_data_in
        else:
            data[name] = all_data_in
    else:
        data = all_data_in

    if not mode in GRID_FILTERING_MODES:
        sys.exit("Unknown mode %s. Please select within %s." % (mode, GRID_FILTERING_MODES))

    #if range_filter != pixel_size or mode=='filter_gauss':
    if mode == 'filter_weight':
        print("Mode : %s, Pixel size : %i, Range : %i" %(mode, pixel_size, range_filter))
        return grid_and_filter_err(lon, lat, data, data_err_or_mode, map_frame, pixel_size, verbose=verbose)
    else: #  mode=='filter_gauss':
        print("Mode : %s, Pixel size : %i, Range : %i" %(mode, pixel_size, range_filter))
        return  grid_and_filter_wrt_distance(lon, lat, data, map_frame, pixel_size, mode, range_filter, verbose=verbose)  
    """
    elif mode == 'filter_mean':
        print("Mode : %s, Pixel size : %i, Range : %i" %(mode, pixel_size, range_filter))
        return grid_and_mean_filter(lon, lat, data, map_frame, pixel_size, verbose=verbose)
    
    elif mode == 'filter_median':
        print("Mode : %s, Pixel size : %i, Range : %i" %(mode, pixel_size, range_filter))
        return grid_and_median_filter(lon, lat, data, map_frame, pixel_size,range_filter, verbose=verbose)
        """
    # UNUSED

    if mode == 'gaussian':
        return  grid_and_gaussian_filter(lon, lat, data, map_frame, pixel_size, range_filter, verbose=verbose)
  
    if mode == 'SCIPY_L':
        return  grid_and_SCIPY_filter(lon, lat, data, 'linear', map_frame, pixel_size,  verbose=verbose)

    if mode == 'SCIPY_C':
        return  grid_and_SCIPY_filter(lon, lat, data, 'cubic', map_frame, pixel_size,  verbose=verbose)

    if mode == 'RBF':
        return  grid_and_RBF_filter(lon, lat, data, data_err_or_mode, map_frame, pixel_size, range_filter, verbose=verbose)
    if mode == 'kevin_filter':
        #an =2015;mois=4
        return grid_and_kevin_median_filter(lon, lat, data, map_frame, pixel_size,range_filter, verbose=verbose)
        #return  gridding(lon,lat,data,lon,mois,an,map_frame,pixel_size=50000, verbose=0)
                

    sys.exit("Unknown mode %s" % mode)

   
# ----------------------------------------------------------------------
# 
# init_grid
# 
def init_grid(lon, lat, all_data_in, map_frame, 
              pixel_size=10000, verbose=0):

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

    if verbose>0:
        print("init duration: %.1f seconds " %(time.time()-start_time_init))

    return lon, lat, all_data, x_grid, y_grid, x_track, y_track, lon_grid_mesh, lat_grid_mesh,  x_grid_mesh, y_grid_mesh, pixels_track, nb_pixels_1D



# ----------------------------------------------------------------------
# 
# grid_and_filter_wrt_distance
# 
def grid_and_filter_wrt_distance(lon, lat, all_data_in, map_frame, pixel_size=10000,mode='filter_gauss',range_filter=50000, verbose=0):
    
    print('\n\tGrid and filter with a filter range greater than the pixel size')
    start_time = time.time()

    # init grid
    lon, lat, all_data, x_grid, y_grid, x_track, y_track, lon_grid_mesh, lat_grid_mesh,  x_grid_mesh, y_grid_mesh, pixels_track, nb_pixels_1D  = init_grid(lon, lat, all_data_in, map_frame, pixel_size, verbose)

 
    # SORT TRACK POINTS    
    # 
    # --------------------------------
    print('\t---> Assign track points to the grid pixels')

    start_time_sort = time.time()

    # For each track point, choose the grid pixels that will use this point to compute the pixel averaged data
    range_filter_in_nb_of_pixels = np.ceil(range_filter/pixel_size)
    # Allocate the 2D jagged list
    grid_data_considered = [[[] for j in range(nb_pixels_1D)] for i in range(nb_pixels_1D)] # empty matrix of lists (shape=gridCoord shape)
    #grid_number_of_data_considered = [[[] for j in range(nb_pixels_1D)] for i in range(nb_pixels_1D)]
    #grid_data_considered = np.empty((nb_pixels_1D,nb_pixels_1D),dtype=object)
    #grid_data_considered.fill([])
    #grid_data_considered = np.frompyfunc(list,1,1)(grid_data_considered)
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

        #grid_number_of_data_considered[x_i,y_i] = 99
        # Place number of the current track point in the grid pixels in range of this point

        # No loop
        """
        foo = lambda l:l.append(k)
        vfunc = np.vectorize(foo)
        vfunc(grid_data_considered[int(min_row_subGrid):int(max_row_subGrid)+1][int(min_col_subGrid):int(max_col_subGrid)+1])"""        
        
        for i in range(int(min_row_subGrid), int(max_row_subGrid)+1):
            for j in range(int(min_col_subGrid), int(max_col_subGrid)+1):
                # Check if a satellite tracks goes over pixel
                #f np.any(np.all(pixels_track-np.array([i,j])[:, np.newaxis] == 0, axis=0)):
                grid_data_considered[i][j].append(k)
    
                """
                dist_to_track = np.sqrt((i-x_i)**2 + (j-y_i)**2)
                min_dist_to_closest_track[i][j] = min(min_dist_to_closest_track[i][j],dist_to_track)
                weight_min_dist_to_closest_track[i][j] = (-1/range_filter_in_nb_of_pixels)*min_dist_to_closest_track[i][j] + 1
                """
                    #grid_number_of_data_considered[i][j] = grid_number_of_data_considered[i][j]+1
                    
    
    # Delete unconsistent data point (seen by too few points)
    """
    unconsistency_conditionq = ((grid_number_of_data_considered <= 99) & (grid_number_of_data_considered > 0))
    idx_unconsistent_pix = np.argwhere(unconsistency_condition)
    for i,j in idx_unconsistent_pix:
            grid_data_considered[i][j] = []"""
            
    #[[del grid_data_considered[i][j] for i in idx_unconsistent_pix[:,0]] for j in idx_unconsistent_pix[:,1]]

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
            #if grid_number_of_data_considered[i][j] < range_filter_in_nb_of_pixels*50: continue

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
            if mode=='filter_gauss':
                weighting = np.exp(-squared_dist/squared_range_filter)
                
                for p_name,data in all_data.items():
                    data_results[p_name][i,j] = np.nansum(data[numbers_of_data_used]*weighting)/np.sum(weighting)
                    data_results[rms_p_name][i,j] = np.std(data[numbers_of_data_used])

            elif mode=='filter_median':
                for p_name,data in all_data.items():
                    data_results[p_name][i,j] = np.nanmedian(data[numbers_of_data_used])
                    data_results[rms_p_name][i,j] =  np.std(data[numbers_of_data_used])

            elif mode=='filter_mean':
                for p_name,data in all_data.items():
                    data_results[p_name][i,j] = np.nanmean(data[numbers_of_data_used])
                    data_results[rms_p_name][i,j] = np.std(data[numbers_of_data_used])

        
    print("grid_and_filter_wrt_distance mode %s range %f pixel %f" % (mode, range_filter, pixel_size))
    if verbose>0:       
        print('filtering duration: %.1f minutes' %((time.time()-start_time_filter)/60))
        print('total duration of the grid_and_filter function: %.1f minutes' %((time.time()-start_time)/60))

    return x_grid, y_grid, lat_grid_mesh, lon_grid_mesh, x_grid_mesh, y_grid_mesh, data_results






# ----------------------------------------------------------------------
# 
# grid_and_mean_filter
# 
def grid_and_mean_filter(lon, lat, all_data_in, map_frame, 
                         pixel_size=50000, verbose=0):
    
    start_time = time.time()

    lon, lat, all_data, x_grid, y_grid, x_track, y_track, lon_grid_mesh, lat_grid_mesh,  x_grid_mesh, y_grid_mesh, pixels_track, nb_pixels_1D = init_grid(lon, lat, all_data_in, map_frame, pixel_size, verbose)

    # SORT TRACK POINTS    
    # --------------------------------
    print('\n\tAssign track points to the grid pixels')

    start_time_sort = time.time()

    # Allocate memory for the 2D jagged list
    grid_data_considered = [[[] for j in range(nb_pixels_1D)] for i in range(nb_pixels_1D)] # empty matrix of lists (shape=gridCoord shape)
    progress_prev = -1
    nb_track_data = pixels_track.shape[1]
    for k in range(nb_track_data):
        if verbose>0:
            progress = np.floor(100*k/nb_track_data)
            if(progress != progress_prev): print("sorting progress: %.0f %%" %progress)
            progress_prev = progress

        pixel_covering_data = pixels_track[:,k] # subscripts of the pixel including the current projected track point
        
        is_pixel_in_the_grid = pixel_covering_data[0]>=0 and pixel_covering_data[0]<nb_pixels_1D and pixel_covering_data[1]>=0 and pixel_covering_data[1]<nb_pixels_1D
        
        # Add the data number in thecorresponding pixel
        if is_pixel_in_the_grid: grid_data_considered[pixel_covering_data[0]][pixel_covering_data[1]].append(k)

    if verbose>0:
        print("sort duration: %.1f minutes " %((time.time()-start_time_sort)/60))
    
    # FILTER           
    # --------------------------------
    print('\n\tSpatial filtering')
    
    start_time_filter = time.time()

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
            numbers_of_data_used = grid_data_considered[i][j]
            
            # If there are no data available to filter on the current pixel i,j, filter the next pixel
            if not numbers_of_data_used: continue
            for p_name,data in all_data.items():
                data_results[p_name][i,j] = np.sum(data[numbers_of_data_used])/len(numbers_of_data_used)
                data_results[rms_p_name][i,j] = np.std(data[numbers_of_data_used])
            
    if verbose>0:
        print('filtering duration: %.1f minutes' %((time.time()-start_time_filter)/60))
        print('total duration of the grid_and_filter function: %.1f minutes' %((time.time()-start_time)/60))

    return x_grid, y_grid, lat_grid_mesh, lon_grid_mesh, x_grid_mesh, y_grid_mesh, data_results





# ----------------------------------------------------------------------
# 
# grid_and_filter_err
# 
def grid_and_filter_err(lon, lat, all_data_in, err_data, map_frame, pixel_size=50000, verbose=0):
    
    print('\n\tGrid and filter with err weight')
    start_time = time.time()

    # init grid
    lon, lat, all_data, x_grid, y_grid, x_track, y_track, lon_grid_mesh, lat_grid_mesh,  x_grid_mesh, y_grid_mesh, pixels_track, nb_pixels_1D = init_grid(lon, lat, all_data_in, map_frame, pixel_size, verbose)

    
    # SORT TRACK POINTS    
    # --------------------------------
    start_time_sort = time.time()

    # Allocate memory for the 2D jagged list
    grid_data_considered = [[[] for j in range(nb_pixels_1D)] for i in range(nb_pixels_1D)] # empty matrix of lists (shape=gridCoord shape)
    progress_prev = -1
    nb_track_data = pixels_track.shape[1]
    for k in range(nb_track_data):
        if verbose>0:
            progress = np.floor(100*k/nb_track_data)
            if(progress != progress_prev): print("sorting progress: %.0f %%" %progress)
            progress_prev = progress

        pixel_covering_data = pixels_track[:,k] # subscripts of the pixel including the current projected track point
        
        is_pixel_in_the_grid = pixel_covering_data[0]>=0 and pixel_covering_data[0]<nb_pixels_1D and pixel_covering_data[1]>=0 and pixel_covering_data[1]<nb_pixels_1D
        
        # Add the data number in the corresponding pixel
        if is_pixel_in_the_grid: grid_data_considered[pixel_covering_data[0]][pixel_covering_data[1]].append(k)

    if verbose>0:
        print("sort duration: %.1f minutes " %((time.time()-start_time_sort)/60))

    
    # FILTER           
    # --------------------------------
    print('\n\tSpatial filtering')
    
    start_time_filter = time.time()

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
            numbers_of_data_used = grid_data_considered[i][j]
            
            # If there are no data available to filter on the current pixel i,j, filter the next pixel
            if not numbers_of_data_used: continue

            # Compute the spatial filter on the current pixel i,j
            # The weighing is inversely proportionnal to the error on the data
            weighting = 1/err_data[numbers_of_data_used]
            for p_name,data in all_data.items():
                data_results[p_name][i,j] = np.sum(data[numbers_of_data_used]*weighting)/np.sum(weighting)
                data_results[rms_p_name][i,j] = np.std(data[numbers_of_data_used])

    if verbose>0:      
        print('filtering duration: %.1f minutes' %((time.time()-start_time_filter)/60))
        print('total duration of the grid_and_filter function: %.1f minutes' %((time.time()-start_time)/60))

    return x_grid, y_grid, lat_grid_mesh, lon_grid_mesh, x_grid_mesh, y_grid_mesh, data_results


# ----------------------------------------------------------------------
# 
# grid_and_scipy_filter
# 
def grid_and_SCIPY_filter(lon, lat, all_data_in, mode, map_frame, 
                          pixel_size,  verbose=0):


    data_results = {}
    from scipy.interpolate import griddata as SCIPY_griddata
    
    lon, lat, all_data, x_grid, y_grid, x_track, y_track, lon_grid_mesh, lat_grid_mesh,  x_grid_mesh, y_grid_mesh, pixels_track, nb_pixels_1D = init_grid(lon, lat, all_data_in, map_frame, pixel_size, verbose)

    points = (x_track, y_track)

    for p_name in all_data.keys():
        data_results[p_name] = SCIPY_griddata(points, np.array(all_data[p_name]), (x_grid_mesh, y_grid_mesh), method=mode)

    return x_grid, y_grid, lat_grid_mesh, lon_grid_mesh, x_grid_mesh, y_grid_mesh, data_results

#>>> grid_z0 = griddata(points, values, (grid_x, grid_y), method='nearest')
#>>> grid_z1 = griddata(points, values, (grid_x, grid_y), method='linear')
#>>> grid_z2 = griddata(points, values, (grid_x, grid_y), method='cubic')

# ----------------------------------------------------------------------
# 
# grid_and_RBF_filter
# 
def grid_and_RBF_filter(lon, lat, all_data_in, rbf_mode, map_frame, 
                        pixel_size, range_filter, verbose=0):

    data_results = {}
    from scipy.interpolate import Rbf

    lon, lat, all_data, x_grid, y_grid, x_track, y_track, lon_grid_mesh, lat_grid_mesh,  x_grid_mesh, y_grid_mesh, pixels_track, nb_pixels_1D = init_grid(lon, lat, all_data_in, map_frame, pixel_size, verbose)


    
    if rbf_mode : mode  = rbf_mode
    else: mode = 'gaussian'


    # use RBF  -> MEMORY ERROR
    #'multiquadric': sqrt((r/self.epsilon)**2 + 1) DEFAULT
    #'inverse': 1.0/sqrt((r/self.epsilon)**2 + 1)
    #'gaussian': exp(-(r/self.epsilon)**2)
    #'linear': r
    #'cubic': r**3
    #'quintic': r**5
    #'thin_plate': r**2 * log(r)
    for p_name in all_data.keys():
        rbf = Rbf(x_track, y_track, np.array(all_data[p_name]), function=mode) #, epsilon=2)
        data_results[p_name] = rbf(x_grid_mesh, y_grid_mesh)

    return x_grid, y_grid, lat_grid_mesh, lon_grid_mesh, x_grid_mesh, y_grid_mesh, data_results

# ----------------------------------------------------------------------
# 
# grid_and_gaussian_filter
# 
def grid_and_gaussian_filter(lon, lat, all_data_in, map_frame, 
                         pixel_size=50000, verbose=0):

    data_results = {}
    start_time = time.time()

    lon, lat, all_data, x_grid, y_grid, x_track, y_track, lon_grid_mesh, lat_grid_mesh,  x_grid_mesh, y_grid_mesh, pixels_track, nb_pixels_1D = init_grid(lon, lat, all_data_in, map_frame, pixel_size, verbose)

    from sklearn.gaussian_process import GaussianProcess
    ptx = x_track
    pty = y_track
    for p_name in all_data.keys():
        z = np.array(all_data[p_name])

        gp = GaussianProcess(regr='quadratic',corr='cubic',theta0=np.min(z),thetaL=min(z),thetaU=max(z),nugget=0.05)
        gp.fit(X=np.column_stack([pty,ptx]),y=z)
        rr_cc_as_cols = np.column_stack([gridy.flatten(), gridx.flatten()])
        data_results[p_name] = gp.predict(rr_cc_as_cols).reshape((ncol,nrow))

    return x_grid, y_grid, lat_grid_mesh, lon_grid_mesh, x_grid_mesh, y_grid_mesh, data_results

#
# ----------------------------------------------------------------------
# 
# grid_and_median_filter
# 
def grid_and_median_filter(lon, lat, all_data_in, map_frame, pixel_size,range_size, verbose=0):
    
    data_results = {}
    start_time = time.time()

    lon, lat, all_data, x_grid, y_grid, x_track, y_track, lon_grid_mesh, lat_grid_mesh,  x_grid_mesh, y_grid_mesh, pixels_track, nb_pixels_1D = init_grid(lon, lat, all_data_in, map_frame, pixel_size, verbose)

    xmax = np.max(x_grid)-pixel_size
    xmin = np.min(x_grid)
    ymax = np.max(y_grid)-pixel_size
    ymin = np.min(y_grid)

    ##XXXX grid_loc = griddata(x_track, y_track, None, xmin,xmax,ymin,ymax, binsize=pixel_size, retbin=False, retloc=True)

    # Allocate the array that will contain the filtered and gridded data
    #data_results=ma.masked_all((nb_pixels_1D,nb_pixels_1D))
    data_results = {}
    for p_name in all_data.keys():
        data_results[p_name] = griddata(x_track, y_track, np.array(all_data[p_name]), xmin,xmax,ymin,ymax, binsize=range_size, retbin=False, retloc=False)

    return x_grid, y_grid, lat_grid_mesh, lon_grid_mesh, x_grid_mesh, y_grid_mesh, data_results

# From:
# http://scipy-cookbook.readthedocs.io/items/Matplotlib_Gridding_irregularly_spaced_data.html
def griddata(x, y, z,xmin,xmax,ymin,ymax, binsize, retbin=True, retloc=True):
    """
    Place unevenly spaced 2D data on a grid by 2D binning (nearest
    neighbor interpolation).
    
    Parameters
    ----------
    x : ndarray (1D)
        The idependent data x-axis of the grid.
    y : ndarray (1D)
        The idependent data y-axis of the grid.
    z : ndarray (1D)
        The dependent data in the form z = f(x,y).
    binsize : scalar, optional
        The full width and height of each bin on the grid.  If each
        bin is a cube, then this is the x and y dimension.  This is
        the step in both directions, x and y. Defaults to 0.01.
    retbin : boolean, optional
        Function returns `bins` variable (see below for description)
        if set to True.  Defaults to True.
    retloc : boolean, optional
        Function returns `wherebins` variable (see below for description)
        if set to True.  Defaults to True.
   
    Returns
    -------
    grid : ndarray (2D)
        The evenly gridded data.  The value of each cell is the median
        value of the contents of the bin.
    bins : ndarray (2D)
        A grid the same shape as `grid`, except the value of each cell
        is the number of points in that bin.  Returns only if
        `retbin` is set to True.
    wherebin : list (2D)
        A 2D list the same shape as `grid` and `bins` where each cell
        contains the indicies of `z` which contain the values stored
        in the particular bin.


    Revisions
    ---------
    2010-07-11  ccampo  Initial version
    """
    # get extrema values.
    # xmin, xmax = x.min(), x.max()
    #ymin, ymax = y.min(), y.max()

    # make coordinate arrays.
    xi      = np.arange(xmin, xmax+binsize, binsize)
    yi      = np.arange(ymin, ymax+binsize, binsize)
    xi, yi = np.meshgrid(xi,yi)

    # make the grid.
    grid           = np.zeros(xi.shape, dtype=x.dtype)
    nrow, ncol = grid.shape
    if retbin: bins = np.copy(grid)

    # create list in same shape as grid to store indices
    if retloc:
        wherebin = np.copy(grid)
        wherebin = wherebin.tolist()

    # fill in the grid.
    for row in range(nrow):
        for col in range(ncol):
            xc = xi[row, col]    # x coordinate.
            yc = yi[row, col]    # y coordinate.

            # find the position that xc and yc correspond to.
            posx = np.abs(x - xc)
            posy = np.abs(y - yc)
            ibin = np.logical_and(posx < binsize/2., posy < binsize/2.)
            ind  = np.where(ibin == True)[0]

            # fill the bin.
            bin = z[ibin]
            if retloc: wherebin[row][col] = ind
            if retbin: bins[row, col] = bin.size
            if bin.size != 0:
                binval         = np.median(bin)
                grid[row, col] = binval
            else:
                grid[row, col] = np.nan   # fill empty bins with nans.

    # return the grid
    if retbin:
        if retloc:
            return grid, bins, wherebin
        else:
            return grid, bins
    else:
        if retloc:
            return grid, wherebin
        else:
            return grid




#
# ----------------------------------------------------------------------
# 
# grid_and_median_filter kevin alg
# 
def grid_and_kevin_median_filter(lon, lat, all_data_in, map_frame,pixel_size,range_filter, verbose=0):
    
    data_results = {}
    #start_time = time.time()

    lon, lat, all_data, x_grid, y_grid, x_track, y_track, lon_grid_mesh, lat_grid_mesh,  x_grid_mesh, y_grid_mesh, pixels_track, nb_pixels_1D = init_grid(lon, lat, all_data_in, map_frame, pixel_size, verbose)

    #xmax = np.max(x_grid)-pixel_size
    #xmin = np.min(x_grid)
    #ymax = np.max(y_grid)-pixel_size
    #ymin = np.min(y_grid)
    #vf = {}
    latmax = np.nanmax(lat)
    for p_name in all_data.keys():
        data_results[p_name]=np.empty((nb_pixels_1D,nb_pixels_1D))
        data_results[p_name][:,:] = np.nan

    for i in range(0,nb_pixels_1D,1):
        #progressbar(i, x2)
        for j in range(0,nb_pixels_1D,1):
            #if (age[i,j]>0):
                if (lat_grid_mesh[i,j]>65) & (lat_grid_mesh[i,j]<latmax):
                    xc=x_grid_mesh[i,j]
                    yc=y_grid_mesh[i,j]
                    D1=np.array(np.sqrt((xc-x_track)*(xc-x_track)+(yc-y_track)*(yc-y_track)))
                    w=D1[D1<range_filter]
                    if len(w)>0:
                        ########## Freeboard
                        for p_name in all_data.keys():
                            fb = all_data[p_name]
                            data_results[p_name][i,j]=np.nanmedian(fb[D1<range_filter],axis=0)

    for p_name in all_data.keys():
        data_results[p_name]=np.ma.masked_invalid(data_results[p_name])
        #data_results[p_name] = vf[p_name]

    return x_grid, y_grid, lat_grid_mesh, lon_grid_mesh, x_grid_mesh, y_grid_mesh, data_results



def append_vec(mat, elt):
    mat.append(elt)
    return None

'''

from netCDF4 import Dataset
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema
from scipy.interpolate import interp1d
from mpl_toolkits.basemap import Basemap
import os
from os.path import *
import math
from netCDF4 import Dataset
import glob
import sys
from ctoh_multi_get_args import ctoh_multi_get_args
from ctoh_read_params import ctoh_read_params
#from progress import *
#from W99 import W99

#
# ----------------------------------------------------------------------
# 
# Kevin grid_and_filter function
# 


def gridding(lon,lat,all_data,peak,mois,an,map_frame,pixel_size=50000, verbose=0):

    data_results = {}
    
    # Initialisation
    lon, lat, all_data, x_grid, y_grid, x_track, y_track, lon_grid_mesh, lat_grid_mesh,  x_grid_mesh, y_grid_mesh, pixels_track, nb_pixels_1D = init_grid(lon, lat, all_data, map_frame, pixel_size, verbose)


    latmax = np.max(lat)
 
    #fb=fb[np.isnan(peak)==0]
    #lon=lon[np.isnan(peak)==0]
    #lat=lat[np.isnan(peak)==0]
    #peak_all=peak_all[np.isnan(peak)==0]
    #peak=peak[np.isnan(peak)==0]

    x2=nb_pixels_1D;x1=nb_pixels_1D
    #nb_pixels_1D = 722
    #x2=722;x1=722
    longitude = lon_grid_mesh
    latitude = lat_grid_mesh
    #longitude = np.zeros((x2,x1))
    #latitude = np.zeros((x2,x1))
    #D3 = np.zeros((x2,x1))
    Xc = np.zeros((x2,x1))
    Yc = np.zeros((x2,x1))
    age = np.zeros((x2,x1))
    agee = np.zeros((x2,x1))
    x = np.zeros((len(lon),1))
    y = np.zeros((len(lon),1))
    #x_grid = np.linspace(x_min_grid, x_max_grid, nb_pixels_1D)
    #y_grid = np.linspace(y_max_grid, y_min_grid, nb_pixels_1D)
    D1 = np.zeros((len(lon),1))
    vf = {}
    """
    vf2 = np.zeros((x2,x1))
    vf3 = np.zeros((x2,x1))
    vf4 = np.zeros((x2,x1))
    vf5 = np.zeros((x2,x1))"""
    w = np.zeros((x2,x1))


    """
    if int(mois)==1:
        file='/data/usrcto/mirror/external_data/nsidc/nsidc0611_seaice_age_v3/grid_nc/'+str(an)+'/iceage.grid.week.'+str(an)+'01*.w03.n.v3.nc'
    elif int(mois)==2:
        file='/data/usrcto/mirror/external_data/nsidc/nsidc0611_seaice_age_v3/grid_nc/'+str(an)+'/iceage.grid.week.'+str(an)+'02*.w07.n.v3.nc'
    elif int(mois)==3:
        file='/data/usrcto/mirror/external_data/nsidc/nsidc0611_seaice_age_v3/grid_nc/'+str(an)+'/iceage.grid.week.'+str(an)+'03*.w11.n.v3.nc'
    elif int(mois)==4:
        file='/data/usrcto/mirror/external_data/nsidc/nsidc0611_seaice_age_v3/grid_nc/'+str(an)+'/iceage.grid.week.'+str(an)+'04*.w15.n.v3.nc'
    elif int(mois)==10:
        file='/data/usrcto/mirror/external_data/nsidc/nsidc0611_seaice_age_v3/grid_nc/'+str(an)+'/iceage.grid.week.'+str(an)+'18*.w42.n.v3.nc'
    elif int(mois)==11:
        file='/data/usrcto/mirror/external_data/nsidc/nsidc0611_seaice_age_v3/grid_nc/'+str(an)+'/iceage.grid.week.'+str(an)+'11*.w46.n.v3.nc'
    elif int(mois)==12:
        file='/data/usrcto/mirror/external_data/nsidc/nsidc0611_seaice_age_v3/grid_nc/'+str(an)+'/iceage.grid.week.'+str(an)+'12*.w50.n.v3.nc'
    file2='/data/usrcto/mirror/external_data/nsidc/nsidc0611_seaice_age_v3/lonlat.nsidc_ease_grid.north.12.5.nc'


    fh = Dataset(file2, mode='r')
    longitude= np.squeeze(np.double(fh.variables['lon'][:]))
    latitude= np.squeeze(np.double(fh.variables['lat'][:]))
    fh.close()

    file=glob.glob(file)
    if len(file)==0:
        file=['/data/usrcto/mirror/external_data/nsidc/nsidc0611_seaice_age_v3/grid_nc/2008/iceage.grid.week.20080314.w11.n.v3.nc']
        print('\n')
        print('ATTENTION: PAS DE GRILLE NSIDC A CETTE PERIODE')
        print('      => GRILLE DE DEFAUT: MARS 2008')
        print('\n')
    fh = Dataset(file[0], mode='r')
    age = np.double(np.array(fh.variables['ice_age'][:]))
    fh.close()
    age=age[0,:,:]
    AGE=np.copy(age)
    agee=np.copy(age)*np.NaN
    agee[age==1]=0
    agee[age>1]=1"""

    lon=np.squeeze(np.array(lon))
    lat=np.squeeze(np.array(lat))
    x=(90-lat)*111000*np.cos(lon/180*np.pi)
    y=(90-lat)*111000*np.sin(lon/180*np.pi)
    Xc=(90-latitude)*111000*np.cos(longitude/180*np.pi)
    Yc=(90-latitude)*111000*np.sin(longitude/180*np.pi)



    ## W99
    #vf3,vf6=W99(mois,an,longitude,latitude,AGE)


    """
    ########### Tri des données se trouvant près de l'eau libre #######
    radius=26000
    for i in range(0,x2,1):
        progressbar(i, x2)
        for j in range(0,x1,1):
           #if (AGE[i,j]==-1):
                if (latitude[i,j]>65) & (latitude[i,j]<81.5):
                    xc=Xc[i,j]
                    yc=Yc[i,j]
                    D1=np.array(np.sqrt((xc-x)*(xc-x)+(yc-y)*(yc-y)))
                    w=D1[D1<radius]
                    if len(w)>0:
                        ########## Freeboard
                        fb[D1<radius]=np.NaN
                        #peak[D1<radius]=np.NaN
                        #peak_all[D1<radius]=np.NaN

    peak=np.squeeze(peak)
    fb=fb[np.isnan(peak)==0]
    x=x[np.isnan(peak)==0]
    y=y[np.isnan(peak)==0]
    #peak_all=peak_all[np.isnan(peak)==0]
    #peak=peak[np.isnan(peak)==0]

    ########### Tri des données se trouvant près de l'eau libre #######
    """


    radius=100000
    for p_name in all_data.keys():
        vf[p_name]=np.empty((x2,x1))
        vf[p_name][:,:] = np.nan
    #vf2=vf2*np.NaN
    #vf3=vf3*np.NaN
    #vf4=vf4*np.NaN
    #vf5=vf5*np.NaN




    for i in range(0,x2,1):
        #progressbar(i, x2)
        for j in range(0,x1,1):
            #if (age[i,j]>0):
                if (latitude[i,j]>65) & (latitude[i,j]<latmax):
                    xc=Xc[i,j]
                    yc=Yc[i,j]
                    D1=np.array(np.sqrt((xc-x)*(xc-x)+(yc-y)*(yc-y)))
                    w=D1[D1<radius]
                    if len(w)>0:
                        ########## Freeboard
                        for p_name in all_data.keys():
                            fb = all_data[p_name]
                            vf[p_name][i,j]=np.nanmedian(fb[D1<radius],axis=0)
                        #vf2[i,j]=np.nanmedian(peak[D1<radius])
                        #vf5[i,j]=np.nanmedian(peak_all[D1<radius])
                        ######### W99
#D2=np.array(np.sqrt((xc-AA)*(xc-AA)+(yc-BB)*(yc-BB)))
#vf3[i,j]=np.nanmean(snow[D2==np.nanmin(D2)])/100
                        ######### Sea ice age
                        #D3=np.array(np.sqrt((xc-Xc)*(xc-Xc)+(yc-Yc)*(yc-Yc)))
                        #w=D3[D3<radius]
                        #vf4[i,j]=np.nanmean(agee[D3<radius])




    """vf2=np.array(vf2)
    vf3=np.array(vf3)
    vf4=np.array(vf4)
    vf4=np.array(vf6)"""

    for p_name in all_data.keys():
        vf[p_name]=np.ma.masked_invalid(vf[p_name])
        data_results[p_name] = vf[p_name]
    x_grid_mesh = Xc
    y_grid_mesh = Yc
    lat_grid_mesh = latitude
    lon_grid_mesh = longitude

    return x_grid, y_grid, lat_grid_mesh, lon_grid_mesh, x_grid_mesh, y_grid_mesh, data_results

    #return longitude,latitude,vf#,vf2,vf3,vf4,vf5,vf6
'''
