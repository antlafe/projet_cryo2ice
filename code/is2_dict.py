
################################
# Icesat-2 dictionnary
################################


# h5dump -n ~/Documents/work/comparison_AL_IS2_CS2/data/IS2_data/ATL10/ATL10-01_20190301004639_09560201_002_01.h5  | grep


def init_dict(dataproduct,BeamN,segment):

    #if 'swath' not in BeamN:
        
    IS2_DATA_DESC = {   
        'ATL07':
        {
            'granules':
            {

            'time'         : '/'+BeamN+'/sea_ice_segments/delta_time',
            'lon'          : '/'+BeamN+'/sea_ice_segments/longitude',
            'lat'          : '/'+BeamN+'/sea_ice_segments/latitude',
            'surface_h'    : '/'+BeamN+'/sea_ice_segments/heights/height_segment_height',
            #'segment_lgt'  : '/'+BeamN+'/sea_ice_segments/heights/height_segment_length_seg',
            'flag_leads'   : '/'+BeamN+'/sea_ice_segments/heights/height_segment_ssh_flag',
            #'surface_type' : '/'+BeamN+'/sea_ice_segments/heights/height_segment_type',
            #'mss'          : '/'+BeamN+'/sea_ice_segments/geophysical/height_segment_mss',
            #'gaussian_w'   : '/'+BeamN+'/sea_ice_segments/heights/height_segment_w_gaussian',
            #'ssh'

            # correction
            #'load_tide'       : '/'+BeamN+'/sea_ice_segments/geophysical/height_segment_load',
            #'ocean_tide'      : '/'+BeamN+'/sea_ice_segments/geophysical/height_segment_ocean',
            #'long_period_tide'   : '/'+BeamN+'/sea_ice_segments/geophysical/height_segment_lpe',
            #'pole_tide'       : '/'+BeamN+'/sea_ice_segments/geophysical/height_segment_pole',
            #'solid_earth_tide': '/'+BeamN+'/sea_ice_segments/geophysical/height_segment_earth',
            #'dac'             : '/'+BeamN+'/sea_ice_segments/geophysical/height_segment_dac',
            },

        },


        'ATL10':
        {

            'granules':
            {
            
            # For each variable data segments (L-m)
            'lat': '/'+BeamN+'/freeboard_beam_segment/beam_freeboard/latitude',
            'lon': '/'+BeamN+'/freeboard_beam_segment/beam_freeboard/longitude',
            'time': '/'+BeamN+'/freeboard_beam_segment/beam_freeboard/delta_time',
            'laser_fb': '/'+BeamN+'/freeboard_beam_segment/beam_freeboard/beam_fb_height',
            'fb_quality': '/'+BeamN+'/freeboard_beam_segment/beam_freeboard/beam_fb_quality_flag',
            

            # group: height_segment
            'Lseg': '/'+BeamN+'/freeboard_beam_segment/height_segments/height_segment_length_seg',

            'h_confidence': '/'+BeamN+'/freeboard_beam_segment/height_segments/height_segment_confidence',
            'gaussian_w': '/'+BeamN+'/freeboard_beam_segment/height_segments/height_segment_w_gaussian',
            #'bsnow_con' : '/'+BeamN+'/freeboard_beam_segment/height_segments/bsnow_con',
            'flag_leads'   : '/'+BeamN+'/freeboard_beam_segment/height_segments/height_segment_ssh_flag',
            'surface_h'    : '/'+BeamN+'/freeboard_beam_segment/height_segments/height_segment_height',
            'surface_type' :  '/'+BeamN+'/freeboard_beam_segment/height_segments/height_segment_type',
            'layer_flag': '/'+BeamN+'/freeboard_beam_segment/height_segments/layer_flag',


            # group: geophysical
            'mss'          : '/'+BeamN+'/freeboard_beam_segment/geophysical/height_segment_mss',
            'dac': '/'+BeamN+'/freeboard_beam_segment/geophysical/height_segment_dac',
            'earth': '/'+BeamN+'/freeboard_beam_segment/geophysical/height_segment_earth',
            'earth_f2m': '/'+BeamN+'/freeboard_beam_segment/geophysical/height_segment_earth_free2mean',
            'load': '/'+BeamN+'/freeboard_beam_segment/geophysical/height_segment_load',
            'lpe': '/'+BeamN+'/freeboard_beam_segment/geophysical/height_segment_lpe',
            'ocean': '/'+BeamN+'/freeboard_beam_segment/geophysical/height_segment_ocean',
            'pole': '/'+BeamN+'/freeboard_beam_segment/geophysical/height_segment_tide_pole',
            

            },

            'swath':{

            # swath param
            'lat': '/'+BeamN+'/freeboard_beam_segment/latitude',
            'lon': '/'+BeamN+'/freeboard_beam_segment/longitude',
            'slasw': '/'+BeamN+'/freeboard_beam_segment/beam_refsurf_height',
            'time': '/'+ BeamN+'/freeboard_beam_segment/delta_time',

            # For each 10Km swath segments
            'fb_swath'      : '/'+BeamN+'/freeboard_beam_segment/beam_fb_length',
            'fb_std_swath'  : '/'+BeamN+'/freeboard_beam_segment/beam_fb_sigma',
            'lat_swath'     : '/'+BeamN+'/freeboard_beam_segment/latitude',
            'lon_swath'     : '/'+BeamN+'/freeboard_beam_segment/longitude',
            'ssh_swath'           : '/'+BeamN+'/freeboard_beam_segment/beam_refsurf_height',
            'ssh_std_swath'       : '/'+BeamN+'/freeboard_beam_segment/beam_refsurf_sigma',
            },

            #'ref_surf': '/'+BeamN+'/freeboard_beam_segment/beam_refsurf_height',
            #'ref_surf_sigma': '/'+BeamN+'/freeboard_beam_segment/beam_refsurf_sigma',


            # estimate of the freeboard height based on entire swath
            #'laser_fb_swath':'/freeboard_swath_segment/'+BeamN+'/swath_freeboard/fbswath_fb_height',

            # For each swath segments (10 km)
            #'lead_length': '/'+BeamN+'/leads/lead_length',
            #'lead_height': '/'+BeamN+'/leads/lead_height',
            #'lead_lon': '/'+BeamN+'/leads/longitude',
            #'lead_lat': '/'+BeamN+'/leads/latitude',

            # Find a way to add these data
            # Foreach ? segments
            #'ssh_swath': '/'+BeamN+'/freeboard_beam_segment/beam_refsurf_height',
            #'lat_swath': '/'+BeamN+'/freeboard_beam_segment/latitude',
            #'lon_swath': '/'+BeamN+'/freeboard_beam_segment/longitude',
            #'lead_height': '/'+BeamN+'/leads/lead_height',

        },

        'ATL12':
        {

            'granules':
            {
                # For each variable data segments (L-m)
                'lat': '/'+BeamN+'/ssh_segments/latitude',
                'lon': '/'+BeamN+'/ssh_segments/longitude',
                'time': '/'+BeamN+'/ssh_segments/delta_time',
                'ssh': '/'+BeamN+'/ssh_segments/heights/h',
                'swh': '/'+BeamN+'/ssh_segments/heights/swh',
                'bin_ssbias' : '/'+BeamN+'/ssh_segments/heights/bin_ssbias',
                
                'ssh_var': '/'+BeamN+'/ssh_segments/heights/h_var',
                #'ssh10': '/'+BeamN+'/ssh_segments/heights/htybin',
                'xseg' : '/'+BeamN+'/ssh_segments/heights/length_seg',
            },

            # estimate of the freeboard height based on entire swath
            #'laser_fb_swath':'/freeboard_swath_segment/'+BeamN+'/swath_freeboard/fbswath_fb_height',
        },
    }
        
    # case swath    
    #else:
        
    #IS2_DATA_DESC = {
            
    #'ATL10':
    #{
    # warning low freq data
    #'time': '/freeboard_swath_segment/delta_time',
    #'lon': '/freeboard_swath_segment/longitude',
    #'lat':  '/freeboard_swath_segment/latitude',
    # Mean of the Freeboard height segments in freeboard swathsegment
    #'fb_swath': '/freeboard_swath_segment/fbswath_fb_height',
    #'fb_swath_length': '/freeboard_swath_segment/fbswath_fb_length',
    #'ssh_swath': '/freeboard_swath_segment/fbswath_refsurf_height',
    #'swath_width':'/freeboard_swath_segment/fbswath_fb_width',
    #'fb_quality':'/freeboard_swath_segment/fbswath_fb_quality_flag',
                
    #           },
    #      }
        

    return IS2_DATA_DESC[dataproduct][segment]
