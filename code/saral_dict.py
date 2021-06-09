################################
# SARAL dictionnary
################################


SARAL_DATA_DESC = {
    'GDR':
    {

        'hf':{
            
        'time':'time_40hz',
        'lon': 'lon_40hz',
        'lat' : 'lat_40hz',
        #'wvf':'waveforms_40hz',
        'agc':'agc_40hz',
        'alt':'alt_40hz',

            },

        'lf':{

        'time':'time',
        'lon': 'lon',
        'lat' : 'lat',
        'iono_corr':'iono_corr_gim',
        'wet_tropo':'model_wet_tropo_corr',
            'dry_tropo':'model_dry_tropo_corr',
        'earth_tide':'solid_earth_tide',
        'pole_tide':'pole_tide',
        'ocean_tide':'ocean_tide_sol1',
        'load_tide':'load_tide_sol1',
        'inv_bar':'inv_bar_corr',
            
        },
    },

    'LEGOS_T50':
    {

        'hf':{
            'lon' : 'lon_40hz',
            'lat' : 'lat_40hz',
            'time': 'time_40hz',
            'radar_fb':'radar_freeboard_40hz',
            'sla': 'ssa_interp_40hz',
            #'sea_ice_fb':'freeboard_40hz',
            'radar_h':'radar_freeboard_raw_40hz',
            'surface_type':'surface_type_40hz',
            #'snow_depth': 'snow_depth_40hz',
            'flag_leads': 'leads_true_40hz',
            'flag_floes': 'floes_true_40hz',
            },
        
        'lf':{
            'lon' : 'lon',
            'lat' : 'lat',
            'time': 'time',

            },
        
    },
}


def get_param_list():

    gdr_list = list()
    for ds in SARAL_DATA_DESC.keys():
        gdr_list.extend(list(SARAL_DATA_DESC[ds].keys()))
    return gdr_list

def get_gdr_list():

    return SARAL_DATA_DESC.keys()


def init_dict(gdr,flag_1hz):

    if flag_1hz:
        return SARAL_DATA_DESC[gdr]['lf']
    else:
        return SARAL_DATA_DESC[gdr]['hf']

