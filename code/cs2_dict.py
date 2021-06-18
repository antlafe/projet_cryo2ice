################################
# Cryosat-2 dictionnary
################################

# 1hz data: '01'

CS2_DATA_DESC = {
    'ESA_BD_GDR':
    {

        'lf':
        {
            
        'lon': 'lon_01',
        'lat': 'lat_01',
        'time':'time_cor_01',

        # correction
        'ssb': 'sea_state_bias_01_ku',
        'load_tide': 'load_tide_01',
        'ocean_tide': 'ocean_tide_01',
        'ocean_eq_tide': 'ocean_tide_eq_01',
        'pole_tide' : 'pole_tide_01',
        'solid_earth_tide': 'solid_earth_tide_01',
        'dac': 'hf_fluct_total_cor_01',
        'geoid': 'geoid_01',
        'sic':  'sea_ice_concentration_01',
        'mss': 'mean_sea_surf_sea_ice_01',
        
        'swh': 'swh_ocean_01_ku',
        'u10': 'wind_speed_alt_01_ku',
        },

        'hf':
        {
        'lon': 'lon_poca_20_ku',
        'lat': 'lat_poca_20_ku',
        'time': 'time_20_ku',
        'radar_fb':'freeboard_20_ku',
        'sla': 'ssha_interp_20_ku',
        #'sea_ice_fb': None,
        'radar_h': 'height_1_20_ku',
        'surface_type':'surf_type_20_ku',
        #'lead_height' : 'height_sea_ice_lead_20_ku',
        'isa': 'height_sea_ice_floe_20_ku',
        'quality_flag' : 'flag_prod_status_20_ku',
        },
            
    },

    'ESA_BD':
    {

         'lf':
        {
            
        'lon': 'lon_01',
        'lat': 'lat_01',
        'time':'time_cor_01',

        # correction
        'ssb': 'sea_state_bias_01_ku',
        #'load_tide': 'load_tide_01',
        #'ocean_tide': 'ocean_tide_01',
        #'ocean_eq_tide': 'ocean_tide_eq_01',
        #'pole_tide' : 'pole_tide_01',
        #'solid_earth_tide': 'solid_earth_tide_01',
        #'dac': 'hf_fluct_total_cor_01',
        #'geoid': 'geoid_01',
        'sic':  'sea_ice_concentration_01',
        'mss': 'mean_sea_surf_sea_ice_01',
        
        'swh': 'swh_ocean_01_ku',
        'u10': 'wind_speed_alt_01_ku',
        },

        'hf':
        {
        'lon': 'lon_poca_20_ku',
        'lat': 'lat_poca_20_ku',
        'time': 'time_20_ku',
        'radar_fb':'freeboard_20_ku',
        'sla': 'ssha_interp_20_ku',
        #'sea_ice_fb': None,
        'radar_h': 'height_1_20_ku',
        'surface_type':'surf_type_20_ku',
        'lead_height' : 'height_sea_ice_lead_20_ku',
        'isa': 'height_sea_ice_floe_20_ku',
        'quality_flag' : 'flag_prod_status_20_ku',
        },
            
    },

    'ESA_BD_SIN1':
    {

        'hf':{
        'lon': 'lon_20_ku',
        'lat': 'lat_20_ku',
        'time': 'time_20_ku',
        #'lat':  'alt_20_ku',
        'alt':  'alt_20_ku',
        'wvf'       :'pwr_waveform_20_ku',
        'coherence' :'coherence_waveform_20_ku',
        'ph_diff'   :'ph_diff_waveform_20_ku',
        'traker'    :'window_del_20_ku',
        #'corr'      :'instr_cor_range_rx',
        'quality_flag' : 'flag_prod_status_20_ku',
           } ,

        'lf':{},
        
            
    },

    'ESA_BD_SIN2':
    {
        'hf':{
            
        'lon': 'lon_poca_20_ku',
        'lat': 'lat_poca_20_ku',
        'time': 'time_20_ku',
        #'alt':  'alt_20_ku',
        #'roll_angle': 'off_nadir_roll_angle_str_01',
        #'sic': 'sea_ice_concentration_01',
        'range1': 'range_1_20_ku',
        #'range2': 'range_2_20_ku',
        #'range3': 'range_3_20_ku',
        'height1': 'height_1_20_ku',
        #'height2': 'height_2_20_ku',
        #'height3': 'height_3_20_ku',
        'flag_corr': 'flag_cor_applied_20_ku',
        #'alt' : 'alt_01',
        'quality_flag' : 'flag_prod_status_20_ku',
        },

        'lf':{},
        
        
    },

    

    

    'ESA_BD_1B':
    {
        'hf':{
        'lon': 'lon_20_ku',
        'lat': 'lat_20_ku',
        'time': 'time_20_ku',
        'wvf':'pwr_waveform_20_ku',
        },

        'lf':{},
            
    },

    'CPOM':
    {
        'hf':{
        'time': 0,
        'lat' : 1,
        'lon' : 2,
        'radar_fb':3,
        'sla':4,
        'ice_conc':5,
        'ice_type':6,
        'quality_flag':7,
            },

        'lf':{},
    },

    'LEGOS_SAM':
    {
        'hf':{
            
        'lon' : 'lon_20hz',
        'lat' : 'lat_20hz',
        'time': 'time_20hz',
        'radar_fb':'radar_freeboard_20hz',
        'sla': 'ssa_interp_20hz',
        'isa': 'ice_surface_anomaly_20hz',
        #'sea_ice_fb':'freeboard_20hz',
        'radar_h':'anomaly_samosa_20hz',
        'surface_type':'surface_type_20hz',
        #'snow_depth': 'snow_depth_20hz',
        'flag_leads': 'leads_true_20hz',
        'pp':'wvf_pulse_peakiness_20hz',
        
        'isa': 'ice_surface_anomaly_raw_20hz',
        'flag_floes': 'floes_true_20hz',
              },

        'lf':{},
    },

    'LEGOS_T50':
    {

        'hf':{
            
        'lon' : 'lon_20hz',
        'lat' : 'lat_20hz',
        'time': 'time_20hz',
        'radar_fb':'radar_freeboard_20hz',
        'sla': 'ssa_interp_20hz',
        'isa': 'ice_surface_anomaly_20hz',
        #'sea_ice_fb':'freeboard_20hz',
        'radar_h':'radar_freeboard_raw_20hz',
        'surface_type':'surface_type_20hz',
        #'snow_depth': 'snow_depth_20hz',
        'flag_leads': 'leads_true_20hz',
        'flag_floes': 'floes_true_20hz',
        },

        'lf':{},
    },

     'LEGOS_PLRM':
    {

        'hf':{
            
        'lon' : 'lon_20_ku',
        'lat' : 'lat_20_ku',
        'time': 'time_20_ku',
        'radar_fb':'radar_freeboard_20hz',
        'isa': 'ice_surface_anomaly_20hz',
        'sla': 'ssa_interp_20hz',
        #'sea_ice_fb':'freeboard_20hz',
        'radar_h':'radar_freeboard_raw_20hz',
        'surface_type':'surface_type_20hz',
        #'snow_depth': 'snow_depth_20hz',
        'flag_leads': 'leads_true_20hz',
        'flag_floes': 'floes_true_20hz',
        },

        'lf':{},
    },

    'AWI':
    {
         'hf':{
             
        'lon' : 'longitude',
        'lat' : 'latitude',
        'time': 'time',
        'radar_fb':'radar_freeboard',
        'sla': 'sea_level_anomaly',
        #'sea_ice_fb':'sea_ice_freeboard',
        'radar_h': 'elevation',
        'surface_type': 'surface_type',
        'snow_depth': 'snow_depth',
        'lew': 'leading_edge_width',
        'pp': 'pulse_peakiness',
        #surface_type,sea_level_anomaly,radar_freeboard,elevation
             },

        'lf':{},

    },

    
    'UOB':
    {
        'hf':{
            
        'lon' : 'Longitude',
        'lat' : 'Latitude',
        'time': 'time',
        'hour': 'Hour',
        'minute': 'Minute',
        'second': 'Second',
        'radar_fb':'Radar_Freeboard',
        'sla': 'Sea_Surface_Height_WGS84',
        #'sea_ice_fb':'sea_ice_freeboard',
        'radar_h': 'Surface_Height_WGS84',
        'flag_leads': 'Lead_Class',
        'surface_type': 'Sea_Ice_Class',
        'roughness': 'Sea_Ice_Roughness_Lognormal',
        #'snow_depth': 'snow_depth',
        },

        'lf':{},
    },

    'NASA':
    {

    },

    }

  

def get_param_list():

    gdr_list = list()
    for ds in CS2_DATA_DESC.keys():
        gdr_list.extend(list(CS2_DATA_DESC[ds].keys()))
    return gdr_list

def get_gdr_list():

    return CS2_DATA_DESC.keys()


def init_dict(gdr,flag_1hz):

    if flag_1hz:
        return CS2_DATA_DESC[gdr]['lf']
    else:
        return CS2_DATA_DESC[gdr]['hf']
