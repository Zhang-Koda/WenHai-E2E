"""
Configuration settings for the data assimilation system.
"""


# Base paths
DATA_PATH = "E:/datacopy_20260128/test_data/enoi"

# Region and depth settings
REGION_SETTINGS = {
    'llon1': 0,
    'llon2': 360,
    'llat1': -80,
    'llat2': 90,
    'depth_index': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39],
    'depth' : [200],
    'is_depth_index': True,
    'offset': 5
}

# Resolution settings
RESOLUTION_SETTINGS = {
    'original_resolution': 0.25,
    'target_resolution': 1
}
# Input and output file settings
FILE_SETTINGS = {
    'gsvc_folder': 'gsvc_data_glorys_climglorys',
    'gsvc_experiment': 'global_4deg_largescale_gsvc_SSTA_SSHA_nob0_1993_2018_icemask_layerdepth_',

    'glorys_folder': '/thetaoa_soa_glorys',
    'glorys_depth_folder': 'depth',
    'glorys_experiment': 'thetaoa_soa_4deg_largescale_',

    'observation_folder': 'observation',
    'observation_experiment': 'merged_obst_',
    
    'clim_thetao_folder': 'clim_data',
    'clim_thetao_experiment': 'clim_thetao',
    'clim_so_folder': 'clim_data',
    'clim_so_experiment': 'clim_so',
    
    'mesoscale_folder': 'meso_gsvc_uncertainty',
    'mesoscale_experiment': 'var_icemask_2019_2023_depth',
    
    'largescale_var_deltaz_folder': 'cov_eta_eta0/',
    'largescale_var_deltaz_experiment': '4deg_large_var_glorys_icemask_2019_2023_depth',
    
    'Y_r_folder': 'gsvc_data_satel_climglorys',
    'Y_r_file': 'global_4deg_largescale_gsvc_SSTA_SSHA_nob0_1993_2018_icemask_layerdepth_',
    
    'mesoscale_gsvc_folder': 'gsvc_data_satel_climglorys',
    'mesoscale_gsvc': 'global_4deg_mesoscale_gsvc_SSTA_SSHA_nob0_1993_2018_icemask_layerdepth_',

    'output_folder': 'data/make_date/large_enoi',
    'output_experiment': 'large_cos_',
}

# Calculation variable settings
VAR_SETTINGS = ['thetaoa', 'soa']
BETA_SETTINGS = ['beta']
OBSERVATION_SETTINGS = ['thetao', 'so']
C_SETTINGS = ['variance_thetaoa', 'variance_soa']

# Time settings
TIME_SETTINGS = {
    'current_date': ['20200103', '20200103'],
    'year_begin': 2019,
    'year_end': 2019,
    'grid_data_bandwidth': (60, 60), # (days_before, days_after)
    'obs_bandwidth': (30, 0)  # (days_before, days_after)
}

# Model settings 
MODEL_SETTINGS = {
    'EnOI': True
}

# Processing settings
THRESHOLD_SETTINGS = {
    'svd_threshold': 1e-6
}

# Block setting
BLOCK_SETTINGS = {
    'num_blocks' : (6, 6),
    'overlap' : (0, 0)
}
# Calculation method settings
CUMPUTERING_SETTINGS = {
    'serial': False,
    'parallel': True,
    'blocks': True
}
# Localization_settings
LOCALIZATION_SETTINGS = {
    'is_localization': False
}


def get_scaling_factor():
    """Calculate scaling factor based on resolution settings."""
    original_res = RESOLUTION_SETTINGS['original_resolution']
    target_res = RESOLUTION_SETTINGS['target_resolution']
    return int(target_res / original_res)