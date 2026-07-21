"""
Main script for running the Ensemble Optimal Interpolation.
"""

import logging
import os
from pathlib import Path
import numpy as np
import xarray as xr
from typing import Dict, Any
import multiprocessing as mp
from functools import partial
import argparse


from config.settings import (
    DATA_PATH, REGION_SETTINGS, TIME_SETTINGS, VAR_SETTINGS, BETA_SETTINGS, OBSERVATION_SETTINGS, C_SETTINGS,
    MODEL_SETTINGS, THRESHOLD_SETTINGS, get_scaling_factor, FILE_SETTINGS, CUMPUTERING_SETTINGS, BLOCK_SETTINGS,LOCALIZATION_SETTINGS
)
from data_assimilation.core import EnsembleOptimalInterpolation
from data_assimilation.preprocessing import DataLoader
from data_assimilation.utils import generate_date_list
import time
import pickle
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DataAssimilationPipeline:
    """Complete pipeline for data assimilation."""

    def __init__(self, region_settings, time_settings):
        self.data_loader = DataLoader(DATA_PATH, FILE_SETTINGS, time_settings, region_settings)
        self.enoi = EnsembleOptimalInterpolation(DATA_PATH, region_settings, time_settings, FILE_SETTINGS, THRESHOLD_SETTINGS, LOCALIZATION_SETTINGS)
        self.sca_factor = get_scaling_factor()
        self.vars = VAR_SETTINGS
        self.beta = BETA_SETTINGS
        self.obs_vars = OBSERVATION_SETTINGS
        self.c_vars = C_SETTINGS
        self.region_settings = region_settings
        self.time_settings = time_settings


    def run_EnOI_pipeline(self):
        """
        Execute the complete data assimilation pipeline.

        Inputs:
            - No direct parameters; uses class attributes initialized in __init__

        Outputs:
            - Dictionary containing analysis results with keys:
              * 'Y': Analysis fields for each variable
              * 'error_cov': Error covariance matrices for each variable

        Description:
            This method orchestrates the entire Ensemble Optimal Interpolation workflow including:
            1. Computing residuals between GSVC model outputs and GLORYS
            2. Loading and processing observational data
            3. Estimating error covariances
            4. Preparing background fields
            5. Performing EnOI analysis
            6. Post-processing and formatting results
            Handles special cases like NaN values and absence of observations.
        """
        try:
            logger.info("Starting complete data assimilation pipeline")

            # Step 1: Compute L residuals

            residuals, lat_gs, lon_gs, isnan = self.compute_L_residuals()


            if isnan:
                analysis_result = self.nan_analysis(residuals, self.vars)
                final_result = self.post_nan_results(analysis_result, self.vars)
                return final_result

            # Step 2: Load and process observations
            observation_data, no_obs = self.enoi.load_EnOI_observations(residuals, self.obs_vars, self.vars)
            if no_obs or ('lon_obs' in observation_data and len(observation_data['lon_obs'])==0):
                background_data = self.enoi.prepare_Y_r(self.sca_factor, self.vars)
                analysis_result = self.no_obs_analysis(observation_data, background_data, self.vars)
            else:

            # Step 3: Estimate error covariance
                Cee, Cdzdz = self.enoi.estimate_error_covariance(
                    observation_data['lon_obs'],
                    observation_data['lat_obs'],
                    observation_data['date_gap_day'],
                    self.c_vars
                )
                observation_data, Cee, Cdzdz = self.enoi._remove_nan_values(observation_data, Cee, Cdzdz, self.obs_vars, self.c_vars, self.vars)
                if len(observation_data['date_gap_day']) == 0:
                    background_data = self.enoi.prepare_Y_r(self.sca_factor, self.vars)
                    analysis_result = self.no_obs_analysis(observation_data, background_data, self.vars)

                else:
            # Step 4: Prepare background field
                    background_data = self.enoi.prepare_background_field(observation_data['lon_obs'],
                                                                         observation_data['lat_obs'],
                                                                         observation_data['date_gap_day'],
                                                                         self.sca_factor,
                                                                         self.vars)
                    background_data, observation_data, Cee, Cdzdz = self.enoi._remove_nan_values_from_background_data(background_data, observation_data, Cee, Cdzdz, self.obs_vars, self.c_vars, self.vars)
    
                    if len(observation_data['date_gap_day']) == 0:
                        background_data = self.enoi.prepare_Y_r(self.sca_factor, self.vars)
                        analysis_result = self.no_obs_analysis(observation_data, background_data, self.vars)
            # Step 5: Perform EnOI analysis
                    else:
                        analysis_result = self.perform_EnOI_analysis(
                            observation_data, background_data, Cee, Cdzdz, self.vars, self.obs_vars, self.c_vars
                        )

            # Step 6: Post-process results
            final_result = self.post_process_results(analysis_result, lat_gs, lon_gs, self.vars)

            logger.info("Data assimilation pipeline completed successfully")
            return final_result

        except Exception as e:
            logger.error(f"Error in data assimilation pipeline: {e}")
            raise


    def compute_L_residuals(self):
        """
        Compute all required residuals for the assimilation.

        Inputs:
            - No direct parameters; uses class attributes (sca_factor, vars)

        Outputs:
            - Dictionary containing residual data with key 'residual_Ldata'
            - Latitude and longitude grids (lat_gs, lon_gs)
            - Boolean flag indicating if NaN values are present (isnan)

        Description:
            Computes residuals between GSVC model outputs and GLORYS, removes samples
            with all NaN values, and prepares residual data for subsequent assimilation steps.
        """
        logger.info("Computing all residual components")

        # Compute temperature residual between GSVC and GLORYS
        residual_Ldata_dict, n_days, lat_gs, lon_gs, isnan = self.enoi.compute_large_residual_between_gsvc_output_and_glorys(self.sca_factor, self.vars)
        logger.info(f"Temperature residual computed")

        # Remove samples with all NaN values
        for key_L in residual_Ldata_dict.keys():

            if isnan:
                residual_Ldata_dict[key_L] = residual_Ldata_dict[key_L].isel(date=0).squeeze()
                residual_Ldata_dict[key_L] = residual_Ldata_dict[key_L].drop_vars('date')
            else:
                index = (np.isnan(residual_Ldata_dict[key_L].values).all(axis=(1, 2)))

                if np.any(index):
                    logger.info(f"Removing {np.sum(index)} samples with all NaN values")
                    residual_Ldata_dict[key_L] = residual_Ldata_dict[key_L].isel(date=~index)
                    residual_Ldata_dict[key_L] = residual_Ldata_dict[key_L].squeeze()

        return {
            'residual_Ldata': residual_Ldata_dict
        }, lat_gs, lon_gs, isnan


    def perform_EnOI_analysis(self, observation_data: Dict[str, Any], background_data: Dict[str, Any],
                          Cee: np.ndarray, Cdzdz: np.ndarray, vars: list, obs_vars: list, c_vars: list):
        """
        Perform the Ensemble Optimal Interpolation analysis.

        Inputs:
            - observation_data: Dictionary containing observational data and metadata
            - background_data: Dictionary containing background field data
            - Cee: Observation error covariance matrix
            - Cdzdz: Background error covariance matrix
            - vars: List of state variables to analyze
            - obs_vars: List of observation variables
            - c_vars: List of covariance configuration variables

        Outputs:
            - Dictionary containing:
              * 'Y_analysis': Analyzed state variables
              * 'error_covariance': Estimated error covariance matrices

        Description:
            Performs Ensemble Optimal Interpolation for each variable by preparing analysis inputs
            and calling the core EnOI analysis routine. Handles multiple variables simultaneously.
        """
        logger.info("Performing Ensemble Optimal Interpolation analysis")

        #observation_data, Cee, Cdzdz = self.enoi._remove_nan_values(observation_data, Cee, Cdzdz, obs_vars, c_vars)
        # Prepare inputs for analysis
        Y_analysis, error_covariance = {}, {}
        if len(vars) == len(obs_vars):
            for i in range(len(vars)):

                analysis_inputs = {
                    'observations': {
                        'obs_dict': observation_data['obs_dict'][obs_vars[i]],
                        'date_gap': observation_data['date_gap_day'],
                        'Cee': Cee[c_vars[i]],
                        'Cdzdz': Cdzdz[c_vars[i]]
                    },
                    'background': {
                        'Y_r': background_data['Y_r'][vars[i]],
                        'H_Y_r': background_data['H_Y_r'][vars[i]]
                    },
                    'residuals': {
                        'L': observation_data['L'][vars[i]],
                        'HL': observation_data['HL'][vars[i]],
                    }
                }


                Y_analysis[vars[i]], error_covariance[vars[i]] = self.enoi.EnOI_analyze(
                    analysis_inputs['observations'],
                    analysis_inputs['background'],
                    analysis_inputs['residuals']
                )

        return {
            'Y_analysis': Y_analysis,
            'error_covariance': error_covariance,
        }

    def no_obs_analysis(self, observation_data: Dict[str, Any], background_data: Dict[str, Any], vars: list):
        """
        Perform analysis when no observations are available.

        Inputs:
            - observation_data: Dictionary containing observational data (empty or minimal)
            - background_data: Dictionary containing background field data
            - vars: List of state variables to analyze

        Outputs:
            - Dictionary containing:
              * 'Y_analysis': Background state variables (no assimilation performed)
              * 'error_covariance': Background error covariance matrices

        Description:
            Handles the case where no valid observations are available by returning
            background fields as analysis results without performing data assimilation.
        """
        logger.info("Performing noobs analysis")
        Y_analysis, error_covariance = {}, {}

        for i in range(len(vars)):

            analysis_inputs = {
                'background': {
                    'Y_r': background_data['Y_r'][vars[i]],
                },
                'residuals': {
                    'L': observation_data['L'][vars[i]],
                }
            }


            Y_analysis[vars[i]], error_covariance[vars[i]] = self.enoi.no_obs_analysis(
                analysis_inputs['background'],
                analysis_inputs['residuals']
            )

        return {
            'Y_analysis': Y_analysis,
            'error_covariance': error_covariance,
        }

    def nan_analysis(self, residuals, vars):
        """
        Perform analysis when NaN values are detected in residuals.

        Inputs:
            - residuals: Dictionary containing residual data with NaN values
            - vars: List of state variables to analyze

        Outputs:
            - Dictionary containing:
              * 'Y_analysis': Residual data used as analysis (no assimilation)
              * 'error_covariance': Residual data used as error estimates

        Description:
            Handles the case where residuals contain NaN values by using the residual data
            directly as analysis results without performing proper data assimilation.
        """
        logger.info("Performing nan analysis")
        Y_analysis, error_covariance = {}, {}
        for i in range(len(vars)):
            Y_analysis[vars[i]] = residuals['residual_Ldata'][vars[i]]
            error_covariance[vars[i]] = residuals['residual_Ldata'][vars[i]]
        return {
            'Y_analysis': Y_analysis,
            'error_covariance': error_covariance,
        }

    def post_nan_results(self,  analysis_result, vars):
        """
        Post-process results from NaN analysis case.

        Inputs:
            - analysis_result: Dictionary containing analysis results from nan_analysis
            - vars: List of state variables

        Outputs:
            - Dictionary containing formatted results with keys 'Y' and 'error_cov'

        Description:
            Formats and renames variables from NaN analysis results for consistent output structure.
        """
        logger.info("Post-nan analysis results")

        Y, error_cov = {}, {}
        for var in vars:
            Y[var] = analysis_result['Y_analysis'][var]
            Y[var] = Y[var].rename(f'Y_{var}')
            error_cov[var] = analysis_result['error_covariance'][var]
            error_cov[var] = error_cov[var].rename(f'variance_{var}')
        return {'Y': Y, 'error_cov': error_cov}

    def post_process_results(self, analysis_result: Dict[str, Any],
                           lat_gs: np.ndarray, lon_gs: np.ndarray, vars: list):
        """
        Post-process and format the analysis results.

        Inputs:
            - analysis_result: Dictionary containing raw analysis results
            - lat_gs: Latitude grid coordinates
            - lon_gs: Longitude grid coordinates
            - vars: List of state variables

        Outputs:
            - Dictionary containing formatted xarray DataArrays with keys 'Y' and 'error_cov'

        Description:
            Converts analysis results into properly formatted xarray DataArrays with geographic
            coordinates and appropriate variable names for output and visualization.
        """
        logger.info("Post-processing analysis results")

        Y, error_cov = {}, {}
        for var in vars:
            Y_2d = analysis_result['Y_analysis'][var].reshape(len(lat_gs), len(lon_gs))
            analysis_da = xr.DataArray(
                Y_2d,
                dims=['lat', 'lon'],
                coords={'lat': lat_gs, 'lon': lon_gs},
                name=f"Y_{var}")
            Y[var] = analysis_da

            error_cov_2d = analysis_result['error_covariance'][var].reshape((len(lat_gs), len(lon_gs)))
            error_da = xr.DataArray(
                error_cov_2d,
                dims=['lat', 'lon'],
                coords={'lat': lat_gs, 'lon': lon_gs},
                name=f"variance_{var}")
            error_cov[var] = error_da

        return {'Y':Y, 'error_cov':error_cov}

def run_pipeline(depth, region_settings, time_settings):
    """
    Run the data assimilation pipeline for a specific depth level.

    Inputs:
        - depth: Depth value for current processing level
        - region_settings: Dictionary containing regional configuration
        - time_settings: Dictionary containing temporal configuration

    Outputs:
        - Dictionary containing analysis results for the specified depth

    Description:
        Wrapper function that creates a DataAssimilationPipeline instance and executes
        the full EnOI pipeline for a single depth level. Used for parallel processing.
    """
    region_settings['depth'] = depth
    EnOI_pipeline = DataAssimilationPipeline(region_settings, time_settings)
    results = EnOI_pipeline.run_EnOI_pipeline()
    
    return results

def process_padding_blocks(time_settings, region_settings, date, depth_list, vars):
    """
    Process data assimilation in spatial blocks with padding for boundary handling.

    Inputs:
        - time_settings: Dictionary containing temporal configuration
        - region_settings: Dictionary containing regional configuration
        - model: Model configuration settings
        - date: Current date being processed
        - depth_list: List of depth levels to process
        - vars: List of state variables to analyze

    Outputs:
        - merged_padding_results: Placeholder for merged results (currently None)

    Description:
        Divides the domain into spatial blocks and processes each block separately using
        parallel computing. Creates regular blocks, padding blocks, and edge blocks to handle
        spatial boundaries and dependencies. Results are saved as numpy files for later merging.
    """
    block_rows, block_cols = BLOCK_SETTINGS['num_blocks']
    llon1, llon2 = region_settings['llon1'], region_settings['llon2']
    llat1, llat2 = region_settings['llat1'], region_settings['llat2']
    h = llat2 - llat1
    w = llon2 - llon1
    base_block_h = h // block_rows
    base_block_w = w // block_cols
    
    out_folder = 'output_base/' + date
    if not os.path.exists(out_folder):
        os.makedirs(out_folder, exist_ok=True) 
    
    block_results = {}
    
    for lat in range(block_rows):
        for lon in range(block_cols):

            start_lat = lat * base_block_h + llat1
            end_lat = (lat + 1) * base_block_h + llat1 if lat < block_rows - 1 else llat2

            start_lon = lon * base_block_w + llon1
            end_lon = (lon + 1) * base_block_w + llon1 if lon < block_cols - 1 else llon2

            time_settings['current_date'] = date
            region_settings['llon1'] = start_lon
            region_settings['llon2'] = end_lon
            region_settings['llat1'] = start_lat
            region_settings['llat2'] = end_lat
            start = time.time()
            run_with_model = partial(run_pipeline, region_settings=region_settings, time_settings=time_settings)
            processes_num = min(len(depth_list), os.cpu_count() or 1)
            with mp.Pool(processes=processes_num) as pool:
                results = pool.map(run_with_model, depth_list)

            ds = build_ds(vars, depth_list, results)
            end = time.time()
            print('one block time:', end-start)
            print('block_lat', lat)
            print('block_lon', lon)
            block_results[(lat,lon)] = ds

    blob_block_results = pickle.dumps(block_results)
    np.save(f"{out_folder}/block.npy", np.frombuffer(blob_block_results, dtype=np.uint8))
       

    padding_llat1 = llat1 - 0.5 * base_block_h
    padding_llat2 = llat2 + 0.5 * base_block_h
    padding_llon1 = llon1 - 0.5 * base_block_w
    padding_llon2 = llon2 + 0.5 * base_block_w
    
    padding_block_results = {}

    for p_lat in range(block_rows + 1):
        for p_lon in range(block_cols + 1):
            start = time.time()
            p_start_lat = p_lat * base_block_h + padding_llat1
            p_end_lat = (p_lat + 1) * base_block_h + padding_llat1 if p_lat < block_rows else padding_llat2

            p_start_lon = p_lon * base_block_w + padding_llon1
            p_end_lon = (p_lon + 1) * base_block_w + padding_llon1 if p_lon < block_cols else padding_llon2

            time_settings['current_date'] = date
            region_settings['llon1'] = p_start_lon
            region_settings['llon2'] = p_end_lon
            region_settings['llat1'] = p_start_lat
            region_settings['llat2'] = p_end_lat
            run_with_model = partial(run_pipeline, region_settings=region_settings, time_settings=time_settings)
            processes_num = min(len(depth_list), os.cpu_count() or 1)
            with mp.Pool(processes=processes_num) as pool:
                results = pool.map(run_with_model, depth_list)

            p_ds = build_ds(vars, depth_list, results)
            end = time.time()
            print('one block time:', end - start)
            print('block_lat', p_lat)
            print('block_lon', p_lon)
            padding_block_results[(p_lat, p_lon)] = p_ds

    blob_padding_block_results = pickle.dumps(padding_block_results)
    np.save(f"{out_folder}/padding_block.npy", np.frombuffer(blob_padding_block_results, dtype=np.uint8))
    
    lat_padding_blocks_results = {}
    for p_lat in range(block_rows + 1):
        for p_lon in range(block_cols):
            p_start_lat = p_lat * base_block_h + padding_llat1
            p_end_lat = (p_lat + 1) * base_block_h + padding_llat1 if p_lat < block_rows else padding_llat2

            start_lon = p_lon * base_block_w + llon1
            end_lon = (p_lon + 1) * base_block_w + llon1 if p_lon < block_cols - 1 else llon2

            time_settings['current_date'] = date
            region_settings['llon1'] = start_lon
            region_settings['llon2'] = end_lon
            region_settings['llat1'] = p_start_lat
            region_settings['llat2'] = p_end_lat
            run_with_model = partial(run_pipeline, region_settings=region_settings, time_settings=time_settings)
            processes_num = min(len(depth_list), os.cpu_count() or 1)
            with mp.Pool(processes=processes_num) as pool:
                results = pool.map(run_with_model, depth_list)

            p_ds = build_ds(vars, depth_list, results)
            lat_padding_blocks_results[(p_lat, p_lon)] = p_ds

    blob_lat_padding_block_results = pickle.dumps(lat_padding_blocks_results)
    np.save(f"{out_folder}/lat_padding_block.npy", np.frombuffer(blob_lat_padding_block_results, dtype=np.uint8))
    
    lon_padding_blocks_results = {}
    for p_lat in range(block_rows):
        for p_lon in range(block_cols + 1):
            start_lat = p_lat * base_block_h + llat1
            end_lat = (p_lat + 1) * base_block_h + llat1 if p_lat < block_rows - 1 else llat2

            p_start_lon = p_lon * base_block_w + padding_llon1
            p_end_lon = (p_lon + 1) * base_block_w + padding_llon1 if p_lon < block_cols else padding_llon2

            time_settings['current_date'] = date
            region_settings['llon1'] = p_start_lon
            region_settings['llon2'] = p_end_lon
            region_settings['llat1'] = start_lat
            region_settings['llat2'] = end_lat
            run_with_model = partial(run_pipeline, region_settings=region_settings, time_settings=time_settings)
            processes_num = min(len(depth_list), os.cpu_count() or 1)
            with mp.Pool(processes=processes_num) as pool:
                results = pool.map(run_with_model, depth_list)

            p_ds = build_ds(vars, depth_list, results)
            lon_padding_blocks_results[(p_lat, p_lon)] = p_ds

    blob_lon_padding_block_results = pickle.dumps(lon_padding_blocks_results)
    np.save(f"{out_folder}/lon_padding_block.npy", np.frombuffer(blob_lon_padding_block_results, dtype=np.uint8))

    merged_padding_results = None
    return merged_padding_results


def build_ds(vars, depth_list, results):
    """
    Build an xarray Dataset from analysis results across multiple depths.

    Inputs:
        - vars: List of state variables
        - depth_list: List of depth levels
        - results: List of analysis results for each depth level

    Outputs:
        - xarray Dataset containing all variables organized by depth dimension

    Description:
        Combines analysis results from multiple depth levels into a single xarray Dataset
        with proper 3D structure (latitude × longitude × depth). Creates separate DataArrays
        for analysis fields and error covariances for each variable.
    """
    arrays_3d_list = []
    for var in vars:
        Y_data_list = []
        variance_date_list = []
        for res in results:
            Y_data_list.append(res['Y'][var])
            variance_date_list.append(res['error_cov'][var])
        for i, depth in enumerate(depth_list):
            Y_data_list[i] = Y_data_list[i].expand_dims({'depth': [depth]})
            variance_date_list[i] = variance_date_list[i].expand_dims({'depth': [depth]})
        arrays_3d_list.append(xr.concat(Y_data_list, dim='depth'))
        arrays_3d_list.append(xr.concat(variance_date_list, dim='depth'))

    first_da = arrays_3d_list[0]
    var_names = [da.name for da in arrays_3d_list]
    ds = first_da.to_dataset(name=var_names[0])
    for i in range(1, len(arrays_3d_list)):
        ds[var_names[i]] = arrays_3d_list[i]

    return ds


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='ENOI')

    parser.add_argument('--current-date', '-d', nargs='+',
                        help='start date and end date (format: YYYYMMDD YYYYMMDD)')

    return parser.parse_args()


def main():
    """Main execution function."""
    args = parse_arguments()
    if args.current_date:
        TIME_SETTINGS['current_date'] = args.current_date

    try:
        logger.info("Starting Data Assimilation System")
        time_settings = TIME_SETTINGS.copy()
        current_date = TIME_SETTINGS['current_date']
        region_settings_final = REGION_SETTINGS

        if region_settings_final['is_depth_index']:

            file_name = Path(f'E:/datacopy_20260128/test_data/public/grid_depth/mercatorglorys12v1_gl12_mean_20240401_R20240403.nc')
            if not file_name.exists():
                raise FileNotFoundError(f"depth file not found: {file_name}")
            ori_depth = xr.open_dataset(file_name)['depth'].values
            depth_list = ori_depth[(np.array(region_settings_final['depth_index']))]
        else:
            depth_list = region_settings_final['depth']
        date_list = generate_date_list(current_date[0], current_date[1])
        vars = VAR_SETTINGS
        output_folder = Path(DATA_PATH) / FILE_SETTINGS['output_folder']
        processes_num = min(len(depth_list), os.cpu_count() or 1)
        for date in date_list:
            time_settings['current_date'] = date
            region_settings = region_settings_final.copy()
            if CUMPUTERING_SETTINGS['blocks'] == True:
                ds = process_padding_blocks(time_settings, region_settings, date, depth_list, vars)
                if not os.path.exists(output_folder):
                    os.makedirs(output_folder, exist_ok=True)
                output_filename = output_folder / f"{FILE_SETTINGS['output_experiment']}{date}.nc"
                if ds is None:
                    print('only get blocks')
                else:
                    ds.to_netcdf(output_filename)

            else:
                run_with_model = partial(run_pipeline, region_settings=region_settings, time_settings=time_settings)
                with mp.Pool(processes=processes_num) as pool:
                    results = pool.map(run_with_model, depth_list)
                if not os.path.exists(output_folder):
                    os.makedirs(output_folder, exist_ok=True)
                output_filename = output_folder / f"{FILE_SETTINGS['output_experiment']}{date}.nc"
                ds = build_ds(vars, depth_list, results)
                ds.to_netcdf(output_filename)


    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        raise

if __name__ == "__main__":
    main()