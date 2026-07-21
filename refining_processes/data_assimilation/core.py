"""
Core data assimilation algorithms.
"""
import numpy as np
import xarray as xr
import pandas as pd
from typing import Tuple, Dict, Any, List
from pathlib import Path
from .utils import svd_inverse
from .preprocessing import DataLoader
import logging
logger = logging.getLogger(__name__)

class EnsembleOptimalInterpolation:
    """Ensemble Optimal Interpolation (EnOI) implementation."""

    def __init__(self, data_path: Path, region_settings: Dict, time_settings: Dict, file_settings: Dict, threshold_settings: Dict, localization_settings: Dict):
        """
        Initialize the Ensemble Optimal Interpolation system.

        Inputs:
            - data_path: Path to data directory
            - region_settings: Dictionary containing regional configuration parameters
            - time_settings: Dictionary containing temporal configuration parameters
            - file_settings: Dictionary containing file I/O configuration
            - smooth_kernel_settings: Tuple for smoothing kernel configuration
            - threshold_settings: Dictionary containing numerical thresholds
            - localization_settings: Dictionary containing localization parameters

        Outputs:
            - None (initializes class attributes)

        Description:
            Sets up the EnOI system by initializing the data loader and storing
            configuration settings for region, time, files, thresholds, and localization.
        """
        self.data_loader = DataLoader(data_path, file_settings, time_settings, region_settings)
        self.region_settings = region_settings
        self.time_settings = time_settings
        self.file_settings = file_settings
        self.threshold_settings = threshold_settings
        self.localization_settings = localization_settings


    def compute_large_residual_between_gsvc_output_and_glorys(self, sca_factor: int, vars: list):
        """
        Compute residual between GSVC model outputs and GLORYS.

        Inputs:
            - sca_factor: Spatial scaling factor for coarsening resolution
            - vars: List of variables to compute residuals for

        Outputs:
            - residual_Ldata_dict: Dictionary of residual DataArrays for each variable
            - n_days: Number of days in the computed residual series
            - lat_gs: Latitude coordinates of the coarse grid
            - lon_gs: Longitude coordinates of the coarse grid
            - isnan: Boolean flag indicating presence of NaN values

        Description:
            Computes residuals between GSVC and GLORYS models over a multi-year period,
            excluding the current year. Applies spatial coarsening and returns residual
            fields along with grid information and NaN status flag.
        """
        current_date = self.time_settings['current_date']
        year_begin, year_end = self.time_settings['year_begin'], self.time_settings['year_end']

        gsvc_dict, glorys_dict = {}, {}
        isnan = False
        for index, year in enumerate(range(year_begin, year_end + 1)):
            if str(year) == str(current_date[:4]):
                continue
            target_date = f"{year}{current_date[4:]}"
            da_gsvc_dict, gsvs_isnan = self.data_loader.load_gsvc_data(target_date, vars)
            da_glorys_dict, glorys_isnan = self.data_loader.load_glorys_data(target_date, vars)

            if index == 0 or str(current_date[:4]) == str(year_begin):
                gsvc_dict = da_gsvc_dict
                glorys_dict = da_glorys_dict
            else:
                for key in da_gsvc_dict:
                    gsvc_dict[key].extend(da_gsvc_dict[key])
                    glorys_dict[key].extend(da_glorys_dict[key])
            if  gsvs_isnan or glorys_isnan:
                isnan = True

        # Concatenate and process data
        residual_Ldata_dict = {}
        for key in gsvc_dict:
            gsvc_Ldata = xr.concat(gsvc_dict[key], dim='date')
            glorys_Ldata = xr.concat(glorys_dict[key], dim='date')
            gsvc_Ldata = gsvc_Ldata.assign_coords(date=glorys_Ldata.date)

            gsvc_Ldata = gsvc_Ldata.coarsen(dim={'lon': sca_factor, 'lat': sca_factor}, boundary='trim').mean()
            glorys_Ldata = glorys_Ldata.coarsen(dim={'lon': sca_factor, 'lat': sca_factor}, boundary='trim').mean()
            lat_gs = gsvc_Ldata.lat.values
            lon_gs = gsvc_Ldata.lon.values
            residual_Ldata_dict[key] = (gsvc_Ldata - glorys_Ldata)
        _, first_value = next(iter(residual_Ldata_dict.items()))
        n_days = first_value.shape[0]
        return residual_Ldata_dict, n_days, lat_gs, lon_gs, isnan


    def load_EnOI_observations(self, residuals: Dict[str, Dict[str, xr.DataArray]], obs_vars: list, vars: list):
        """
        Load and process observational data for EnOI assimilation.

        Inputs:
            - residuals: Dictionary containing residual data from model comparison
            - obs_vars: List of observation variable names
            - vars: List of state variable names

        Outputs:
            - interpolated_data: Dictionary containing processed observations and interpolated residuals
            - no_obs: Boolean flag indicating if no observations were found

        Description:
            Loads observational data within spatial-temporal bounds, removes climatology
            and mesoscale signals, concatenates observations across time, and interpolates
            model residuals to observation locations. Handles cases with no valid observations.
        """
        logger.info("Loading and processing observations")

        llon1, llon2 = self.region_settings['llon1'], self.region_settings['llon2']
        llat1, llat2 = self.region_settings['llat1'], self.region_settings['llat2']
        depth = self.region_settings['depth']
        current_date = self.time_settings['current_date']
        obs_bandwidth = self.time_settings['obs_bandwidth']

        # Load observation dataset
        ds_obs, obs_dates, no_obs = self.data_loader.load_observation_data(current_date, obs_bandwidth)
        if no_obs:
            L = {}
            for var in vars:
                L[var] = residuals['residual_Ldata'][var].values.reshape(
                    (residuals['residual_Ldata'][var].shape[0], -1)).T
                land_mask = np.isnan(L[var]).any(axis=1)
                L[var][land_mask, :] = 0
            interpolated_data = {
                'L': L,
            }
            return interpolated_data, no_obs
        # Process observations
        lon_obs, lat_obs, date_obs = [], [], []
        obs_vars_dict = {v:[] for v in obs_vars if v in ds_obs.data_vars}
        clim_dict = {}

        if 'thetao' in obs_vars:
            clim_dict['thetao'] = self.data_loader.load_clim_thetao_data(depth)
        if 'so' in obs_vars:
            clim_dict['so'] = self.data_loader.load_clim_so_data(depth)

        da_mesoscale = self.data_loader.load_mesoscale_gsvc(current_date, obs_bandwidth, depth)

        for date_str in obs_dates:
            clim_day = pd.to_datetime(date_str, format="%Y%m%d").replace(year=2012).dayofyear - 1
            if date_str not in ds_obs.date:
                continue

            ds_date = ds_obs.sel(date=date_str)

            if len(ds_date.lon.values) == 0:
                continue
            lon1, lat1 = ds_date.lon.values, ds_date.lat.values
            index = (lon1 > llon1) & (lon1 < llon2) & (lat1 > llat1) & (lat1 < llat2)

            if np.sum(index) > 0:
                lon_obs.append(lon1[index])
                lat_obs.append(lat1[index])

                # Remove climatology
                for i, var in enumerate(obs_vars):
                    obs_vars_dict[var].append(ds_date[var].sel(depths=depth)[index]-clim_dict[var].isel(time=clim_day).interp(lon=('points',lon1[index]), lat=('points',lat1[index])).values -
                                              da_mesoscale[vars[i]].sel(date=date_str).interp(lon=('points',lon1[index]), lat=('points',lat1[index])).values)
                date_obs.append(np.full(len(lon1[index]), date_str))

        # Concatenate and clean
        observation_data = self._concatenate_observations(lon_obs, lat_obs, obs_vars_dict, date_obs, current_date)

        # Interpolate residuals to observation locations
        interpolated_data = self._interpolate_L_residuals_to_obs(
            residuals, observation_data, vars
        )

        return interpolated_data, no_obs


    def estimate_error_covariance(self, lon_obs, lat_obs, date_gap_day, c_vars):
        """
        Estimate observation and temporal error covariance matrices.

        Inputs:
           - lon_obs: Longitudes of observation points
           - lat_obs: Latitudes of observation points
           - date_gap_day: Time differences between observations and analysis time
           - c_vars: List of covariance variable names

        Outputs:
           - Cee: Representation error covariance matrices for each variable
           - Cdzdz: Temporal error covariance matrices for each variable

        Description:
           Estimates observation error covariance from mesoscale variability and
           temporal error covariance from large-scale variability patterns.
           Uses interpolation to observation locations and handles different time lags.
        """
        depth = self.region_settings['depth']
        Cee = {}
        for var in c_vars:
            da_var = self.data_loader.load_mesoscale_var(depth, var)
            argo_var = da_var.interp(lon=('points', lon_obs), lat=('points', lat_obs)).values

            Cee[var] = argo_var
        deltaz_var = self.data_loader.load_largescale_var_deltaz(depth)

        Cdzdz = {v: [] for v in c_vars}
        for var in c_vars:
            unique_abs = np.unique(np.abs(date_gap_day))
            result_var = np.zeros_like(lon_obs, dtype=float)
            for abs_val in unique_abs:
                mask = np.abs(date_gap_day) == abs_val
                if abs_val == 0:
                    result_var[mask] = 0
                else:
                    points_ds = xr.Dataset({
                        'lat': (['points'], lat_obs[mask]),
                        'lon': (['points'], lon_obs[mask])
                    })
                    result_var[mask] = deltaz_var[var].sel(lag_days=abs_val).interp(points_ds).values
            Cdzdz[var] = result_var

        return Cee, Cdzdz

    def prepare_background_field(self, lon_obs, lat_obs, date_gap_day, sca_factor, vars):
        """
        Prepare the background field for assimilation.

        Inputs:
            - lon_obs: Longitudes of observation points
            - lat_obs: Latitudes of observation points
            - date_gap_day: Time differences for background selection
            - sca_factor: Spatial scaling factor
            - vars: List of state variables

        Outputs:
            - Dictionary containing:
              * 'Y_r': Background field on model grid
              * 'H_Y_r': Background field interpolated to observation locations

        Description:
            Loads and prepares the background field (GSVC reconstruction) for the
            assimilation. Interpolates the background field to observation locations
            and returns both gridded and interpolated versions.
        """
        logger.info("Preparing background field")

        depth = self.region_settings['depth']
        current_date = self.time_settings['current_date']

        # Load background field (GSVC reconstruction)
        Y_r, H_Y_r = self.data_loader.load_Y_r_and_H_Y_r(current_date, depth, date_gap_day, lon_obs, lat_obs, sca_factor, vars)

        return {
            'Y_r': Y_r,
            'H_Y_r': H_Y_r,
        }

    def prepare_Y_r(self, sca_factor, vars):
        """
        Prepare background field without observation-specific interpolation.

        Inputs:
            - sca_factor: Spatial scaling factor
            - vars: List of state variables

        Outputs:
            - Dictionary containing 'Y_r': Background field on model grid

        Description:
            Loads the background field (GSVC reconstruction) for cases without
            observation data, such as when no valid observations are available.
        """
        logger.info("Preparing Y_r")
        depth = self.region_settings['depth']
        current_date = self.time_settings['current_date']
        Y_r = self.data_loader.load_Y_r(current_date, depth, sca_factor, vars)
        return {
            'Y_r': Y_r
        }


    def _concatenate_observations(self, lon_obs: List, lat_obs: List,
                                 obs_dict: Dict, date_obs: List, current_date: str):
        """
        Concatenate observation arrays and compute temporal differences.

        Inputs:
            - lon_obs: List of longitude arrays for each time step
            - lat_obs: List of latitude arrays for each time step
            - obs_dict: Dictionary of observation variable arrays for each time step
            - date_obs: List of date arrays for each time step
            - current_date: Current analysis date string

        Outputs:
            - Dictionary containing concatenated and processed observation data:
              * 'lon_obs', 'lat_obs': Flattened coordinate arrays
              * 'obs_dict': Concatenated observation variables
              * 'date_obs': Concatenated date array
              * 'date_gap_day': Days difference from current date

        Description:
            Concatenates observation data across multiple time steps and computes
            the temporal gap (in days) between each observation and the current analysis time.
        """
        lon_concatenated = np.concatenate(lon_obs)
        lat_concatenated = np.concatenate(lat_obs)
        for key in obs_dict:
            obs_dict[key] = np.concatenate(obs_dict[key])
        date_concatenated = np.concatenate(date_obs)

        # Calculate day differences
        date_gap_day = (pd.to_datetime(date_concatenated, format="%Y%m%d") - pd.to_datetime(current_date, format="%Y%m%d")).days

        return {
            'lon_obs': lon_concatenated,
            'lat_obs': lat_concatenated,
            'obs_dict': obs_dict,
            'date_obs': date_concatenated,
            'date_gap_day': date_gap_day
        }

    def _remove_nan_values(self, observation_data, Cee, Cdzdz, obs_vars, c_vars, vars):
        """
        Remove invalid data points containing NaN values.

        Inputs:
            - observation_data: Dictionary containing observation data
            - Cee: Representation error covariance matrices
            - Cdzdz: Temporal error covariance matrices
            - obs_vars: List of observation variable names
            - c_vars: List of covariance variable names
            - vars: List of state variable names

        Outputs:
            - Filtered observation_data, Cee, Cdzdz with NaN entries removed

        Description:
            Identifies and removes data points that contain NaN values in any critical
            variable (coordinates, observations, or covariance matrices) to ensure
            numerical stability in subsequent computations.
        """
        valid_idx = ~(np.isnan(observation_data['lon_obs']) | np.isnan(observation_data['lat_obs']) | np.isnan(observation_data['obs_dict'][obs_vars[0]]) | np.isnan(Cee[c_vars[0]]) | np.isnan(Cdzdz[c_vars[0]]) | np.isnan(observation_data['obs_dict'][obs_vars[1]]) | np.isnan(Cee[c_vars[1]]) | np.isnan(Cdzdz[c_vars[1]]))
        observation_data['lon_obs'] = observation_data['lon_obs'][valid_idx]
        observation_data['lat_obs'] = observation_data['lat_obs'][valid_idx]
        observation_data['date_gap_day'] = observation_data['date_gap_day'][valid_idx]
        for var in vars:
            observation_data['HL'][var] = observation_data['HL'][var][valid_idx]
        for var in obs_vars:
            observation_data['obs_dict'][var] = observation_data['obs_dict'][var][valid_idx]

        for var in c_vars:
            Cee[var] = Cee[var][valid_idx]
            Cdzdz[var] = Cdzdz[var][valid_idx]
        return observation_data, Cee, Cdzdz

    def _remove_nan_values_from_background_data(self, background_data, observation_data, Cee, Cdzdz, obs_vars, c_vars, vars):
        """
        Remove invalid data points based on background field quality.

        Inputs:
            - background_data: Dictionary containing background field data
            - observation_data: Dictionary containing observation data
            - Cee: Representation error covariance matrices
            - Cdzdz: Temporal error covariance matrices
            - obs_vars: List of observation variable names
            - c_vars: List of covariance variable names
            - vars: List of state variable names

        Outputs:
            - Filtered background_data, observation_data, Cee, Cdzdz

        Description:
            Removes data points where background field interpolation produced NaN values,
            ensuring that only observations with valid background estimates are used
            in the assimilation. Also converts covariance matrices to diagonal form.
        """
        valid_idx = ~(np.isnan(background_data['H_Y_r'][vars[0]]) | np.isnan(background_data['H_Y_r'][vars[1]]))
        observation_data['lon_obs'] = observation_data['lon_obs'][valid_idx]
        observation_data['lat_obs'] = observation_data['lat_obs'][valid_idx]
        observation_data['date_gap_day'] = observation_data['date_gap_day'][valid_idx]
        for var in vars:
            observation_data['HL'][var] = observation_data['HL'][var][valid_idx]
            background_data['H_Y_r'][var] = background_data['H_Y_r'][var][valid_idx]
        for var in obs_vars:
            observation_data['obs_dict'][var] = observation_data['obs_dict'][var][valid_idx]
        # observation_data['thetao_obs'] = observation_data['thetao_obs'][valid_idx]
        for var in c_vars:
            Cee[var] = np.diag(Cee[var][valid_idx])
            Cdzdz[var] = np.diag(Cdzdz[var][valid_idx])
        return background_data, observation_data, Cee, Cdzdz


    def _interpolate_L_residuals_to_obs(self, residuals: Dict[str, Dict[str, xr.DataArray]],
                                     observation_data: Dict, vars: list):
        """
        Interpolate residual ensemble data to observation locations.

        Inputs:
            - residuals: Dictionary containing residual ensemble DataArrays
            - observation_data: Dictionary containing observation locations and metadata
            - vars: List of state variables to interpolate

        Outputs:
            - Dictionary containing interpolated residuals and processed observations:
              * 'lon_obs', 'lat_obs': Valid observation coordinates
              * 'obs_dict': Valid observation values
              * 'HL': Interpolated residuals at observation locations
              * 'L': Original residual ensemble on model grid
              * 'valid_obs_mask': Mask indicating valid observations

        Description:
            Interpolates model residual ensembles to observation locations using either
            nearest-neighbor localization or bilinear interpolation. Handles land points
            and removes observations where interpolation failed.
        """
        L = {}
        for var in vars:
            L[var] = residuals['residual_Ldata'][var].values.reshape((residuals['residual_Ldata'][var].shape[0], -1)).T
        # Handle land points (NaN values)
            land_mask = np.isnan(L[var]).any(axis=1)
            L[var][land_mask, :] = 0

        # Interpolate to observation locations
        HL = {}
        if self.localization_settings['is_localization']:
            lon_obs = observation_data['lon_obs']
            lat_obs = observation_data['lat_obs']
            for var in vars:
                lon_l = residuals['residual_Ldata'][var].lon.values
                lat_l = residuals['residual_Ldata'][var].lat.values
                lon_grid, lat_grid = np.meshgrid(lon_l, lat_l)
                lon_flat = lon_grid.reshape(-1)
                lat_flat = lat_grid.reshape(-1)
                H_rec = np.full((len(lon_obs), len(lon_flat)), 0)
                for i in range(len(lon_obs)):
                    dist = np.sqrt((lon_flat-lon_obs[i])**2 + (lat_flat-lat_obs[i])**2)
                    min_ind = np.argmin(dist)
                    H_rec[i, min_ind] = 1
                distance_matrix = (abs(lon_flat[:, np.newaxis] - lon_obs[np.newaxis, :])<10) & (abs(lat_flat[:, np.newaxis] - lat_obs[np.newaxis, :]) < 5)
                distance_matrix = distance_matrix.T
                H_rec[~distance_matrix] = 0
                HL[var] = H_rec @ L[var]
                valid_obs = ~(np.isnan(HL[var][:, 0]))
                HL[var] = HL[var][valid_obs, :]
        else:
            for var in vars:
                HL[var] = residuals['residual_Ldata'][var].interp(lon=('points', observation_data['lon_obs']), lat=('points', observation_data['lat_obs'])).values.T
            # Remove observations where interpolation failed
                valid_obs = ~(np.isnan(HL[var][:, 0]))
                HL[var] = HL[var][valid_obs, :]        
        
        for key in observation_data['obs_dict']:
            observation_data['obs_dict'][key] = observation_data['obs_dict'][key][valid_obs]

        interpolated_data = {
            'lon_obs': observation_data['lon_obs'][valid_obs],
            'lat_obs': observation_data['lat_obs'][valid_obs],
            'obs_dict': observation_data['obs_dict'],
            'date_obs': observation_data['date_obs'][valid_obs],
            'date_gap_day': observation_data['date_gap_day'][valid_obs],
            'L': L,
            'HL': HL,
            'valid_obs_mask': valid_obs
        }

        return interpolated_data


    def EnOI_analyze(self, observations: Dict[str, Any], background: Dict[str, Any],
                residuals: Dict[str, Any]):
        """
        Perform Ensemble Optimal Interpolation analysis update.

        Inputs:
            - observations: Dictionary containing observation data and error covariances
            - background: Dictionary containing background field and interpolated version
            - residuals: Dictionary containing residual ensembles (L and HL)

        Outputs:
            - Y_analysis: Analyzed state estimate on model grid
            - error_covariance: Diagonal of analysis error covariance matrix

        Description:
            Implements the core EnOI algorithm: computes innovation, constructs
            observation error covariance, calculates Kalman gain, performs analysis
            update, and estimates analysis error covariance. Includes robust error
            handling and numerical stability checks.
        """
        logger.info("Performing Ensemble Optimal Interpolation analysis")

        try:

            # Extract inputs
            thetao_obs = observations['obs_dict']
            H_Y_r = background['H_Y_r']
            Y_r = background['Y_r']

            L = residuals['L']  # Shape: (n_grid, n_ensemble)
            HL = residuals['HL']  # Shape: (n_obs, n_ensemble)

            Cee = observations['Cee']
            Cdzdz = observations['Cdzdz']

            # Ensemble size
            n_ensemble = HL.shape[1]
            logger.info(f"Using ensemble size: {n_ensemble}")

            if n_ensemble < 2:
                raise ValueError(f"Ensemble size must be at least 2, got {n_ensemble}")

            # Compute innovation (observation minus background)
            innovation = thetao_obs - H_Y_r
            logger.info(f"Innovation statistics: mean={innovation.mean():.3f}, std={innovation.std():.3f}")

            # Compute observation error covariance matrix B
            B = self._EnOI_compute_observation_covariance(HL, Cee, Cdzdz, n_ensemble)

            # Compute Kalman gain components
            kalman_gain,  cross_cov_l= self._EnOI_compute_kalman_gain(L, HL, B, n_ensemble)

            # Perform analysis update
            Y_analysis = self._perform_analysis_update(Y_r, kalman_gain, innovation)

            # Compute analysis error covariance
            error_covariance = self._EnOI_compute_error_covariance(L, kalman_gain, cross_cov_l, n_ensemble)

            logger.info("Analysis completed successfully")
            return Y_analysis, np.diag(error_covariance)

        except Exception as e:
            logger.error(f"Error in analysis: {e}")
            raise

    def no_obs_analysis(self, background: Dict[str, Any],
                residuals: Dict[str, Any]):
        """
        Perform analysis when no observations are available.

        Inputs:
            - background: Dictionary containing background field
            - residuals: Dictionary containing residual ensemble (L)

        Outputs:
            - Y_analysis: Background field (no update performed)
            - error_covariance: Estimated error covariance from ensemble

        Description:
            Handles the special case where no valid observations are available
            by returning the background field as the analysis and estimating
            error covariance directly from the residual ensemble.
        """
        logger.info("Performing no obs analysis")
        try:
            # Extract inputs
            Y_r = background['Y_r']

            L = residuals['L']  # Shape: (n_grid, n_ensemble)


            # Ensemble size
            n_ensemble = L.shape[1]
            logger.info(f"Using ensemble size: {n_ensemble}")

            if n_ensemble < 2:
                raise ValueError(f"Ensemble size must be at least 2, got {n_ensemble}")

            Y_analysis = Y_r
            error_covariance = 1/(n_ensemble-1) * L @ L.T

            logger.info("Analysis completed successfully")
            return Y_analysis, np.diag(error_covariance)

        except Exception as e:
            logger.error(f"Error in analysis: {e}")
            raise


    def _perform_analysis_update(self, Y_r: np.ndarray, kalman_gain: np.ndarray,
                                 innovation: np.ndarray):
        """
        Perform the analysis update: Y_a = Y_r + K * (y_obs - H*Y_r).

        Inputs:
            - Y_r: Background field estimate (n_grid,)
            - kalman_gain: Kalman gain matrix (n_grid, n_obs)
            - innovation: Observation-minus-background (n_obs,)

        Outputs:
            - Y_analysis: Updated state estimate after assimilation

        Description:
            Applies the standard Kalman filter analysis equation to update the
            background estimate using the Kalman gain and innovation. Includes
            sanity checks for NaN values and unrealistic magnitudes.
        """
        analysis_increment = kalman_gain @ innovation
        Y_analysis = Y_r + analysis_increment

        # Check for reasonable values
        if np.any(np.isnan(Y_analysis)):
            logger.warning("Analysis contains NaN values")
        if np.any(np.abs(Y_analysis) > 100):  # Unrealistic temperature anomalies
            logger.warning("Analysis contains potentially unrealistic values")

        logger.debug(f"Analysis update completed: mean increment={analysis_increment.mean():.3f}")
        return Y_analysis


    def _EnOI_compute_observation_covariance(self, HL: np.ndarray,
                                        Cee: np.ndarray, Cdzdz: np.ndarray,
                                        n_ensemble: int):
        """
        Compute the observation error covariance matrix B.

        Inputs:
            - HL: H-operator applied to residual ensemble (n_obs, n_ensemble)
            - Cee: Representation error covariance (n_obs, n_obs)
            - Cdzdz: Temporal error covariance (n_obs, n_obs)
            - n_ensemble: Size of the ensemble

        Outputs:
            - B: Total observation error covariance matrix (n_obs, n_obs)

        Description:
            Constructs the total observation error covariance matrix by combining
            ensemble covariance, representation error, and temporal error components.
            Includes numerical stability considerations through optional regularization.
        """
        # Ensemble covariance components
        ensemble_cov_hl = (1.0 / (n_ensemble - 1)) * HL @ HL.T


        # Total observation error covariance
        B = ensemble_cov_hl + Cee + Cdzdz

        # Add small regularization for numerical stability
        # regularization = 1e-8 * np.eye(B.shape[0])
        # B += regularization

        logger.debug(f"Observation covariance computed: shape={B.shape}, cond_number={np.linalg.cond(B):.2e}")
        return B

    def _EnOI_compute_kalman_gain(self, L: np.ndarray,
                             HL: np.ndarray, B: np.ndarray, n_ensemble: int):
        """
        Compute the Kalman gain matrix.

        Inputs:
            - L: L-residual ensemble (n_grid, n_ensemble)
            - HL: H-operator applied to residual ensemble (n_obs, n_ensemble)
            - B: Observation error covariance matrix (n_obs, n_obs)
            - n_ensemble: Size of the ensemble

        Outputs:
            - kalman_gain: Kalman gain matrix (n_grid, n_obs)
            - cross_cov_l: Cross-covariance between state and observations (n_grid, n_obs)

        Description:
            Computes the Kalman gain matrix K = P * H^T * B^(-1) where P is the
            background error covariance. Uses SVD-based pseudoinverse for numerical
            stability and handles potential matrix inversion failures.
        """
        try:
            # Compute cross-covariance between state and observations
            cross_cov_l = (1.0 / (n_ensemble - 1)) * L @ HL.T


            # Compute Kalman gain: K = PH^T * B^{-1}
            B_inv = svd_inverse(B, pseudo=True, threshold=self.threshold_settings['svd_threshold'])
            kalman_gain = cross_cov_l @ B_inv

            logger.debug(f"Kalman gain computed: shape={kalman_gain.shape}")
            return kalman_gain, cross_cov_l

        except np.linalg.LinAlgError as e:
            logger.error(f"Matrix inversion failed in Kalman gain computation: {e}")
            raise

    def _EnOI_compute_error_covariance(self, L: np.ndarray, kalman_gain: np.ndarray,
                                  cross_cov_l:np.ndarray, n_ensemble: int):
        """
        Compute the analysis error covariance.

        Inputs:
            - L: L-residual ensemble (n_grid, n_ensemble)
            - kalman_gain: Kalman gain matrix (n_grid, n_obs)
            - cross_cov_l: Cross-covariance matrix (n_grid, n_obs)
            - n_ensemble: Size of the ensemble

        Outputs:
            - analysis_covariance: Analysis error covariance matrix (n_grid, n_grid)

        Description:
            Computes the analysis error covariance using the standard Kalman filter
            formula: P_a = P_b - K * H * P_b. Includes error handling and provides
            conservative fallback estimate if computation fails.
        """
        try:
            # Background error covariance M
            background_cov_l = (1.0 / (n_ensemble - 1)) * L @ L.T
            # Analysis error covariance: P_a = P_b - K * H * P_b
            # Using the identity: P_a = P_b - K * (H * P_b)
            # where H * P_b = cross_covariance.T
            analysis_covariance = background_cov_l - kalman_gain @ cross_cov_l.T

            return analysis_covariance

        except np.linalg.LinAlgError as e:
            logger.error(f"Error in covariance computation: {e}")
            # Return a conservative estimate if computation fails
            logger.warning("Using conservative error estimate")
            return np.ones(L.shape[0]) * 0.1  # Default variance