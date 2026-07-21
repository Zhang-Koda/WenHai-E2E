"""
Data preprocessing functions.
"""

import xarray as xr
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Dict

class DataLoader:
    """Class for loading and preprocessing various data sources."""

    def __init__(self, data_path: Path, file_settings: Dict, time_settings: Dict, region_settings: Dict):
        """
        Initialize the DataLoader with configuration settings.

        Inputs:
            - data_path: Root directory path for all data files
            - file_settings: Dictionary containing folder names and experiment identifiers
            - time_settings: Dictionary with temporal parameters (year ranges, bandwidths)
            - region_settings: Dictionary defining spatial boundaries and depth

        Outputs:
            - None (initializes class attributes for data loading operations)

        Description:
            Sets up the data loader with paths and configuration parameters needed
            to locate and access various oceanographic datasets including model outputs,
            reanalysis products, climatologies, and observational data.
        """
        self.data_path = Path(data_path)
        self.file_settings = file_settings
        self.time_settings = time_settings
        self.region_settings = region_settings


    def load_gsvc_data(self, target_date: str, vars: list):
        """
        Load GSVC  model data for specified variables and date range.

        Inputs:
            - target_date: Target date string in 'YYYYMMDD' format
            - vars: List of variable names to load (e.g., ['thetao', 'so'])

        Outputs:
            - vars_dict: Dictionary containing loaded DataArrays for each requested variable
            - Boolean flag indicating if all data is NaN (True if no valid data)

        Description:
            Loads GSVC model output data for a date range centered around target_date.
            Handles leap year cases (Feb 29) with linear interpolation. Manages data
            spanning multiple years and applies spatial subsetting and padding. Uses
            Zarr format for efficient chunked access to large datasets.
        """
      
        llon1, llon2 = self.region_settings['llon1'], self.region_settings['llon2']
        llat1, llat2 = self.region_settings['llat1'], self.region_settings['llat2']
        depth = self.region_settings['depth']
        engine='zarr'
        grid_data_bandwidth = self.time_settings['grid_data_bandwidth']
        current_date = target_date
        year_begin, year_end = self.time_settings['year_begin'], self.time_settings['year_end']
        if (int(target_date[:4]) % 4 != 0) & (target_date[-4:] == '0229'):
            before_date =  f"{target_date[:4]}0228"
            after_date = f"{target_date[:4]}0301"
            date_range = self.generate_leap_date_range(current_date, grid_data_bandwidth, year_begin, year_end)
        else:
            date_range = self.generate_date_range(current_date, grid_data_bandwidth, year_begin, year_end)
        years, year_flag = self.check_years_in_date_array(date_range)
        list_date_range = date_range.strftime('%Y%m%d').tolist()
        vars_dict = {v: [] for v in vars}
        if year_flag:
            file_path = self.data_path / self.file_settings['gsvc_folder'] / f"{self.file_settings['gsvc_experiment']}{depth:.1f}m_{years[0]}_{years[1]}.zarr"
            if not file_path.exists():
                raise FileNotFoundError(f"GSVC data file not found: {file_path}")
            da1 = xr.open_dataset(file_path, engine=engine)
            vars = [v for v in vars if v in da1.data_vars]

            for var in vars:
                var_concat = da1[var]
                if (int(target_date[:4]) % 4 != 0) & (target_date[-4:] == '0229'):
                    hole_date = var_concat.sel(date=list_date_range)
                    padding_date = 0.5 * hole_date.sel(date=before_date) + 0.5 * hole_date.sel(date=after_date)
                    padding_date = padding_date.expand_dims(date=[f'{target_date[:4]}0229'])
                    concat_date = xr.concat([hole_date.sel(date=slice(None, before_date)), padding_date, hole_date.sel(date=slice(after_date, None))],dim='date')
                    concat_date = self.is_padding(concat_date, llon1, llon2, llat1, llat2)
                    concat_date = concat_date.sel(lon=slice(llon1, llon2), lat=slice(llat1, llat2))
                    vars_dict[var].append(concat_date)
                else:
                    cut_date = var_concat.sel(date=list_date_range)
                    cut_date = self.is_padding(cut_date, llon1, llon2, llat1, llat2)
                    vars_dict[var].append(cut_date.sel(lon=slice(llon1, llon2), lat=slice(llat1, llat2)))
            if vars_dict[var][0].isnull().all().item():
                return vars_dict, True
        else:
            file_path = self.data_path / self.file_settings['gsvc_folder'] / f"{self.file_settings['gsvc_experiment']}{depth:.1f}m_{years[0]}.zarr"
            if not file_path.exists():
                raise FileNotFoundError(f"GSVC data file not found: {file_path}")

            da = xr.open_dataset(file_path, engine=engine)
            vars = [v for v in vars if v in da.data_vars]
            for var in vars:
                if (int(target_date[:4]) % 4 != 0) & (target_date[-4:] == '0229'):
                    hole_date = da[var].sel(date=list_date_range, lon=slice(llon1, llon2), lat=slice(llat1, llat2), depth=depth)
                    padding_date = 0.5 * hole_date.sel(date=before_date) + 0.5 * hole_date.sel(date=after_date)
                    padding_date = padding_date.expand_dims(date=[f'{target_date[:4]}0229'])
                    concat_date = xr.concat([hole_date.sel(date=slice(None, before_date)), padding_date, hole_date.sel(date=slice(after_date, None))],dim='date')
                    concat_date = self.is_padding(concat_date, llon1, llon2, llat1, llat2)
                    vars_dict[var].append(concat_date)
                else:

                    cut_date = da[var].sel(date=list_date_range)
                    cut_date = self.is_padding(cut_date, llon1, llon2, llat1, llat2)
                    vars_dict[var].append(cut_date.sel(lon=slice(llon1, llon2), lat=slice(llat1, llat2)))
            if vars_dict[var][0].isnull().all().item():
                return vars_dict, True

        return vars_dict, False

    def load_glorys_data(self, target_date: str, vars: list):
        """
        Load GLORYS reanalysis data for specified variables and date range.

        Inputs:
            - target_date: Target date string in 'YYYYMMDD' format
            - vars: List of variable names to load (e.g., ['thetao', 'so'])

        Outputs:
            - vars_dict: Dictionary containing loaded DataArrays for each requested variable
            - Boolean flag indicating if all data is NaN (True if no valid data)

        Description:
            Loads GLORYS ocean reanalysis data with similar functionality to load_gsvc_data.
            Handles leap years, multi-year data spans, spatial subsetting, and padding.
            Uses depth-specific folder structure and Zarr storage format for efficient access.
        """
        llon1, llon2 = self.region_settings['llon1'], self.region_settings['llon2']
        llat1, llat2 = self.region_settings['llat1'], self.region_settings['llat2']
        depth = self.region_settings['depth']
        engine='zarr'
        grid_data_bandwidth = self.time_settings['grid_data_bandwidth']
        current_date = target_date
        year_begin, year_end = self.time_settings['year_begin'], self.time_settings['year_end']

        if (int(target_date[:4]) % 4 != 0) & (target_date[-4:] == '0229'):
            before_date = f"{target_date[:4]}0228"
            after_date = f"{target_date[:4]}0301"
            date_range = self.generate_leap_date_range(current_date, grid_data_bandwidth, year_begin, year_end)
        else:
            date_range = self.generate_date_range(current_date, grid_data_bandwidth, year_begin, year_end)
        years, year_flag = self.check_years_in_date_array(date_range)
        list_date_range = date_range.strftime('%Y%m%d').tolist()
        vars_dict = {v: [] for v in vars}
        if year_flag:
            file_path = self.data_path / self.file_settings['glorys_folder'] / f"{self.file_settings['glorys_depth_folder']}{depth:.1f}" / f"{self.file_settings['glorys_experiment']}{years[0]}_{years[1]}.zarr"
            if not file_path.exists():
                raise FileNotFoundError(f"GLORYS data file not found: {file_path}")
            da1 = xr.open_dataset(file_path, engine=engine)
            vars = [v for v in vars if v in da1.data_vars]
            for var in vars:
                var_concat = da1[var]
                if (int(target_date[:4]) % 4 != 0) & (target_date[-4:] == '0229'):
                    hole_date = var_concat.sel(date=list_date_range)
                    padding_date = 0.5 * hole_date.sel(date=before_date) + 0.5 * hole_date.sel(date=after_date)
                    padding_date = padding_date.expand_dims(date=[f'{target_date[:4]}0229'])
                    concat_date = xr.concat([hole_date.sel(date=slice(None, before_date)), padding_date,
                                             hole_date.sel(date=slice(after_date, None))], dim='date')
                    concat_date = self.is_padding(var_concat, llon1, llon2, llat1, llat2)
                    concat_date = concat_date.sel(lon=slice(llon1, llon2), lat=slice(llat1, llat2))
                    vars_dict[var].append(concat_date)
                else:
                    cut_date = var_concat.sel(date=list_date_range)
                    cut_date = self.is_padding(cut_date, llon1, llon2, llat1, llat2)
                    vars_dict[var].append(cut_date.sel(lon=slice(llon1, llon2), lat=slice(llat1, llat2)))
            if vars_dict[var][0].isnull().all().item():
                return vars_dict, True

        else:
            file_path = self.data_path / self.file_settings['glorys_folder'] / f"{self.file_settings['glorys_depth_folder']}{depth:.1f}" / f"{self.file_settings['glorys_experiment']}{years[0]}.zarr"
            if not file_path.exists():
                raise FileNotFoundError(f"GLORYS data file not found: {file_path}")

            da = xr.open_dataset(file_path, engine=engine)
            vars = [v for v in vars if v in da.data_vars]
            for var in vars:
                if (int(target_date[:4]) % 4 != 0) & (target_date[-4:] == '0229'):
                    hole_date = da[var].sel(date=list_date_range, lon=slice(llon1, llon2), lat=slice(llat1, llat2))
                    padding_date = 0.5 * hole_date.sel(date=before_date) + 0.5 * hole_date.sel(date=after_date)
                    padding_date = padding_date.expand_dims(date=[f'{target_date[:4]}0229'])
                    concat_date = xr.concat([hole_date.sel(date=slice(None, before_date)), padding_date, hole_date.sel(date=slice(after_date, None))],dim='date')
                    concat_date = self.is_padding(concat_date, llon1, llon2, llat1, llat2)
                    vars_dict[var].append(concat_date)

                else:
                    cut_date = da[var].sel(date=list_date_range)
                    cut_date = self.is_padding(cut_date, llon1, llon2, llat1, llat2)
                    vars_dict[var].append(cut_date.sel(lon=slice(llon1, llon2), lat=slice(llat1, llat2)))
            if vars_dict[var][0].isnull().all().item():
                return vars_dict, True

        return vars_dict, False


    def load_clim_thetao_data(self, depth: int):
        """
        Load climatological temperature (THETAO) data at specified depth.

        Inputs:
            - depth: Target depth for climatological temperature data

        Outputs:
            - da_clim_thetao: DataArray containing climatological temperature

        Description:
            Loads depth-specific climatological temperature data from Zarr storage.
            Applies spatial padding to handle regional boundary conditions and
            returns data subsetted to the analysis region.
        """
        llon1, llon2 = self.region_settings['llon1'], self.region_settings['llon2']
        llat1, llat2 = self.region_settings['llat1'], self.region_settings['llat2']
        engine = 'zarr'
        clim_thetao_file = self.data_path / self.file_settings['clim_thetao_folder']/ f"{self.file_settings['clim_thetao_experiment']}{depth:.1f}.zarr"
        if not clim_thetao_file.exists():
            raise FileNotFoundError(f"CLIM THETAO data file not found: {clim_thetao_file}")
        da_clim_thetao = xr.open_dataset(clim_thetao_file, engine=engine).clim_thetao
        da_clim_thetao = self.is_padding(da_clim_thetao, llon1, llon2, llat1, llat2)
        return da_clim_thetao

    def load_clim_so_data(self, depth: int):
        """
        Load climatological salinity (SO) data at specified depth.

        Inputs:
            - depth: Target depth for climatological salinity data

        Outputs:
            - da_clim_so: DataArray containing climatological salinity

        Description:
            Loads depth-specific climatological salinity data from Zarr storage.
            Applies spatial padding and returns data subsetted to the analysis region.
            Similar to load_clim_thetao_data but for salinity variable.
        """
        llon1, llon2 = self.region_settings['llon1'], self.region_settings['llon2']
        llat1, llat2 = self.region_settings['llat1'], self.region_settings['llat2']
        engine = 'zarr'
        depth = round(depth, 1)
        clim_so_file = self.data_path / self.file_settings['clim_so_folder'] / f"{self.file_settings['clim_so_experiment']}{depth:.1f}.zarr"
        if not clim_so_file.exists():
            raise FileNotFoundError(f"CLIM THETAO data file not found: {clim_so_file}")
        da_clim_so = xr.open_dataset(clim_so_file, engine=engine).clim_so
        da_clim_so = self.is_padding(da_clim_so, llon1, llon2, llat1, llat2)
        return da_clim_so

    def load_mesoscale_gsvc(self, current_date, obs_bandwidth: Tuple, depth):
        """
        Load mesoscale GSVC data for observation error estimation.

        Inputs:
            - current_date: Current analysis date string
            - obs_bandwidth: Temporal window for observation data (days before/after)
            - depth: Target depth for mesoscale data

        Outputs:
            - da_mesoscale: Dataset containing mesoscale velocity data

        Description:
            Loads mesoscale velocity data from historical period (2019 to current year)
            for use in representing observation errors. Handles multi-year data spans
            and applies spatial subsetting and padding for regional analysis.
        """
        llon1, llon2 = self.region_settings['llon1'], self.region_settings['llon2']
        llat1, llat2 = self.region_settings['llat1'], self.region_settings['llat2']
        engine='zarr'
        year_begin, year_end = 2019, int(current_date[:4])
        date_range = self.generate_date_range(current_date, obs_bandwidth, year_begin, year_end)
        years, year_flag = self.check_years_in_date_array(date_range)
        if year_flag:
            year_list = self.split_date_range_by_year(date_range)
            mesoscale_gsvc_file1 = self.data_path / self.file_settings['mesoscale_gsvc_folder'] / f"{self.file_settings['mesoscale_gsvc']}{depth:.1f}m_{years[0]}.zarr"
            if not mesoscale_gsvc_file1.exists():
                raise FileNotFoundError(f"MESOSCALE GSVC data file not found: {mesoscale_gsvc_file1}")
            mesoscale_gsvc_file2 = self.data_path / self.file_settings['mesoscale_gsvc_folder'] / f"{self.file_settings['mesoscale_gsvc']}{depth:.1f}m_{years[1]}.zarr"
            if not mesoscale_gsvc_file2.exists():
                raise FileNotFoundError(f"MESOSCALE GSVC data file not found: {mesoscale_gsvc_file2}")
            da1 = xr.open_dataset(mesoscale_gsvc_file1, engine=engine).sel(date=year_list[0])
            da2 = xr.open_dataset(mesoscale_gsvc_file2, engine=engine).sel(date=year_list[1])
            da_mesoscale = xr.concat((da1, da2), dim='date')
            da_mesoscale = self.is_padding(da_mesoscale, llon1, llon2, llat1, llat2)
            da_mesoscale = da_mesoscale.sel(lon=slice(llon1, llon2), lat=slice(llat1, llat2))
        else:
            list_date_range = date_range.strftime('%Y%m%d').tolist()
            mesoscale_gsvc_file = self.data_path / self.file_settings['mesoscale_gsvc_folder'] / f"{self.file_settings['mesoscale_gsvc']}{depth:.1f}m_{years[0]}.zarr"
            if not mesoscale_gsvc_file.exists():
                raise FileNotFoundError(f"MESOSCALE GSVC data file not found: {mesoscale_gsvc_file}")
            da_mesoscale = xr.open_dataset(mesoscale_gsvc_file, engine=engine).sel(date=list_date_range)
            da_mesoscale = self.is_padding(da_mesoscale, llon1, llon2, llat1, llat2)
            da_mesoscale = da_mesoscale.sel(lon=slice(llon1, llon2), lat=slice(llat1, llat2))
        return da_mesoscale


    def load_observation_data(self, current_date: str, obs_bandwidth: Tuple):
        """
        Load observational data from Argo floats or other measurement platforms.

        Inputs:
            - current_date: Current analysis date string
            - obs_bandwidth: Temporal window for observation data (days before/after)

        Outputs:
            - da_obs: Dataset containing observational data
            - list_date_range: List of dates in the observation period
            - no_obs: Boolean flag indicating if no valid observations were found

        Description:
            Loads observational data (typically Argo profiles) within temporal window.
            Handles multi-year data spans and applies sophisticated spatial padding
            to handle global coordinate systems and regional boundaries. Returns
            flag indicating if observations are available for the period.
        """
        llon1, llon2 = self.region_settings['llon1'], self.region_settings['llon2']
        llat1, llat2 = self.region_settings['llat1'], self.region_settings['llat2']
        year_begin, year_end = 2019, int(current_date[:4])
        date_range = self.generate_date_range(current_date, obs_bandwidth, year_begin, year_end)
        years, year_flag = self.check_years_in_date_array(date_range)
        list_date_range = date_range.strftime('%Y%m%d').tolist()
        if year_flag:
            year_list = self.split_date_range_by_year(date_range)
            observation_data_file1 = self.data_path / self.file_settings[
                'observation_folder'] / f"{self.file_settings['observation_experiment']}{years[0]}.nc"
            if not observation_data_file1.exists():
                raise FileNotFoundError(f"OBSERVATION data file not found: {observation_data_file1}")
            observation_data_file2 = self.data_path / self.file_settings[
                'observation_folder'] / f"{self.file_settings['observation_experiment']}{years[1]}.nc"
            if not observation_data_file2.exists():
                raise FileNotFoundError(f"OBSERVATION data file not found: {observation_data_file2}")
            ds1 = xr.open_dataset(observation_data_file1)
            ds2 = xr.open_dataset(observation_data_file2)
            valid_date_1 = [v for v in year_list[0] if v in ds1.date.values]
            valid_date_2 = [v for v in year_list[1] if v in ds2.date.values]
            da1 = ds1.sel(date=valid_date_1)
            da2 = ds2.sel(date=valid_date_2)
            da_obs = xr.concat((da1, da2), dim='date')
            da_obs, no_obs = self.is_obs_padding(da_obs, llon1, llon2, llat1, llat2)

        else:
            observation_data_file = self.data_path / self.file_settings['observation_folder'] /f"{self.file_settings['observation_experiment']}{years[0]}.nc"
            if not observation_data_file.exists():
                raise FileNotFoundError(f"OBSERVATION data file not found: {observation_data_file}")
            ds_obs = xr.open_dataset(observation_data_file)
            valid_date = [v for v in list_date_range if v in ds_obs.date.values]
            da_obs = ds_obs.sel(date=valid_date)
            da_obs, no_obs = self.is_obs_padding(da_obs, llon1, llon2, llat1, llat2)
        return da_obs, list_date_range, no_obs

    def load_mesoscale_var(self, depth, var):
        """
        Load mesoscale variance data for specified variable and depth.

        Inputs:
            - depth: Target depth for variance calculation
            - var: Variable name ('thetao' or 'so')

        Outputs:
            - da_var: DataArray containing mesoscale variance for the variable

        Description:
            Loads precomputed mesoscale variance data used for representing observation
            errors in data assimilation. Applies spatial padding and subsetting to
            match the analysis region boundaries.
        """
        llon1, llon2 = self.region_settings['llon1'], self.region_settings['llon2']
        llat1, llat2 = self.region_settings['llat1'], self.region_settings['llat2']
        engine='zarr'

        mesoscale_var_file = self.data_path / self.file_settings['mesoscale_folder'] / f"{self.file_settings['mesoscale_experiment']}{depth:.1f}.zarr"
        if not mesoscale_var_file.exists():
            raise FileNotFoundError(f"MESOSCALE data file not found: {mesoscale_var_file}")
        da_var = xr.open_dataset(mesoscale_var_file, engine=engine)[var]
        da_var = self.is_padding(da_var, llon1, llon2, llat1, llat2).sel(lon=slice(llon1,llon2), lat=slice(llat1,llat2))
        return  da_var

    def load_largescale_var_deltaz(self, depth):
        """
        Load large-scale variance data for temporal error covariance.

        Inputs:
            - depth: Target depth for variance data

        Outputs:
            - deltaz_var: Dataset containing temporal variance components

        Description:
            Loads large-scale variance data that varies with time lag (delta-z).
            Used for computing temporal error covariance in data assimilation.
            Applies spatial padding for regional boundary handling.
        """
        llon1, llon2 = self.region_settings['llon1'], self.region_settings['llon2']
        llat1, llat2 = self.region_settings['llat1'], self.region_settings['llat2']
        engine='zarr'

        largescale_var_deltaz_file = self.data_path / self.file_settings['largescale_var_deltaz_folder'] / f"{self.file_settings['largescale_var_deltaz_experiment']}{depth:.1f}.zarr"
        if not largescale_var_deltaz_file.exists():
            raise FileNotFoundError(f"LARGESCALE VAR DELTAZ data file not found: {largescale_var_deltaz_file}")
        deltaz_var = xr.open_dataset(largescale_var_deltaz_file, engine=engine)
        deltaz_var = self.is_padding(deltaz_var, llon1, llon2, llat1, llat2)
        return  deltaz_var

    def load_Y_r_and_H_Y_r(self, current_date: str, depth: int, date_gap_day, lon_obs, lat_obs, sca_factor, vars):
        """
        Load background field and interpolate to observation locations.

        Inputs:
            - current_date: Current analysis date
            - depth: Target depth
            - date_gap_day: Array of time differences for background selection
            - lon_obs, lat_obs: Observation coordinates
            - sca_factor: Spatial scaling factor for coarsening
            - vars: List of state variables

        Outputs:
            - Y_r: Background field on model grid (flattened)
            - H_Y_r: Background field interpolated to observation locations

        Description:
            Loads the background field (Y_r) and creates its observation-space
            representation (H_Y_r) using interpolation. Handles multiple time levels
            for background selection and applies spatial coarsening. Essential for
            data assimilation analysis step.
        """
        llon1, llon2 = self.region_settings['llon1'], self.region_settings['llon2']
        llat1, llat2 = self.region_settings['llat1'], self.region_settings['llat2']
        engine='zarr'

        date_range = [(pd.to_datetime(current_date, format='%Y%m%d') + pd.Timedelta(days=days)) for days in date_gap_day]
        date_range_load = date_range + [pd.to_datetime(current_date, format='%Y%m%d')]
        years, year_flag = self.check_years_in_date_array(date_range_load)
        date_list = [day.strftime('%Y%m%d') for day in date_range]
        if year_flag:
            year_list = self.split_date_range_by_year(date_range_load)
            Y_r_file1 = self.data_path / self.file_settings['Y_r_folder'] / f"{self.file_settings['Y_r_file']}{depth:.1f}m_{years[0]}.zarr"
            if not Y_r_file1.exists():
                raise FileNotFoundError(f"LARGESCALE VAR DELTAZ data file not found: {Y_r_file1}")
            Y_r_file2 = self.data_path / self.file_settings['Y_r_folder'] / f"{self.file_settings['Y_r_file']}{depth:.1f}m_{years[1]}.zarr"
            if not Y_r_file2.exists():
                raise FileNotFoundError(f"LARGESCALE VAR DELTAZ data file not found: {Y_r_file2}")
            da1 = xr.open_dataset(Y_r_file1, engine=engine).sel(date=np.unique(year_list[0]))
            da2 = xr.open_dataset(Y_r_file2, engine=engine).sel(date=np.unique(year_list[1]))
            da = xr.concat((da1, da2), dim='date')
            da = self.is_padding(da, llon1, llon2, llat1, llat2)
        else:
            date_list_load = [day.strftime('%Y%m%d') for day in date_range_load]
            Y_r_file = self.data_path / self.file_settings['Y_r_folder'] / f"{self.file_settings['Y_r_file']}{depth:.1f}m_{years[0]}.zarr"
            if not Y_r_file.exists():
                raise FileNotFoundError(f"LARGESCALE VAR DELTAZ data file not found: {Y_r_file}")
            da = xr.open_dataset(Y_r_file, engine=engine).sel(date=np.unique(date_list_load))
            da = self.is_padding(da, llon1, llon2, llat1, llat2)

        da= da.coarsen(dim={'lon': sca_factor, 'lat': sca_factor}, boundary='trim').mean()
        Y_r = {}
        for var in vars:
            Y_r[var] = da[var].sel(date=current_date, lon=slice(llon1, llon2), lat=slice(llat1, llat2)).values.reshape(-1)

        H_Y_r = {v: [] for v in vars}

        for var in vars:
            unique_values = np.unique(date_list)
            result_var = np.zeros_like(lon_obs, dtype=float)
            for value in unique_values:
                mask = np.array(date_list) == value
                points_ds = xr.Dataset({
                    'lat': (['points'], lat_obs[mask]),
                    'lon': (['points'], lon_obs[mask])
                })
                result_var[mask] = da[var].sel(date=value).interp(points_ds).values

            H_Y_r[var] = result_var

        return Y_r, H_Y_r

    def load_Y_r(self, current_date: str, depth: int, sca_factor, vars):
        """
        Load background field without observation interpolation.

        Inputs:
            - current_date: Current analysis date
            - depth: Target depth
            - sca_factor: Spatial scaling factor for coarsening
            - vars: List of state variables

        Outputs:
            - Y_r: Background field on model grid (flattened)

        Description:
            Loads the background field (Y_r) for cases without observation data.
            Applies spatial coarsening and reshapes to flattened vector format.
            Used when no valid observations are available for assimilation.
        """
        llon1, llon2 = self.region_settings['llon1'], self.region_settings['llon2']
        llat1, llat2 = self.region_settings['llat1'], self.region_settings['llat2']
        engine='zarr'

        Y_r_file = self.data_path / self.file_settings['Y_r_folder'] / f"{self.file_settings['Y_r_file']}{depth:.1f}m_{current_date[:4]}.zarr"
        if not Y_r_file.exists():
            raise FileNotFoundError(f"LARGESCALE VAR DELTAZ data file not found: {Y_r_file}")
        da = xr.open_dataset(Y_r_file, engine=engine).sel(date=current_date)
        da = self.is_padding(da, llon1, llon2, llat1, llat2)
        da = da.coarsen(dim={'lon': sca_factor, 'lat': sca_factor}, boundary='trim').mean()
        Y_r = {}
        for var in vars:
            Y_r[var] = da[var].sel(lon=slice(llon1, llon2), lat=slice(llat1, llat2)).values.reshape(-1)
        return Y_r


    def generate_date_range(self, target_date_str, grid_data_bandwidth, year_begin, year_end):
        """
        Generate date range for data loading with temporal bandwidth.

        Inputs:
            - target_date_str: Center date string 'YYYYMMDD'
            - grid_data_bandwidth: Tuple (days_before, days_after) for temporal window
            - year_begin, year_end: Year boundaries for filtering

        Outputs:
            - filtered_date_range: Pandas DatetimeIndex of valid dates

        Description:
            Creates a temporal window around target date and filters by year range.
            Used to define the time period for loading model and observation data.
        """
        target_date = pd.to_datetime(target_date_str, format='%Y%m%d')

        date_range = pd.date_range(
            start=target_date - pd.Timedelta(days=grid_data_bandwidth[0]),
            end=target_date + pd.Timedelta(days=grid_data_bandwidth[1]),
            freq='D'
        )

        year_mask = (date_range.year >= year_begin) & (date_range.year <= year_end)
        filtered_date_range = date_range[year_mask]

        return filtered_date_range

    def generate_leap_date_range(self, target_date_str, grid_data_bandwidth, year_begin, year_end):
        """
        Generate date range handling leap year February 29th specially.

        Inputs:
            - target_date_str: Date string that may be Feb 29 in non-leap year
            - grid_data_bandwidth: Temporal window specification
            - year_begin, year_end: Year boundaries

        Outputs:
            - filtered_date_range: DatetimeIndex with March 1st as center

        Description:
            Special handling for Feb 29th in non-leap years by shifting center
            to March 1st to avoid missing data issues. Ensures continuous
            temporal coverage for data loading.
        """
        new_date_str = f'{target_date_str[:4]}0301'
        target_date = pd.to_datetime(new_date_str, format='%Y%m%d')

        date_range = pd.date_range(
            start=target_date - pd.Timedelta(days=grid_data_bandwidth[0]),
            end=target_date + pd.Timedelta(days=grid_data_bandwidth[1]-1),
            freq='D'
        )


        year_mask = (date_range.year >= year_begin) & (date_range.year <= year_end)
        filtered_date_range = date_range[year_mask]


        return filtered_date_range

    def check_years_in_date_array(self, target_date_range):
        """
        Check if date range spans multiple years and identify unique years.

        Inputs:
            - target_date_range: Pandas DatetimeIndex to analyze

        Outputs:
            - years: Sorted list of unique years in date range
            - has_two_years: Boolean indicating if span exceeds one year

        Description:
            Analyzes temporal span of date range to determine if data loading
            needs to handle multiple yearly files. Critical for proper file
            path construction and data concatenation.
        """
        original_dates = target_date_range

        years = sorted(list(set([date.year for date in original_dates])))
        has_two_years = len(years) == 2

        return years, has_two_years


    def split_date_range_by_year(self, date_range):
        """
        Split date range into separate lists for each year.

        Inputs:
            - date_range: Pandas DatetimeIndex to split

        Outputs:
            - year_groups: List of date lists, one per year

        Description:
            Groups consecutive dates by year while preserving order. Used for
            loading and concatenating data from multiple yearly files while
            maintaining temporal sequence.
        """

        date_series = pd.Series(date_range).drop_duplicates()

        df_dates = pd.DataFrame({'date': date_series})
        df_dates['year'] = df_dates['date'].dt.year

        year_groups = []
        for year, group in df_dates.groupby('year', sort=True): 

            date_list = group['date'].dt.strftime('%Y%m%d').tolist()
            year_groups.append(date_list)

        return year_groups


    def is_padding(self, da, llon1, llon2, llat1, llat2):
        """
        Apply spatial padding to handle regional boundary conditions.

        Inputs:
            - da: DataArray to pad
            - llon1, llon2: Regional longitude boundaries
            - llat1, llat2: Regional latitude boundaries

        Outputs:
            - da_c: DataArray with spatial padding applied

        Description:
            Implements sophisticated spatial padding to handle global coordinate
            systems and regional analysis domains. Uses mirroring and extrapolation
            techniques to extend data beyond regional boundaries for boundary
            condition handling in analysis.
        """
        lon_min, lon_max = 0, 360
        lat_min, lat_max = -80, 90

        if llon1 < lon_min:
            da_s = da.sel(lon=slice(llon1 + da.lon.values.max(), da.lon.values.max()))
            # before concat cut lon first
            da_r = da.sel(lon=slice(da.lon.values.min(), llon2))
            da_s = da_s.assign_coords(lon=da_s.lon - da.lon.values.max())
            if da_s.lon.values[-1] == da_r.lon.values[0]:
                da_s = da_s.isel(lon=slice(None, -1))
            da_c = xr.concat((da_s, da_r), dim='lon')
            if llat1 < lat_min:
                da_s = da_c.sel(lat=slice(da.lat.values.min(), -llat1 + 2 * da.lat.values.min()))
                da_s = da_s.assign_coords(lat=2 * da.lat.values.min()-da_s.lat)
                da_s = da_s.isel(lat=slice(None, None, -1))
                # before concat cut lat first
                da_t = da_c.sel(lat=slice(da_c.lat.values.min(), llat2)).isel(lat=slice(1, None))
                da_c = xr.concat((da_s, da_t), dim='lat')
                # already slice to block
                return da_c
            elif llat2 > lat_max:
                da_s = da_c.sel(lat=slice(2 * da.lat.values.max() - llat2, da.lat.values.max()))
                da_s = da_s.assign_coords(lat=2 * da.lat.values.max()-da_s.lat)
                da_s = da_s.isel(lat=slice(None, None, -1))
                da_b = da_c.sel(lat=slice(llat1, da.lat.values.max())).isel(lat=slice(None, -1))
                da_c = xr.concat((da_b, da_s), dim='lat')
                return da_c

            else:
                da_c = da_c.sel(lat=slice(llat1, llat2))
                return da_c

        elif llon2 > lon_max:
            da_s = da.sel(lon=slice(da.lon.values.min(), llon2-da.lon.values.max()))
            da_s = da_s.assign_coords(lon=da_s.lon + da.lon.values.max())
            da_l = da.sel(lon=slice(llon1, da.lon.values.max()))
            if da_s.lon.values[0] == da_l.lon.values[-1]:
                da_l = da_l.isel(lon=slice(None, -1))
            da_c = xr.concat((da_l, da_s), dim='lon')
            if llat1 < lat_min:
                da_s = da_c.sel(lat=slice(da.lat.values.min(), -llat1 + 2 * da.lat.values.min()))
                da_s = da_s.assign_coords(lat=2 * da.lat.values.min() - da_s.lat)
                da_s = da_s.isel(lat=slice(None, None, -1))
                # before concat cut lat first
                da_t = da_c.sel(lat=slice(da_c.lat.values.min(), llat2)).isel(lat=slice(1, None))
                da_c = xr.concat((da_s, da_t), dim='lat')
                # already slice to block
                return da_c
            elif llat2 > lat_max:
                da_s = da_c.sel(lat=slice(2 * da.lat.values.max() - llat2, da.lat.values.max()))
                da_s = da_s.assign_coords(lat=2 * da.lat.values.max() - da_s.lat)
                da_s = da_s.isel(lat=slice(None, None, -1))
                da_b = da_c.sel(lat=slice(llat1, da.lat.values.max())).isel(lat=slice(None, -1))
                da_c = xr.concat((da_b, da_s), dim='lat')
                return da_c

            else:
                da_c = da_c.sel(lat=slice(llat1, llat2))
                return da_c

        elif llat1 < lat_min:
            da_c = da.sel(lon=slice(llon1, llon2))
            da_s = da.sel(lon=slice(llon1, llon2), lat=slice(da.lat.values.min(), -llat1 + 2 * da.lat.values.min()))
            da_s = da_s.assign_coords(lat=2 * da.lat.values.min() - da_s.lat)
            da_s = da_s.isel(lat=slice(None, None, -1))
            # before concat cut lat first
            da_t = da_c.sel(lat=slice(da_c.lat.values.min(), llat2)).isel(lat=slice(1, None))
            da_c = xr.concat((da_s, da_t), dim='lat')
            return da_c

        elif llat2 > lat_max:
            da_c = da.sel(lon=slice(llon1, llon2))
            da_s = da_c.sel(lat=slice(2 * da.lat.values.max() - llat2, da.lat.values.max()))
            da_s = da_s.assign_coords(lat=2 * da.lat.values.max() - da_s.lat)
            da_s = da_s.isel(lat=slice(None, None, -1))
            da_b = da_c.sel(lat=slice(llat1, da.lat.values.max())).isel(lat=slice(None, -1))
            da_c = xr.concat((da_b, da_s), dim='lat')
            return da_c

        else:
            da_c = da.sel(lat=slice(llat1, llat2), lon=slice(llon1, llon2))
            return da_c

    def is_obs_padding(self, da, llon1, llon2, llat1, llat2):
        """
        Apply spatial padding specifically for observation data.

        Inputs:
            - da: Observation DataArray to pad
            - llon1, llon2: Regional longitude boundaries
            - llat1, llat2: Regional latitude boundaries

        Outputs:
            - da_c: Observation DataArray with padding, possibly concatenated
            - flag: Boolean indicating if no valid observations remain

        Description:
            Specialized padding for observation data that handles coordinate
            wrapping and polar regions. May concatenate data from different
            geographic regions to cover analysis domain. Returns flag if no
            observations fall within padded domain.
        """
        flag = False
        if llon1 < 0:
            lon_mask = (da.lon < llon2) | (da.lon > llon1 + 360)
            if llat1 < -80:
                lat_mask = da.lat < llat2
                mask = lat_mask & lon_mask
                if mask.sum().item() == 0:
                    mask2 = (da.lat < (160 + llat1)) & lon_mask
                    if mask2.sum().item() == 0:
                        flag = True
                        return da, flag
                    else:
                        da = da.where(mask2, drop=True)
                        da['lon'] = da['lon'].where(da['lon'] < llon2, da['lon'] - 360)
                        da['lat'] = da['lat'] + (llat1 + 80)
                        return da, flag
                else:
                    da = da.where(mask, drop=True)
                    da['lon'] = da['lon'].where(da['lon'] < llon2, da['lon'] - 360)
                    mask2 = da.lat < (160 + llat1)
                    if mask2.sum().item() == 0:
                        return da, flag
                    else:
                        da_down = da.copy()
                        da_down = da_down.where(mask2, drop=True)
                        da_down['lat'] = da_down['lat'] + (llat1 + 80)
                        da_c = xr.concat([da_down, da], dim='numobs')
                        return da_c, flag
            elif llat2 > 90:
                lat_mask = da.lat > llat1
                mask = lat_mask & lon_mask
                if mask.sum().item() == 0:
                    mask2 = (da.lat > (180 - llat2)) & lon_mask
                    if mask2.sum().item() == 0:
                        flag = True
                        return da, flag
                    else:
                        da = da.where(mask2, drop=True)
                        da['lon'] = da['lon'].where(da['lon'] < llon2, da['lon'] - 360)
                        da['lat'] = da['lat'] + (180 - llat2)
                        return da, flag
                else:
                    da = da.where(mask, drop=True)
                    da['lon'] = da['lon'].where(da['lon'] < llon2, da['lon'] - 360)
                    mask2 = da.lat > (180 - llat2)
                    if mask2.sum().item() == 0:
                        return da, flag
                    else:
                        da_up = da.copy()
                        da_up = da_up.where(mask2, drop=True)
                        da_up['lat'] = da_up['lat'] + (180 - llat2)
                        da_c = xr.concat([da, da_up], dim='numobs')
                        return da_c, flag
            else:
                lat_mask = (da.lat > llat1) & (da.lat < llat2)
                mask = lat_mask & lon_mask
                if mask.sum().item() == 0:
                    flag = True
                    return da, flag
                da_c = da.where(mask, drop=True)
                da_c['lon'] = da_c['lon'].where(da_c['lon'] < llon2, da_c['lon'] - 360)
                return da_c, flag

        elif llon2 > 360:
            lon_mask = (da.lon > llon1) | (da.lon < llon2 - 360)
            if llat1 < -80:
                lat_mask = da.lat < llat2
                mask = lat_mask & lon_mask
                if mask.sum().item() == 0:
                    mask2 = (da.lat < (160 + llat1)) & lon_mask
                    if mask2.sum().item() == 0:
                        flag = True
                        return da, flag
                    else:
                        da = da.where(mask2, drop=True)
                        da['lon'] = da['lon'].where(da['lon'] > llon1, da['lon'] + 360)
                        da['lat'] = da['lat'] + (llat1 + 80)
                        return da, flag
                else:
                    da = da.where(mask, drop=True)
                    da['lon'] = da['lon'].where(da['lon'] > llon1, da['lon'] + 360)
                    mask2 = da.lat < (160 + llat1)
                    if mask2.sum().item() == 0:
                        return da, flag
                    else:
                        da_down = da.copy()
                        da_down = da_down.where(mask2, drop=True)
                        da_down['lat'] = da_down['lat'] + (llat1 + 80)
                        da_c = xr.concat([da_down, da], dim='numobs')
                        return da_c, flag
            elif llat2 > 90:
                lat_mask = da.lat > llat1
                mask = lat_mask & lon_mask
                if mask.sum().item() == 0:
                    mask2 = (da.lat > (180 - llat2)) & lon_mask
                    if mask2.sum().item() == 0:
                        flag = True
                        return da, flag
                    else:
                        da = da.where(mask2, drop=True)
                        da['lon'] = da['lon'].where(da['lon'] > llon1, da['lon'] + 360)
                        da['lat'] = da['lat'] + (180 - llat2)
                        return da, flag
                else:
                    da = da.where(mask, drop=True)
                    da['lon'] = da['lon'].where(da['lon'] > llon1, da['lon'] + 360)
                    mask2 = da.lat > (180 - llat2)
                    if mask2.sum().item() == 0:
                        return da, flag
                    else:
                        da_up = da.copy()
                        da_up = da_up.where(mask2, drop=True)
                        da_up['lat'] = da_up['lat'] + (180 - llat2)
                        da_c = xr.concat([da, da_up], dim='numobs')
                        return da_c, flag
            else:
                lat_mask = (da.lat > llat1) & (da.lat < llat2)
                mask = lat_mask & lon_mask
                if mask.sum().item() == 0:
                    flag = True
                    return da, flag
                da_c = da.where(mask, drop=True)
                da_c['lon'] = da_c['lon'].where(da_c['lon'] > llon1, da_c['lon'] + 360)
                return da_c, flag
        elif llat1 < -80:
            lon_mask = (da.lon > llon1) & (da.lon < llon2)
            lat_mask = da.lat < llat2
            mask = lat_mask & lon_mask
            if mask.sum().item() == 0:
                mask2 = (da.lat < (160 + llat1)) & lon_mask
                if mask2.sum().item() == 0:
                    flag = True
                    return da, flag
                else:
                    da = da.where(mask2, drop=True)
                    da['lat'] = da['lat'] + (llat1 + 80)
                    return da, flag
            else:
                da = da.where(mask, drop=True)
                mask2 = da.lat < (160 + llat1)
                if mask2.sum().item() == 0:
                    return da, flag
                else:
                    da_down = da.copy()
                    da_down = da_down.where(mask2, drop=True)
                    da_down['lat'] = da_down['lat'] + (llat1 + 80)
                    da_c = xr.concat([da_down, da], dim='numobs')
                    return da_c, flag
        elif llat2 > 90:
            lon_mask = (da.lon > llon1) & (da.lon < llon2)
            lat_mask = da.lat > llat1
            mask = lat_mask & lon_mask
            if mask.sum().item() == 0:
                mask2 = (da.lat > (180 - llat2)) & lon_mask
                if mask2.sum().item() == 0:
                    flag = True
                    return da, flag
                else:
                    da = da.where(mask2, drop=True)
                    da['lat'] = da['lat'] + (180 - llat2)
                    return da, flag
            else:
                da = da.where(mask, drop=True)
                mask2 = da.lat > (180 - llat2)
                if mask2.sum().item() == 0:
                    return da, flag
                else:
                    da_up = da.copy()
                    da_up = da_up.where(mask2, drop=True)
                    da_up['lat'] = da_up['lat'] + (180 - llat2)
                    da_c = xr.concat([da, da_up], dim='numobs')
                    return da_c, flag

        else:
            lon_mask = (da.lon > llon1) & (da.lon < llon2)
            lat_mask = (da.lat > llat1) & (da.lat < llat2)
            mask = lat_mask & lon_mask
            if mask.sum().item() == 0:
                flag = True
                return da, flag
            da_c = da.where(mask, drop=True)
            return da_c, flag

