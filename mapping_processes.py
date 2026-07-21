import os
import time
import datetime
import argparse
import logging
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple, Optional, Generator
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import xarray as xr
from multiprocessing import Pool

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings('ignore')

# =============================================================================
# 1. Configuration Layer
# =============================================================================

@dataclass
class RegressionConfig:
    """Holds all configuration parameters to avoid hardcoding."""
    # Run arguments
    ilon: int
    depth_idx: int = 0
    
    # Time & Space Settings
    year_range: Tuple[int, int] = (1993, 2018)
    climatology_year: int = 2012
    resolution: float = 0.25
    
    # Kernel Bandwidths
    bw_time: int = 30
    bw_space_x: int = 2
    bw_space_y: int = 1
    
    # Paths
    base_data_path: Path = Path('/home/zhangxiang/code/20251206_global_enoi')
    data_source_path: Path = Path('/home/zhangxiang/data/20251206_global_enoi_data')
    output_base_path: Path = Path('/home/zhangxiang/result')
    
    # Variable Names
    var_name: str = 'thetaoa'
    deg: int = 4
    scale: str = 'largescale'

    @property
    def bw_time_window(self) -> int:
        return 2 * self.bw_time

    @property
    def output_dir(self) -> Path:
        """Dynamically generates the output directory path."""
        # Note: Ideally, depth should be passed dynamically, but sticking to logic structure
        return self.output_base_path / (
            f"{self.var_name[:-1]}_global_{self.deg}deg_{self.scale}_"
            f"bw{self.bw_time}_{self.bw_space_x}_{self.bw_space_y}_2truncation_nobeta0_"
            f"{self.year_range[0]}_{self.year_range[1]}"
        )

# =============================================================================
# 2. Utility Layer
# =============================================================================

class MathUtils:
    """Static class for pure mathematical operations."""

    @staticmethod
    def weighted_regression_no_intercept(X1: np.ndarray, X2: np.ndarray, Y: np.ndarray, weight: np.ndarray) -> np.ndarray:
        """
        Performs Weighted Least Squares (WLS) without intercept.
        Returns [beta1, beta2].
        """
        # Flatten and Validate
        X1, X2, Y, W = map(np.ravel, [X1, X2, Y, weight])
        mask = ~(np.isnan(X1) | np.isnan(X2) | np.isnan(Y) | np.isnan(W))
        
        if not mask.any():
            return np.full(2, np.nan)

        # Apply weights (WLS transformation: multiply by sqrt(w))
        sqrt_w = np.sqrt(W[mask])
        wx1 = X1[mask] * sqrt_w
        wx2 = X2[mask] * sqrt_w
        wy  = Y[mask]  * sqrt_w

        # Construct Normal Equations (X.T * X) * beta = (X.T * Y)
        # Using explicit summation is often faster for small fixed dimensions (2x2) than np.dot
        xtx = np.array([
            [np.sum(wx1**2),       np.sum(wx1*wx2)],
            [np.sum(wx1*wx2),      np.sum(wx2**2)]
        ])
        xty = np.array([np.sum(wx1*wy), np.sum(wx2*wy)])

        try:
            # Cholesky is faster and numerically stable for positive-definite matrices
            L = np.linalg.cholesky(xtx)
            beta = np.linalg.solve(L.T, np.linalg.solve(L, xty))
        except np.linalg.LinAlgError:
            try:
                beta = np.linalg.lstsq(xtx, xty, rcond=1e-10)[0]
            except:
                beta = np.full(2, np.nan)
        
        return beta

    @staticmethod
    def generate_gaussian_kernel(t_size: int, y_size: int, x_size: int) -> np.ndarray:
        """Generates a 3D Gaussian kernel (Time, Lat, Lon)."""
        wt = np.exp(-np.linspace(-2, 2, t_size)**2)
        wy = np.exp(-np.linspace(-2, 2, y_size)**2)
        wx = np.exp(-np.linspace(-2, 2, x_size)**2)
        
        # Outer product to create 3D volume
        spatial = np.outer(wy, wx)
        kernel = wt[:, None, None] * spatial[None, :, :]
        
        kernel[kernel < 0.01] = np.nan
        return kernel


class GeoUtils:
    """Static class for geographic array manipulations."""

    @staticmethod
    def pad_circular(da: xr.DataArray, dim: str, res: float, pad_deg: float) -> xr.DataArray:
        """Wraps longitude."""
        pad_num = int(pad_deg / res)
        # Calculate new coordinates manually to ensure monotonicity
        coords = da[dim].values
        step = coords[1] - coords[0]
        left = coords[0] - step * np.arange(pad_num, 0, -1)
        right = coords[-1] + step * np.arange(1, pad_num + 1)
        
        padded = da.pad({dim: (pad_num, pad_num)}, mode='wrap')
        return padded.assign_coords({dim: np.concatenate([left, coords, right])})

    @staticmethod
    def pad_reflect(da: xr.DataArray, dim: str, res: float, pad_deg: float) -> xr.DataArray:
        """Reflects latitude."""
        pad_num = int(pad_deg / res)
        coords = da[dim].values
        step = coords[1] - coords[0]
        left = coords[0] - step * np.arange(pad_num, 0, -1)
        right = coords[-1] + step * np.arange(1, pad_num + 1)
        
        padded = da.pad({dim: (pad_num, pad_num)}, mode='symmetric')
        return padded.assign_coords({dim: np.concatenate([left, coords, right])})

# =============================================================================
# 3. Data Layer
# =============================================================================

class DataLoader:
    """Handles file I/O, padding, and data stacking."""
    
    def __init__(self, config: RegressionConfig):
        self.cfg = config

    def get_reference_data(self) -> np.ndarray:
        """Loads depth information."""
        fpath = self.cfg.base_data_path / 'complement_data_19930101.nc'
        if not fpath.exists():
            raise FileNotFoundError(f"Reference file not found: {fpath}")
        with xr.open_dataset(fpath) as ds:
            return ds.depth.values

    def load_and_preprocess_year(self, year: int, bounds: dict, depth_val: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Worker function to load a single year's data.
        NOTE: This must be pickle-able (top-level or static) if using 'spawn', 
        but 'fork' (default on Linux) works with instance methods usually.
        """
        # Define paths
        p_ssta = self.cfg.data_source_path / 'ssta' / f'ssta_{self.cfg.deg}deg_{self.cfg.scale}_{year}.nc'
        p_ssha = self.cfg.data_source_path / 'ssha' / f'ssha_{self.cfg.deg}deg_{self.cfg.scale}_{year}.nc'
        p_sta  = self.cfg.data_source_path / f'depth{depth_val:.1f}' / f'{self.cfg.var_name}_{self.cfg.deg}deg_{self.cfg.scale}_{year}.nc'

        def _process(path: Path, var_key: str, is_lon_dim: bool = False) -> np.ndarray:
            with xr.open_dataset(path) as ds:
                da = ds[var_key]
                # Normalize coord names if necessary, or assume helper handles it
                # Here we strictly follow the logic: Pad Lon -> Pad Lat -> Slice
                
                # Identify dim names
                d_lon = 'lon' if 'lon' in da.coords else 'longitude'
                d_lat = 'lat' if 'lat' in da.coords else 'latitude'

                da = GeoUtils.pad_circular(da, d_lon, self.cfg.resolution, self.cfg.bw_space_x * 2 + 1)
                da = GeoUtils.pad_reflect(da, d_lat, self.cfg.resolution, self.cfg.bw_space_y * 2 + 1)
                
                # Selection slice
                return da.sel({
                    d_lat: slice(bounds['lat_min'], bounds['lat_max']),
                    d_lon: slice(bounds['lon_min'], bounds['lon_max'])
                }).values

        return (
            _process(p_ssta, 'ssta'),
            _process(p_ssha, 'ssha'),
            _process(p_sta, self.cfg.var_name)
        )

# =============================================================================
# 4. Processing Layer
# =============================================================================

class RegressionProcessor:
    """Main controller for the regression analysis."""
    
    def __init__(self, config: RegressionConfig):
        self.cfg = config
        self.loader = DataLoader(config)
        self.depths = self.loader.get_reference_data()

    def run(self):
        """Entry point for the processing logic."""
        depth_val = self.depths[self.cfg.depth_idx]
        
        # Calculate Logic Blocks
        # Assuming 0-360 range divided into 5 blocks
        total_lon_range = np.arange(0, 360, 1)
        block_size = 360 // 5
        
        # Determine current block range
        # ilon passed from args is the relative index in the sub-block? 
        # Based on original code: ilon = args.ilon + ii_lon * ...
        # Simplified here: We assume the external loop calls this class per block,
        # or we iterate blocks here. Let's iterate blocks here to match original logic.
        
        for block_idx in range(5):
            real_lon_idx = self.cfg.ilon + block_idx * block_size
            
            # Define specific area for this block
            lon_start = total_lon_range[real_lon_idx]
            lon_end = total_lon_range[real_lon_idx] + 1
            
            logger.info(f"Processing Depth: {depth_val}m | Lon Block: {real_lon_idx} ({lon_start}-{lon_end})")
            
            # Setup output directory
            current_out_dir = self.cfg.output_dir / f"depth{depth_val:.1f}" / str(real_lon_idx)
            current_out_dir.mkdir(parents=True, exist_ok=True)
            
            # Check completion
            if len(list(current_out_dir.glob('*.npy'))) >= 366:
                logger.info("Block already processed. Skipping.")
                continue

            self._process_single_block(lon_start, lon_end, depth_val, current_out_dir)

    def _process_single_block(self, lon_s: int, lon_e: int, depth_val: float, out_dir: Path):
        t0 = time.time()
        
        # 1. Define Padding Bounds
        pad_lat = self.cfg.bw_space_y * 2
        pad_lon = self.cfg.bw_space_x * 2
        
        bounds_pad = {
            'lat_min': -80 - pad_lat, 'lat_max': 90 + pad_lat,
            'lon_min': lon_s - pad_lon, 'lon_max': lon_e + pad_lon
        }
        
        # 2. Get Coordinate Grids (using a sample file)
        # This is needed to map pixels back to lat/lon
        sample_file = self.cfg.base_data_path / 'raw_data_19930101.nc'
        with xr.open_dataset(sample_file) as ds:
            # Recreate the padding logic just to get coords
            # (In a real app, I'd cache this or separate it better)
            da = GeoUtils.pad_circular(ds.sst, 'longitude', self.cfg.resolution, pad_lon + 1)
            da = GeoUtils.pad_reflect(da, 'latitude', self.cfg.resolution, pad_lat + 1)
            
            lat_full = da.latitude.sel(latitude=slice(bounds_pad['lat_min'], bounds_pad['lat_max'])).values
            lon_full = da.longitude.sel(longitude=slice(bounds_pad['lon_min'], bounds_pad['lon_max'])).values
            
            # Create computation mask (2x2 coarse)
            ds_mask = da.coarsen(dim={'longitude': 2, 'latitude': 2}, boundary='trim').mean()
            ds_mask = ds_mask.sel(latitude=slice(-80, 90), longitude=slice(lon_s, lon_e))
            
            mask_valid = ~np.isnan(ds_mask.values)
            lat_cal = ds_mask.latitude.values
            lon_cal = ds_mask.longitude.values

        # 3. Load All Years (Parallel IO)
        years = list(range(self.cfg.year_range[0], self.cfg.year_range[1] + 1))
        logger.info(f"Loading data for {len(years)} years...")
        
        # Use partial to pass constant args to map
        load_args = [(y, bounds_pad, depth_val) for y in years]
        with Pool(processes=4) as pool:
            # Wrapper to unpack arguments for starmap
            data_raw = pool.starmap(self.loader.load_and_preprocess_year, load_args)

        # Stack arrays: Result shape (Years*Days, Lat, Lon)
        # Note: Optimization - Pre-allocate arrays if memory is tight, but concat is fine here
        ssta = np.concatenate([x[0] for x in data_raw], axis=0)
        ssha = np.concatenate([x[1] for x in data_raw], axis=0)
        sta  = np.concatenate([x[2] for x in data_raw], axis=0)

        # Append Safety NaN Frame (for boundary dates)
        pad_frame = np.full((1, ssta.shape[1], ssta.shape[2]), np.nan)
        ssta = np.concatenate([ssta, pad_frame], axis=0)
        ssha = np.concatenate([ssha, pad_frame], axis=0)
        sta  = np.concatenate([sta, pad_frame], axis=0)

        # 4. Prepare Kernels and Indices
        # Kernel
        kernel_3d = MathUtils.generate_gaussian_kernel(
            2 * self.cfg.bw_time_window + 1,
            int(1/self.cfg.resolution * self.cfg.bw_space_y * 2)*2 + 1,
            int(1/self.cfg.resolution * self.cfg.bw_space_x * 2)*2 + 1
        )
        kernel_stack = np.tile(kernel_3d, (len(years), 1, 1))

        # Calculation Indices
        # ii, jj are indices in the OUTPUT (Calculated) grid
        ii, jj = np.where(mask_valid)
        calc_coords = list(zip(ii, jj)) # List of (row, col)

        # Date Mapping
        base_date = datetime.date(self.cfg.year_range[0], 1, 1)
        total_days_data = (ssta.shape[0] - 1) # Excluding pad frame

        logger.info("Starting regression loop...")
        
        # 5. Regression Loop (Daily)
        for day_idx in range(366):
            target_f = out_dir / f'beta_d{day_idx}.npy'
            if target_f.exists(): continue
            if day_idx == 59: continue # Skip Leap Day logic if desired (matches original)

            # --- A. Time Slicing ---
            # Find indices for this climatological day across all years
            clim_date = datetime.date(self.cfg.climatology_year, 1, 1) + datetime.timedelta(days=day_idx)
            time_indices = []
            
            for y in years:
                # Construct date for specific year
                # Handle Feb 29 for non-leap years? (Original code strictness kept simple here)
                try:
                    center_dt = datetime.date(y, clim_date.month, clim_date.day)
                except ValueError:
                    continue # Skip Feb 29 in non-leap years

                center_idx = (center_dt - base_date).days
                # Add window +/- bandwidth
                win_indices = np.arange(center_idx - self.cfg.bw_time_window, 
                                        center_idx + self.cfg.bw_time_window + 1)
                time_indices.append(win_indices)
            
            if not time_indices:
                continue

            time_indices = np.concatenate(time_indices)
            # Boundary checks -> map to last NaN frame
            time_indices[time_indices < 0] = -1
            time_indices[time_indices >= total_days_data] = -1

            # Extract Data Cube for this day
            ssta_t = ssta[time_indices]
            ssha_t = ssha[time_indices]
            sta_t  = sta[time_indices]

            # --- B. Spatial Regression (Parallel) ---
            beta_map = np.full((len(lat_cal), len(lon_cal), 2), np.nan)

            # Define the worker closure here to capture local variables
            # Note: ThreadPool shares memory, so this is efficient (no copying big arrays)
            def solve_pixel(coord):
                r, c = coord
                # Map output grid (coarse) back to input grid (fine + padded)
                # Find index of lat_cal[r] in lat_full
                # Uses argmin for nearest neighbor (robust floating point match)
                r_in = np.argmin(np.abs(lat_full - lat_cal[r]))
                c_in = np.argmin(np.abs(lon_full - lon_cal[c]))
                
                # Slicing window
                k_t, k_y, k_x = kernel_3d.shape
                sl_y = slice(r_in - k_y//2, r_in + k_y//2 + 1)
                sl_x = slice(c_in - k_x//2, c_in + k_x//2 + 1)
                
                return MathUtils.weighted_regression_no_intercept(
                    ssta_t[:, sl_y, sl_x],
                    ssha_t[:, sl_y, sl_x],
                    sta_t[:, sl_y, sl_x],
                    kernel_stack # Weights
                )

            with ThreadPoolExecutor(max_workers=24) as executor:
                results = executor.map(solve_pixel, calc_coords)
            
            # Fill result map
            for (r, c), res in zip(calc_coords, results):
                beta_map[r, c] = res

            np.save(target_f, beta_map)

        logger.info(f"Block finished in {time.time() - t0:.2f}s")


# =============================================================================
# 5. Main Entry Point
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Standardized Global Regression")
    parser.add_argument("--ilon", type=int, required=True, help="Longitude block index")
    args = parser.parse_args()

    # Create Configuration
    config = RegressionConfig(ilon=args.ilon)
    
    # Initialize and Run Processor
    try:
        processor = RegressionProcessor(config)
        processor.run()
    except Exception as e:
        logger.exception("Fatal error occurred during execution")
        raise