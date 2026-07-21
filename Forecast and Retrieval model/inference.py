from datetime import datetime, timedelta
import argparse
import ctypes
import glob
import multiprocessing
import os
import sysconfig
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import xarray as xr
from metpy.calc import specific_humidity_from_dewpoint
from metpy.units import units
from aerobulk.flux import noskin_np


# help onnxruntime-gpu find the CUDA/cuDNN libraries shipped with pip torch

def _preload_cuda_libs():
    sp = sysconfig.get_paths()['purelib']
    for name in ('libcudnn.so.9', 'libcublasLt.so.12', 'libcublas.so.12',
                 'libcufft.so.11', 'libcurand.so.10', 'libnvrtc.so.12'):
        for pattern in (os.path.join(sp, 'torch', 'lib', name),
                        os.path.join(sp, 'nvidia', '*', 'lib', name)):
            hits = glob.glob(pattern)
            if hits:
                try:
                    ctypes.CDLL(hits[0], mode=ctypes.RTLD_GLOBAL)
                except OSError:
                    pass
                break


_preload_cuda_libs()
import onnxruntime as ort

# pass in arguments

parser = argparse.ArgumentParser()
parser.add_argument('--init_path', type=str, default='sample_fusion_zos.nc')
parser.add_argument('--forcing_path', type=str, default='sample_IFS_d0083.nc')
parser.add_argument('--output_path', type=str, default='./output')
parser.add_argument('--model_dir', type=str, default=os.path.dirname(os.path.abspath(__file__)))
parser.add_argument('--nproc', type=int, default=16)
args = parser.parse_args()

# normalization, mask and grid

ds_mask = xr.open_dataset(os.path.join(args.model_dir, 'mask.nc'))
mask40 = ds_mask['mask'].values.astype(bool)
longitude = ds_mask.longitude.values
latitude = ds_mask.latitude.values
depth = ds_mask.depth.values
ds_mask.close()
mask81 = np.concatenate([mask40, mask40, mask40[:1]], axis=0)
mask80 = mask81[:80]
sst_mask = mask40[0]

max_GLORYS = np.load(os.path.join(args.model_dir, 'max_GLORYS.npy'))
min_GLORYS = np.load(os.path.join(args.model_dir, 'min_GLORYS.npy'))
max_flux = np.load(os.path.join(args.model_dir, 'max_flux.npy')).reshape(-1, 1, 1)
min_flux = np.load(os.path.join(args.model_dir, 'min_flux.npy')).reshape(-1, 1, 1)
max_tsz = max_GLORYS[80:].reshape(-1, 1, 1)
min_tsz = min_GLORYS[80:].reshape(-1, 1, 1)
max_uv = max_GLORYS[:80].reshape(-1, 1, 1)
min_uv = min_GLORYS[:80].reshape(-1, 1, 1)

# bulk formulae functions


def _noskin_chunk(chunk_args):
    return noskin_np(*chunk_args, 'ncar', 2, 10, 4, False)


def calc_bulk_flux(sst_norm, i):
    # read in surface atmospheric and ocean variables
    ds = xr.open_dataset(args.forcing_path).isel(time=i)
    sst = ((max_GLORYS[80] - min_GLORYS[80]) * sst_norm.astype(np.float32)
           + min_GLORYS[80]) * sst_mask + 273.15
    sst = sst * sst_mask

    def getvar(name):
        return np.nan_to_num(ds[name].values) * sst_mask

    d2m = getvar('d2m')
    mtpr = getvar('mtpr')
    ssr = getvar('ssr')
    strd = getvar('strd')
    t2m = getvar('t2m')
    u10 = getvar('u10')
    v10 = getvar('v10')
    msl = getvar('MSL') if 'MSL' in ds else getvar('msl')
    ds.close()
    ssr /= 3600  # J m-2 to W m-2, hourly
    strd /= 3600
    mtpr /= 1000  # mm/s (kg/m^2/s) to m/s
    h2m = specific_humidity_from_dewpoint(msl * units.Pa, d2m * units.K).to('kg/kg').magnitude
    h2m = np.nan_to_num(h2m)

    # bulk formulae, land points are masked out by NaN in sst
    sst_nan = np.where(sst_mask, sst, np.nan)
    nlon = sst.shape[1]
    bounds = np.linspace(0, nlon, args.nproc + 1).astype(int)
    chunks = [(sst_nan[:, a:b], t2m[:, a:b], h2m[:, a:b],
               u10[:, a:b], v10[:, a:b], msl[:, a:b])
              for a, b in zip(bounds[:-1], bounds[1:])]
    pool = multiprocessing.Pool(processes=args.nproc)
    results = pool.map(_noskin_chunk, chunks)
    pool.close()
    pool.join()
    qe, qh, taux, tauy, evap = [np.concatenate(v, axis=1) for v in zip(*results)]
    evap /= 1000  # to m/s

    # net surface thermal (longwave) radiation
    sigma = 5.67e-8  # W/m^2 per K^4
    ql = strd - sigma * (sst ** 4)
    qs = ssr

    # normalization
    bulk_flux = np.stack((ql, qs, qh, qe, taux, tauy, evap, mtpr), axis=0)
    bulk_flux = np.nan_to_num(bulk_flux)
    bulk_flux = (bulk_flux - min_flux) / (max_flux - min_flux)
    bulk_flux *= sst_mask
    return bulk_flux.astype(np.float16).clip(0, 1)[None]


# set up onnxruntime

use_cuda = 'CUDAExecutionProvider' in ort.get_available_providers()
if use_cuda:
    # grow the arena on demand and shrink it after every run, so the two
    # sessions can alternate on one GPU without each holding its peak memory
    cuda_opts = {'arena_extend_strategy': 'kSameAsRequested',
                 'cudnn_conv_algo_search': 'HEURISTIC'}
    providers = [('CUDAExecutionProvider', cuda_opts), 'CPUExecutionProvider']
else:
    providers = ['CPUExecutionProvider']
sess_tsssh = ort.InferenceSession(os.path.join(args.model_dir, 'WenHai_tsssh.onnx'), providers=providers)
sess_uv = ort.InferenceSession(os.path.join(args.model_dir, 'WenHai_uv.onnx'), providers=providers)
print('onnxruntime providers:', sess_tsssh.get_providers())
name_in1 = sess_tsssh.get_inputs()[0].name
name_in2 = sess_tsssh.get_inputs()[1].name

run_opts = ort.RunOptions()
if use_cuda:
    run_opts.add_run_config_entry('memory.enable_memory_arena_shrinkage', 'gpu:0')

# read data

ds = xr.open_dataset(args.init_path)
init_date = datetime.fromtimestamp(
    (ds.time[0].values - np.datetime64('1970-01-01T00:00:00')) / np.timedelta64(1, 's'))
init = np.concatenate([ds.thetao.values[0], ds.so.values[0], ds.zos.values[0][None]], axis=0)
ds.close()

# initial condition

init = (init - min_tsz) / (max_tsz - min_tsz)
init = (np.nan_to_num(init) * mask81).astype(np.float16)

# set forecast length to number of days in provided forcing file, the last time
# step only drives the bulk fluxes of the final U/V diagnosis
nday = len(xr.open_dataset(args.forcing_path).time) - 1


def save_netcdf(output, uv, fcst_date, lead):
    # write forecasts in NetCDF format
    full = output.astype(np.float32) * (max_tsz - min_tsz) + min_tsz
    full[~mask81] = np.nan
    uv = (uv.astype(np.float32) * mask80) * (max_uv - min_uv) + min_uv
    uv[~mask80] = np.nan
    t, s, ssh = full[:40], full[40:80], full[-1]
    u, v = uv[:40], uv[40:80]

    time = np.array([f'{fcst_date[:4]}-{fcst_date[4:6]}-{fcst_date[6:]}'], dtype='datetime64[ns]')
    coords3d = dict(depth=(['depth'], depth),
                    latitude=(['latitude'], latitude),
                    longitude=(['longitude'], longitude))
    coords2d = dict(latitude=(['latitude'], latitude),
                    longitude=(['longitude'], longitude))
    dims3d = ['depth', 'latitude', 'longitude']
    dims2d = ['latitude', 'longitude']

    def da(data, dims, coords, standard_name, long_name, units):
        return xr.DataArray(data, dims=dims, coords=coords,
                            attrs=dict(standard_name=standard_name, long_name=long_name,
                                       units=units)).expand_dims(time=time)

    ds = xr.Dataset({
        'thetao': da(t, dims3d, coords3d, 'sea_water_potential_temperature', 'Temperature', 'degrees_C'),
        'so': da(s, dims3d, coords3d, 'sea_water_salinity', 'Salinity', '1e-3'),
        'uo': da(u, dims3d, coords3d, 'eastward_sea_water_velocity', 'Eastward velocity', 'm s-1'),
        'vo': da(v, dims3d, coords3d, 'northward_sea_water_velocity', 'Northward velocity', 'm s-1'),
        'zos': da(ssh, dims2d, coords2d, 'sea_surface_height_above_geoid', 'Sea surface height', 'm')})
    ds.depth.attrs = dict(standard_name='depth', long_name='Depth', units='m',
                          positive='down', axis='Z')
    ds.latitude.attrs = dict(standard_name='latitude', long_name='Latitude',
                             units='degrees_north', axis='Y')
    ds.longitude.attrs = dict(standard_name='longitude', long_name='Longitude',
                              units='degrees_east', axis='X')
    ds.attrs = dict(Conventions='CF-1.4', title='WenHai-E2E ocean forecast',
                    source='WenHai-E2E', lead=lead)
    save_path = os.path.join(args.output_path, f'fcst{fcst_date}')
    os.makedirs(save_path, exist_ok=True)
    ds.to_netcdf(os.path.join(save_path, f'fcst{fcst_date}_lead{lead}_byWenHai-E2E.nc'),
                 unlimited_dims=['time'])
    ds.close()


for i in range(-1, nday):
    fcst_date = (init_date + timedelta(days=1 + i)).strftime('%Y%m%d')

    # set autoregressive initial condition
    input_x = init if i == -1 else last_step

    if i == -1:
        # lead 0, no prognostic step, U/V diagnosed from the initial state
        bulk_flux = calc_bulk_flux(input_x[0], 0)
        output = input_x.copy()
    else:
        # inference
        delta = sess_tsssh.run(None, {name_in1: input_x[None], name_in2: bulk_flux}, run_opts)[0][0]
        output = ((input_x + delta) * mask81).clip(0, 1)
        bulk_flux = calc_bulk_flux(output[0], i + 1)

    uv = sess_uv.run(None, {name_in1: output[None], name_in2: bulk_flux}, run_opts)[0][0]
    last_step = output

    save_netcdf(output, uv, fcst_date, i + 1)
    print(f'lead {i + 1} ({fcst_date}) done')

print('forecast finished:', args.output_path)
