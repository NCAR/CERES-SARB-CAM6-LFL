import os
import sys
import argparse
import logging
import yaml
import numpy as np
import pandas as pd
import xarray as xr
from scipy.interpolate import RegularGridInterpolator
from pprint import pprint

from utils import fill_date_template

from microphysics import mix_hygroscopicity, wet_radius, water_fraction, mix_indices, ext_lookup, layer_optical_depth
from microphysics import m_to_um, um_to_m

np.set_printoptions(threshold=np.inf)
xr.set_options(display_max_rows=128)



def read_aerosol_optics(filename):

    logging.info(filename)
    with open(args.aerosol, 'r') as f:
        aerosol_config = yaml.safe_load(f)
        # pprint(aerosol_config)

    file_su = os.path.expandvars(
        aerosol_config['Types']['SU']['filename'])
    logging.info(file_su)
    ds_su = xr.open_dataset(file_su)

    file_oc = os.path.expandvars(
        aerosol_config['Types']['POM']['filename'])
    logging.info(file_oc)
    ds_oc = xr.open_dataset(file_oc)

    file_bc = os.path.expandvars(
        aerosol_config['Types']['BC']['filename'])
    logging.info(file_bc)
    ds_bc = xr.open_dataset(file_bc)

    file_du = os.path.expandvars(
        aerosol_config['Types']['DU']['filename'])
    logging.info(file_du)
    ds_du = xr.open_dataset(file_du)

    file_ss = os.path.expandvars(
        aerosol_config['Types']['SS']['filename'])
    logging.info(file_ss)
    ds_ss = xr.open_dataset(file_ss)

    file_wat = os.path.expandvars(
        aerosol_config['Types']['WAT']['filename'])
    logging.info(file_wat)
    ds_wat = xr.open_dataset(file_wat)

    aerosol_optics = {'SU': ds_su,
            'POM': ds_oc, 'SOA': ds_oc, 'BC': ds_bc,
            'DU': ds_du, 'SS': ds_ss,
            'WAT': ds_wat}

    return aerosol_optics


def process_file(aerosol_optics, wvl,
        species_info, mode_info, ds_mode, ds_fine_table,
        filename, aux_filename):

    logging.info(aux_filename)
    ds_aux = xr.open_dataset(aux_filename)
    logging.info(ds_aux)
    ntime, nlev, nlat, nlon = ds_aux['RH'].values.shape
    logging.info((ntime, nlev, nlat, nlon))

    logging.info(filename)
    ds = xr.open_dataset(filename)
    logging.info(ds)
    ds_dims = {'time': ntime, 'lev': nlev, 'lat': nlat, 'lon': nlon}

    """
    Extract numpy q arrays and flatten
    """
    q_values = dict()
    refractive_re = dict()
    refractive_im = dict()
    num_mix_ratio = ds['num_' + mode_info['label']].values.flatten()
    if ('SU' in mode_info['types']):
        refreal = aerosol_optics['SU']['refreal'][0,0,:]
        refimag = aerosol_optics['SU']['refimag'][0,0,:]
        table_wvls = aerosol_optics['SU']['lambda'].values
        wvl_idx = np.argmin(np.abs(table_wvls - um_to_m * wvl))
        refractive_re['SU'] = refreal[wvl_idx]
        refractive_im['SU'] = refimag[wvl_idx]
        q_values['SU'] = ds['so4_' + mode_info['label']].values.flatten()
    if ('POM' in mode_info['types']):
        refreal = aerosol_optics['POM']['refreal'][0,0,:]
        refimag = aerosol_optics['POM']['refimag'][0,0,:]
        table_wvls = aerosol_optics['POM']['lambda'].values
        wvl_idx = np.argmin(np.abs(table_wvls - um_to_m * wvl))
        refractive_re['POM'] = refreal[wvl_idx]
        refractive_im['POM'] = refimag[wvl_idx]
        q_values['POM'] = ds['pom_' + mode_info['label']].values.flatten()
    if ('SOA' in mode_info['types']):
        refreal = aerosol_optics['SOA']['refreal'][0,0,:]
        refimag = aerosol_optics['SOA']['refimag'][0,0,:]
        table_wvls = aerosol_optics['SOA']['lambda'].values
        wvl_idx = np.argmin(np.abs(table_wvls - um_to_m * wvl))
        refractive_re['SOA'] = refreal[wvl_idx]
        refractive_im['SOA'] = refimag[wvl_idx]
        q_values['SOA'] = np.zeros(ntime * nlev * nlat * nlon)
        for i in range(mode_info['soa_flavors']):
            q_values['SOA'] += ds['soa' + str(i+1) + '_' + mode_info['label']].values.flatten()
    if ('BC' in mode_info['types']):
        refreal = aerosol_optics['BC']['refreal'][0,0,:]
        refimag = aerosol_optics['BC']['refimag'][0,0,:]
        table_wvls = aerosol_optics['BC']['lambda'].values
        wvl_idx = np.argmin(np.abs(table_wvls - um_to_m * wvl))
        refractive_re['BC'] = refreal[wvl_idx]
        refractive_im['BC'] = refimag[wvl_idx]
        q_values['BC'] = ds['bc_' + mode_info['label']].values.flatten()
    if ('DU' in mode_info['types']):
        refreal = aerosol_optics['DU']['refreal'][0,0,:]
        refimag = aerosol_optics['DU']['refimag'][0,0,:]
        table_wvls = aerosol_optics['DU']['lambda'].values
        wvl_idx = np.argmin(np.abs(table_wvls - um_to_m * wvl))
        refractive_re['DU'] = refreal[wvl_idx]
        refractive_im['DU'] = refimag[wvl_idx]
        q_values['DU'] = ds['dst_' + mode_info['label']].values.flatten()
    if ('SS' in mode_info['types']):
        refreal = aerosol_optics['SS']['refreal'][0,0,:]
        refimag = aerosol_optics['SS']['refimag'][0,0,:]
        table_wvls = aerosol_optics['SS']['lambda'].values
        wvl_idx = np.argmin(np.abs(table_wvls - um_to_m * wvl))
        refractive_re['SS'] = refreal[wvl_idx]
        refractive_im['SS'] = refimag[wvl_idx]
        q_values['SS'] = ds['ncl_' + mode_info['label']].values.flatten()

    refreal = aerosol_optics['WAT']['watern'][:]
    refimag = aerosol_optics['WAT']['wateri'][:]
    table_wvls = aerosol_optics['WAT']['wavelength1'].values
    wvl_idx = np.argmin(np.abs(table_wvls - wvl))
    refractive_re['WAT'] = refreal[wvl_idx]
    refractive_im['WAT'] = refimag[wvl_idx]
    RH_values = ds_aux['RH'].values.flatten()
    # rwet_mode(time, nmde, lev, lat, lon)
    r_w_values = m_to_um \
        * ds['rwet_mode'].values[:,mode_info['index'],:,:,:].flatten()
    print(r_w_values.shape)
    print(RH_values.shape)

    """
    Compute layer pressure thickness
    """
    hyai_values = ds['hyai'].values
    hybi_values = ds['hybi'].values
    P0 = ds['P0'].values
    logging.info('P0 ' + str(P0))
    ps_2d = ds['PS'].values
    p_3d = np.zeros([ntime, nlev+1, nlat, nlon], dtype=np.float32)
    pdel_3d = np.zeros([ntime, nlev, nlat, nlon], dtype=np.float32)
    for l in range(ntime):
        for k in range(nlev+1):
            p_3d[l,k,:,:] = hyai_values[k] * P0 \
                          + hybi_values[k] * ps_2d[l,:,:]
    pdel_3d[:,0:nlev,:,:] = p_3d[:,1:nlev+1,:,:] - p_3d[:,0:nlev,:,:]
    pdel_values = pdel_3d.flatten()

    """
    Compute mixture hygroscopicity
    """
    B_mixed = mix_hygroscopicity(species_info, q_values, RH_values)

    """
    Compute wet radius and water fraction
    """
    r_d = 0.5 * float(ds_mode['dgnum'])
    # r_w = wet_radius(r_d, B_mixed, RH_values)

    # vmr_d, vmr_wat = water_fraction(species_info, q_values, r_d, r_w)
    vmr_d, vmr_wat = water_fraction(species_info, q_values, r_d, r_w_values)

    """
    Compute mixture refractive indices
    """
    n_re, n_im = mix_indices(ntime * nlev * nlat * nlon,
        species_info, refractive_re, refractive_im, q_values, vmr_wat)

    """
    Lookup extinction, absorption, asymmetry
    """
    ext_field, abs_field, asm_field = ext_lookup(ntime * nlev * nlat * nlon,
        n_re, n_im, r_w_values, ds_fine_table)

    """
    Compute layer optical depth
    """
    tau_field = layer_optical_depth(ntime * nlev * nlat * nlon,
        pdel_values, num_mix_ratio, ext_field)

    """
    Output
    """
    da_pdel = xr.DataArray(
             pdel_values.reshape(ntime, nlev, nlat, nlon), dims=ds_dims)
    da_pdel.attrs['units'] = 'Pa'
    da_RH = xr.DataArray(
            RH_values.reshape(ntime, nlev, nlat, nlon), dims=ds_dims)
    da_B = xr.DataArray(
            B_mixed.reshape(ntime, nlev, nlat, nlon), dims=ds_dims)
    da_r_w = m_to_um * xr.DataArray(
            r_w_values.reshape(ntime, nlev, nlat, nlon), dims=ds_dims)
    da_r_w.attrs['units'] = 'um'
    da_vmr_d = xr.DataArray(
            vmr_d.reshape(ntime, nlev, nlat, nlon), dims=ds_dims)
    da_vmr_d.attrs['units'] = 'cm^3 g-1'
    da_vmr_wat = xr.DataArray(
            vmr_wat.reshape(ntime, nlev, nlat, nlon), dims=ds_dims)
    da_vmr_wat.attrs['units'] = 'cm^3 g-1'
    da_n_re = xr.DataArray(
            n_re.reshape(ntime, nlev, nlat, nlon), dims=ds_dims)
    da_n_im = xr.DataArray(
            n_im.reshape(ntime, nlev, nlat, nlon), dims=ds_dims)
    da_ext = xr.DataArray(
            ext_field.reshape(ntime, nlev, nlat, nlon), dims=ds_dims)
    da_ext.attrs['units'] = 'um^2'
    da_abs = xr.DataArray(
            abs_field.reshape(ntime, nlev, nlat, nlon), dims=ds_dims)
    da_abs.attrs['units'] = 'um^2'
    da_asm = xr.DataArray(
            asm_field.reshape(ntime, nlev, nlat, nlon), dims=ds_dims)
    da_tau = xr.DataArray(
            tau_field.reshape(ntime, nlev, nlat, nlon), dims=ds_dims)
    da_tau_column = da_tau.sum(dim='lev')
    print('ext',
          da_ext.values.min(),
          da_ext.values.max(),
          da_ext.values.mean())
    print('tau_column',
          da_tau_column.values.min(),
          da_tau_column.values.max(),
          da_tau_column.values.mean())

    ds_out = xr.Dataset({
        'pdel': da_pdel,
        'RH': da_RH,
        'B': da_B,
        'r_w': da_r_w,
        'vmr_d': da_vmr_d,
        'vmr_wat': da_vmr_wat,
        'n_re': da_n_re,
        'n_im': da_n_im,
        'ext': da_ext,
        'abs': da_abs,
        'asm': da_asm,
        'tau': da_tau,
        'tau_column': da_tau_column},
        coords=ds.coords)
    filename_out = filename.replace('cam', 'cam_optics.'
        + mode_info['label'].replace('a', 'mode') + '_' + mode_info['band'])
    logging.info('writing ' + filename_out)
    ds_out.to_netcdf(filename_out)


if __name__ == '__main__':

    """
    Parse command line arguments
    """     
    parser = argparse.ArgumentParser()
    parser.add_argument('--logfile', type=str,
        default=sys.stdout,
        help='log file (default stdout)')
    parser.add_argument('--debug', action='store_true',
        help='set logging level to debug')
    parser.add_argument('--datadir', type=str,
        default=os.path.join(os.getenv('HOME'), 'Data'),
        help='top-level data directory (default $HOME/Data)')
    parser.add_argument('--aerosol', type=str,
        default=os.path.join('aerosol.yaml'),
        help='yaml aerosol file')
    parser.add_argument('--start', type=str,
        default='20100101',
        help='start date (yyyymmdd)')
    parser.add_argument('--end', type=str,
        default='20100103',
        help='end date (yyyymmdd)')
    parser.add_argument('--file_pattern', type=str,
        # default='FL_qaers_GEOSIT_MODIS.cam.h0.YYYY-MM-DD-00000.nc')
        default='jan-2010.cam.h0.YYYY-MM-DD-00000.nc')
    parser.add_argument('--aux_file_pattern', type=str,
        default='RH.cam.h0.YYYY-MM-DD-00000.nc')
    parser.add_argument('--scheme', type=str, default='MAM4')
    parser.add_argument('--mode', type=str, default='a1')
    parser.add_argument('--band', type=str, default='sw1')
    args = parser.parse_args()

    """
    Setup logging
    """
    logging_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(stream=args.logfile, level=logging_level)

    with open(args.aerosol, 'r') as f:
        aerosol_config = yaml.safe_load(f)

    logging.info('%s:mode:%s' % (args.scheme, args.mode))
    mode_info = aerosol_config[args.scheme][args.mode]
    mode_info['label'] = args.mode
    mode_info['index'] = int(args.mode[1]) - 1
    mode_info['band'] = args.band
    pprint(mode_info)
    species_info = aerosol_config['Types']
    pprint(species_info)
    ds_mode = xr.open_dataset(os.path.expandvars(mode_info['filename_rrtm']))
    logging.info(ds_mode)
    ds_table = xr.open_dataset(os.path.expandvars(mode_info['filename_sarb']))
    logging.info(ds_table)
    ds_fine_table = xr.open_dataset(os.path.expandvars(
        mode_info['filename_sarb'].replace('larc', args.band + '_larc')))
    logging.info(ds_fine_table)

    aerosol_optics = read_aerosol_optics(args.aerosol)

    # get band wavelength
    band_idx = int(args.band[2])
    if 'sw' in args.band:
        wvl = (ds_table['LFL_SW_bands'].values[band_idx]
            + ds_table['LFL_SW_bands'].values[band_idx - 1]) / 2
    if 'lw' in args.band:
        wvl = (ds_table['LFL_LW_bands'].values[band_idx]
            + ds_table['LFL_LW_bands'].values[band_idx - 1]) / 2
    logging.info('wavelength:%.2f' % wvl)

    dates = pd.date_range(start=args.start, end=args.end, freq='D')
    logging.info(dates)

    # for date in dates[0:1]:
    for date in dates[0:1]:
        date_str = date.strftime('%Y-%m-%b-%d-%j')
        logging.info(date_str)
        filepath = os.path.join(args.datadir, 'CAM6',
            fill_date_template(args.file_pattern, date_str))
        aux_filepath = os.path.join(args.datadir, 'CAM6',
            fill_date_template(args.aux_file_pattern, date_str))
        process_file(aerosol_optics, wvl,
            species_info, mode_info, ds_mode, ds_fine_table,
            filepath, aux_filepath)

