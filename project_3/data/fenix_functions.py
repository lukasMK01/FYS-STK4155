'''
Created by: Lukas Monrad-Krohn on 2026-03-13

This script contains all functions used for FENIX post-processing and later plotting
'''


#######################################################
# Imports
######################################################
import pandas as pd
import numpy as np
# import glob
import xarray as xr
# import datetime
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
# import sys
from spectral import *
import spectral.io.envi as envi
from osgeo import gdal
import gc

import re
from pathlib import Path
from datetime import datetime
import subprocess



#####################################################
# Functions
#####################################################

def rgb_to_gray(arr):
    '''
     last updated: --- by Tim Carlsen
     
     from Jäkel et al. (2017) Cloud-side ACRIDICON
     RGB = 0.2126 · R + 0.7152 · G + 0.0722 · B
     
     Calculates grayscale image from RGB channels
     '''

    return(0.2126 * arr[0,:,:] + 0.7152 * arr[1,:,:] + 0.0722 * arr[2,:,:])
    # used in CELLO for MSI RGB-true image
    # return(0.2989 * arr[0,:,:] + 0.5870 * arr[1,:,:] + 0.1140 * arr[2,:,:])


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


def list_files_with_ending(root, endings, follow_symlinks=False):
    """
    last updated: 2026-03-13 by Lukas Monrad-Krohn

    Recursively list files under `root` whose names end with one of `endings`.

    Parameters
    ----------
    root : str
        Directory to search.
    endings : str or iterable of str
        Single ending like '.csv' or an iterable like ['.csv', '.nc', '.txt'].
        Matching is case-insensitive; each ending should include the leading dot.
    follow_symlinks : bool
        If True, follow symbolic links (passed to os.walk).

    Returns
    -------
    list
        Full paths to matching files (using OS native separator).
    """
    # normalize endings to a lowercase tuple for fast checking
    if isinstance(endings, str):
        ends = (endings.lower(),)
    else:
        ends = tuple(e.lower() for e in endings)

    matches = []
    for dirpath, dirnames, filenames in os.walk(root, followlinks=follow_symlinks):
        for fname in filenames:
            fname_lower = fname.lower()
            for e in ends:
                if fname_lower.endswith(e):
                    matches.append(os.path.join(dirpath, fname))
                    break  # no need to check other endings for this file
    return matches


def dark_current_correction(arr, n_bands=364, n_dark_samples=-150):
    """
    last updated: 2026-03-13 by Lukas Monrad-Krohn

    Perform dark current correction by subtracting the mean of the last `n_dark_samples` from each pixel and wavelength.

    Parameters
    ----------
    arr : numpy.ndarray
        Input array of shape (n_samples, n_pixels, n_bands).
    n_dark_samples : int
        Number of samples at the end of the array to use for estimating dark current.

    Returns
    -------
    numpy.ndarray
        Dark current corrected array of the same shape as input.
    """
    print(f'Checking dims: along={arr.shape[0]}, across={arr.shape[1]}, bands={arr.shape[2]} (expected bands={n_bands})')

    new_arr = arr.copy()
    dark = np.zeros((arr.shape[1], arr.shape[2]))  # shape: (n_pixels, n_bands)
    dark_std = np.zeros((arr.shape[1], arr.shape[2]))  # shape: (n_pixels, n_bands)

    for i in range(arr.shape[1]):
        for j in range(arr.shape[2]):
            # print(f'i={i}, j={j}')
            dark[i,j] = np.nanmean(arr[n_dark_samples:, i, j])
            dark_std[i,j] = np.nanstd(arr[n_dark_samples:, i, j])
            new_arr[:, i, j] = arr[:, i, j] - dark[i,j]

    return new_arr, dark, dark_std


def post_process_fenix(fenix_path, real_start_time=None, real_end_time=None):
    ''' function to post-process Fenix data, including dark correction, 
        radiometric calibration.

        last updated: 2026-05-25 by Lukas Monrad-Krohn
    Input:
        - file: path to the raw Fenix data file
    Output:
        - processed data array
    '''
    #####################################################
    # Read data from .hdr file with spectral
    #####################################################
    # open image
    
    img = open_image(fenix_path)
    bands = img.bands.centers
    split_band_idx = np.where(np.array(bands) >= 976.01)[0][0]+1

    # get metadata
    date = img.metadata['acquisition date'][-10:]
    pc_start_time = img.metadata['start time'][-8:]
    pc_end_time = img.metadata['stop time'][-8:]
    fps = float(img.metadata['fps'])
    vnir_exposure = float(img.metadata['tint1'])
    swir_exposure = float(img.metadata['tint2'])
    print(f"Start time from metadata: {pc_start_time}, end time from metadata: {pc_end_time}, fps: {fps}")

    pc_start_dtime = pd.to_datetime(date + pc_start_time, format='%Y-%m-%d%H:%M:%S')
    pc_end_dtime = pd.to_datetime(date + pc_end_time, format='%Y-%m-%d%H:%M:%S')
    pc_duration = (pc_end_dtime - pc_start_dtime).total_seconds()


    # get data values
    try:
        arr = np.float32(img.asarray()) #(time, pixels, wavelengths)
        # arr = np.float32(arr) # convert to float for processing
    except Exception as e:
        print(f"Error reading Fenix data with spectral: {e}")
        print("Trying to read with gdal instead...")

        gdal.GetDriverByName('EHdr').Register()

        # Open BIL raw file and save as array
        img = gdal.Open(fenix_path[:-3]+'raw')
        arr = np.float32(img.ReadAsArray())
        arr = np.transpose(arr, (1, 2, 0))

    
    dim_wvl = np.shape(arr)[2]
    dim_across = np.shape(arr)[1]
    dim_time = np.shape(arr)[0]
    print(f"Dimensions: {dim_wvl} wavelengths, {dim_across} across pixels, {dim_time} time pixels")



    # compare times
    if real_start_time is not None and real_end_time is not None:
        real_start_dtime = pd.to_datetime(real_start_time, format='%Y-%m-%d %H:%M:%S')
        real_end_dtime = pd.to_datetime(real_end_time, format='%Y-%m-%d %H:%M:%S')
        real_duration = (real_end_dtime - real_start_dtime).total_seconds()
        print(f"start diff: {(real_start_dtime - pc_start_dtime).total_seconds()} seconds, \n \
              end diff: {(real_end_dtime - pc_end_dtime).total_seconds()} seconds, \n \
              duration diff: {real_duration - pc_duration} seconds")
        start_time = real_start_dtime
        end_time = real_end_dtime
        duration = real_duration
    else:
        start_time = pc_start_dtime
        end_time = pc_end_dtime
        duration = pc_duration


    #####################################################
    # Quicklook of RGB image without corrections for checking
    #####################################################
    # rgb_img = np.array([arr[:, :, find_nearest(bands, 700)], arr[:, :, find_nearest(bands, 546.1)], arr[:, :, find_nearest(bands, 435.8)]])

    # fig, ax = plt.subplots(1,1,figsize = (7, 3))
    # ax.set_title('Grayscale image from RGB channels (no corrections)', fontsize = 10)
    # ax.set_xlabel('Time pixels')
    # ax.set_ylabel('Across pixels')
    # pc = ax.pcolormesh(np.arange(dim_time), np.arange(dim_across), rgb_to_gray(rgb_img).T, cmap = plt.get_cmap('gray'))
    # cbar = plt.colorbar(pc, ax=ax)
    # cbar.set_label('Counts')
    # plt.show()

    ############################################################
    # Dark correction
    ############################################################

    arr, dark, dark_std = dark_current_correction(arr, n_bands=dim_wvl, n_dark_samples=-150)



    ############################################################
    # apply radiometric calibration to convert DN to radiances (yW cm-2 nm-1 sr-1)
    ############################################################
    print(f"spectral binning: {img.metadata['binning']}, spatial binning: {img.metadata['binning2']}")
    calib_path = f'input/Radiometric_{img.metadata["binning"][0]}x{img.metadata["binning"][1]}_{img.metadata["binning2"][0]}x{img.metadata["binning2"][1]}.cal'

    # with gdal
    calib = gdal.Open(calib_path)
    cal = calib.ReadAsArray()
    # meta_cal = calib.GetMetadata()

    cal = np.transpose(cal, (1, 2, 0)) # transpose to match dimensions of data (wavelengths, across-track pixels)

    print('Calibration file: ', np.shape(cal))
    print('Raw data file: ', np.shape(arr))

    arr_rad = arr
    # arr_rad = arr * cal[0][np.newaxis,:,:]
    # calculate for vnir and swir to account for different exposure times
    arr_rad[:,:,:split_band_idx] = arr_rad[:,:,:split_band_idx] * cal[0][np.newaxis,:,:split_band_idx] / vnir_exposure
    arr_rad[:,:,split_band_idx:] = arr_rad[:,:,split_band_idx:] * cal[0][np.newaxis,:,split_band_idx:] / swir_exposure
    del arr, cal
    gc.collect()
    
            
    print('Calibrated rad file: ', np.shape(arr_rad))

    ### convert from yW cm-2 nm-1 sr-1 to mW m-2 nm-1 sr-1
    arr_rad = arr_rad * 10

    output_dict = {'arr': arr_rad, 'bands': bands, 'start_time': start_time, 'end_time': end_time,
                    'fps': fps, 'dim_wvl': dim_wvl, 'dim_across': dim_across, 'dim_time': dim_time}
    return output_dict





def read_three_line_records(fname):
    # read file and keep only non-empty lines (striped)
    with open(fname, 'r') as f:
        lines = [ln.rstrip() for ln in f if ln.strip() != '']

    if len(lines) % 3 != 0:
        raise ValueError(f"Expected number of non-empty lines to be a multiple of 3, got {len(lines)}")

    records = []
    max_first_extras = 0
    max_second_extras = 0
    max_third_extras = 0

    for i in range(0, len(lines), 3):
        l1 = lines[i].strip()
        l2 = lines[i+1].strip()
        l3 = lines[i+2].strip()

        # split on whitespace and parse floats (robust to multiple spaces)
        nums1 = [float(x) for x in l1.split()]
        nums2 = [float(x) for x in l2.split()]
        nums3 = [float(x) for x in l3.split()]

        if len(nums1) < 1:
            raise ValueError(f"Line {i+1} has no numbers")
        wl = nums1[0]
        extras1 = nums1[1:]

        # take azimuth as first number in line 2, but keep any extra numbers
        az = nums2[0] if len(nums2) >= 1 else np.nan
        extras2 = nums2[1:]

        # parse third line: cos_zenith, optional val, radiance (or variable)
        cos_zenith = nums3[0] if len(nums3) >= 1 else np.nan
        # capture the rest (could be [val, radiance] or other)
        extras3 = nums3[1:]

        max_first_extras = max(max_first_extras, len(extras1))
        max_second_extras = max(max_second_extras, len(extras2))
        max_third_extras = max(max_third_extras, len(extras3))

        records.append((wl, extras1, az, extras2, cos_zenith, extras3))

    # Build DataFrame columns with appropriate padding
    rows = []
    for wl, extras1, az, extras2, cos_zenith, extras3 in records:
        row = {'wavelength': wl, 'azimuth': az, 'cos_zenith': cos_zenith}
        # extras from line 1: extra1_0, extra1_1, ...
        for j in range(max_first_extras):
            row[f'extra1_{j}'] = extras1[j] if j < len(extras1) else np.nan
        # extras from line 2
        for j in range(max_second_extras):
            row[f'extra2_{j}'] = extras2[j] if j < len(extras2) else np.nan
        # extras from line 3
        for j in range(max_third_extras):
            row[f'extra3_{j}'] = extras3[j] if j < len(extras3) else np.nan

        rows.append(row)

    df = pd.DataFrame(rows)
    # Optional: try to name common extras if present (e.g. radiance is often the last extra3)
    if max_third_extras >= 1:
        df = df.rename(columns={f'extra3_{max_third_extras-1}': 'radiance'})
    return df





# regex to find YYYYMMDD_HHMMSS in filename
_filename_time_re = re.compile(r'.*?(\d{8}_\d{6}).*')

def _parse_time_from_filename(fname):
    """Parse YYYYMMDD_HHMMSS from filename and return a pandas.Timestamp."""
    m = _filename_time_re.match(Path(fname).name)
    if not m:
        raise ValueError(f"Filename {fname} does not contain YYYYMMDD_HHMMSS pattern")
    ts_str = m.group(1)
    return pd.to_datetime(ts_str, format='%Y%m%d_%H%M%S')

def read_libradtran_radiances(fname, n_az, n_zen, *,
                              expect_blocks=None,
                              strict_grid_check=True,
                              parse_time_from_filename=True):
    """
    Read libradtran output with repeated blocks:
      <wavelength> <extra1> <extra2> ...
      <az1> <az2> ... <azN>         (n_az values)
      <cosz1> <rad_az1_1> <rad_az1_2> ... <rad_az1_n_az>
      ...
      (n_zen lines of cosz + n_az radiances)
    Repeats for each wavelength.

    Added behaviour:
      - If parse_time_from_filename is True, extract a time from the filename
        pattern YYYYMMDD_HHMMSS and add a leading time dimension of length 1.
        The returned radiance DataArray will have dims
        ('time','wavelength','azimuth','cos_zenith').
      - Extras on the wavelength line remain 1D with the wavelength coordinate.

    Parameters
    ----------
    fname : str or Path
        Path to the text file.
    n_az : int
        Number of azimuth columns per wavelength block.
    n_zen : int
        Number of zenith (cos_zenith) rows per wavelength block.
    expect_blocks : int or None
        Optional expected number of wavelength blocks; if provided, a check is performed.
    strict_grid_check : bool
        If True (default) asserts that azimuth and cos_zenith arrays are identical
        across all blocks. If False, uses the first found grid.
    parse_time_from_filename : bool
        If True (default), parse time from filename and add time dimension.

    Returns
    -------
    xr.Dataset
        Dataset with:
          - radiance: DataArray (time, wavelength, azimuth, cos_zenith)
          - wavelength (coordinate)
          - azimuth (coordinate)
          - cos_zenith (coordinate)
          - extra1_0, extra1_1, ... 1D variables (dimension wavelength) for other numbers on the wavelength line.
          - time coordinate of length 1 (if parse_time_from_filename True)
    """
    fname = Path(fname)

    # read all non-empty lines trimmed of trailing newline
    with open(fname, 'r') as f:
        lines = [ln.rstrip() for ln in f if ln.strip() != '']

    # Parse through lines block by block
    i = 0
    wl_list = []
    az_grid = None
    cosz_grid = None
    radiance_blocks = []   # list of arrays shape (n_zen, n_az)
    extra1_list = []       # list of lists for extras on wavelength line

    block_len = 2 + n_zen
    # quick check for plausibility:
    if len(lines) % block_len != 0:
        raise ValueError(f"Total non-empty lines ({len(lines)}) is not a multiple of block length "
                         f"{block_len}. Check n_az and n_zen.")

    while i < len(lines):
        # Line 1: wavelength + extras
        l1 = lines[i].strip()
        tokens1 = l1.split()
        if len(tokens1) < 1:
            raise ValueError(f"Missing wavelength at block starting line {i+1}")
        wl = float(tokens1[0])
        extras1 = [float(tok) for tok in tokens1[1:]]  # may be empty
        wl_list.append(wl)
        extra1_list.append(extras1)

        # Line 2: azimuths (n_az values)
        i += 1
        l2 = lines[i].strip()
        tokens2 = l2.split()
        azs = [float(tok) for tok in tokens2]
        if len(azs) != n_az:
            raise ValueError(f"Expected {n_az} azimuth values on line {i+1}, got {len(azs)}")
        
        azs = [float(tok) for tok in tokens2]
        # azs_count = number of azimuths explicitly printed (e.g. 6)
        if len(azs) != n_az:
            raise ValueError(f"Expected {n_az} azimuth values on line {i+1}, got {len(azs)}")

        # We'll allow for an implicit azimuth radiance column on each data line.
        # Variable to mark whether an implicit azimuth (e.g. az=0.0) is present.
        implicit_az_present = False
        implicit_az_value = 0.0  # change if needed

        # Prepare temporary storage for block radiances
        # But we don't yet know whether implicit az is present; allocate for the maximal case:
        # if implicit present radiance shape will be (n_zen, n_az + 1), else (n_zen, n_az)
        # We'll inspect first data line to decide; to make logic simple we'll read first data line tokens here
        peek_idx = i + 1
        peek_tokens = lines[peek_idx].strip().split()
        # peek_tokens should have at least 1 + n_az (cos + n_az) normally
        if len(peek_tokens) == 1 + n_az + 1:
            implicit_az_present = True
            total_rad_cols = n_az + 1
        else:
            implicit_az_present = False
            total_rad_cols = n_az

        # if implicit az is present, insert it into az_grid at the desired position.
        # The file example suggests the implicit column sits before the listed azimuths,
        # so we'll insert it at the start (position 0). Adjust if your implicit column belongs elsewhere.
        if az_grid is None:
            if implicit_az_present:
                az_grid = np.concatenate(([implicit_az_value], azs))  # shape (n_az+1,)
            else:
                az_grid = np.array(azs)
        else:
            # If az_grid already exists from previous blocks, ensure consistency
            compare_az = np.array(azs)
            if implicit_az_present:
                compare_az = np.concatenate(([implicit_az_value], compare_az))
            if strict_grid_check and not np.allclose(az_grid, compare_az, atol=1e-8, rtol=0):
                raise ValueError("Azimuth grid mismatch at block starting with wavelength {wl}.")

        # Now read the n_zen data lines, extracting radiances into block_rads
        block_rads = np.empty((n_zen, total_rad_cols), dtype=float)
        block_cosz = []
        for j in range(n_zen):
            i += 1
            toks = lines[i].strip().split()
            if len(toks) < 1 + total_rad_cols:
                raise ValueError(f"Expected cos_zenith + {total_rad_cols} radiance values on line {i+1}, got {len(toks)} tokens")
            cosz = float(toks[0])
            # If implicit column present, toks[1] is radiance for implicit az, toks[2:2+n_az] are for listed azs
            if implicit_az_present:
                rad_vals = [float(toks[1])] + [float(t) for t in toks[2:2 + n_az]]
            else:
                rad_vals = [float(t) for t in toks[1:1 + n_az]]
            block_cosz.append(cosz)
            block_rads[j, :] = rad_vals


        if cosz_grid is None:
            cosz_grid = np.array(block_cosz)
        else:
            if strict_grid_check and not np.allclose(cosz_grid, block_cosz, atol=1e-8, rtol=0):
                raise ValueError(f"cos_zenith grid mismatch at wavelength {wl}.")
        radiance_blocks.append(block_rads)

        # finished this block
        i += 1

    # assemble arrays
    n_wl = len(wl_list)
    wl_arr = np.array(wl_list)

    # radiance_blocks is list of (n_zen, n_az) arrays in same order as wl_list
    radiance = np.stack(radiance_blocks, axis=0)   # shape (n_wl, n_zen, n_az)
    # reorder to (wavelength, azimuth, cos_zenith) as requested:
    # currently radiance[wavelength, z_index, a_index]
    # we want dims (wavelength, azimuth, cos_zenith) -> transpose accordingly
    radiance = np.transpose(radiance, (0, 2, 1))   # shape (n_wl, n_az, n_zen)

    # If requested, parse time from filename and add a leading time dimension (length 1)
    time_coord = None
    if parse_time_from_filename:
        time_coord = _parse_time_from_filename(fname)
        # expand radiance to have leading time dim
        radiance = radiance[np.newaxis, ...]  # shape becomes (1, n_wl, n_az, n_zen)

    # create xarray dataset
    ds = xr.Dataset()

    if parse_time_from_filename:
        ds = ds.assign_coords({'time': ('time', [pd.to_datetime(time_coord)])})

    ds = ds.assign_coords({
        'wavelength': ('wavelength', wl_arr),
        'azimuth': ('azimuth', az_grid),
        'cos_zenith': ('cos_zenith', cosz_grid)
    })

    # Assign radiance with dims including time if present
    if parse_time_from_filename:
        ds['radiance'] = (('time', 'wavelength', 'azimuth', 'cos_zenith'), radiance)
    else:
        ds['radiance'] = (('wavelength', 'azimuth', 'cos_zenith'), radiance)

    # handle extras from the wavelength line: they may vary in number per line;
    # pad to common length and create variables extra1_0, extra1_1, ...
    max_extras = max(len(e) for e in extra1_list)
    if max_extras > 0:
        extra_arr = np.full((n_wl, max_extras), np.nan, dtype=float)
        for idx, e in enumerate(extra1_list):
            for j, val in enumerate(e):
                extra_arr[idx, j] = val
        for j in range(max_extras):
            # keep extras 1D along wavelength (unchanged for single-file reader)
            ds[f'extra1_{j}'] = (('wavelength',), extra_arr[:, j])

    # set attrs
    ds['radiance'].attrs['units'] = 'mW m-2 sr-1 nm-1'
    ds['wavelength'].attrs['units'] = 'nm'
    ds['azimuth'].attrs['units'] = 'deg'
    ds['cos_zenith'].attrs['description'] = 'cosine of zenith angle'

    return ds


def time_key_from_filename(fname):
    # helper to extract time string as in your function's regex
    _time_re = re.compile(r'.*?(\d{8}_\d{6}).*')
    m = _time_re.match(Path(fname).name)
    if not m:
        raise ValueError(f"No timestamp in filename: {fname}")
    return m.group(1)

def build_timeset(files, n_az, n_zen):
    """
    files: list of filenames (both swir and vnir mixed)
    n_az, n_zen: grid sizes to pass to read_libradtran_radiances
    returns: xr.Dataset concatenated over time and wavelength
    """
    files = [Path(f) for f in files]
    # Group by timestamp key
    groups = {}
    for f in files:
        key = time_key_from_filename(f.name)
        groups.setdefault(key, []).append(f)

    per_time_ds = []
    time_coords = []

    for key in sorted(groups):
        group_files = groups[key]
        # prefer to read both vnir and swir if present
        # identify by filename containing 'vnir' or 'swir' (case-insensitive)
        vnir = [f for f in group_files if 'vnir' in f.name.lower()]
        swir = [f for f in group_files if 'swir' in f.name.lower()]

        datasets_to_concat = []

        # convenience function to read and squeeze time dimension out
        def read_and_squeeze(fpath):
            ds = read_libradtran_radiances(fpath, n_az=n_az, n_zen=n_zen, parse_time_from_filename=True)
            # ds['radiance'] dims are ('time','wavelength','azimuth','cos_zenith')
            # squeeze out time (length 1) to get ('wavelength','azimuth','cos_zenith')
            ds_s = ds.isel(time=0, drop=True)
            return ds_s

        if swir:
            ds_sw = read_and_squeeze(swir[0])
            datasets_to_concat.append(ds_sw)
        if vnir:
            ds_vn = read_and_squeeze(vnir[0])
            datasets_to_concat.append(ds_vn)

        if not datasets_to_concat:
            # nothing for this time (shouldn't happen)
            continue

        # Concatenate along wavelength. We assume azimuth and cos_zenith grids match.
        # If they do not match, set compat or regrid as needed.
        try:
            ds_wl = xr.concat(datasets_to_concat, dim='wavelength', data_vars='all', coords='minimal', compat='equals')
        except Exception:
            # be more permissive on coords (e.g. attributes differ)
            ds_wl = xr.concat(datasets_to_concat, dim='wavelength', data_vars='all', coords='minimal', compat='override')

        # Re-add the time coordinate as length-1 dimension using parsed timestamp
        timestamp = pd.to_datetime(key, format='%Y%m%d_%H%M%S')
        ds_time = ds_wl.expand_dims({'time': [timestamp]})
        per_time_ds.append(ds_time)
        time_coords.append(timestamp)

    # Combine all per-time datasets along time
    if not per_time_ds:
        raise ValueError("No datasets read")

    ds_all = xr.concat(per_time_ds, dim='time', data_vars='all', coords='minimal', compat='override', join='outer')
    # optional: sort by time (should already be sorted)
    ds_all = ds_all.sortby('time').sortby('wavelength')

    return ds_all




def compute_solar_angles(
    timestamps,
    latitude=47.0667,
    longitude=-7.5667,
    std_longitude=-15,
    zenith_bin='/nird/home/lukasjm/libRadtran-2.0.6/bin/zenith',
):
    """
    Run libRadtran's `zenith` tool once per timestamp and collect the
    solar zenith / azimuth angles into a DataFrame.

    Parameters
    ----------
    timestamps : array-like of pandas.Timestamp
        Times to evaluate. Assumed to already be in the local standard
        time matching `std_longitude` (the zenith tool takes plain
        day/month/hour/minute/second, not a UTC offset).
    latitude, longitude, std_longitude : float
        Passed straight through to zenith's -a, -o, -s flags.
    zenith_bin : str
        Path to the zenith executable.

    Returns
    -------
    pandas.DataFrame
        Indexed by the original timestamps, with columns:
        'solar_zenith' (degrees) and 'solar_azimuth' (degrees).
    """
    records = []

    for ts in timestamps:
        ts = pd.Timestamp(ts)

        cmd = [
            zenith_bin,
            '-a', str(latitude),
            '-o', str(longitude),
            '-s', str(std_longitude),
            '-y', str(ts.year),
            str(ts.day), str(ts.month), str(ts.hour), str(ts.minute), str(ts.second),
        ]

        result = subprocess.run(
            cmd, capture_output=True, text=True, check=True
        )

        # the tool prints several human-readable lines, then a final
        # machine-readable line: "HH:MM:SS   zenith   azimuth"
        lines = [line for line in result.stdout.splitlines() if line.strip()]
        last_line = lines[-1]
        parts = last_line.split()

        records.append({
            'timestamp': ts,
            'solar_zenith': float(parts[1]),
            'solar_azimuth': float(parts[2]),
        })

    df = pd.DataFrame(records).set_index('timestamp')
    return df



def load_srf_definition(wls_filepaths):
    """Load AisaFENIX channel centers and FWHM from .wls calibration files.

    Each file has two whitespace-separated columns: wavelength [nm] and
    FWHM [nm], no header. Multiple files (e.g. VNIR + SWIR detector) are
    concatenated and sorted by wavelength.

    Parameters
    ----------
    wls_filepaths : list of str
        Paths to one or more AisaFENIX .wls calibration files.

    Returns
    -------
    pandas.DataFrame
        Columns "wavelength" and "fwhm", sorted by wavelength.
    """
    tables = [
        pd.read_csv(path, sep=r"\s+", header=None, names=["wavelength", "fwhm"])
        for path in wls_filepaths
    ]
    srf_table = pd.concat(tables, ignore_index=True)
    srf_table = srf_table.sort_values("wavelength").reset_index(drop=True)
    return srf_table


def slit_function_weights(wavelength_grid, center, fwhm, shape="gaussian"):
    """Evaluate a normalized slit function (spectral response) on a grid.

    Parameters
    ----------
    wavelength_grid : array-like
        Wavelengths [nm] at which to evaluate the slit function.
    center : float
        Center wavelength [nm] of the instrument channel.
    fwhm : float
        Full width at half maximum [nm] of the channel's response.
    shape : {"gaussian", "triangular"}
        Shape assumed for the spectral response function.

    Returns
    -------
    numpy.ndarray
        Weights, normalized to sum to 1.
    """
    distance = np.asarray(wavelength_grid) - center

    if shape == "gaussian":
        sigma = fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
        weights = np.exp(-0.5 * (distance / sigma) ** 2)
    elif shape == "triangular":
        # Linear taper reaching zero at +/- fwhm from the center, which by
        # construction gives half the peak weight at +/- fwhm/2 (i.e. the
        # correct FWHM).
        weights = np.clip(1.0 - np.abs(distance) / fwhm, a_min=0.0, a_max=None)
    else:
        raise ValueError(f"Unknown slit function shape '{shape}', "
                          "expected 'gaussian' or 'triangular'")

    return weights / weights.sum()


def convolve_to_aisafenix(
    ds,
    wls_filepaths,
    shape="gaussian",
    n_fwhm=4,
    oversampling=10,
    wavelength_dim="wavelength",
):
    """Convolve simulated radiance spectra to the AisaFENIX spectral channels.

    For every AisaFENIX channel, the simulated spectrum is interpolated onto
    a finely-sampled local wavelength axis spanning `n_fwhm` FWHM around the
    channel center, and averaged using a Gaussian or triangular slit
    function built from the channel's FWHM.

    Parameters
    ----------
    ds : xarray.Dataset
        libRadtran output containing a "wavelength" [nm] coordinate (e.g.
        with data variables like "radiance").
    wls_filepaths : list of str
        Paths to AisaFENIX .wls calibration files (wavelength, FWHM).
    shape : {"gaussian", "triangular"}
        Slit function shape to assume.
    n_fwhm : float
        Width of the slit function window, in units of FWHM, centered on
        the channel (e.g. n_fwhm=4 means the window spans center +/- 2*FWHM).
    oversampling : int
        Number of interpolation points per FWHM used to sample the slit
        function. Higher values give a more accurate convolution.
    wavelength_dim : str
        Name of the wavelength dimension/coordinate in `ds`.

    Returns
    -------
    xarray.Dataset
        `ds` with the wavelength dimension replaced by the AisaFENIX
        channel centers, convolved with the channel slit functions.
    """
    srf_table = load_srf_definition(wls_filepaths)

    sim_wavelength_min = float(ds[wavelength_dim].min())
    sim_wavelength_max = float(ds[wavelength_dim].max())

    convolved_channels = []
    for center, fwhm in zip(srf_table["wavelength"], srf_table["fwhm"]):
        half_window = 0.5 * n_fwhm * fwhm
        n_points = max(int(round(oversampling * n_fwhm)), 3)
        local_wavelengths = np.linspace(
            center - half_window, center + half_window, n_points
        )

        # skip channels whose slit function window falls outside the range
        # covered by the simulated spectrum (nothing to interpolate onto)
        if local_wavelengths.min() < sim_wavelength_min or local_wavelengths.max() > sim_wavelength_max:
            warnings.warn(
                f"Skipping AisaFENIX channel at {center:.2f} nm: its slit "
                "function window falls outside the simulated spectral range."
            )
            continue

        weights = slit_function_weights(local_wavelengths, center, fwhm, shape=shape)
        weights = xr.DataArray(
            weights, dims=wavelength_dim, coords={wavelength_dim: local_wavelengths}
        )

        # interpolate the simulated spectrum onto the local slit-function grid ...
        local_spectrum = ds.interp({wavelength_dim: local_wavelengths})

        # ... and integrate over the slit function (weighted average)
        channel_spectrum = (local_spectrum * weights).sum(wavelength_dim)
        channel_spectrum = channel_spectrum.assign_coords({wavelength_dim: center})
        convolved_channels.append(channel_spectrum)

    ds_convolved = xr.concat(convolved_channels, dim=wavelength_dim)
    return ds_convolved