import numpy as np
import os
import datetime as dt
import shutil
from datetime import timedelta
import xarray as xr

def check_restart_files(hourly_hwpdir, fcst_dates):
    hwp_avail_hours = []
    hwp_non_avail_hours = []

    for cycle in fcst_dates:
        restart_file = f"{cycle[:8]}.{cycle[8:10]}0000.phy_data.nc"
        file_path = os.path.join(hourly_hwpdir, restart_file)

        if os.path.exists(file_path):
            print(f'Restart file available for: {restart_file}')
            hwp_avail_hours.append(cycle)
        else:
            print(f'Copy restart file for: {restart_file}')
            hwp_non_avail_hours.append(cycle)

    print(f'Available restart at: {hwp_avail_hours}, Non-available restart files at: {hwp_non_avail_hours}')
    return(hwp_avail_hours, hwp_non_avail_hours)

def copy_missing_restar(nwges_dir, hwp_non_avail_hours, hourly_hwpdir):
    restart_avail_hours = []
    restart_nonavail_hours_test = []

    for cycle in hwp_non_avail_hours:
        YYYYMMDDHH = dt.datetime.strptime(cycle, "%Y%m%d%H")
        prev_hr = YYYYMMDDHH - timedelta(hours=1)
        prev_hr_st = prev_hr.strftime("%Y%m%d%H"):q
        
        source_restart_dir = os.path.join(nwges_dir, prev_hr_st, 'fcst_fv3lam', 'RESTART')
        target_file_name = f"{cycle[:8]}.{'000000' if cycle.endswith('00') else cycle[8:10]+'0000'}.phy_data.nc"
        source_file_path = os.path.join(source_restart_dir, target_file_name)
        target_file_path = os.path.join(hourly_hwpdir, target_file_name)

        try:
            if os.path.exists(source_file_path):
                with xr.open_dataset(source_file_path) as ds:
                    ds=ds.rrfs_hwp_ave
                    ds.to_netcdf(target_file_path)
                    restart_avail_hours.append(cycle)
                    print(f'Restart file copied: {target_file_name}')
            else:
                raise FileNotFoundError
        except (FileNotFoundError, AttributeError) as e:
            restart_nonavail_hours_test.append(cycle)
            print(f'Issue with file {target_file_name}: {e}')

    return(restart_avail_hours, restart_nonavail_hours_test)

def process_hwp(fcst_dates, hourly_hwpdir, cols, rows, intp_dir, rave_to_intp):
    hwp_ave = []

    for cycle in fcst_dates:
        print(f'Processing restart file for date: {cycle}')
        file_path = os.path.join(hourly_hwpdir, f"{cycle[:8]}.{cycle[8:10]}0000.phy_data.nc")
        rave_path = os.path.join(intp_dir, f"{rave_to_intp}{cycle}00_{cycle}00.nc")

        # Check if both restart and rave files are available
        if os.path.exists(file_path) and os.path.exists(rave_path):
            try:
                with xr.open_dataset(file_path) as nc:
                    hwp_values = nc.rrfs_hwp_ave.values.ravel().tolist()
                    hwp_ave.append(hwp_values)
                    print(f'HWP TEST: {nc.rrfs_hwp_ave[0, 895, 637].values}')
            except FileNotFoundError:
                # Restart file not available for that time period, values are then set to nan
                pass
        else:
            print('Restart file non-available, setting HWP to nan')

    # Calculate the mean HWP values if available
    if hwp_ave:
        hwp_ave_arr = np.nanmean(hwp_ave, axis=0).reshape(cols, rows)
        print(f'HWP MEAN: {hwp_ave_arr[895, 637]}')
    else:
        hwp_ave_arr = np.zeros((cols, rows))

    xarr_hwp = xr.DataArray(hwp_ave_arr)

    return(hwp_ave_arr, xarr_hwp)