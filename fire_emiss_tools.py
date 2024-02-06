import os
import numpy as np
import xarray as xr
from datetime import datetime
from netCDF4 import Dataset
import interp_tools as i_tools

def averaging_FRP(fcst_dates, cols, rows, intp_dir, rave_to_intp, veg_map, tgt_area, beta, fg_to_ug):
    base_array = np.zeros((cols*rows))
    frp_daily = []
    ebb_smoke_total = base_array
    ef_map = xr.open_dataset(veg_map)
    emiss_factor = ef_map.emiss_factor.values
    target_area = tgt_area.values

    for cycle in fcst_dates:
        file_path = os.path.join(intp_dir, f'{rave_to_intp}{cycle}00_{cycle}00.nc')
        
        if os.path.exists(file_path):
            with xr.open_dataset(file_path) as nc:
                open_fre = nc.FRE[0, :, :].values
                open_frp = nc.frp_avg_hr[0, :, :].values
                ebb_hourly = open_fre * emiss_factor * beta * fg_to_ug
                ebb_to_add = ebb_hourly / target_area
                ebb_to_add_filtered = xr.where(open_frp > 0, ebb_to_add, 0)
                ebb_smoke_total += ebb_to_add_filtered.ravel()
                
                frp_to_add = xr.where(open_frp > 0, open_frp, np.nan)
                frp_daily.append(frp_to_add)

    if frp_daily:
        frp_avg = np.nanmean(frp_daily, axis=0)
        frp_avg_reshaped = xr.where(frp_avg > 0, frp_avg, 0)
        ebb_total = ebb_smoke_total.reshape(cols, rows)
        ebb_total_reshaped = ebb_total / 86400
    else:
        zero_array = np.zeros((cols, rows))
        frp_avg_reshaped = zero_array
        ebb_total_reshaped = zero_array

    return(frp_avg_reshaped, ebb_total_reshaped)

def estimate_fire_duration(intp_avail_hours, intp_dir, fcst_dates, current_day, cols, rows, rave_to_intp):
    # There are two steps here.
    #   1) First day simulation no RAVE from previous 24 hours available (fire age is set to zero)
    #   2) previus files are present (estimate fire age as the difference between the date of the current cycle and the date whe the fire was last observed whiting 24 hours)
    t_fire = np.zeros((cols, rows))

    for date_str in fcst_dates:
        date_file = int(date_str[:10])
        print('Date processing for fire duration',date_file)
        file_path = os.path.join(intp_dir, f'{rave_to_intp}{date_str}00_{date_str}00.nc')
        
        if os.path.exists(file_path):
            with xr.open_dataset(file_path) as open_intp:
                FRP = open_intp.frp_avg_hr[0, :, :].values
                dates_filtered = np.where(FRP > 0, date_file, 0)
                t_fire = np.maximum(t_fire, dates_filtered)

    t_fire_flattened = t_fire.flatten()
    t_fire_flattened = [int(i) if i != 0 else 0 for i in t_fire_flattened]

    try:
        fcst_t = datetime.strptime(current_day, '%Y%m%d%H')
        hr_ends = [datetime.strptime(str(hr), '%Y%m%d%H') if hr != 0 else 0 for hr in t_fire_flattened]
        te = [(fcst_t - i).total_seconds()/3600 if i != 0 else 0 for i in hr_ends]
    except ValueError:
        te = np.zeros((rows, cols))

    return(te)

def save_fire_dur(cols, rows, te):
    fire_dur = np.array(te).reshape(cols, rows)
    return(fire_dur)

def produce_emiss_file(xarr_hwp, frp_avg_reshaped, intp_dir, current_day, tgt_latt, tgt_lont, ebb_tot_reshaped, fire_age, cols, rows):
    # Filter HWP
    filtered_hwp = xarr_hwp.where(frp_avg_reshaped > 0, 0)

    # Produce emiss file
    file_path = os.path.join(intp_dir, f'SMOKE_RRFS_data_{current_day}00.nc')

    with Dataset(file_path, 'w') as fout:
        i_tools.create_emiss_file(fout, cols, rows)
        i_tools.Store_latlon_by_Level(fout, 'geolat', tgt_latt, 'cell center latitude', 'degrees_north', '2D', '-9999.f', '1.f')
        i_tools.Store_latlon_by_Level(fout, 'geolon', tgt_lont, 'cell center longitude', 'degrees_east', '2D', '-9999.f', '1.f')
 
        print('Storing different variables')
        i_tools.Store_by_Level(fout,'frp_davg','Daily mean Fire Radiative Power','MW','3D','0.f','1.f')
        fout.variables['frp_davg'][0, :, :] = frp_avg_reshaped
        i_tools.Store_by_Level(fout,'ebb_rate','Total EBB emission','ug m-2 s-1','3D','0.f','1.f') 
        fout.variables['ebb_rate'][0, :, :] = ebb_tot_reshaped
        i_tools.Store_by_Level(fout,'fire_end_hr','Hours since fire was last detected','hrs','3D','0.f','1.f')
        fout.variables['fire_end_hr'][0, :, :] = fire_age
        i_tools.Store_by_Level(fout,'hwp_davg','Daily mean Hourly Wildfire Potential', 'none','3D','0.f','1.f')
        fout.variables['hwp_davg'][0, :, :] = filtered_hwp

    return "Emissions file created successfully"
