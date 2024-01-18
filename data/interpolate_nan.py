# reference: https://stackoverflow.com/questions/37662180/interpolate-missing-values-2d-python
import xarray as xr
import numpy as np
from scipy.interpolate import griddata

def interp_nan():
    ds = xr.open_dataset('./saved_aod_2023.nc')
    aod = ds['aod'].values # (time, y, x)
    # print shape
    print(aod.shape)
    # interpolate nan values
    ntime, ny, nx = aod.shape

    x = np.arange(nx)
    y = np.arange(ny)

    xx, yy = np.meshgrid(x, y)

    for i in range(ntime):
        print("processing time step: ", i)
        aod_i = aod[i,:,:]
        # if no nan values, skip
        if not np.isnan(aod_i).any():
            continue
        max_interp_time = 16
        # loop until all nan values are interpolated or max_interp_time is reached
        while np.isnan(aod_i).any() and max_interp_time > 0:
            # mask invalid values
            array = np.ma.masked_invalid(aod_i)
            xx1 = xx[~array.mask]
            yy1 = yy[~array.mask]
            newarr = array[~array.mask]

            aod_i = griddata((xx1, yy1), newarr.ravel(),
                        (xx, yy),
                        method='cubic')
            max_interp_time -= 1
        # if there are still nan values, use nearest neighbor interpolation
        if np.isnan(aod_i).any():
            aod_i = griddata((xx1, yy1), newarr.ravel(),
                        (xx, yy),
                        method='nearest')
        aod[i,:,:] = aod_i
    return aod


if __name__ == "__main__":
    aod = interp_nan()
    ds = xr.open_dataset('./saved_aod_20230101.nc')
    ds['aod'] = (('time', 'y', 'x'), aod)
    ds.to_netcdf('./saved_aod_20230101_interp_cubic.nc')
