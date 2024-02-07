import pandas as pd
import numpy as np
import rasterio as rio
import os
import geoplot as gplt
import geoplot.crs as gcrs
import geopandas as gpd
from geodatasets import get_path
import matplotlib.pyplot as plt
from shapely.geometry import Polygon
import matplotlib.patches as patches
from matplotlib.widgets import Cursor
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.cluster import DBSCAN
from datetime import datetime
from dateutil.parser import parse
import pytz
from weather import weather


class analysis:
    """
    For analysis of run 11 onwards. Format for data file names was changed from run 11 onwards.
    """

    def __init__(self, path, date_string: str = "18-04-2016 01:00:00 +08", boundary_file="sg_map.geojson") -> None:
        self.results_dir = path
        self.domain_bounds = gpd.read_file(boundary_file)
        self.rural_count = pd.read_pickle("rural_count.pkl")
        self.land_sea = gpd.read_file("land_sea.geojson")
        self.ts = parse(timestr=date_string)

# ----------------------------------------- Functions for extracting raw data into a dataframe/GeoDataFrame ------------------#

    def load_tiff(self, fn: str) -> pd.DataFrame:
        with rio.open(fn) as dataset:
            # specify band number in the next line. The urban-heat-island tiff file has 2 bands.
            # Band 1 contains the actual UHI data while band 2 specifies the locations of rural and urban areas.
            # The values in band 2 are zero for urban
            # and one for rural areas.
            val = dataset.read(1)
            no_data = 991
            data = [(dataset.xy(x, y)[0], dataset.xy(x, y)[1], val[x, y])
                    for x, y in np.ndindex(val.shape) if val[x, y] != no_data]
            lon = [i[0] for i in data]
            lat = [i[1] for i in data]
            d = [i[2] for i in data]
            res = pd.DataFrame({"lon": lon, 'lat': lat, "data": d})
        return res

    def get_timestamp(self, hour: int) -> str:
        """
        Constructs SGT timestamp for given date and hour. 
        Returns UTC timestamp as a string in the format yyyymmddhhmmss.

        """
        tsn = self.ts
        tsn = tsn.replace(hour=hour)
        if tsn.tzinfo is None or tsn.tzinfo.utcoffset(tsn) is None:
            tsu = pytz.timezone(
                "Asia/Singapore").localize(tsn).astimezone(pytz.utc)
        else:
            tsu = tsn.astimezone(pytz.utc)

        tsr = tsu.strftime("%Y-%m-%d %H:%M:%S").replace("-",
                                                        "").replace(" ", "").replace(":", "")
        return tsr + "UTC"

    def get_temp_uhi(self, run: int, hour: int, param: str = "\\2m_air_temperature_") -> gpd.GeoDataFrame:
        """
        Gets air temperature data for the specified run and timestamp by default. 

        Example illustrating format for ts: "20160418010000UTC"

        """
        ts = self.get_timestamp(hour)
        fnp = self.results_dir + \
            str(run) + "\\wrfrun" + param + ts + ".pkl"
        if (not os.path.exists(fnp)):
            fnt = fnp.replace("pkl", "tiff")
            res = self.load_tiff(fnt)
            res.to_pickle(fnp)
        else:
            res = pd.read_pickle(fnp)
        gdf = gpd.GeoDataFrame(res, geometry=gpd.points_from_xy(
            res.lon, res.lat), crs="EPSG:4326")
        drop_index = (self.land_sea.loc[self.land_sea["mask"] == 1, :]).index
        gdf = gdf.drop(drop_index)
        return gdf

    def calculate_uhi(self, run: int, hour: int) -> gpd.GeoDataFrame:
        param = "\\2m_air_temperature_"
        rc = self.rural_count.copy(deep=True)
        df = self.get_temp_uhi(run, hour, param)
        rc = rc.drop(columns='data', errors="ignore")
        rc = rc.merge(df, on=["lon", "lat"])
        ave_rural_temp = (
            rc["data"]*rc["count"]).sum()/rc["count"].sum()
        print(ave_rural_temp, df["data"].min(), df["data"].max())
        s1 = df["data"] - ave_rural_temp
        df = df.assign(uhi=s1)
        gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(
            df.lon, df.lat), crs="EPSG:4326")
        return gdf

    def get_heat_flux(self, run: int, hour: int, layer: int) -> gpd.GeoDataFrame:
        """
        Gets sensible heat flux for a particular run and hour
        """
        hour_string = str(hour)
        if hour < 10:
            hour_string = '0' + hour_string

        layer_string = str(layer)

        if layer < 10:
            layer_string = '0' + layer_string

        fnp = self.results_dir + \
            str(run) + "\\real" + "\\SH_EXT_d04_k" + \
            layer_string + "_t" + hour_string + ".pkl"

        if (not os.path.exists(fnp)):
            fnt = fnp.replace("pkl", "tiff")
            res = self.load_tiff(fnt)
            res.to_pickle(fnp)
        else:
            res = pd.read_pickle(fnp)

        gdf = gpd.GeoDataFrame(res, geometry=gpd.points_from_xy(
            res.lon, res.lat), crs="EPSG:4326")

        drop_index = (self.land_sea.loc[self.land_sea["mask"] == 1, :]).index
        gdf = gdf.drop(drop_index)
        return gdf

    def get_wind_data(self, run: int, hour: int) -> pd.DataFrame:
        ts = self.get_timestamp(hour)
        wsfn = self.results_dir + \
            str(run) + "\\wrfrun" + "\\10m_wind_speed_" + ts + ".tiff"
        wdfn = wsfn.replace("wind_speed", "wind_direction")

        wind_pkl = self.results_dir + \
            str(run) + "\\wrfrun" + "\\10m_wind_data_" + ts + ".pkl"

        if (not os.path.exists(wind_pkl)):
            ws = self.load_tiff(wsfn)
            ws = ws.rename(columns={"data": "wind_speed"})
            wd = self.load_tiff(wdfn)
            wd = wd.rename(columns={"data": "wind_direction"})
            wind_data = pd.merge(ws, wd, on=["lat", "lon"])
            wind_data.to_pickle(wind_pkl)
            # Cannot ignore grid cells in sea for wind data because the wind speed and directions
            # at those locations may influence the temperatures in land areas
            # drop_index = (self.land_sea.loc[self.land_sea["mask"] == 1, :]).index
            # wind_data = wind_data.drop(drop_index)
        else:
            wind_data = pd.read_pickle(wind_pkl)
        return wind_data

# ---------------------------------------Functions for performing calculations on extracted data -----#

    def get_min_ave_max(self, run: int, hour: int,
                        data_bounds: list[float] = [
                            103.65, 1.22, 103.80, 1.36],
                        exclude_bounds: list[list[float]] = [[]],
                        param="temperature") -> list[float]:
        """
        Return the min, average and max values of the specified parameter for all points within the region defined by 
        data_bounds and outside of exclude_bounds at the required time.
        """
        if param.lower() == "temperature":
            gdf = self.get_temp_uhi(run, hour)
        else:
            gdf = self.calculate_uhi(run, hour)

        for region in exclude_bounds:
            gdc = gdf.clip(region)
            gdf = gdf.drop(gdc.index)

        gds = gdf.clip(data_bounds)["data"]
        if param.lower() != "temperature":
            gds = gdc["uhi"]
        res = [gds.min(), gds.mean(), gds.max()]
        return res

    def get_mid_regions_coords(self, regions_bounds: list[list[float]]) -> list[list[float]]:
        """
        Returns a list containing the coordinates (lon, lat) of the middle grid cell in each
        of the specified regions.

        If param == 'index', returns a list containing the indices of the middle grid in each of the specified 
        regions in the dataframe obtained by processing the tiff file for each timestamp.

        """
        gdf = self.get_temp_uhi(run=37, hour=0)
        coords = []
        for ind, region in enumerate(regions_bounds):
            gdc = gdf.clip(region)
            nl = np.floor(len(gdc)/2.0)
            gdc = gdc.sort_values(by=['lon', 'lat'])
            gdc = gdc.reset_index(drop=True)
            coords.append([gdc.loc[nl, 'lon'], gdc.loc[nl, 'lat']])
        return coords

    def get_mid_regions_indices(self, regions_bounds: list[list[float]]) -> list[int]:
        """

        Returns a list containing the indices of the middle grid in each of the specified 
        regions in the dataframe obtained by processing the tiff file for each timestamp.

        It has been verified that the values of indices depend only on the specified region
        and not the value of run used in the first line of this function.

        """
        gdf = self.get_temp_uhi(run=45, hour=0)
        inds = []
        for ind, region in enumerate(regions_bounds):
            gdc = gdf.clip(region)
            nl = np.floor(len(gdc)/2.0)
            gdc = gdc.sort_values(by=['lon', 'lat'])
            gdc = gdc.reset_index()
            inds.append(gdc.loc[nl, 'index'])
        return inds

    def calculate_error(self, r1: np.ndarray, r2: np.ndarray) -> np.ndarray:
        """
        Returns the mean absolute difference (MAD) and root mean square difference (RMSD)
        across all rows of the input arrays for each column.

        First row of output contains MAD and second row contains RMSD.

        """
        if r1.shape != r2.shape:
            raise TypeError("The two input arrays are of different shapes")

        nc = r1.shape[1]
        res = np.zeros((2, nc))
        abs_diff = np.abs(r2 - r1)
        mad = np.mean(abs_diff, axis=0)
        rmsd = np.sqrt(np.mean(abs_diff*abs_diff, axis=0))
        res[0, :] = mad
        res[1, :] = rmsd
        return res

    def get_error_single(self, run=40, regions_bounds: list[list[float]] = [],
                         actual_temperature_data: str = "openmeteo_temperature.txt") -> np.ndarray:
        """
        Compute mean absolute error and root mean square error for the
        coordinates corresponding to the middle grid cell of each region.
        """

        # Load observed temperature values
        observed = np.loadtxt(actual_temperature_data)

        # Get indices of the middle grid cell for each region
        indices = self.get_mid_regions_indices(regions_bounds)

        # Get actual values from data for specified run
        nl = len(regions_bounds)
        simulated = np.zeros((24, nl))
        for hour in range(24):
            gdf = self.get_temp_uhi(run=run, hour=hour)
            gdc = gdf.loc[indices, "data"]
            simulated[hour, :] = gdc.values
        error = self.calculate_error(observed, simulated)
        return error

    def get_region_coords(self, regions_bounds: list[list[float]] = []) -> np.ndarray:
        """
        Returns a list of coodinates of all grid cells that fall within at least one of
        the specified regions
        """

        run, hour = 37, 0
        gdf = self.get_temp_uhi(run=run, hour=hour)
        res = np.zeros((0, 2))
        for ind, region in enumerate(regions_bounds):
            gdc = gdf.clip(region)
            coords = gdc.loc[:, ['lon', 'lat']].values
            res = np.concatenate((res, coords))
        return res

    def get_openmeteo_temperature(self, regions_bounds: list[list[float]] = []) -> np.ndarray:
        """
        Retrieves temperature data from OpenMeteo API for the specified date
        """

        # Get list of grid cell coordinates in each region
        coords = self.get_region_coords(regions_bounds)
        # Send request to OpenMeteo API for the required date
        start_date = self.ts.strftime('%Y-%m-%d')
        wr = weather()
        observed_temp = wr.get_temp(date=start_date, locations=coords)
        return observed_temp

    def get_simulation_temperature(self, run=40, ngc: float = 25, regions_bounds: list[list[float]] = []) -> np.ndarray:
        """
        Returns numpy array containing the simulation temperature at each grid cell in each region
        """

        nr = len(regions_bounds)
        res = np.zeros((24, ngc*nr))

        for hour in range(24):
            gdf = self.get_temp_uhi(run=run, hour=hour)
            for ind, region in enumerate(regions_bounds):
                gdc = gdf.clip(region)
                si = ind*ngc
                li = (ind + 1)*ngc
                res[hour, si:li] = gdc["data"].values
        return res

    def get_error_regions(self, regions_bounds: list[list[float]] = []) -> np.ndarray:

        ngc = 25
        nr = len(regions_bounds)
        error = np.zeros((2, nr))
        observed = self.get_openmeteo_temperature(
            regions_bounds=regions_bounds)
        simulated = self.get_simulation_temperature(
            ngc=ngc, regions_bounds=regions_bounds)
        abs_diff = np.abs(observed - simulated)
        abs_diff_sq = abs_diff*abs_diff
        # Take average of 24 by ngc blocks of the difference arrays
        abs_diff = np.reshape(abs_diff, (24, ngc, -1))
        abs_diff_sq = np.reshape(abs_diff_sq, (24, ngc, -1))
        error[0, :] = np.mean(abs_diff, axis=(0, 1))
        error[1, :] = np.sqrt(np.mean(abs_diff_sq, axis=(0, 1)))
        return error

    def get_total_heat(self, run: int = 37, hour: int = 0, data_bounds: list[float] = [
            103.65, 1.22, 103.80, 1.36], min_layer: int = 0, max_layer: int = 8) -> float:
        """
        Returns total heat across all grid cells lying within the specified region
        and between the min_layer and max_layer.


        """

        total_heat = 0.0
        grid_cell_area = 300*300

        for layer in range(min_layer, 1 + max_layer):
            gdf = self.get_heat_flux(run=run, hour=hour, layer=layer)
            gdf = gdf.assign(heat=gdf["data"]*grid_cell_area)
            layer_heat = gdf.clip(data_bounds)["heat"].sum()
            total_heat += layer_heat

        return total_heat


# ----------------------------------------- Functions for plotting extracted data ------------------#

    def plot_temp_time(self, run: int, times: list[int],
                       data_bounds: list[float] = [
        103.65, 1.22, 103.80, 1.36],
        exclude_bounds: list[list[float]] = [[]],
            param="temperature") -> plt.figure:
        temp_data = np.zeros((len(times), 3))
        for ind, t in enumerate(times):
            res = self.get_min_ave_max(
                run, t, data_bounds, exclude_bounds, param)
            temp_data[ind][:] = np.asarray(res)
        fig, ax = plt.subplots()
        handles = []
        labels = ["Minimum", "Mean", "Maximum"]
        markers = ['o-', 's-', 'd-', '^-']
        for i in range(3):
            line, = ax.plot(times, temp_data[:, i], markers[i])
            handles.append(line)
        ax.set_xlabel("Time (h)")
        ax.set_ylabel(param + " ($^\circ$C)")
        ax.set_title(param + " ($^\circ$C)")
        ax.legend(handles, labels)
        return fig

    def plot_ave_temp_runs(self, runs: list[int], times: list[int],
                           data_bounds: list[float] = [
                               103.6, 1.17, 103.85, 1.35],
                           exclude_bounds: list[list[float]] = [[]], param: str = "temperature",
                           legend_labels: list[str] = []) -> plt.figure:
        temp_data = np.zeros((len(runs), len(times)))
        for ir, r in enumerate(runs):
            for it, t in enumerate(times):
                res = self.get_min_ave_max(
                    r, t, data_bounds, exclude_bounds, param)
                temp_data[ir, it] = res[1]

        for i in range(1, len(runs)):
            td = np.abs(temp_data[i, :] - temp_data[0, :])
            mt = np.argmax(td)
            print(mt, td[mt])

        fig, ax = plt.subplots()
        handles = []
        markers = ['o-', 's-', 'p-', 'h-', 'd-', '^-', 'v-', '<-', '>-', '*-']
        for i in range(len(runs)):
            line, = ax.plot(times, temp_data[i, :], markers[i])
            handles.append(line)
        ax.set_xlabel("Time (hours)")
        ax.set_ylabel(" $T_a$($^\circ$C)")
        # ax.set_title(param + " ($^\circ$C)")
        ax.legend(handles, legend_labels, frameon=False)
        return fig

    def plot_ave_diff_runs(self, ax: plt.Axes, runs: list[int], times: list[int],
                           data_bounds: list[float] = [
                               103.6, 1.17, 103.85, 1.35],
                           ylim: tuple[float] = (None, None),
                           exclude_bounds: list[list[float]] = [], param: str = "temperature",
                           legend_labels: list[str] = [],
                           title: str = ""):
        """
        Plot difference in average temperature 
        of grid cells within the specified region.
        """
        temp_data = np.zeros((len(runs), len(times)))
        temp_diff = np.zeros((len(runs) - 1, len(times)))
        for ir, r in enumerate(runs):
            for it, t in enumerate(times):
                res = self.get_min_ave_max(
                    r, t, data_bounds, exclude_bounds, param)
                temp_data[ir, it] = res[1]

        lt = 1000
        ht = -1000
        time_lt = -1
        time_ht = -1
        for i in range(1, len(runs)):
            temp_diff[i-1, :] = temp_data[i, :] - temp_data[0, :]
            lti = np.argmin(temp_diff[i-1, :])
            hti = np.argmax(temp_diff[i-1, :])
            ltr = temp_diff[i-1, lti]
            htr = temp_diff[i-1, hti]
            if ltr < lt:
                lt = ltr
                time_lt = lti
            if htr > ht:
                ht = htr
                time_ht = hti

        print(time_lt, ", ", lt, ", ", time_ht, ", ", ht)

        handles = []
        markers = ['o-', 's-', 'p-', 'h-', 'd-', '^-', 'v-', '<-', '>-', '*-']
        for i in range(len(runs) - 1):
            line, = ax.plot(times, temp_diff[i, :], markers[i])
            handles.append(line)
        ax.set_xlabel("Time (hours)")
        ax.set_ylabel(r" $ \Delta T_a$ ($^\circ$C)")
        ax.set_ylim(ylim)
        # ax.set_title(param + " ($^\circ$C)")
        ax.legend(handles, legend_labels, frameon=False)
        ax.set_title(title)
        return handles

    def plot_ave_regions(self, runs: list[int], times: list[int],
                         data_bounds: list[float] = [
                             103.6, 1.17, 103.85, 1.35],
                         ylim: tuple[float] = (None, None),
                         regions_bounds: list[list[float]] = [],
                         legend_labels: list[str] = []) -> plt.figure:
        """
        Plot average temperature difference for multiple runs.
        """
        nr = len(regions_bounds)
        nc = int(np.ceil(nr/2))
        fig, axes = plt.subplots(2, nc, figsize=(15, 10))

        for i in range(2):
            for j in range(nc):
                axes[i][j].set_visible(False)

        handles = []
        for ind, region in enumerate(regions_bounds):
            title = "R" + str(ind+1)
            (i, j) = np.unravel_index(ind, (2, nc))
            ax = axes[i][j]
            ax.set_visible(True)
            handles = self.plot_ave_diff_runs(ax=ax, runs=runs, times=times,
                                              data_bounds=region, ylim=ylim,
                                              title=title)
        fig.legend(handles, legend_labels, frameon=False,
                   ncol=1, loc=(0.7, 0.1), fontsize=16)
        plt.tight_layout()
        return fig

    def add_box(self, ax, bounds: list) -> None:
        # Create a Rectangle patch
        width = bounds[2] - bounds[0]
        height = bounds[3] - bounds[1]
        rect = patches.Rectangle(
            (bounds[0], bounds[1]), width, height, linewidth=1.5,
            edgecolor='r', facecolor='none')
        # Add the patch to the Axes
        ax.add_patch(rect)

    def plot_temp_spatial(self, run: int, hour: int,
                          data_bounds: list[float] = [
                              103.65, 1.22, 103.80, 1.36],
                          exclude_bounds: list[list[float]] = [[]],
                          xticks: list[float] = [],
                          yticks: list[float] = [],
                          locations_lon: list[float] = [],
                          locations_lat: list[float] = [],
                          markersize=None,
                          regions_bounds: list[list[float]] = [[]],
                          regions_labels: list[str] = [],
                          vmin=None, vmax=None, title: str = "",
                          param="Temperature") -> plt.figure:
        """
        Plot all temperature data in the region defined by data_bounds 
        excluding all points in exclude_bounds.

        Also plots the points specified in locations_lon and locations_lat. 
        """

        if param.lower() == "temperature":
            gdf = self.get_temp_uhi(run, hour)
        else:
            gdf = self.calculate_uhi(run, hour)

        fig = plt.figure()
        ax = self.domain_bounds.clip(data_bounds).plot(
            color="white", edgecolor="black")
        # Solution relying on geopandas within and sjoin functions doesn't work correctly. See analyze_output.ipynb for more details.

        for region in exclude_bounds:
            gdc = gdf.clip(region)
            gdf = gdf.drop(gdc.index)

        plot_var = "data"
        if param.lower() != "temperature":
            plot_var = "uhi"

        gdt = gdf.clip(data_bounds)
        print("Minimum temperature = ", gdt["data"].min())
        print("Maximum temperature = ", gdt["data"].max())

        gdt.plot(plot_var, ax=ax,
                 vmin=vmin, vmax=vmax,
                 markersize=markersize, legend=True)

        for ind, region in enumerate(regions_bounds):
            self.add_box(ax, region)
            offset = 0.002
            label = regions_labels[ind]
            label_loc = [region[0] - offset, region[3] + offset]
            ax.text(label_loc[0], label_loc[1], label)

        ax.scatter(locations_lon, locations_lat, c='r')

        if (len(xticks) > 0):
            ax.set_xticks(xticks)
        if (len(yticks) > 0):
            ax.set_yticks(yticks)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_title(param + " ($^\circ$C). " + title)
        return fig

    def plot_heat_spatial(self, run: int, hour: int,
                          layer: int,
                          data_bounds: list[float] = [
                              103.65, 1.22, 103.80, 1.36],
                          exclude_bounds: list[list[float]] = [[]],
                          xticks: list[float] = [],
                          yticks: list[float] = [],
                          vmin=None, vmax=None, title: str = "") -> plt.figure:
        """
        Plot all temperature data in the region defined by data_bounds 
        excluding all points in exclude_bounds 
        """

        hour_string = str(hour)
        if hour < 10:
            hour_string = '0' + hour_string

        layer_string = str(layer)

        if layer < 10:
            layer_string = '0' + layer_string

        fnp = self.results_dir + \
            str(run) + "\\real" + "\\SH_EXT_d04_k" + \
            layer_string + "_t" + hour_string + ".pkl"

        if (not os.path.exists(fnp)):
            fnt = fnp.replace("pkl", "tiff")
            res = self.load_tiff(fnt)
            res.to_pickle(fnp)
        else:
            res = pd.read_pickle(fnp)

        gdf = gpd.GeoDataFrame(res, geometry=gpd.points_from_xy(
            res.lon, res.lat), crs="EPSG:4326")
        gdf = pd.concat([gdf, self.land_sea["mask"]], axis=1)
        drop_index = (
            gdf.loc[(gdf["mask"] == 1) & (gdf["data"] < 0.5), :]).index
        gdf = gdf.drop(drop_index)

        fig = plt.figure()
        ax = self.domain_bounds.clip(data_bounds).plot(
            color="white", edgecolor="black")

        for region in exclude_bounds:
            gdc = gdf.clip(region)
            gdf = gdf.drop(gdc.index)

        plot_var = "data"

        gdt = gdf.clip(data_bounds)
        print("Minimum heat flux (W/m^2) = ", gdt["data"].min())
        print("Maximum heat flux (W/m^2) = ", gdt["data"].max())

        gdt.plot(plot_var, ax=ax,
                 vmin=vmin, vmax=vmax, legend=True)
        if (len(xticks) > 0):
            ax.set_xticks(xticks)
        if (len(yticks) > 0):
            ax.set_yticks(yticks)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_title(" Heat flux ($W/m^2$). " + title)
        return fig

    def plot_wind_data(self, run: int, hour: int,
                       vmin=None, vmax=None, scale: float = 0.1, freq: int = 10, title: str = "",
                       data_bounds: list[float] = [103.65, 1.26, 103.75, 1.31],
                       xticks: list[float] = [103.64, 103.68, 103.72, 103.76, 103.80]) -> plt.figure:

        wind_data = self.get_wind_data(run=run, hour=hour)
        geo_wind_data = gpd.GeoDataFrame(wind_data, geometry=gpd.points_from_xy(
            wind_data.lon, wind_data.lat), crs="EPSG:4326")
        # Transform wind direction from degrees to radians
        geo_wind_dir_rad = geo_wind_data["wind_direction"].apply(
            lambda x: x*np.pi/180)
        geo_wind_data.loc[:, "wind_direction"] = geo_wind_dir_rad

        geo_wind_data = geo_wind_data.clip(data_bounds)
        geo_wind_data = geo_wind_data.iloc[::freq]
        fig = plt.figure()
        ax = self.domain_bounds.clip(data_bounds).plot(
            color="white", edgecolor="black")

        nws = -1.0*geo_wind_data['wind_speed']
        wsx = geo_wind_data['wind_direction'].apply(np.sin)
        wsy = geo_wind_data['wind_direction'].apply(np.cos)

        print(geo_wind_data['wind_speed'].min())
        print(geo_wind_data['wind_speed'].max())

        qq = plt.quiver(geo_wind_data['lon'], geo_wind_data['lat'],
                        nws*wsx, nws*wsy, geo_wind_data['wind_speed'],
                        cmap=plt.cm.jet, scale=scale)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        if (len(xticks) > 0):
            ax.set_xticks(xticks)
        ax.set_title("Wind speed (m/s) and direction. " + title)
        # create an axes on the right side of ax. The width of cax will be 5%
        # of ax and the padding between cax and ax will be fixed at 0.05 inch.
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(qq, cmap=plt.cm.jet, cax=cax)
        plt.clim(vmin, vmax)
        return fig

    def plot_ave_region(self, data_bounds: list[float] = [
            103.65, 1.22, 103.80, 1.36],
            exclude_bounds: list[list[float]] = [[]], title="Singapore mainland") -> plt.figure:
        """
        Plot the grid cells within Singapore mainland for which the average temperature is calculated.
        """

        fig = plt.figure()
        ax = self.domain_bounds.plot(color="white", edgecolor="black")

        gdf = self.get_temp_uhi(run=17, hour=0)
        for region in exclude_bounds:
            gdc = gdf.clip(region)
            gdf = gdf.drop(gdc.index)

        gdf.clip(data_bounds).plot(ax=ax)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_title(title)
        print(len(gdf.clip(data_bounds)))
        return fig

    def plot_multiple_regions(self, data_bounds: list[float] = [],
                              regions_bounds: list[list[float]] = [[]],
                              title="Singapore mainland") -> plt.Figure:
        """
        Plot the grid cells lying within one or more user-specified regions
        """
        fig = plt.figure()
        ax = self.domain_bounds.clip(data_bounds).plot(
            color="white", edgecolor="black")

        gdf = self.get_temp_uhi(run=17, hour=0)
        num_cells = []
        lab = "R"
        for ind, region in enumerate(regions_bounds):
            gdc = gdf.clip(region)
            num_cells.append(len(gdc))
            gdc.plot(ax=ax, markersize=5)
            label = lab + str(ind + 1)
            offset = 0.002
            label_loc = [region[0] - offset, region[3] + offset]
            ax.text(label_loc[0], label_loc[1], label)

        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_title(title)
        print(num_cells)
        return fig

    def plot_rural_loc(self) -> plt.figure:
        """
        Display locations of grid cells classified as rural on a map
        """
        df = pd.read_pickle("rural_urban.pkl")
        rural_loc = df.loc[df["data"] == 1, [
            "lon", "lat"]].reset_index(drop=True)
        gdf = gpd.GeoDataFrame(
            rural_loc, geometry=gpd.points_from_xy(rural_loc.lon, rural_loc.lat), crs="EPSG:4326"
        )
        fig = plt.figure()
        ax = self.domain_bounds.plot(color="white", edgecolor="black")
        gdf.plot(ax=ax)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        return fig

    def plot_clusters(self, run: int, hour: int, data_bounds: list[float] = [],
                      min_temperature: float = 35.0, min_samples: int = 2, eps: float = 100.0) -> plt.figure:
        """
        Identify clusters of grid cells whose 
        temperature is larger than the specified threshold
        """
        gdf = self.get_temp_uhi(run, hour)
        gdf = gdf.loc[gdf["data"] > min_temperature, :]
        # Re-project the Geodataframe to a CRS whose units are in meters
        gdc = gdf.to_crs("3857")
        # Create x and y coordinate columns (with units in meters)
        gdc['x'] = gdc.geometry.x
        gdc['y'] = gdc.geometry.y
        # Create a numpy array where each row is a coordinate pair
        coords = gdc[['x', 'y']].values
        # Cluster using DBScan
        db = DBSCAN(eps=eps, min_samples=min_samples).fit(coords)
        cluster_labels = pd.Series(db.labels_).rename('cluster')
        gdt = pd.concat([gdf.reset_index(drop=True), cluster_labels], axis=1)
        fig = plt.figure()
        ax = self.domain_bounds.clip(data_bounds).plot(
            color="white", edgecolor="black")
        gdt.clip(data_bounds).plot("cluster", ax=ax)
        return fig


# ----- These functions are run once only for the purpose of obtaining the rural_count dataframe and land_sea geodataframe
# and saving them for future use.------#


    def grid_type(self, fn: str) -> None:
        with rio.open(fn) as dataset:
            # specify band number in the next line. The urban-heat-island tiff file has 2 bands. The values in band 2 are zero for urban
            # and one for rural areas.
            val = dataset.read(2)
            no_data = 999
            data = [(dataset.xy(x, y)[0], dataset.xy(x, y)[1], val[x, y])
                    for x, y in np.ndindex(val.shape) if val[x, y] != no_data]
            lon = [i[0] for i in data]
            lat = [i[1] for i in data]
            d = [i[2] for i in data]
            res = pd.DataFrame({"lon": lon, 'lat': lat, "data": d})
        res.to_pickle("rural_urban.pkl")

    def get_rural_count(self) -> None:
        run = 1
        hour = 0
        param = "\\urban-heat-island_"
        fnp = self.results_dir + str(run) + param + str(hour) + ".tiff"

        if not os.path.exists("rural_urban.pkl"):
            self.grid_type(fnp)
        df = pd.read_pickle("rural_urban.pkl")
        rural_loc = df.loc[df["data"] == 1, [
            "lon", "lat"]].reset_index(drop=True)
        gdf = gpd.GeoDataFrame(
            rural_loc, geometry=gpd.points_from_xy(rural_loc.lon, rural_loc.lat), crs="EPSG:4326"
        )
        fnp = fnp.replace("urban-heat-island", "air-temperature")
        res = self.get_all_data(fnp)
        gdt = gpd.GeoDataFrame(
            res, geometry=gpd.points_from_xy(res.lon, res.lat), crs="EPSG:4326"
        )
        domain_bounds = gdt.to_crs(24500).total_bounds
        grid_res = 300.0
        nx = np.round(
            (domain_bounds[2] - domain_bounds[0])/grid_res).astype(int)
        ny = np.round(
            (domain_bounds[3] - domain_bounds[1])/grid_res).astype(int)
        gdf = self.get_grid_cell(gdf, domain_bounds, nx, grid_res)
        gdt = self.get_grid_cell(gdt, domain_bounds, nx, grid_res)
        s4 = gdf["grid_cell"].value_counts().rename("count")
        final_res = gdt.merge(s4, left_on="grid_cell", right_index=True)
        final_res.drop(final_res[final_res["data"] > 90].index, inplace=True)
        final_res.to_pickle("rural_count.pkl")

    def get_all_data(self, fn: str) -> pd.DataFrame:
        with rio.open(fn) as dataset:
            # specify band number in the next line. The urban-heat-island tiff file has 2 bands. The values in band 2 are zero for urban
            # and one for rural areas.
            val = dataset.read(1)
            no_data = 991
            data = [(dataset.xy(x, y)[0], dataset.xy(x, y)[1], val[x, y])
                    for x, y in np.ndindex(val.shape) if val[x, y] != no_data]
            lon = [i[0] for i in data]
            lat = [i[1] for i in data]
            d = [i[2] for i in data]
            res = pd.DataFrame({"lon": lon, 'lat': lat, "data": d})
        return res

    def get_land_sea(self):
        run = 1
        hour = 0
        param = "\\air-temperature_"
        fnp = self.results_dir + str(run) + param + str(hour) + ".tiff"
        res = self.get_all_data(fnp)
        s1 = res["data"].map(lambda x: 1 if x == 999 else 0)
        res = res.assign(mask=s1)
        gdf = gpd.GeoDataFrame(res, geometry=gpd.points_from_xy(
            res.lon, res.lat), crs="EPSG:4326")
        gdf.drop("data", axis=1)
        gdf.to_file('land_sea.geojson', driver='GeoJSON')

    def get_grid_cell(self, gdf: gpd.GeoDataFrame, domain_bounds: list[float], nx: int, grid_res: float) -> pd.DataFrame:
        s1 = ((gdf.to_crs(24500).geometry.x -
              domain_bounds[0])/grid_res).round().astype(int)
        s2 = ((gdf.to_crs(24500).geometry.y -
              domain_bounds[1])/grid_res).round().astype(int)
        s3 = s1 + s2*nx
        gdf = gdf.assign(grid_cell=s3)
        return gdf

### ---------------- Helper function for renaming all files in directory-----------------------------------#

    def rename_timestamps(self, run: int) -> None:
        fmt = "%Y%m%d%H%M%S"
        path = self.results_dir + str(run) + "\\wrfrun"
        files_list = os.listdir(path=path)
        for file in files_list:
            file_name = str(file)
            i1 = file_name.rfind("_")
            i2 = file_name.find("UTC")
            time_string = file_name[i1 + 1:i2]
            timestamp = datetime.strptime(time_string, fmt)
            tsu = pytz.timezone(
                "Asia/Singapore").localize(timestamp).astimezone(pytz.utc)
            time_string_new = tsu.strftime(fmt)
            ofn = self.results_dir + str(run) + "\\wrfrun\\" + file_name
            nfn = ofn.replace(time_string, time_string_new)
            os.rename(ofn, nfn)
