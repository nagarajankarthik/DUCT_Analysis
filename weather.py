import requests
from datetime import datetime, timedelta
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


class weather:
    """
    Collect data from openmeteo and write it to a dataframe
    """

    def __init__(self):
        pass

    def get_data(self, lat: float = 1.286, lon: float = 103.685,
                 start_date: str = "2016-07-01", end_date: str = "2016-07-31",
                 params: list[str] = ['winddirection_10m', 'windspeed_10m', 'temperature_2m', 'relativehumidity_2m', 'cloudcover']) -> pd.DataFrame:
        payload = {'latitude': str(lat), 'longitude': str(lon), 'start_date': start_date, 'end_date': end_date,
                   'temperature_unit': 'celsius', 'windspeed_unit': 'ms', 'timezone': 'auto',
                   'hourly': params}
        r = requests.get(
            'https://archive-api.open-meteo.com/v1/era5', params=payload)
        res = r.json()
        df = pd.DataFrame(res['hourly'])
        df['time'] = pd.to_datetime(df['time'], format='%Y-%m-%dT%H:%M')
        return df

    def get_temp(self, date: str = "2016-04-18",
                 locations: np.ndarray = np.zeros((0, 2)), fn: str = "openmeteo_temperature.txt") -> np.ndarray:
        """
        Gets timeseries of temperature for the specified date and locations.
        Location coordinates must be specified as [lon, lat] in EPSG:4326 CRS.
        """
        param = 'temperature_2m'
        nl = len(locations)
        res = np.zeros((24, nl))

        for ind, location in enumerate(locations):
            lon = locations[ind][0]
            lat = locations[ind][1]
            df = self.get_data(lat, lon, start_date=date,
                               end_date=date, params=[param])
            df['hour'] = df['time'].dt.hour
            res[:, ind] = df.loc[:, param]
        np.savetxt(fn, res)
        return res

    def get_wind_data(self, req_hours: list[int] = [0, 23]) -> pd.DataFrame:
        """
        Returns wind data for hours specified in req_hours on all days of the month.
        """
        param = 'windspeed_10m'
        df = self.get_data(params=[param])
        df['hour'] = df['time'].dt.hour
        dfh = df[df['hour'].isin(req_hours)]
        dfs = dfh.sort_values(by=[param], ignore_index=False)
        return dfs

    def get_daily_wind_data(self) -> pd.DataFrame:
        """
        Returns average wind speed for each day of the month
        """
        param = 'windspeed_10m'
        df = self.get_data(params=[param])
        df['day'] = df['time'].dt.day
        d1 = df.loc[:, ['day', param]]
        dg = d1.groupby(by=['day']).mean()
        dgs = dg.sort_values(by=[param], ignore_index=False)
        return dgs
