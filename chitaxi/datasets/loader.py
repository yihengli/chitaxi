import os
import pandas as pd
from chitaxi.utils import config
from chitaxi.datasets.cleaner import ChiTaxiFormat
from datetime import date


def get_data_taxi(year=None, start=None, end=None):
    """ Read the data from HDF storage. If year is given, start and end will
    be overwritten

    Args:
        year (int, optional): Defaults to None. e.g. 2016
        start (str, optional): Defaults to None. e.g. '20130101'
        end (str, optional): Defaults to None. e.g. '20161231'

    Returns:
        pandas.DataFrame
    """
    path_hdf = config.get_path_taxi()
    time_col = ChiTaxiFormat().TIME[0].lower().replace(' ', '_')

    if year:
        start = date(year, 1, 1).strftime('%Y%m%d')
        end = date(year, 12, 31).strftime('%Y%m%d')

    if start and end:
        return pd.read_hdf(path_hdf,
                           'table',
                           where='{}>={} & {}<={}'.format(
                               time_col, start, time_col, end))


def get_data_taxi_speed(year=None):
    """ Load the data per year in the fastest speed. The backend is implmented
    by using feather format. If the feather format hasn't been created, this
    function may take some time to initialize it.

    Note: As the yearly data is potentially huge. (4 - 5G), always make sure
    you have enough RAM to read through the data

        year (int, optional): Defaults to None. e.g. 2015
    """
    path_feather = config.get_path_feather_taxi(year)

    if os.path.exists(path_feather):
        return pd.read_feather(path_feather)
    else:
        df = get_data_taxi(year=year)
        df.to_feather(path_feather)
        return df
