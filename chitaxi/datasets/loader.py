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
