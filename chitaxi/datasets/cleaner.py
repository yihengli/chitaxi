import os
import pandas as pd
import numpy as np
import warnings
import time
from scipy import stats
from chitaxi.utils import logger
from chitaxi.utils import config


logger = logger.get_logger()


class ChiTaxiFormat():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    MAPPER = os.path.join(dir_path, 'column_remapping.json')

    ALL = [
        'Taxi ID',
        'Trip Start Timestamp',
        'Trip End Timestamp',
        'Trip Seconds',
        'Trip Miles',
        'Fare',
        'Tips',
        'Tolls',
        'Extras',
        'Trip Total',
        'Payment Type',
        'Company',
        'Pickup Centroid Latitude',
        'Pickup Centroid Longitude',
        'Dropoff Centroid Latitude',
        'Dropoff Centroid Longitude'
    ]

    COLS = [col.lower().replace(' ', '_') for col in ALL]

    METRIC = 'trip_total'
    ID = 'taxi_id'

    TIME = [
        'Trip Start Timestamp',
        'Trip End Timestamp'
    ]

    NUMBERS = [
        'Fare',
        'Tips',
        'Tolls',
        'Extras',
        'Trip Total'
    ]

    CATS = [
        'Payment Type',
        'Company'
    ]

    GEO = [
        'Pickup Centroid Latitude',
        'Pickup Centroid Longitude',
        'Dropoff Centroid Latitude',
        'Dropoff Centroid Longitude'
    ]


def clean_chitax_csv(path, h5=config.get_path_taxi(), path_mapper=None):
    """ Clean the data from the original taxi CSV files

    Args:
        path ([type]): [description]
        h5 ([type], optional): Defaults to config.get_path_taxi().
        path_mapper ([type], optional): Defaults to None. [description]
    """
    logger.info('Reading File {}'.format(path))
    cols = ChiTaxiFormat()

    df = pd.read_csv(path, usecols=cols.ALL)

    # Cerate datetime values
    df[cols.TIME[0]] = pd.to_datetime(df[cols.TIME[0]],
                                      format='%m/%d/%Y %I:%M:%S %p')
    df[cols.TIME[1]] = pd.to_datetime(df[cols.TIME[1]],
                                      format='%m/%d/%Y %I:%M:%S %p')
    # Clean numerical dollar values
    df[cols.NUMBERS] = df[cols.NUMBERS].apply(
        lambda col: col.str.replace('$', '').astype(np.float), axis=0)

    if not path_mapper:
        path_mapper = cols.MAPPER
    ids = pd.read_json(path_mapper)[['taxi_id']].reset_index()

    df = pd.merge(ids, df, left_on='taxi_id', right_on='Taxi ID')
    df.drop(['Taxi ID', 'taxi_id'], axis=1, inplace=True)
    df[cols.CATS].astype('category')
    df.columns = cols.COLS

    logger.info('Finished data cleaning, now switched to saving HDF5')
    h5_path = os.path.join(config.get_config()['data'], h5)
    if not os.path.exists(h5_path):
        df.to_hdf(h5_path,
                  'table',
                  append=True,
                  format='table',
                  data_columns=cols.COLS,
                  min_itemsize={'company': 50})
    else:
        logger.info("HD5 Exists, we are appeding to the existing h5")
        store = pd.HDFStore(os.path.join(config.get_config()['data'], h5))
        # Specifically using append method since each dataset has uneven
        # company column strings size
        store.append('table',
                     df,
                     data_columns=cols.COLS,
                     min_itemsize={'company': 50})
    logger.info('HDF converted')


def haversine_np(lon1, lat1, lon2, lat2):
    """ Calculate mile distance given pickup and dropoff latitude and longitude
    for any NA given, return 0

    Args:
        lat1 (numpy.array): [description]
        lon1 (numpy.array): [description]
        lat2 (numpy.array): [description]
        lon2 (numpy.array): [description]

    Returns:
        numpy.array: distance in miles
    """
    lon1, lat1, lon2, lat2 = map(np.deg2rad, [lon1, lat1, lon2, lat2])

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2

    c = np.arcsin(np.sqrt(a))
    miles = 7918 * c
    return np.nan_to_num(miles)


def haversine_np_df(data):
    """ Mile distance calculations, but can directly apply on a dataframe

    Args:
        data (pandas.DataFrame): [description]

    Returns:
        numpy.array
    """
    geo = [geo.lower().replace(' ', '_') for geo in ChiTaxiFormat().GEO]
    coordinates = data[geo].values
    lat1 = coordinates[:, 0]
    lng1 = coordinates[:, 1]
    lat2 = coordinates[:, 2]
    lng2 = coordinates[:, 3]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return haversine_np(lng1, lat1, lng2, lat2)


def report_details(f):
    """ Decorator that will report the drop ratio etc.

    Args:
        msg (str, optional): Defaults to None. Notification of the type
            of outlier removing function
    """
    def wrapper(*args, **kargs):
        if len(args) >= 2:
            data = args[1]
        elif 'data' in kargs.keys():
            data = kargs['data']

        ori_len = len(data)
        logger.info("Orignal - Rows: {}".format(ori_len))
        data = f(*args, **kargs)

        logger.info("After Drroping - Rows: {}".format(len(data)))
        logger.info("Drop Ratio: {0:.5%}".format(1 - len(data) / ori_len))
        return data
    return wrapper


class Outlier():
    """ A set of tools to detect outliers from chicago taxi dataset
    """

    @report_details
    def drop_duplicates(self, data, info='duplications'):
        fmt = ChiTaxiFormat()
        logger.info("Dropping duplications")

        return data.drop_duplicates(subset=fmt.COLS, keep=False)

    @report_details
    def drop_zero_values(self, data, metric):
        logger.info("Dropping 0 value {}".format(metric))

        index = data.index[(data[metric] == 0) |
                           (data[metric].apply(pd.isnull))]
        return data.drop(index)

    @report_details
    def drop_at_threshhold(self, data, metric, thresh, islarger=True):
        if islarger:
            logger.info("Dropping any {} more than {}".format(metric, thresh))
            index = data.index[data[metric] >= thresh]
        else:
            logger.info("Dropping any {} less than {}".format(metric, thresh))
            index = data.index[data[metric] <= thresh]

        return data.drop(index)

    @report_details
    def drop_null_values(self, data, metric):
        logger.info("Dropping NA values at {}".format(metric))

        index = data.index[data[metric].apply(pd.isnull)]
        return data.drop(index)

    def model_fares_outliers(self, data, rules=[]):
        """

        Args:
            data ([type]): [description]
            rules (dict, optional): Defaults to {}. Sample format is
                [{"thresh": 200, "values": ("abs_fare_diff", 50)},
                 {"thresh": np.inf, "values": ("abs_pct_diff", 1)}]
                This means for trip_total ranges from 0 - 200, fare_diff > 50
                will be marked as outliers. For trip_total 200+, abs_pct_diff>1
                will be marked as outliers.

        Returns:
            pandas.DataFrame: Additional columns such as 
            * modeled price fares,
            * the diffrence between modeled price and actual price
            * the absolute percentage diffrence between two prices
            * marked as outliers if satisfied the rules
        """

        # calculate the distance based on pickup and dropoff location as a
        # reference in some cases, trip_miles is not acurate.
        data["model_miles"] = haversine_np_df(data)

        # calculate a fare, based on maximum of trip_miles and model_miles.
        data["model_fare"] = 3.25 + 2.25 * data[["trip_miles", "model_miles"]]\
            .max(axis=1)+0.25*data["trip_seconds"]/36 + 3.5

        # fare difference and absolute percent change, will be used to check
        # outlier later
        data["abs_fare_diff"] = abs(data["fare"]-data["model_fare"])
        data["abs_pct_diff"] = abs(data["abs_fare_diff"]/data["model_fare"])

        data['Model_outliers'] = 0
        for i, rule in enumerate(rules):
            outliers = data[(data['trip_total'] <= rule["thresh"]) &
                            (data[rule["values"][0]] >= rule["values"][1])]
            logger.info(
                "{0:.5%} satisfied rules {1}"
                .format(len(outliers)/len(data), i))
            data.loc[outliers.index, 'Model_outliers'] = 1

        return data

    def get_summary(self, data, time_low, time_high, IQR=1.5):
        line = {}

        trip_total = data["trip_total"]
        quan = np.log(trip_total).quantile([0, 0.25, 0.5, 0.75, 1])
        iqr = stats.iqr(np.log(trip_total).values)
        if time_low >= 3600:
            fmt = '%H hr'
        else:
            fmt = '%M min'
        line['time'] = ' ~ '.join((
            time.strftime(fmt, time.gmtime(time_low)),
            time.strftime(fmt, time.gmtime(time_high))))
        line['lower'] = np.exp(quan[0.25] - IQR * iqr)
        line['upper'] = np.exp(quan[0.75] + IQR * iqr)

        line['avg_fare'] = np.mean(data["fare"])
        line['avg_model_fare'] = np.mean(data["model_fare"])

        to_drop = data[(data["trip_total"] < line['lower']) |
                       (data["trip_total"] > line['upper'])]

        logger.info("{}: {:.5%} are outside IQR range"
                    .format(line['time'], len(to_drop)/len(data)))

        return line, list(to_drop.index)

    def time_range_outliers(self, data, IQR=1.5):
        lines = []
        drop_index_all = []

        # Within 1 hour, check every 10 minutes
        for i in range(6):
            slice_data = data[(data["trip_seconds"] >= i*600) &
                              (data["trip_seconds"] < (i+1)*600)]
            line, drop_index = self.get_summary(slice_data,
                                                i*600,
                                                (i+1)*600,
                                                IQR)
            lines.append(line)
            drop_index_all += drop_index

        # After first hour, go by hourly
        for i in range(1, int(data["trip_seconds"].max()/3600)+1):
            slice_data = data[(data["trip_seconds"] >= i*3600) &
                              (data["trip_seconds"] < (i+1)*3600)]
            line, drop_index = self.get_summary(slice_data,
                                                i*3600,
                                                (i+1)*3600,
                                                IQR)
            lines.append(line)
            drop_index_all += drop_index

        data["IQR_outlier"] = np.zeros(len(data))
        data.loc[drop_index_all, "IQR_outlier"] = 1

        report = pd.DataFrame(lines)
        return data, report

    @report_details
    def drop_outliers_one(self, data, cols=['Model_outliers', 'IQR_outlier']):
        outliers = data[cols][data[cols] == 1].dropna().index
        return data.drop(outliers)
