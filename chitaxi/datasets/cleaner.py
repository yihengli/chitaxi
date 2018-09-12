import os
import pandas as pd
import numpy as np
from chitaxi.utils import config


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
    # TODO: Use a logger instead of print all the items

    print('Reading File {}'.format(path))
    cols = ChiTaxiFormat()

    df = pd.read_csv(path, usecols=cols.ALL)

    # Cerate datetime values
    df[cols.TIME[0]] = pd.to_datetime(df[cols.TIME[0]],
                                      format='%m/%d/%Y %H:%M:%S %p')
    df[cols.TIME[1]] = pd.to_datetime(df[cols.TIME[1]],
                                      format='%m/%d/%Y %H:%M:%S %p')
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

    print('Clean Done')
    df.to_hdf(os.path.join(config.get_config()['data'], h5),
              'table',
              append=True,
              format='table',
              data_columns=cols.COLS)

    print('HDF converted')
