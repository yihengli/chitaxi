import pandas as pd
import numpy as np
import datetime
import multiprocessing as mp
from functools import partial
from pandas.tseries.holiday import USFederalHolidayCalendar
from sklearn.model_selection import train_test_split
from chitaxi.datasets.cleaner import ChiTaxiFormat
from chitaxi.datasets import loader
from chitaxi.utils import logger


logger = logger.get_logger()


def check_groupby(f):
    def wrapper(*args, **kargs):
        if args[0].GROUP_BY is None:
            logger.warning("The data hasn't been resampled yet, "
                           "use set_resample first")
            return None
        else:
            return f(*args, **kargs)
    return wrapper


class Builder():
    fmt = ChiTaxiFormat()
    ID = fmt.ID
    METRIC = fmt.METRIC
    TIME = fmt.TIME[0].lower().replace(' ', '_')

    def extract_y(self, data, freq=None):
        """ Extract the labels from the data that is cleaned from
        Outlier.end_to_end_clean(). In addition to aggregated labels, this
        function will also extract labels per freq

        Args:
            data (pandas.DataFrame): cleaned after outlier detections
            freq (str, Optional): frequency per label

        Returns:
            pandas.DataFrame: the first column is the aggregated Y, Y per freq
            afterward
        """
        aggregated = data.groupby([self.ID])[self.METRIC].sum()
        aggregated.name = 'Y_total'
        if freq is None:
            return aggregated
        else:
            temp = data.groupby(
                [self.ID, pd.Grouper(freq=freq, key=self.TIME)])[self.METRIC]\
                .sum().unstack().fillna(0)
            temp.columns = ["Y_" + str(i+1) for i in range(temp.shape[1])]
            return pd.DataFrame(aggregated).join(temp)

    def merge_data(self, X, y, dropna=True):
        """ Both datasets should have taxi_id as their index.
        """
        logger.info("Merging features and labels...")
        original = len(X)
        df = pd.merge(X, y, left_index=True, right_index=True, how='left')
        nans = df.iloc[:, -1].isna().sum()

        logger.info("We have {} unique ids in 2015".format(original))
        logger.info("{} of them ({:.5%}) dropped (nan) in 2016"
                    .format(nans, nans/original))
        if not dropna:
            df.iloc[:, -1].fillna(0, inplace=True)
        else:
            df.dropna(inplace=True)
        return df

    def train_split(self, data, ylabel, testsize=0.2, seed=1, filename=None):
        Y = data[[ylabel]]
        X = data.drop(ylabel, axis=1)

        X_train, X_test, y_train, y_test =\
            train_test_split(X, Y, test_size=testsize, random_state=seed)

        if filename:
            loader.save_as_feather(X_train, filename + "_Xtrain.feather")
            loader.save_as_feather(X_test, filename + "_Xtest.feather")
            loader.save_as_feather(y_train, filename + "_Ytrain.feather")
            loader.save_as_feather(y_test, filename + "_Ytest.feather")

        return X_train, X_test, y_train, y_test


class FeatureExtraction():
    fmt = ChiTaxiFormat()
    ID = fmt.ID
    METRIC = fmt.METRIC
    GROUP_BY = None

    def _workday_process(self, data, holidays):
        # workday=1, Mon-Fri and not a holiday
        data = data.copy()
        data.loc[:, "workday"] = np.zeros(len(data))
        data.loc[:, "workday"] = [1 - int(ts.dayofweek / 5)
                                  for ts in data["trip_start_timestamp"]]
        data.loc[[ts in holidays
                  for ts in data["trip_start_timestamp"]
                  .dt.date], "workday"] = 0
        return data

    def assign_workdays(self, data, start, end, speed=False):
        cal = USFederalHolidayCalendar()
        holidays = cal.holidays(start=start, end=end)
        if speed:
            num_process = mp.cpu_count()
            chunk_size = int(data.shape[0] / num_process)
            chunks = [data.loc[data.index[i: i + chunk_size]]
                      for i in range(0, data.shape[0], chunk_size)]
            pools = mp.Pool(num_process)
            res = pools.map(
                partial(self._workday_process, holidays=holidays), chunks)
            return pd.concat(res)
        else:
            return self._workday_process(data, holidays)

    def assign_daytime(self, data, night_end=6, night_start=19):
        data["daytime"] = np.zeros(len(data))
        data["daytime"] = [int((ts.time() > datetime.time(night_end, 0, 0)) &
                               (ts.time() < datetime.time(night_start, 0, 0)))
                           for ts in data["trip_start_timestamp"]]
        return data

    def set_resample(self, data, freq='m'):
        self.GROUP_BY = data.groupby([
            self.ID, pd.Grouper(key="trip_start_timestamp", freq=freq)])

    @check_groupby
    def get_trip_counts(self):
        data = self.GROUP_BY
        res = data.count().unstack()[self.METRIC]
        res.columns = ["count_" + str(i + 1) for i in range(res.shape[1])]
        res = res.fillna(0)
        return res

    @check_groupby
    def get_trip_total(self):
        data = self.GROUP_BY
        res = data.sum().unstack()[self.METRIC]
        res.columns = ["total_sum_" + str(i + 1) for i in range(res.shape[1])]
        res = res.fillna(0)
        return res

    def get_workday_ratio(self, data, freq):
        by_taxi_workday = data.groupby([self.ID, "workday"])\
            .resample(freq, on="trip_start_timestamp")

        taxi_workday = by_taxi_workday.sum().unstack()[self.METRIC].unstack()
        new_col = [col for cols in
                   [["total_workday0_"+str(m+1), "total_workday1_"+str(m+1)]
                    for m in range(int(taxi_workday.shape[1]/2))]
                   for col in cols]
        taxi_workday.columns = new_col
        taxi_workday = taxi_workday.fillna(0)

        # Convert the trip sum to ratios
        for m in range(int(taxi_workday.shape[1]/2)):
            total = taxi_workday["total_workday0_"+str(m+1)]\
                + taxi_workday["total_workday1_"+str(m+1)]
            taxi_workday["total_workday0_"+str(m+1)] =\
                taxi_workday["total_workday0_"+str(m+1)] / total
            taxi_workday["total_workday1_"+str(m+1)] =\
                taxi_workday["total_workday1_"+str(m+1)] / total

        taxi_workday = taxi_workday.fillna(0)
        return taxi_workday

    def get_daytime_ratio(self, data, freq):
        by_taxi_daytime = data.groupby([self.ID, "daytime"])\
            .resample(freq, on="trip_start_timestamp")

        taxi_daytime = by_taxi_daytime.sum().unstack()[self.METRIC].unstack()
        new_col = [col for cols in
                   [["total_daytime0_"+str(m+1), "total_daytime1_"+str(m+1)]
                    for m in range(int(taxi_daytime.shape[1]/2))]
                   for col in cols]
        taxi_daytime.columns = new_col
        taxi_daytime = taxi_daytime.fillna(0)

        # Convert the trip sum to ratios
        for m in range(int(taxi_daytime.shape[1]/2)):
            total = taxi_daytime["total_daytime0_"+str(m+1)]\
                + taxi_daytime["total_daytime1_"+str(m+1)]
            taxi_daytime["total_daytime0_"+str(m+1)] =\
                taxi_daytime["total_daytime0_"+str(m+1)] / total
            taxi_daytime["total_daytime1_"+str(m+1)] =\
                taxi_daytime["total_daytime1_"+str(m+1)] / total

        taxi_daytime = taxi_daytime.fillna(0)
        return taxi_daytime

    @check_groupby
    def get_quantiles(self):
        data = self.GROUP_BY

        taxi_25p = data[self.METRIC].quantile(0.25).unstack()
        new_col = ["total_25p_"+str(m+1) for m in range(taxi_25p.shape[1])]
        taxi_25p.columns = new_col
        taxi_25p = taxi_25p.fillna(0)

        taxi_50p = data[self.METRIC].quantile(0.5).unstack()
        new_col = ["total_50p_"+str(m+1) for m in range(taxi_25p.shape[1])]
        taxi_50p.columns = new_col
        taxi_50p = taxi_50p.fillna(0)

        taxi_75p = data[self.METRIC].quantile(0.75).unstack()
        new_col = ["total_75p_"+str(m+1) for m in range(taxi_25p.shape[1])]
        taxi_75p.columns = new_col
        taxi_75p = taxi_75p.fillna(0)

        res = taxi_25p.merge(taxi_50p, left_on=self.ID,
                             right_on=self.ID, how="outer")
        res = res.merge(taxi_75p, left_on=self.ID,
                        right_on=self.ID, how="outer")

        taxi_mean = data.mean().unstack()["trip_total"]
        new_col = ["total_mean_"+str(m+1) for m in range(taxi_25p.shape[1])]
        taxi_mean.columns = new_col
        taxi_mean = taxi_mean.fillna(0)
        res = res.merge(taxi_mean, left_on=self.ID,
                        right_on=self.ID, how='outer')
        return res

    def get_break_hours(self, data, freq):
        df = data.groupby(
            [self.ID, pd.Grouper(key='trip_start_timestamp', freq=freq),
             pd.Grouper(key='trip_start_timestamp', freq='h')])[['fare']]\
            .count()
        # Revise the second index to avoid repeated index
        df.index = df.index.rename('time', level=2)

        # Final output format
        res = pd.DataFrame({
            'max_break_hrs': 0,
            'avg_break_hrs': 0
        }, index=df.index)
        res = res.groupby(level=(0, 1)).sum()

        # Loop through each taxi, each month's data, sort dates, get diff
        for n1, df1 in df.groupby(level=(0, 1)):
            df1 = df1.reset_index(level=2).sort_values('time').diff()
            maxi = np.max(df1['time'])
            avg = np.mean(df1['time'])
            # For only one trip month, assume break is half month
            # For no trip month, assume 30 days break
            if maxi != np.datetime64('NaT'):
                res.loc[n1]['max_break_hrs'] =\
                    maxi.days * 24 + maxi.seconds // 3600
                res.loc[n1]['avg_break_hrs'] =\
                    avg.days * 24 + avg.seconds // 3600
            elif len(df1['time']) == 1:
                res.loc[n1]['max_break_hrs'] = 15 * 24
                res.loc[n1]['avg_break_hrs'] = 15 * 24
            else:
                res.loc[n1]['max_break_hrs'] = 30 * 24
                res.loc[n1]['avg_break_hrs'] = 30 * 24

        # Format the final result
        res = res.unstack().fillna(30*24)
        new_col = ["max_break_hrs_"+str(m+1) for m in range(res.shape[1]//2)]
        new_col += ["avg_break_hrs_"+str(m+1) for m in range(res.shape[1]//2)]
        res.columns = new_col

        return res

    @check_groupby
    def get_trip_seconds(self):
        taxi_seconds = self.GROUP_BY.mean().unstack()["trip_seconds"]
        new_col = ["seconds_mean_"+str(m+1)
                   for m in range(taxi_seconds.shape[1])]
        taxi_seconds.columns = new_col
        taxi_seconds = taxi_seconds.fillna(0)

        return taxi_seconds

    @check_groupby
    def get_trip_miles(self):
        taxi_miles = self.GROUP_BY.mean().unstack()["trip_miles"]
        new_col = ["miles_mean_"+str(m+1)
                   for m in range(taxi_miles.shape[1])]
        taxi_miles.columns = new_col
        taxi_miles = taxi_miles.fillna(0)

        return taxi_miles
