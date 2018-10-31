import pandas as pd
from analyze_panel_data.utils import panel_to_multiindex
import numpy as np


def split_panel_into_data_and_target_and_fill_missing(
        panel_with_time_as_major_axis, fill_missing_value=0.0,
        num_lags=1, lag_label='lag', target_is_difference=False,
        as_sequence=False):
    """Compute data and target dataframes from a panel dataset that has time
    on the major_axis.

    A new level is added to the columns that indicates the lag of that
    quantity. For example, if the lag is 1, then the value in that cell is
    the quantity 1 time step ago. We drop (item, major_axis) pairs with
    missing values along the entire minor_axis by dropping rows from the
    MultiIndex DataFrame that contain all missing values.

    Parameters
    ----------
    panel_with_time_as_major_axis : pandas Panel
        Each item in the 'items' axis is a different timeseries. Time is on the
        major_axis.

    num_lags : int, optional, default: 1
        The number of time steps in the past to put in the dataframe `data`.
        For example, num_lags=1 means predicting time step t from time step
        t - 1.

    fill_missing_value : float, default: 0.0
            The value to use to fill missing values.

    lag_label : str, optional, default 'lag'
        The name of the new level in the column for the integer lag.

    target_is_difference : bool, optional, default: False
        Whether to make the `target` the change from the previous time step
        to the next.

    as_sequence : bool, default: False
        Whether the data should have shape
        `(num_samples, num_lags, num_features)` instead of shape
        `(num_samples, num_lags * num_features)`.

    Returns
    -------
    data, target : MultiIndex DataFrames of the same length
        The data and target to be used in regression. The columns of target are
        the minor_axis of the panel, with a new level added with name
        `lag_label` that encodes the shift in time of the values in that
        column.
    """
    return (
        panel_with_time_as_major_axis
        .pipe(panel_to_multiindex)
        .dropna(how='all', axis=0)
        .fillna(fill_missing_value)
        .pipe(split_multiindex_dataframe_into_data_target, num_lags,
              as_sequence=as_sequence,
              lag_label=lag_label, target_is_difference=target_is_difference))


def split_panel_into_data_and_target_dataframes(
        panel_with_time_as_major_axis, as_sequence=False,
        num_lags=1, lag_label='lag', target_is_difference=False):
    """Compute data and target dataframes from a panel dataset that has time
    on the major_axis.

    A new level is added to the columns that indicates the lag of that
    quantity. For example, if the lag is 1, then the value in that cell is
    the quantity 1 time step ago. We drop (item, major_axis) pairs with
    missing values along the entire minor_axis by dropping rows from the
    MultiIndex DataFrame that contain all missing values.

    Parameters
    ----------
    panel_with_time_as_major_axis : pandas Panel
        Each item in the 'items' axis is a different timeseries. Time is on the
        major_axis.

    num_lags : int, optional, default: 1
        The number of time steps in the past to put in the dataframe `data`.
        For example, num_lags=1 means predicting time step t from time step
        t - 1.

    lag_label : str, optional, default 'lag'
        The name of the new level in the column for the integer lag.

    target_is_difference : bool, optional, default: False
        Whether to make the `target` the change from the previous time step
        to the next.

    as_sequence : bool, default: False
        Whether the data should have shape
        `(num_samples, num_lags, num_features)` instead of shape
        `(num_samples, num_lags * num_features)`.

    Returns
    -------
    data, target : MultiIndex DataFrames of the same length
        The data and target to be used in regression. The columns of target are
        the minor_axis of the panel, with a new level added with name
        `lag_label` that encodes the shift in time of the values in that
        column.
    """
    return (
        panel_with_time_as_major_axis
        .pipe(panel_to_multiindex)
        .dropna(how='all', axis=0)
        .pipe(split_multiindex_dataframe_into_data_target, num_lags,
              as_sequence=as_sequence,
              lag_label=lag_label, target_is_difference=target_is_difference))


def split_multiindex_dataframe_into_data_target(
        midx_dataframe, num_lags, lag_label='lag', target_is_difference=False,
        as_sequence=False):
    """Split a MultiIndex DataFrame into data and target dataframes by shifting
    the given dataframe a certain number of lags into the past.

    The given dataframe is shifted by 1, 2, ..., num_lags many lags to form the
    `data` dataframe and by 0 lags to form the target data. These lags are
    recoreded in a new level in the columns of the resulting dataframes.
    """
    df_grouped_by_items = midx_dataframe.groupby(level=0)

    data = shift_grouped_dataframes_by_lags(
        df_grouped_by_items, range(1, num_lags + 1),
        num_lags, lag_label=lag_label)

    target = shift_grouped_dataframes_by_lags(
        df_grouped_by_items, [0], num_lags, lag_label)

    if target_is_difference:
        target = subtract_lag_1_data_from_target(data, target)
    if as_sequence:
        data = np.stack(
            [data.loc[:, lag] for lag in range(1, num_lags + 1)], axis=1)
    return data, target


def subtract_lag_1_data_from_target(data, target):
    target_new = pd.DataFrame(target.values - data.loc[:, 1].values)
    target_new.index = target.index
    target_new.columns = target.columns
    return target_new


def shift_grouped_dataframes_by_lags(
        grouped_dataframes, lags, num_lags, lag_label='lag'):
    return pd.concat(
        [(shift_timeseries_by_lags(sub_dataframe, lags, lag_label=lag_label)
            .iloc[num_lags:])
         for __, sub_dataframe in grouped_dataframes],
        axis=0)


def shift_timeseries_by_lags(df, lags, lag_label='lag'):
    """Shift a pandas DataFrame by certain numbers of lags.

    The index is kept the same; a new dimension in the columns is added with
    label given by `lag_label` that specifies how many time steps back in time
    the value in this cell corresponds to.

    Parameters
    ----------
    df : pandas DataFrame
        The index of `df` is the time axis

    lags : list of ints
        The number of lags by which to shift the time-series.

    lag_label : str, optional, default: 'lag'
        The label to put in the MultiIndex columns

    Returns
    -------
    df_shifted : pandas DataFrame
        The DataFrame `df` lagged by the amounts in `lags`.  The DataFrame has
        a new dimension in the columns containing the contents of `lags`,
        and it's labeled `lag_label`.
    """
    if len(lags) == 0:
        return df
    else:
        df_shifted = pd.concat({lag: df.shift(lag) for lag in lags}, axis=1)
        column_names = list(df_shifted.columns.names.copy())
        column_names[0] = lag_label
        df_shifted = df_shifted.rename_axis(column_names, axis=1)
        return df_shifted
