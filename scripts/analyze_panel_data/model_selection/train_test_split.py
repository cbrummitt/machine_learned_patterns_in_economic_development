"""Train-test splits of panel datasets.

Functions:
train_test_split_of_panels_items_and_major_axes
random_train_test_split_of_panels
sequential_train_test_split_of_panels
"""
#   Charlie Brummitt <brummitt@gmail.com> Github: cbrummitt
#   Andres Gomez Lievano <andres_gomez@hks.harvard.edu>
from sklearn.model_selection import train_test_split
from sklearn.model_selection._split import _validate_shuffle_split
from analyze_panel_data.utils import panel_to_multiindex
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import load_data.synthetic_exports_data as synthetic_exports_data
import warnings
from sklearn.externals.six import with_metaclass
from abc import ABCMeta, abstractmethod
from sklearn.model_selection._split import BaseCrossValidator
import numbers


def train_test_split_of_panels_items_and_major_axes(*panels, **kwargs):
    """Create a random train-test split of pandas Panels on the first two axes.

    This method of splitting into train and test sets assumes that observations
    defined as (item, major_axis) pairs are independent.

    Parameters
    ----------
    *panels : sequence of panels with the same shape[0] and same shape[1].
        The unit of observation is an item in a certain point in time.

    test_size : float, int, or None (default is None)
        If float, should be between 0.0 and 1.0 and represent the
        proportion of the dataset to include in the test split. If
        int, represents the absolute number of test samples. If None,
        the value is automatically set to the complement of the train size.
        If train size is also None, test size is set to 0.25.

    train_size : float, int, or None (default is None)
        If float, should be between 0.0 and 1.0 and represent the
        proportion of the dataset to include in the train split. If
        int, represents the absolute number of train samples. If None,
        the value is automatically set to the complement of the test size.

    random_state : int or RandomState
        Pseudo-random number generator state used for random sampling.

    Returns
    -------
    splitting : list of DataFrames, length = 2 * len(panels)
        List containing train-test splits of the given panels.

    Notes
    -----
    If the panel has some (item, major_axis) pairs with missing values along
    the entire minor_axis, then the multiindex dataframes will have rows of
    all missing values.
    """
    multiindex_dataframes = map(panel_to_multiindex, panels)
    return train_test_split(*multiindex_dataframes, **kwargs)


def random_train_test_split_of_panels(*panels, **kwargs):
    """Create a random train-test split on pandas Panels along the axis
    specified by the keyword argument 'axis'.

    Parameters
    ----------
    *panels : sequence of panels with the same length along the axis specified
    by the keyword argument 'axis'

    axis : {items (0), major_axis (1), minor_axis (2)}
        The axis along which to split the panel. If axis is not given as a
        keyword argument, then the default is to split on the 'items' axis.

    test_size : float, int, or None (default is None)
        If float, should be between 0.0 and 1.0 and represent the
        proportion of the dataset to include in the test split. If
        int, represents the absolute number of test samples. If None,
        the value is automatically set to the complement of the train size.
        If train size is also None, test size is set to 0.25.

    train_size : float, int, or None (default is None)
        If float, should be between 0.0 and 1.0 and represent the
        proportion of the dataset to include in the train split. If
        int, represents the absolute number of train samples. If None,
        the value is automatically set to the complement of the test size.

    random_state : int or RandomState
        Pseudo-random number generator state used for random sampling.

    Returns
    -------
    splitting : list of DataFrames, length = 2 * len(panels)
        List containing train-test splits of the given panels, converted to
        MultiIndex DataFrames.
    """
    tr = transpose_panel_items_with_axis(kwargs.pop('axis', 'items'))
    splitting = train_test_split(*map(tr, panels), **kwargs)

    return [panel_to_multiindex(tr(panel)) for panel in splitting]


def sequential_train_test_split_of_panels(*panels, **kwargs):
    """Return a sequential train-test split of pandas Panels along the axis
    specified by the keyword argument 'axis'. The training set is the first
    n_train samples; the test_set is the remaining remaining samples.

    Parameters
    ----------

    *panels : sequence of panels with the same length along the axis specified
    by the keyword argument 'axis'

    axis : {items (0), major_axis (1), minor_axis (2)}
        The axis along which to split the panel. If axis is not given as a
        keyword argument, then the default is to split on the 'items' axis.

    test_size : float, int, or None (default is None)
        If float, should be between 0.0 and 1.0 and represent the
        proportion of the dataset to include in the test split. If
        int, represents the absolute number of test samples. If None,
        the value is automatically set to the complement of the train size.
        If train size is also None, test size is set to 0.25.

    train_size : float, int, or None (default is None)
        If float, should be between 0.0 and 1.0 and represent the
        proportion of the dataset to include in the train split. If
        int, represents the absolute number of train samples. If None,
        the value is automatically set to the complement of the test size.

    Returns
    -------
    splitting : list of DataFrames, length = 2 * len(panels)
        List containing train-test splits of the given panels. The training set
        contains the first n_train samples, while the test set contains the
        remaining samples, where n_train is computed from train_size and
        test_size as described above. The resulting Panels are converted to
        MultiIndexDataFrames.

    See also
    --------
    sklearn.model_selection.TimeSeriesSplit
        Variant of k-fold cross validation that uses nested training sets
        that contain successively more training samples in time.
    """
    tr = transpose_panel_items_with_axis(kwargs.pop('axis', 'items'))

    transposed_panels = [tr(panel) for panel in panels]

    n_samples = transposed_panels[0].shape[0]
    train_size = kwargs.pop('train_size', None)
    test_size = kwargs.pop('test_size', None)

    if test_size is None and train_size is None:
        test_size = 0.25

    n_train, n_test = _validate_shuffle_split(n_samples,
                                              test_size, train_size)

    splitting = [
        [transposed_panel.iloc[:n_train],
         transposed_panel.iloc[n_train:]]
        for transposed_panel in transposed_panels]

    splitting_flattened = [panel_to_multiindex(tr(panel))
                           for train_test_pair in splitting
                           for panel in train_test_pair]

    return splitting_flattened


def transpose_panel_items_with_axis(axis_to_swap_with_items):
    """Given an axis_to_swap_with_items in ['items', 'major_axis', 'minor_axis']
    or in [0, 1, 2], return a function that transposes a panel so that the
    'items' axis is swapped with the the axis `axis_to_swap_with_items`.
    """
    panel_axes = ['items', 'major_axis', 'minor_axis']

    if axis_to_swap_with_items in [0, 1, 2]:
        index_axis_to_swap_with_items = axis_to_swap_with_items
    else:
        index_axis_to_swap_with_items = (
            panel_axes.index(axis_to_swap_with_items))

    order_axes = permute_by_indices(panel_axes,
                                    (0, index_axis_to_swap_with_items))

    def tr(panel):
        return panel.transpose(*order_axes)

    return tr


def permute_by_indices(list_of_things, *list_of_index_transpositions):
    """Given a list_of_things and a list of pairs of transpositions of indices
    [(i, j), (k, m), ...], return the list_of_things with the i-th an j-th
    values swapped, the k-th- and m-th values swapped, and so on.

    Examples
    --------
    >>> permute_by_indices(['a', 'b', 'c'], [(0, 1)])
    ['b', 'a', 'c']

    >>> permute_by_indices(['a', 'b', 'c'], [(0, 2), (1, 2)])
    ['c', 'a', 'b']
    """
    result = list_of_things
    for i, j in list_of_index_transpositions:
        result[j], result[i] = result[i], result[j]
    return result


class _BaseTimeSeriesSplit(with_metaclass(ABCMeta, BaseCrossValidator)):
    """Base class for KFold, GroupKFold, and StratifiedKFold"""

    @abstractmethod
    def __init__(self, n_splits, shuffle, random_state):
        if not isinstance(n_splits, numbers.Integral):
            raise ValueError('The number of folds must be of Integral type. '
                             '%s of type %s was passed.'
                             % (n_splits, type(n_splits)))
        n_splits = int(n_splits)

        if n_splits < 1:
            raise ValueError(
                "k-fold cross-validation requires at least one"
                " train/test split by setting n_splits=1 or more,"
                " got n_splits={0}.".format(n_splits))

        if not isinstance(shuffle, bool):
            raise TypeError("shuffle must be True or False;"
                            " got {0}".format(shuffle))

        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

    def get_n_splits(self, X=None, y=None, groups=None):
        """Returns the number of splitting iterations in the cross-validator

        Parameters
        ----------
        X : object
            Always ignored, exists for compatibility.

        y : object
            Always ignored, exists for compatibility.

        groups : object
            Always ignored, exists for compatibility.

        Returns
        -------
        n_splits : int
            Returns the number of splitting iterations in the cross-validator.
        """
        return self.n_splits


class MultiTimeSeriesSplit(_BaseTimeSeriesSplit):
    """Generalize `sklearn.model_selection.TimeSeriesSplit` to account for
    multiple trajectories with overlapping times.

    Parameters
    ----------
    n_train_test_sets : int, default: 3
        Number of train/test sets . Must be at least 1.

    first_split_fraction : float between 0 and 1 exclusive or `None` (default)
        The size of the first training set expressed as a fraction of the
        size of the training set. If None, then all the splits have the
        same size.

    level_of_index_for_time_values : int or string or None, default: None
        The level of the index of the time-series to use as the times.
        If the data `X` given to `split()` is a MultiIndex DataFrame, and if
        `level_of_index_for_time_values` is not None, and if `times` is not
        given to the constructor, then the level specified
        by this string or int is used as the times.

    times : None or list of times of length n_samples, containing int,
        float, pandas Periods, pandas TimeStamps; optional, default : None
            The time stamp of each sample by which to split the data `X`.

    verbose : 0 or 1
        Whether to print messages that confirm that the level for times was
        used.
    """

    def __init__(self, n_train_test_sets=3, first_split_fraction=None,
                 times=None,
                 level_of_index_for_time_values=None, verbose=0):
        if n_train_test_sets < 1:
            raise ValueError('The number of train/test sets must be >= 1; '
                             'got {}.'.format(n_train_test_sets))
        # The data is split into `n_split + 1` many folds, so that there are
        # `n_split` many train-test splits. `sklearn.model_section._BaseKFold`
        # expects the number of folds in its constructor.
        super(MultiTimeSeriesSplit, self).__init__(
            n_splits=n_train_test_sets, shuffle=False, random_state=None)
        self.n_train_test_sets = n_train_test_sets
        self.first_split_fraction = first_split_fraction
        self.level_of_index_for_time_values = level_of_index_for_time_values
        filename_template = ('MTSS_{n_train_test_sets}train_test_sets'
                             '_{first_split_fraction}firstfraction')
        self.filename = (
            filename_template.format(
                n_train_test_sets=n_train_test_sets,
                first_split_fraction=first_split_fraction)).replace('.', 'p')
        self.verbose = verbose
        self.times = times

    def split(self, X, y=None, times=None, drop_empty=True):
        """Split the data and target into successively larger nested splits
        according to their times.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The design matrix. `X` can be a Pandas DataFrame with an index
            or MultiIndex that can provide the times.

        y : array-like, shape (n_samples, n_output_features), optional

        times : None or time index, optional, default: None
            List of times of the samples by which to split them. If not None,
            then `self.times` is set to this `times`.

        drop_empty : bool, default: True
            Whether to yield emtpy train-test splits in which one or both of
            the train, test set (likely the test set) is empty. This can
            happen if the distribution of times is pathological.

        Yields
        ------
        train_indices, test_indices : lists of indices
            The indices of the samples in the train and test sets.

        Notes
        -----
        If `times` was not provided to the constructor, then the times are
        attempted to be obtained from `X` as follows. (a) If X has a MultiIndex
        index and we know from the constructor's argument
        `level_of_index_for_time_values` which level of the index is the times,
        then use that level's values. (b) If X has an index, then use that as
        the times. (c) Otherwise assume that the times are `0`, `1`, ...,
        `X.shape[1]`.

        Warnings
        --------
        If the data `X` is not a MultiIndex DataFrame and if a non-None keyword
        arguemnt was given for `level_of_index_for_time_values` upon
        constructing `MultiTimeSeriesSplit`, then a warning is raised to
        alert the user that they did not pass a design matrix `X` with an
        index.

        If `drop_empty` is True and one of the train-test splits has an empty
        set, then a warning is raised indicating that one of the train-test
        splits is being skipped.

        Raises
        ------
        A ValueError is raised if times was provided in the constructor
        and `split` was given an `X` with length different from `len(times)`.
        """
        n_samples = len(X)
        indices = np.arange(n_samples)

        if self.verbose:
            print('\n\n\n STARTING `split` METHOD OF MTSS\n\n')
            print('Given input to split: times =', times)

        if times is not None:
            if self.verbose:
                print('`split` was given times as input; using that.')
            self.times = times

        if self.verbose:
            print('len(X)  =', len(X))
            print('self.times is None:', self.times is None)
            if self.times is not None:
                print('len(self.times) = ', len(self.times))

        if self.times is None or len(self.times) != len(X):
            if hasattr(X, 'index'):
                X_is_MultiIndex = hasattr(X.index, 'levels')
                know_level_of_times = bool(self.level_of_index_for_time_values)
                if X_is_MultiIndex and know_level_of_times:
                    if self.verbose:
                        print('Getting time values from level {}'.format(
                            self.level_of_index_for_time_values))
                    if self.verbose:
                        print('Executing `self.times = '
                              'X.index.get_level_values`')
                    self.times = X.index.get_level_values(
                        self.level_of_index_for_time_values)
                elif X_is_MultiIndex and not know_level_of_times:
                    self.times = indices
                else:
                    if self.verbose:
                        print('Executing `self.times = X.index`')
                    self.times = X.index
            else:
                if self.level_of_index_for_time_values:
                    warnings.warn('A level of the index corresponding to time '
                                  'values was given, but the given data `X` '
                                  'does not have an `index`. Taking the times '
                                  'to be the indices 0, 1, ..., `len(X) - 1`.')
                else:
                    warnings.warn('Setting the `times` attribute to '
                                  '0, 1, ..., `len(X) - 1`.')
                if self.verbose:
                    print('Warning: Setting `self.times` to '
                          '0, 1, ..., `len(X) - 1`.')
                self.times = indices

        if self.verbose:
            print('after the big if statement, self.times has length',
                  len(self.times))
            print('and it equals ', self.times)

        n_folds = self.n_train_test_sets + 1
        if n_folds > n_samples:
            raise ValueError(
                ("Cannot have number of folds = {folds} "
                 "(= n_train_test_sets + 1) greater than the "
                 "number of samples: {samp}.").format(
                    folds=n_folds, samp=n_samples))

        quantiles = compute_list_of_quantiles(n_folds,
                                              self.first_split_fraction)
        samples_to_quantile_buckets = qcut_of_time_index(self.times, quantiles)
        if self.verbose:
            print('just made samples_to_quantile_buckets; its shape is',
                  samples_to_quantile_buckets.shape)

        samples_to_quantile_buckets.cat.rename_categories(
            np.arange(n_folds), inplace=True)

        buckets = samples_to_quantile_buckets.cat.categories

        if self.verbose:
            print('samples_to_quantile_buckets.shape = ',
                  samples_to_quantile_buckets.shape)
            print('len(indices) = ', len(indices))

        for test_bucket in buckets[1:]:
            in_train = (samples_to_quantile_buckets.values < test_bucket)
            in_test = (samples_to_quantile_buckets.values == test_bucket)
            if self.verbose:
                print('indices[in_train] = ', indices[in_train])
                print('indices[in_test] = ', indices[in_test])
            if not drop_empty or (sum(in_train) > 0 and sum(in_test) > 0):
                yield (indices[in_train], indices[in_test])
            else:
                msg = 'Skipping train-test split number {num}'.format(
                    num=test_bucket)
                if sum(in_train) == 0 and sum(in_test) == 0:
                    msg += ' because the train and test splits are both empty.'
                elif sum(in_train) == 0 and sum(in_test) > 0:
                    msg += ' because the training set is empty.'
                else:
                    msg += ' because the test set is empty.'
                warnings.warn(msg)


def compute_list_of_quantiles(n_folds, fraction_in_first_split=None):
    """Split the unit interval into `n_folds` segments and return a list
    of the form [0, x_1, x_2, ..., 1].

    If fraction_in_first_split is None, then split the unit interval into
    segments of equal size. If fraction_in_first_split is not None, then
    the first segment is fraction_in_first_split, and the interval
    (fraction_in_first_split, 1] is split into equal-size pieces."""
    if fraction_in_first_split is None:
        quantiles = list(
            np.linspace(0, 1.0, num=n_folds + 1))
    else:
        if not (0 <= fraction_in_first_split <= 1):
            raise ValueError(
                '`fraction_in_first_split` must be in the unit interval; '
                'got {}'.format(fraction_in_first_split))
        quantiles = (
            [0] +
            list(np.linspace(fraction_in_first_split, 1.0, num=n_folds)))
    return quantiles


def qcut_of_time_index(time_index, quantiles):
    """Compute quantiles of a pandas Index that may be a PeriodIndex,
    and return the assignment of samples to buckets as a Series.
    """
    min_time = min(time_index)
    relative_times = pd.Series(
        [time - min_time for time in time_index],
        index=time_index)
    try:
        return pd.qcut(relative_times, quantiles)
    except TypeError:
        return pd.qcut(relative_times.astype(int), quantiles)
    else:
        raise ValueError('Cannot compute quantile cut on dtype {dtype}'.format(
            dtype=relative_times.dtype))


def plot_train_test_split(panel, train_indices, test_indices, ax,
                          colors=['r', 'b', 'g']):
    item_to_yaxis = {item: index
                     for index, item in dict(enumerate(panel.items)).items()}
    midx_df = panel_to_multiindex(panel).dropna(how='all')

    years = midx_df.index.get_level_values(1).astype(int) + 1970

    for indx in range(len(midx_df)):
        if indx in train_indices:
            color = colors[0]
        elif indx in test_indices:
            color = colors[1]
        else:  # neither train nor test
            color = colors[2]
        item, __ = midx_df.iloc[indx].name
        year = years[indx]
        ax.barh(item_to_yaxis[item], 1, left=int(year), color=color)
    unique_years = np.unique(years)
    ax.set_xticks(unique_years + .5)
    ax.set_xticklabels(unique_years)
    ax.set_yticks(np.arange(len(panel.items)))
    ax.set_yticklabels([item for item, pos in
                        sorted(item_to_yaxis.items(), key=lambda x: x[1])])

    ax.set_ylabel("whether data exists at that year")
    ax.set_xlabel("year")


def demonstrate_MultiTimeSeriesSplit(n_train_test_sets=3,
                                     legend_position=(1.15, 3)):
    """Create a horizontal bar chart illustrating MultiTimeSeriesSplit on a
    small synthetic dataset."""
    panel = synthetic_exports_data.medium_panel
    midx_df = panel_to_multiindex(panel).dropna(how='all')

    n_train_test_sets = n_train_test_sets
    mtss = MultiTimeSeriesSplit(n_train_test_sets=n_train_test_sets,
                                level_of_index_for_time_values='year')
    split_indices = list(mtss.split(midx_df))

    n_train_test_sets = len(split_indices)
    fig, ax = plt.subplots(nrows=n_train_test_sets)
    colors = ['#69D2E7', '#FA6900', '#E0E4CC']

    for i in range(len(split_indices)):
        print('train-test split number:', i)
        print('train indices:', split_indices[i][0])
        print('test indices:', split_indices[i][1])
        print()
        if n_train_test_sets > 1:
            axis = ax[i]
        else:
            axis = ax
        plot_train_test_split(panel, *split_indices[i], axis, colors=colors)
        axis.set_ylabel('')
        axis.set_xlabel('')
        axis.set_title('train-test split {}'.format(i))
    plt.tight_layout()
    train_label = mpl.patches.Patch(color=colors[0], label='train')
    test_label = mpl.patches.Patch(color=colors[1], label='test')
    other_label = mpl.patches.Patch(color=colors[2], label='neither')
    plt.legend(handles=[train_label, test_label, other_label],
               bbox_to_anchor=legend_position)
    plt.show()
