import numpy as np
import pandas as pd
from ..utils import panel_to_multiindex
# import sklearn.decomposition
# from ..visualization.dimension_reduction import feature_loading_heatmap
# import functools
# import inspect
# import abc
# from sklearn.feature_selection import SelectKBest
from sklearn.preprocessing import FunctionTransformer
from sklearn.decomposition import PCA


# class MeanStd(sklearn.base.BaseEstimator, sklearn.base.TransformerMixin):
#     """Reduce 2D data to its row mean and row standard deviation."""

#     def __init__(self):
#         pass

#     def fit(self, X, y=None):
#         return self

#     def transform(self, X, y=None):
#         X = sklearn.utils.check_array(X)
#         means = np.mean(X, axis=1)
#         stds = np.std(X, axis=1)
#         return np.concatenate(
#             (means[:, np.newaxis], stds[:, np.newaxis]), axis=1)


def reduce_dim_of_panel(panel, dim_reducer, cross_validators=None,
                        drop_rows_missing_all=True, fill_missing_value=None,
                        column_name='component'):
    """Reduce dimensions of a pandas Panel and return a MultiIndex DataFrame.

    It also drops rows that are missing all their values, and it optionally
    fills any remaining missing values.

    Parameters
    ----------
    panel : a pandas Panel

    dim_reducer : a dimension reducer such as sklearn.decomposition.PCA
        This object must have the attributes `fit` and `transform`
        (and `fit_transform`).

    cross_validators : None or cross validator or list of cross validators
        If not None, then fit `dim_reducer` to the first training set. If
        doing nested cross validation, then set `cross_validators` to be
        `[outer_cross_validator, inner_cross_validator]` because they are
        used to split the data in left-to-right order.

    column_name : str, optional, default: 'principal_component'
        The name to assign to the columns of the returned DataFrame

    drop_rows_missing_all : bool, default: True
        Whether to drop rows missing all their values after converting the
        panel to a MultiIndex DataFrame.

    fill_missing_value : None or float
        If fill_missing_value is not None, then fill missing data with
        `fill_missing_value`.

    Returns
    -------
    dataframe_reduced_dim : pandas DataFrame
        The dimension-reduced data
    """
    df = panel_to_multiindex(panel)
    if drop_rows_missing_all:
        df = df.dropna(how='all', axis=0)
    if fill_missing_value is not None:
        df = df.fillna(fill_missing_value)

    no_dimension_reduction = (
        (dim_reducer is None) or
        (isinstance(dim_reducer, FunctionTransformer) and
         dim_reducer.func is None))
    if no_dimension_reduction:
        return df
    else:
        if cross_validators is not None:
            # train_indices = next(split_train_test.split(df))[0]
            first_training_set = (
                get_first_training_set_from_nested_cross_validation(
                    cross_validators, df))
            dim_reducer.fit(first_training_set)
        else:
            dim_reducer.fit(df)

        dataframe_reduced_dim = pd.DataFrame(dim_reducer.transform(df))
        dataframe_reduced_dim.index = df.index
        dataframe_reduced_dim.columns.name = column_name
        return dataframe_reduced_dim


def get_first_training_set_from_nested_cross_validation(cvs, data):
    """Use recursion to get the indices of the first training set in nested
    cross-validation.

    Parameters
    ----------
    cvs : None or cross validator or a list of cross validators
        If None, then the data is returned.
        If `cvs` is a single cross validator, it is first wrapped with a list.
        If `cvs` is a list, they should be ordered from left to right as the
        outer-most to inner-most cross validator.

    data : pandas DataFrame
    """
    if cvs is None:
        return data
    try:
        len(cvs)
    except TypeError:
        cvs = [cvs]
    if len(cvs) == 0:
        return data
    else:
        cv = cvs.pop(0)
        train_indices, __ = next(cv.split(data))
        return get_first_training_set_from_nested_cross_validation(
            cvs, data.iloc[train_indices])


# def descending_scores(X, y=None):
#     """Return the array `[n_columns, n_columns - 1, ..., 1]`, where `n_columns`
#     is the number of columns of `X`.

#     This is used for selecting the first `k` columns in a matrix using
#     `SelectKBest`."""
#     return np.array(range(X.shape[1], 0, -1))


# def select_first_columns(k=2):
#     """Return an estimator that selects the first `k` columns.

#     The inverse transform pads with columns of zeros.

#     Parameters
#     ----------
#     k : int
#         The number of columns to select.
#     """
#     return SelectKBest(descending_scores, k=k)


# class PCA_Labeled(sklearn.decomposition.PCA):
#     """Wrapper of PCA that gives a filename and list of component labels."""

#     def __init__(self, dimension_labels=None, **kwargs):
#         # TODO: put PCA's arguments into `__init__` explicitly
#         super().__init__(**kwargs)
#         if dimension_labels is None:
#             self._dimension_labels = ['principal component {}'.format(i)
#                                       for i in range(self.n_components)]
#         else:
#             self._dimension_labels = dimension_labels

#     def fit(self, X, y=None):
#         """Ensures that the average loading in each component is positive."""
#         super().fit(X, y=y)
#         for i, loadings in enumerate(self.components_):
#             self.components_[i] *= np.sign(np.mean(loadings))

#     @property
#     def filename(self):
#         """A string for use in file names."""
#         return 'PCA_{}'.format(self.n_components)

#     @property
#     def dimension_labels(self):
#         """Names of the reduced dimensions."""
#         return self._dimension_labels

#     @dimension_labels.setter
#     def dimension_labels(self, labels):
#         if len(labels) >= self.n_components:
#             self._dimension_labels = labels[:self.n_components]
#         else:
#             raise ValueError(
#                 'the number of labels does not match `n_components`: '
#                 'expected {}, got {}'.format(self.n_components, len(labels)))

#     @property
#     def dimension_labels_dict(self):
#         """Dictionary mapping the index of the dimension to its label."""
#         return dict(zip(range(self.n_components), self.dimension_labels))

#     def feature_loading_heatmap(self, original_feature_labels, **kwargs):
#         return feature_loading_heatmap(
#             self.components_, original_feature_labels, **kwargs)


# class LabeledDimensionReducer(object):
#     """Wraps a dimension reducer to provide dimension labels, file name, plots.
#     """

#     def __init__(self, dim_reducer, labels=None, **kws):
#         self.dim_reducer = dim_reducer
#         functools.update_wrapper(wrapper=self, wrapped=dim_reducer)

#         if labels is None:
#             labels = ['component {}'.format(i)
#                       for i in range(self.dim_reducer.n_components)]
#         self._labels = labels

#         for name, method in inspect.getmembers(dim_reducer, callable):
#             if not name.startswith('_'):
#                 setattr(self, name, method)

#     def __repr__(self):
#         return '{cls}({dr}, labels={dl})'.format(
#             cls=self.__class__.__name__,
#             dr=self.dim_reducer.__repr__(), dl=self.labels)

#     def __getattr__(self, attr):
#         """Getting an attribute that LabeledDimensionReducer does no have is
#         passed along to `self.dim_reducer`."""
#         return getattr(self.dim_reducer, attr)

#     @property
#     def filename(self):
#         """A short name for use in file names."""
#         return self.__str__()

#     @property
#     def labels(self):
#         """Labels of the reduced dimensions."""
#         return self._labels

#     @labels.setter
#     def labels(self, labels):
#         """Set the labels of the dimensions.

#         If the number of labels given `len(labels)` exceeds the number of
#         components of the dimension reducer, then the extra ones are ignored.
#         If the number of labels given is smaller than the number of components
#         of the dimension reducer, then the first `len(labels)` components are
#         renamed.
#         """
#         slice_n_components = slice(0, min(self.n_components, len(labels)))
#         self._labels[slice_n_components] = labels[slice_n_components]
#         assert len(self._labels) == self.n_components

#     @property
#     def dimension_labels_dict(self):
#         """Dictionary mapping the index of the dimension to its label."""
#         return dict(zip(range(self.n_components), self._labels))

#     def feature_loading_heatmap(self, original_feature_labels, **kwargs):
#         """Plot the loadings on the features."""
#         return feature_loading_heatmap(
#             self.components_, original_feature_labels, **kwargs)


# class LabeledDimensionReducerMixin(object):
#     """Provide dimension labels, file name, plots to a dimension reducer."""

#     def __init__(self, labels=None):
#         if labels is None:
#             labels = ['component %s' % i for i in range(self.n_components)]
#         self._labels = labels

#     def __str__(self):
#         return '{cls}_{n_comp}'.format(
#             cls=self.__class__.__name__, n_comp=self.n_components)

#     @abc.abstractproperty
#     def n_components(self):
#         """The number of reduced dimensions."""
#         pass

#     def components_(self):
#         """The weights of reduced dimensions in terms of the original dimensions
#         """
#         pass

#     @property
#     def filename(self):
#         """A short name for use in file names."""
#         return self.__str__()

#     @property
#     def labels(self):
#         """Labels of the reduced dimensions."""
#         return self._labels

#     @labels.setter
#     def labels(self, labels):
#         """Set the labels of the dimensions.

#         If the number of labels given `len(labels)` exceeds the number of
#         components of the dimension reducer, then the extra ones are ignored.
#         If the number of labels given is smaller than the number of components
#         of the dimension reducer, then the first `len(labels)` components are
#         renamed.
#         """
#         slice_n_components = slice(0, min(self.n_components, len(labels)))
#         self._labels[slice_n_components] = labels[slice_n_components]
#         assert len(self._labels) == self.n_components

#     @property
#     def dimension_labels_dict(self):
#         """Dictionary mapping the index of the dimension to its label."""
#         return dict(zip(range(self.n_components), self._labels))

#     def feature_loading_heatmap(self, original_feature_labels, **kwargs):
#         """Plot the loadings on the features."""
#         return feature_loading_heatmap(
#             self.components_, original_feature_labels, **kwargs)


# class LabeledPCA(sklearn.decomposition.PCA, LabeledDimensionReducerMixin):
#     def __init__(self, labels=None, **kwargs):
#         sklearn.decomposition.PCA.__init__(self, **kwargs)
#         LabeledDimensionReducerMixin.__init__(self, labels=labels)


class PCAsomeColumns(PCA):
    """PCA that doesn't transform some columns. Works only with DataFrames.

    It also standardizes the signs of the loadings (which are arbitrary) so
    that the mean of each loading vector is positive.
    """

    def __init__(self, exclude_columns=None, n_components=None,
                 copy=True, whiten=False, svd_solver='auto', tol=0.0,
                 iterated_power='auto', random_state=None):
        if exclude_columns is None:
            self.exclude_columns = []
        else:
            self.exclude_columns = list_wrap(exclude_columns)
        super(PCAsomeColumns, self).__init__(
            n_components=n_components,
            copy=True, whiten=whiten,
            svd_solver=svd_solver, tol=tol,
            iterated_power=iterated_power,
            random_state=random_state)

    @property
    def filename(self):
        filename = 'PCAsomeColumns_n_components={}'.format(self.n_components)
        if len(self.exclude_columns) > 0:
            filename += (
                '_exclude_columns_' +
                '_'.join(col for col in self.exclude_columns))
        return filename

    def fit(self, X, y=None):
        self.include_columns = [
            c for c in X.columns.values if c not in self.exclude_columns]
        super(PCAsomeColumns, self).fit(X.loc[:, self.include_columns])
        for i, loadings in enumerate(self.components_):
            # Standardize the signs (which are arbitary) so that mean(loadings)
            # is positive
            self.components_[i] *= np.sign(np.mean(loadings))

    def transform(self, X, y=None):
        X_transformed = super(PCAsomeColumns, self).transform(
            X.loc[:, self.include_columns])
        transformed_by_pca = pd.DataFrame(
            X_transformed,
            columns=['component_{}'.format(i)
                     for i in range(self.n_components)],
            index=X.index)
        result = pd.concat(
            [transformed_by_pca, X.loc[:, self.exclude_columns]], axis=1)
        return result

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X, y)

    def inverse_transform(self, X, y=None):
        X_inverse_transformed = super(PCAsomeColumns, self).inverse_transform(
            X.loc[:, [c for c in X.columns.values
                      if c not in self.exclude_columns]])
        X_inverse_transformed = pd.DataFrame(
            X_inverse_transformed,
            columns=self.include_columns,
            index=X.index)
        result = pd.concat(
            [X_inverse_transformed, X.loc[:, self.exclude_columns]], axis=1)
        return result


def list_wrap(x):
    if isinstance(x, str):
        return [x]
    try:
        return list(x)
    except TypeError:
        return [x]
