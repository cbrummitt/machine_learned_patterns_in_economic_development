# import pandas as pd
import logging
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler


logger = logging.getLogger(__name__)


def subtract_shuffled_null_model_from_panel(panel, shift_null_model=False):
    """Standardize a panel dataset by subtracting the expected value in a null
    model.

    Given a panel dataset of export values, with items as countries,
    time as major_axis, and products as minor_axis, create a new panel in which
    an expected value in a null model is subtracted from the data. The null
    model assumes that a country's exports are allocated to different products
    in the same proportions as those products' total exports are compared to
    the total exports of all products.

    Parameters
    ----------
    panel : pandas Panel
        A panel dataset with `items` being the names of different trajectories
        (people, countries, etc.), time as `major_axis`, and features as
        `minor_axis`.

    shift_null_model : bool, optional, default: False
        Whether to shift the null model by one time step so that data is not
        normalized by data that depends on itself.

    Returns
    -------
    panel_normalized_null_model : pandas Panel
        A normalized panel in which an expected value is subtracted from each
        entry in `panel`. The new normalized panel is essentially
            `panel - (panel.sum(axis='minor_axis') * panel.sum('items') /
                      panel.sum('items').sum(axis=1)).
    """
    panel_normalized_null_model = panel.copy()

    sum_across_items = panel.sum('items')
    sum_across_items_and_features = panel.sum('items').sum(axis=1)
    share_of_each_feature = (
        sum_across_items.div(
            sum_across_items_and_features, axis='index')
        .shift(int(shift_null_model)))

    for item in panel.items:
        sum_across_features = panel.loc[item].sum(axis=1)

        expected = (
            share_of_each_feature).mul(
                sum_across_features, axis='index')

        panel_normalized_null_model.loc[item] -= expected
    return panel_normalized_null_model


class IteratedLog1p(BaseEstimator, TransformerMixin):
    """Transforms features by applying log1p a certain number of times.

    Parameters
    ----------
    n : int, default: 1
        The number of times to apply numpy.log1p to the data
    """

    def __init__(self, n=1, pseudolog=False):
        if n < 0:
            raise ValueError('`n` must be positive; got {}'.format(n))
        self.n = n

    def _transformed_filename(self, filename):
        if self.n == 1:
            return 'log1p_{}'.format(filename)
        else:
            return 'log1p_applied_{}_times_to_{}'.format(self.n, filename)

    def _transformed_name(self, name):
        if self.n == 1:
            return 'log(1 + {})'.format(name)
        else:
            return r'log1p^{' + str(self.n) + '}' + '(' + name + ')'

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        """Apply `numpy.log1p` to `X` `n` times."""
        result = X.copy()
        for i in range(self.n):
            result = np.log1p(result)

        if isinstance(X, pd.DataFrame):
            result = pd.DataFrame(result, index=X.index, columns=X.columns)

        if hasattr(X, 'name'):
            result.name = self._transformed_name(X.name)
        if hasattr(X, 'filename'):
            result.filename = self._transformed_filename(X.filename)

        return result

    def inverse_transform(self, X):
        """Apply `np.exp(X) - 1` `n` times."""
        result = X.copy()
        for i in range(self.n):
            result = np.exp(X) - 1.0
        if isinstance(X, pd.DataFrame):
            result = pd.DataFrame(result, index=X.index, columns=X.columns)
        return result


class PseudoLog(BaseEstimator, TransformerMixin):
    """Transforms features by applying arcsinh(x / 2).
    """

    def __init__(self):
        pass

    def _transformed_filename(self, filename):
        return 'pseudolog_{}'.format(filename)

    def _transformed_name(self, name):
        return 'pseudolog(1 + {})'.format(name)

    def _transformed_math_name(self, name):
        return 'arcsinh({} / 2)'.format(name)

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        """Apply `arcsinh(x / 2)` to `X`."""
        result = np.arcsinh(X / 2.0)

        if isinstance(X, pd.DataFrame):
            result = pd.DataFrame(result, index=X.index, columns=X.columns)

        if hasattr(X, 'name'):
            result.name = self._transformed_name(X.name)
        if hasattr(X, 'filename'):
            result.filename = self._transformed_filename(X.filename)

        return result

    def inverse_transform(self, X):
        """Apply `np.exp(X) - 1` `n` times."""
        result = 2.0 * np.sinh(X)

        if isinstance(X, pd.DataFrame):
            result = pd.DataFrame(result, index=X.index, columns=X.columns)
        return result


class ScaledLogPositiveData(BaseEstimator, TransformerMixin):
    """Applies x -> (log(x) + min(x[x > 0])) / min(x[x > 0]) to x[x > 0].
    """

    def __init__(self, X_min_pos=None):
        if X_min_pos == 1.:
            raise ValueError('X_min_pos cannot be 1.')
        self.X_min_pos = X_min_pos
        self.X_min_pos_computed_from_data = False

    def _transformed_filename(self, filename):
        return 'scaled_log_positive_data_{}'.format(filename)

    def _transformed_name(self, name):
        return '1 + log({X}) / log({Xminpos}) ({Xminpos} - 1)'.format(
            X=name, Xminpos=self._min_pos_repr(name))

    def _min_pos_repr(self, name):
        return_number = (self.X_min_pos is not None and
                         not self.X_min_pos_computed_from_data)
        if return_number:
            return '{:.1f}'.format(self.X_min_pos)
        return '{}_{{min. pos.}}'.format(name)

    def _transformed_math_name(self, name):
        t = r'1 + \log \left ({X} \right ) / \log ({Xminpos}) ({Xminpos} - 1)'
        return t.format(X=name, Xminpos=self._min_pos_repr(name))

    def fit(self, X, y=None):
        """Fit the scaler by computing the smallest positive value or by using
        X_min_pos if specified.
        """
        X_array = np.array(X)
        if self.X_min_pos is None:
            self.X_min_pos_computed_from_data = True
            self.X_min_pos = X_array[X_array > 0].min()
            logger.info('Computed X_min_pos = {}'.format(self.X_min_pos))
        self._is_fitted = True
        return self

    def _check_fitted(self):
        if not self._is_fitted:
            raise ValueError('Transformer not yet fitted. Call `fit` first.')

    def transform(self, X, y=None, fillna=0.0):
        """Apply (log(Xp) + m) / m, where m = log(X_min_pos) / (X_min_pos - 1)
        and X_min_pos is the smallest positive value in X, applied to the
        positive data Xp.
        """
        self._check_fitted()
        X_tr = np.array(X.fillna(fillna) if hasattr(X, 'fillna') else X)
        positive = X_tr > 0
        m = np.log(self.X_min_pos) / (self.X_min_pos - 1.0)
        logger.info('m = {:.6f}'.format(m))
        X_tr[positive] = (np.log(X_tr[positive]) + m) / m

        if isinstance(X, pd.DataFrame):
            X_tr = pd.DataFrame(X_tr, index=X.index, columns=X.columns)
        if isinstance(X, pd.Panel):
            X_tr = pd.Panel(X_tr, items=X.items, major_axis=X.major_axis,
                            minor_axis=X.minor_axis)
        if isinstance(X, pd.DataFrame) or isinstance(X, pd.Panel):
            if hasattr(X, 'name'):
                X_tr.name = self._transformed_name(X.name)
            if hasattr(X, 'filename'):
                X_tr.filename = self._transformed_filename(X.filename)

        return X_tr

    def inverse_transform(self, X):
        """Apply exp(X * X_min_pos - X_min_pos)."""
        self._check_fitted()
        X_inv = np.array(X)
        positive = (X_inv > 0)
        X_inv[positive] = np.exp(
            X_inv[positive] * self.X_min_pos - self.X_min_pos)

        if isinstance(X, pd.DataFrame):
            X_inv = pd.DataFrame(X_inv, index=X.index, columns=X.columns)
        return X_inv


class LabeledStandardScaler(StandardScaler):
    def _transformed_name(self, name):
        return 'StandardScaler({})'.format(name)

    def _transformed_filename(self, name):
        return 'StandardScaler_{}'.format(name)

    def transform(self, X, y=None):
        result = super().transform(X)

        if isinstance(X, pd.DataFrame):
            result = pd.DataFrame(result, index=X.index, columns=X.columns)

        if hasattr(X, 'name'):
            result.name = self._transformed_name(X.name)
        if hasattr(X, 'filename'):
            result.filename = self._transformed_filename(X.filename)

        return result
