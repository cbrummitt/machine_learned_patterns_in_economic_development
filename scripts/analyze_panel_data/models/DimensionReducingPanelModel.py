import abc
import datetime
import itertools
import os
import time
import warnings
from operator import attrgetter

import imageio
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import sklearn.metrics
import sympy as sym
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.spatial import ConvexHull
from sklearn.base import clone
from sklearn.externals import joblib
from sklearn.metrics.scorer import make_scorer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

import dill as pickle
from analyze_panel_data.preprocessing.dimension_reduction import \
    reduce_dim_of_panel
from analyze_panel_data.utils import split_pipeline_at_step

from ..model_selection.split_data_target import \
    split_panel_into_data_and_target_and_fill_missing
from ..utils import hash_string, multiindex_to_panel
from ..visualization import inferred_model as vis_model
from ..visualization.inferred_model import (
    aggregate_dimensions_of_grid_points_and_velocities,
    bounding_grid, make_axis_labels,
    mask_arrays_with_convex_hull)
from ..visualization.utils import (convert_None_to_empty_dict_else_copy,
                                   create_fig_ax, maybe_save_fig,
                                   shifted_color_map)

regression_metrics = [
    sklearn.metrics.r2_score,
    sklearn.metrics.mean_squared_error,
    sklearn.metrics.mean_absolute_error]

scoring_values_sklearn = [
    'neg_mean_absolute_error',
    'neg_mean_squared_error',
    'neg_median_absolute_error',
    'r2']

MAX_LENGTH_FILE_NAME = 200


class DimensionReducingModel(metaclass=abc.ABCMeta):
    """An estimator that optionally reduces dimensions as an intermediate step.
    """
    @abc.abstractmethod
    def __init__(self, model, dim_reducer, predictor_from_reduced_dimensions):
        """
        model : estimator with `fit` and `predict` methods

        dim_reducer : estimator
            The part of the model that reduces dimensions.

        predictor_from_reduced_dimensions : estimator
            The part of the model that predicts the target from inputs that
            are the output of `dim_reducer`.
        """
        self.model = model
        self.dim_reducer = dim_reducer
        self.predictor_from_reduced_dimensions = (
            predictor_from_reduced_dimensions)

    @abc.abstractmethod
    def reduce_dimensions(self, X):
        """Reduce dimensions of the matrix X."""

    def __str__(self):
        template = ('{cls}(dim_reducer={dim_reducer}, '
                    'predictor_from_reduced_dimensions={pred})')
        return template.format(
            cls=self.__class__.__name__, dim_reducer=self.dim_reducer,
            pred=self.predictor_from_reduced_dimensions)

    def predict_from_reduced_dimensions(self, X_dim_reduced):
        """Predict the target from a matrix of dimension-reduced data."""
        return self.predictor_from_reduced_dimensions.predict(
            np.array(X_dim_reduced))

    def predict_and_reduce_from_reduced_dimensions(self, X_dim_reduced):
        return self.reduce_dimensions(
            self.predict_from_reduced_dimensions(np.array(X_dim_reduced)))


class ModelThatDoesNotReduceDimensions(DimensionReducingModel):
    """A model that does not reduce dimensions as an intermediate step.

    Its dimension reducer is the identity function.
    """

    def __init__(self, model):
        super(ModelThatDoesNotReduceDimensions, self).__init__(
            model=model, dim_reducer=FunctionTransformer(None),
            predictor_from_reduced_dimensions=model)

    def reduce_dimensions(self, X):
        """Reduce dimensions of the matrix X.

        This model does not reduce dimensions as an intermediate step."""
        return X


class DimensionReducingPipeline(DimensionReducingModel):
    """An sklearn Pipeline that may reduce dimensions as an intermediate step.
    """

    def __init__(self, pipeline, step_of_dimension_reducer):
        """Create a dimension-reducing Pipeline model.

        Parameters
        ----------
        pipeline : sklearn Pipeline
            The model

        step_of_dimension_reducer : str
            The name of the step in the pipeline that reduces dimensions
        """
        reducer, predictor = split_pipeline_at_step(
            pipeline, step_of_dimension_reducer)
        super(DimensionReducingPipeline, self).__init__(
            model=pipeline, dim_reducer=reducer,
            predictor_from_reduced_dimensions=predictor)

    def reduce_dimensions(self, X):
        return self.dim_reducer.transform(X)


class _BaseDimensionReducingPanelModel(metaclass=abc.ABCMeta):
    """Train a model to predict a panel dataset, and analyze those predictions.

    Note: do not use this class. Use a class that inherits from it, such as
    `DimensionReducingPanelModelSKLearn` or `DimensionReducingPanelModelKeras`.
    """

    def __init__(
            self,
            panel,
            model_with_hyperparameter_search,
            model_predicts_change,
            validate_on_full_dimensions=False,
            num_lags=1, as_sequence=False,
            dim_reducer_preprocessing=None,
            fit_preprocessing_dimension_reduction_to_train=True,
            target_transformer=None,
            fill_missing_value=0.0,
            metric=sklearn.metrics.mean_squared_error,
            metric_greater_is_better=False,
            results_directory='results', overwrite_existing_results=False,
            already_fitted_model_in_directory=None):
        """A panel dataset and model of it, with dimension reduction optionally
        as preprocessing step, a part of the model, and as a postprocessing
        step.

        Parameters
        ----------
        panel : pandas Panel
            The time-series data we want to train a model to predict.
            * items denote different trajectories;
            * major_axis is time;
            * minor_axis is features of those times at those times.

        model_with_hyperparameter_search : a model wrapped with a
        hyperparameter search such as GridSearchCV or RandomizedGridSearchCV,
        or `None`
            The model we want to train to predict the panel data (i.e., to
            predict the next time step given `num_lags` many previous
            time steps). This must be a model wrapped with a hyperparameter
            search object such as `sklearn.model_selection.RandomizedSearchCV`.
            It should have a `cv` attribute that is a cross validator.
            If None, then a filename in which a hyperparameter search's best
            model has been saved should be given in the input
            `already_fitted_model_in_directory`.

        model_predicts_change : bool
            Whether the model predicts the change between one time step and the
            next (`model_predicts_change == True`) or predicts the value at the
            next time step.

        validate_on_full_dimensions : bool, default: False
            Whether the validation score should be the score on the
            original, full-dimensionsal data and target (without being
            transformed by `dim_reducer_preprocessing`) or on the data and
            target after they are transformed by `dim_reducer_preprocessing`.

        num_lags : int, default: 1
            The number of time lags to use for prediction. For example, if
            `num_lags` is `1`, then the model predicts year `t` using the state
            at year `t - 1`.

        as_sequence : bool, default: False
            Whether the data should have shape
            `(num_samples, num_lags, num_features)` instead of shape
            `(num_samples, num_lags * num_features)`. Set this to `True`
            for recurrent neural networks, for example.

        dim_reducer_preprocessing : transformer or None, default: None
            A dimension reducer such as sklearn.decomposition.PCA. This is
            applied as a preprocessing step in an unsupervised way. If None,
            then no dimension reduction is done as a preprocessing step.

        fit_preprocessing_dimension_reduction_to_train : bool, default: True
            Whether to fit `dim_reducer_preprocessing` to just the first
            training split of `cv`. It is a good idea to set this to `True` in
            order to not cheat by letting the dimension reduction "see" the
            data in the future compared to the first training set.

        target_transformer : None or function
            Function to apply to the target before fitting. This should be
            an `sklearn.preprocessing.FunctionTransformer` with an inverse
            function provided (`inverse_func`) so that predictions can be made
            by calling the inverse function on the model's predictions.
            This transformer should have a 'filename' attribute with a
            descriptive string, for use in saving the data to the hard disk.

        fill_missing_value : float, default: 0.0
            The value to use to fill missing values.

        metric : a metric for evaluating model performance
            A scoring or loss function whose signature is
            `y_true, y_pred, sample_weight=None` plus other optional keyword
            arguments. E.g., `sklearn.metrics.mean_squared_error`.
            This metric is used both for fitting the model and for evaluating
            the model on out-of-sample data.
            If `validate_on_full_dimensions` is true, then in cross
            validation the scorer is changed so that it measures the score
            on the full dataset (without being transformed by the
            preprocessing dimension reduction).

        metric_greater_is_better : bool, default: False
            Whether the metric is a score function (higher output is better)
            or a loss function (lower output is better; this is the default).

        results_directory : name of the directory in which to put results
            The results will be put into 'results_directory/{self.filename}/'
            where `self.filename` is created by combining the `filename`
            attributes of `panel`, `model_with_hyperparameter_search`, and
            `dim_reducer_preprocessing`, as well as `num_lags` and
            `validate_on_full_dimensions` (see Notes below).

        overwrite_existing_results : bool, default: False
            Whether to overwrite existing results if a
            `DimensionReducingPanelModel` with this results_path and filename
            (see note below) already exists.

        already_fitted_model_in_directory : None or str
            If a string, then a fitted model is found in the path
            `os.path.join(results_directory, already_fitted_model_in_directory,
                          'best_model')`
            This model is loaded into `self.best_model`. In this case, the
            input for `model_with_hyperparameter_search` should be None.

        Notes
        -----
        A file name for this object is created using the `filename` attribute
        of `panel`, `model_with_hyperparameter_search`, `cv`,
        `model_predicts_change`, `dim_reducer_preprocessing` (as well as
        `num_lags` and `validate_on_full_dimensions`).
        If those objects do not have a `filename` attribute, then no error is
        thrown, and a generic fallback string is used (such as 'model' if
        `model_with_hyperparameter_search` does not have a `filename`
        attribute).
        """
        self.panel = panel
        self.model = model_with_hyperparameter_search
        self.model_predicts_change = model_predicts_change
        self.validate_on_full_dimensions = validate_on_full_dimensions
        self.reduces_dim_preprocessing = dim_reducer_preprocessing is not None
        self.dim_reducer_preprocessing = (
            FunctionTransformer(None) if dim_reducer_preprocessing is None
            else dim_reducer_preprocessing)
        self.target_transformer = (
            FunctionTransformer(None) if target_transformer is None
            else target_transformer)
        self.fill_missing_value = fill_missing_value
        self.as_sequence = as_sequence
        self.num_lags = num_lags
        self.fit_preprocessing_dimension_reduction_to_train = (
            fit_preprocessing_dimension_reduction_to_train)
        self.metric = metric
        self.metric_greater_is_better = metric_greater_is_better

        self.overwrite_existing_results = overwrite_existing_results
        self.attributes_to_join_to_make_filename = [
            'model', 'panel',
            'dim_reducer_preprocessing', 'target_transformer']

        self._results_directory = results_directory
        if already_fitted_model_in_directory is not None:
            # The model has already been fitted and is located in the path
            # results_path/already_fitted_model_in_directory/best_model
            self._filename = already_fitted_model_in_directory
            if not os.path.exists(os.path.join(self.results_path)):
                m = ('A directory `{d}` was given, but this directory is'
                     'not found in the results directory `{res}`.')
                raise ValueError(m.format(d=already_fitted_model_in_directory,
                                          res=results_directory))
            os.makedirs(self.results_path, exist_ok=True)
            os.makedirs(self.animations_path, exist_ok=True)
            os.makedirs(self.plot_path, exist_ok=True)
            os.makedirs(self.path_to_best_model_directory, exist_ok=True)
            self.load_best_model()
        else:
            self._filename = self._create_filename()
            self.create_results_paths()
            self.best_model = None
            self.fit_time = None
        self.reduce_dimensions_preprocessing_and_fill_missing_values()
        self.compute_data_target()
        # self.set_up_times_in_cv_and_model_cv_if_needed()
        # self._set_times_attribute_if_needed(self.cv)
        if hasattr(self.model, 'cv'):
            self._set_times_attribute_if_needed(self.model.cv)

        set_scoring_to_evaluate_on_full_dimensions = (
            self.validate_on_full_dimensions and
            hasattr(self.model, 'scoring') and self.reduces_dim_preprocessing)
        if set_scoring_to_evaluate_on_full_dimensions:
            print('Setting scoring to validate on unreduced dimensions')
            self.model.scoring = self._create_scorer_for_full_dimensions()
        elif hasattr(self.model, 'scoring'):
            self.model.scoring = make_scorer(
                self.metric, greater_is_better=metric_greater_is_better)

        # These are set to not-None values by methods:
        self.model_dim_reducer = None
        self.model_predictor_from_reduced_dim = None
        self.dimension_reducing_model = None
        self._n_reduced_dimensions_in_model = (
            'The model has not yet been fitted or not yet split into a '
            'dimension reducer and predictor. First fit the model using the '
            'method `fit_model_to_entire_dataset_and_save_best_model`. Then '
            'run `split_best_model_into_dimension_reducer_and_predictor` '
            'with the appropriate parameters.')

    def _get_filename_of_attribute(self, attribute):
        """Get the `filename` attribute of an attribute of `self`.

        This method is used to create a brief filename using the method
        `_create_filename`."""
        attribute_value = self.__getattribute__(attribute)
        if (attribute_value is None or
            (isinstance(attribute_value, FunctionTransformer) and
             attribute_value.func is None)):
            return 'None'
        else:
            return getattr(attribute_value, 'filename', attribute)

    def _create_filename(self):
        """Return a string that summarizes this panel, model, and other
        parameters."""
        filename = '__'.join(
            '{a}={val}'.format(a=a, val=self._get_filename_of_attribute(a))
            for a in self.attributes_to_join_to_make_filename)
        extra = ('__validate_on_full_dimensions={val_full}__num_lags={lag}'
                 '__model_predicts_change={change}__metric={metric}')
        filename += extra.format(
            val_full=self.validate_on_full_dimensions, lag=self.num_lags,
            change=self.model_predicts_change,
            metric=getattr(self.metric, '__name__', str(self.metric)))
        return filename

    @property
    def filename(self):
        """Return a string to use in file names, hashed if it is too long."""
        if len(self._filename) > MAX_LENGTH_FILE_NAME:
            return hash_string(self._filename)
        else:
            return self._filename

    @property
    def results_path(self):
        return os.path.join(self._results_directory, self.filename)

    @property
    def animations_path(self):
        return os.path.join(self.results_path, 'animations')

    @property
    def plot_path(self):
        return os.path.join(self.results_path, 'plots')

    @property
    def path_to_best_model_directory(self):
        """Return the path to the directory where the best model is saved."""
        return os.path.join(self.results_path, 'best_model')

    @property
    def best_model_filename(self):
        return 'best_model'

    @property
    def path_to_best_model(self):
        return os.path.join(self.path_to_best_model_directory,
                            self.best_model_filename)

    @property
    def cv_results_path(self):
        return os.path.join(self.path_to_best_model_directory, 'cv_results')

    @property
    def fit_time_path(self):
        return os.path.join(self.path_to_best_model_directory, 'fit_time')

    @property
    def n_reduced_dimensions_in_model(self):
        """The number of dimensions after the model's internal dimension
        reducer is applied."""
        return self._n_reduced_dimensions_in_model

    def create_results_paths(self):
        if os.path.exists(self.results_path):
            msg = 'A directory already exists at the results path {}.'.format(
                self.results_path)
            if self.overwrite_existing_results:
                msg += (' Continuing anyway (and overwriting results) '
                        'because `overwrite_existing_results` is `True`. '
                        ' If you fit the model, it will overwrite the '
                        'previously saved fit.')
                warnings.warn(msg)
            else:
                raise RuntimeError(
                    msg + ' Stopping now because `overwrite_existing_results`'
                    ' is `False`. To load the existing results, set '
                    '`overwrite_existing_results=True` but do not fit the'
                    ' model.')
        os.makedirs(self.results_path, exist_ok=True)
        os.makedirs(self.animations_path, exist_ok=True)
        os.makedirs(self.plot_path, exist_ok=True)
        os.makedirs(self.path_to_best_model_directory, exist_ok=True)
        with open(os.path.join(self.results_path,
                               'description.txt'), 'w') as f:
            f.write(self._filename)

    def __str__(self):
        """Return a shorthand description of just the most important inputs.
        """
        return '{cls_name}({inputs})'.format(
            cls_name=self.__class__.__name__,
            inputs=', '.join(
                '\n\n\t{a}={val}'.format(a=a, val=self.__getattribute__(a))
                for a in self.attributes_to_join_to_make_filename))

    def reduce_dimensions_preprocessing_and_fill_missing_values(self):
        """Preprocess the panel data by reducing dimensions in an unsupervised
        way."""
        if (self.fit_preprocessing_dimension_reduction_to_train and
                hasattr(self.model, 'cv')):
            cross_validators = [self.model.cv]
        else:
            cross_validators = None  # fit `dim_reducer` to the entire data

        self.df_dim_reduced = reduce_dim_of_panel(
            self.panel, self.dim_reducer_preprocessing,
            fill_missing_value=self.fill_missing_value,
            cross_validators=cross_validators)
        self.n_reduced_dimensions_after_preprocessing = (
            self.df_dim_reduced.shape[1])
        self.panel_dim_reduced = multiindex_to_panel(self.df_dim_reduced)

        assert not pd.isnull(
            self.df_dim_reduced.dropna(axis=0, how='all')).any().any()

    def compute_data_target(self):
        """Split the panel into data (what we predict from) and target (what
        we predict).

        Do this for both the original data before its dimensions are reduced
        as a preprocessing step and for the panel after its dimensions are
        reduced."""
        self.X, self.y = split_panel_into_data_and_target_and_fill_missing(
            self.panel_dim_reduced, num_lags=self.num_lags,
            as_sequence=self.as_sequence,
            target_is_difference=self.model_predicts_change)
        self.X_full_dim, self.y_full_dim = (
            split_panel_into_data_and_target_and_fill_missing(
                self.panel, num_lags=self.num_lags,
                target_is_difference=self.model_predicts_change))

        self.y_full_dim = pd.DataFrame(
            self.target_transformer.transform(self.y_full_dim),
            index=self.y_full_dim.index, columns=self.y_full_dim.columns)

        self.y = pd.DataFrame(
            self.target_transformer.transform(self.y),
            index=self.y.index, columns=self.y.columns)

        assert not pd.isnull(self.X).any().any()
        assert not pd.isnull(self.y).any().any()
        assert not pd.isnull(self.X_full_dim).any().any()
        assert not pd.isnull(self.y_full_dim).any().any()

    def _set_times_attribute_if_needed(self, cv):
        need_to_set_time_attribute_of_cv = (
            hasattr(cv, 'level_of_index_for_time_values') and
            cv.times is None and
            hasattr(self.X, 'index') and hasattr(self.X.index, 'levels'))

        if need_to_set_time_attribute_of_cv:
            level_name = cv.level_of_index_for_time_values
            if level_name not in self.X.index.names:
                raise ValueError(
                    'The level name {name} is not a level in the index '
                    'of `self.X`; it should be an element of {names}'
                    ''.format(level_name=level_name, names=self.X.index.names))
            print('Setting times attribute of {}'.format(cv))
            cv.times = self.X.index.get_level_values(level_name)

    def _create_scorer_for_full_dimensions(self):
        def scoring_on_full_dimensions(estimator, X, y):
            pred = self.target_transformer.inverse_transform(
                estimator.predict(X))
            sign = (1 if self.metric_greater_is_better else -1)
            try:
                return sign * self.metric(
                    self.dim_reducer_preprocessing.inverse_transform(pred),
                    self.dim_reducer_preprocessing.inverse_transform(y))
            except Exception:
                return np.nan
        return scoring_on_full_dimensions

    def check_that_best_model_exists(self):
        if self.best_model is None:
            raise ValueError(
                'The model has not yet been fitted. First run the method '
                '`fit_model_to_entire_dataset_and_save_best_model`.')

    def check_that_best_model_has_been_split_into_reducer_predictor(self):
        if self.dimension_reducing_model is None:
            raise ValueError(
                'The best model has not been split into a dimension reducer'
                ' and predictor from reduced dimensions. Call the method '
                '`split_best_model_into_dimension_reducer_and_predictor` '
                'with the appropriate parameters.')

    def fit_model_to_entire_dataset_and_save_best_model(
            self, warnings_action='default', **fit_kws):
        """Fit the model to the entire dataset and save the best model.

        Parameters
        ----------
        warnings_action : {'default', 'ignore'}
            Parameter passed to `warnings.simplefilter`; set it to 'ignore' to
            suppress warning messages.

        **fit_kws : keyword arguments
            Keyword arguments passed to the `fit` method of the model.

        Returns
        -------
        fit_time : float
            The time elapsed during fitting.
        """
        fit_time = self._fit_model_to_entire_dataset(
            warnings_action=warnings_action, **fit_kws)
        if hasattr(self.model, 'best_estimator_'):
            self.best_model = self.model.best_estimator_
        else:
            self.best_model = self.model
        self.save_best_model()
        self.save_cv_results()
        return fit_time

    def _fit_model_to_entire_dataset(
            self, warnings_action='default', **fit_kws):
        """Fit the model to the entire dataset and return the time elapsed."""
        if hasattr(self.X, 'index'):
            times = self.X.index.get_level_values(1)
        else:
            times = self.y.index.get_level_values(1)
        with warnings.catch_warnings():
            warnings.simplefilter(warnings_action)
            t0 = time.time()
            self.model.fit(
                np.array(self.X), np.array(self.y), times, **fit_kws)
            fit_time = time.time() - t0
            fit_time_str = str(datetime.timedelta(seconds=fit_time))
            print('Fit time: {} seconds = {}'.format(fit_time, fit_time_str))
        self.fit_time = fit_time
        return fit_time

    @abc.abstractmethod
    def save_best_model(self):
        """Save the best model (in the hyperparameter search) to the given path

        Every method that overrides this method should call this method in
        order to save the results of cross validation and the version of
        sci-kit learn.
        """
        self.save_cv_results()
        self.save_sklearn_version()
        self.save_fit_time()

    @abc.abstractmethod
    def load_best_model(self):
        """Load the best model from a file.

        Every method that overrides this method should call this method in
        order to load the results of cross validation and the fit time.
        """
        self.load_cv_results()
        self.load_fit_time()

    def load_saved_best_model_if_already_computed_else_fit_and_save(
            self, warnings_action='default', **fit_kws):
        warnings.warn(
            '`load_saved_best_model_if_already_computed_else_fit_and_save`'
            ' is deprecated in favor of the shorter method name '
            '`load_best_model_if_already_computed_else_fit`.')
        return self.load_best_model_if_already_computed_else_fit(
            warnings_action=warnings_action, **fit_kws)

    def load_best_model_if_already_computed_else_fit(
            self, warnings_action='default', **fit_kws):
        """Load the best model from a file if possible; else fit and save it.

        If the model has already been fitted and saved to a file, then this
        method loads that result. Otherwise, it fits the model and saves the
        best model (i.e., the model with the best hyperparameters).
        """
        this_panel_model_has_already_been_fitted_and_saved_to_a_file = (
            os.path.exists(self.path_to_best_model))
        if this_panel_model_has_already_been_fitted_and_saved_to_a_file:
            self.load_best_model()
        else:
            self.fit_model_to_entire_dataset_and_save_best_model(
                warnings_action=warnings_action, **fit_kws)

    def save_cv_results(self):
        """Save the results of the sklearn hyperparameter search to a file."""
        if hasattr(self.model, 'cv_results_'):
            try:
                with open(self.cv_results_path + '.pkl', 'wb') as f:
                    pickle.dump(self.model.cv_results_, f)
            except Exception:
                pass
            pd.DataFrame(self.model.cv_results_).to_csv(
                self.cv_results_path + '.csv')
            self.cv_results = self.model.cv_results_

    def load_cv_results(self):
        """Load the results of the sklearn hyperparameter search.

        It also computes the index of the best estimator, which enables the
        use of the attributes `best_score_` and `best_params_`.
        """
        try:
            with open(self.cv_results_path + '.pkl', 'rb') as f:
                cv_results = pickle.load(f)
        except EOFError:
            cv_results_dict = pd.read_csv(
                self.cv_results_path + '.csv',
                index_col=0).to_dict(orient='list')
            cv_results = {
                k: np.array(v) for k, v in cv_results_dict.items()}

        if self.model is not None:
            self.model.cv_results_ = cv_results
            self.model.best_index_ = np.where(
                self.model.cv_results_['rank_test_score'] == 1)[0][0]
        self.cv_results = cv_results
        self.best_index = np.where(
            self.cv_results['rank_test_score'] == 1)[0][0]

    def save_fit_time(self):
        """Save the time it took to fit the model."""
        assert self.fit_time is not None
        with open(self.fit_time_path + '.txt', 'w') as f:
            f.write(str(self.fit_time))
        with open(self.fit_time_path + '.pkl', 'wb') as f:
            pickle.dump(self.fit_time, f)

    def load_fit_time(self):
        """Load the time it took to fit the model (in seconds)."""
        try:
            with open(self.fit_time_path + '.pkl', 'rb') as f:
                self.fit_time = pickle.load(f)
        except EOFError:
            with open(self.fit_time_path + '.txt', 'r') as f:
                self.fit_time = f.read()

    def save_sklearn_version(self):
        """Save the version of sci-kit learn to a text file."""
        path_sklearn_v = os.path.join(
            self.path_to_best_model_directory, 'sklearn_version.txt')
        with open(path_sklearn_v, 'w') as f:
            f.write(str(sklearn.__version__))

    def cv_score_on_unreduced_dimensions(
            self, filename='cv_score_on_unreduced_dimensions', force=False):
        """Compute cross-validation scores on the unreduced dimensions using
        the best model found in hyperparameter search.

        The model's `cv` attribute is used to split the data into training
        and testing sets, and the predictions are re-computed and then
        transformed by the inverse of the preprocessing dimension reducer
        `self.dim_reducer_preprocessing`.

        It assigns the result to `self.val_scores_on_unreduced_dimensions`, and
        it pickles the resulting scores to the given filename in the path where
        the best model is saved (`self.path_to_best_model_directory`).

        If `force` is `False` and the scores are already saved to the given
        filename, then the scores are loaded from that file.
        """
        self.check_that_best_model_exists()
        path_pickle = os.path.join(self.path_to_best_model_directory,
                                   filename + '.pkl')
        if os.path.exists(path_pickle) and not force:
            with open(path_pickle, 'rb') as f:
                self.val_scores_on_unreduced_dimensions = pickle.load(f)
        else:
            self.val_scores_on_unreduced_dimensions = (
                self._compute_cv_scores_on_unreduced_dimensions())
            with open(path_pickle, 'wb') as f:
                pickle.dump(self.val_scores_on_unreduced_dimensions, f)
            path_text_file = os.path.join(self.path_to_best_model_directory,
                                          filename + '.txt')
            with open(path_text_file, 'w') as f:
                f.write(str(self.val_scores_on_unreduced_dimensions))
        return self.val_scores_on_unreduced_dimensions

    def _compute_cv_scores_on_unreduced_dimensions(self):
        if not self.reduces_dim_preprocessing:
            # No need to re-compute test scores on the cross validaiton splits
            test_score_keys = [
                c for c in self.cv_results
                if c[:5] == 'split' and c[-11:] == '_test_score']
            best_index = np.where(
                self.cv_results['rank_test_score'] == 1)[0][0]
            return np.array(
                [self.cv_results[key][best_index]
                 for key in test_score_keys])
        scores = []
        # Need to clone because we re-fit to the training set
        copy_best_model = clone(self.best_model)
        for train_indices, test_indices in self.model.cv.split(
                self.X, self.y, self.X.index.get_level_values(1)):
            copy_best_model.fit(
                np.array(self.X)[train_indices],
                np.array(self.y)[train_indices])
            predictions = self.target_transformer.inverse_transform(
                copy_best_model.predict(np.array(self.X)[test_indices]))
            predictions = pd.DataFrame(
                predictions, index=self.y.iloc[test_indices].index,
                columns=self.y.columns).loc[:, 0]
            predictions_full_dimensions = (
                self.dim_reducer_preprocessing.inverse_transform(predictions))
            score = self.metric(
                predictions_full_dimensions,
                self.target_transformer.inverse_transform(
                    np.array(self.y_full_dim)[test_indices]))
            scores.append(score)
        return scores

    @abc.abstractmethod
    def split_best_model_into_dimension_reducer_and_predictor(
            self, step_of_dimension_reducer=None):
        """Split the model into a dimension reducer and predictor.

        Use the class `DimensionReducingModel` or a descendant of it. Assign
        the result to `self.dimension_reducing_model`.

        When overriding this method, at the end of the method (after
        `self.dimension_reducing_model` has been set) the method
        `_compute_n_reduced_dimensions_in_model` should be called,
        so that the attribute `_n_reduced_dimensions_in_model` is set.
        """

    def _compute_n_reduced_dimensions_in_model(self):
        """Compute the number of dimensions after the model's dimension reducer
        is applied by doing it on the first row of data."""
        self._n_reduced_dimensions_in_model = (
            self.dimension_reducing_model.reduce_dimensions(
                np.array(self.X)[0].reshape(1, -1)).shape[1])

    def results_dict(self):
        """Return a dictionary of results of the model's predictive performance.
        """
        self.check_that_best_model_exists()
        results = {
            'best_score_': self.model.best_score_,
            'mean_cv_score_on_unreduced_dimensions': np.mean(
                self.cv_score_on_unreduced_dimensions())}

        attributes = [
            'fit_time', 'n_reduced_dimensions_after_preprocessing', 'num_lags',
            'n_reduced_dimensions_in_model', 'validate_on_full_dimensions']
        for attribute in attributes:
            results[attribute] = self.__getattribute__(attribute)

        try:
            results['n_random_search_iterations'] = self.model.n_iter
        except AttributeError:
            results['n_random_search_iterations'] = np.nan

        for attribute in self.attributes_to_join_to_make_filename:
            results[attribute] = self._get_filename_of_attribute(attribute)

        best_index = np.where(
            self.cv_results['rank_test_score'] == 1)[0][0]
        best_model_series = pd.DataFrame(self.cv_results).iloc[
            best_index]
        for a in [
                'mean_fit_time', 'mean_score_time', 'mean_test_score',
                'mean_train_score', 'std_fit_time', 'std_score_time',
                'std_test_score', 'std_train_score']:
            results[a] = best_model_series.loc[a]

        results['panel_model'] = self

        return results

    def compute_low_dimensional_predictions_on_a_grid(
            self, extra_dimension_reducer=None, n_points_each_dim=30,
            mask_velocity_to_convex_hull_of_data=True):
        """Compute low-dimensional predictions on a grid of points.

        The grid is computed as follows:
        * The data's dimensions are reduced by `self.dim_reducer_preprocessing`
        * and then by the model's intermediate dimension-reducing step
        `self.dimension_reducing_model.reduce_dimensions`
        * and then by `extra_dimension_reducer`.
        The result should have < 30 dimensions, and ideally < 10 dimensions.
        * Then a grid of equally spaced points is created.

        Then the predictions are computed as follows:
        * first apply the `inverse_transform` of `extra_dimension_reducer`
        * then predict from reduce dimensions and reduce dimensions using the
        `predict_and_reduce_from_reduced_dimensions` method of
        `self.dimension_reducing_model`
        * then reduce dimensions using `extra_dimension_reducer`.

        Parameters
        ----------
        extra_dimension_reducer : dimension reducer with `fit`, `transform`,
        `fit_transform`, and `inverse_transform` methods
            An extra dimension reducer to reduce dimensions enough to make
            a grid of points. It is fit on the output of the model's internal
            dimension reducer applied to the data `self.X.loc[:, 1]` (i.e., the
            data with the last time point removed from every item).

        n_points_each_dim : scalar or tuple of length equal to the number of
        columns in the output of `extra_dimension_reducer`
            The number of grid points to use each dimension. If this parameter
            is a scalar, then it is duplicated for every column of the output
            of `extra_dimension_reducer`.

        Returns
        -------
        grid_points : list of 1D arrays of shape given by `n_points_each_dim`
            1D list s of locatiaons of the grid points in the (reduced)
            dimensions.
            If `n_points_each_dim` is a scalar, then each element of this
            list is an array with `n_points_each_dim` many numbers. Otherwise,
            the shape of `grid_points[i]` is `n_points_each_dim[i],`.

        velocities : list of ND arrays of shape specified by
        `n_points_each_dim`
            The predicted velocities at each grid point.
        """
        # Need to grab the value 1 time step ago (`data.loc[:, 1]`)
        self.check_that_best_model_exists()
        self.check_that_best_model_has_been_split_into_reducer_predictor()
        if list(self.X.columns.levels[0]) != [1]:
            raise NotImplementedError('This method only handles models that '
                                      'have `num_lags` equal to `1`.')
        # Reduce dimensions with the model's dimension reducer
        data_dim_reduced = self.dimension_reducing_model.reduce_dimensions(
            self.X.loc[:, 1])

        # Reduce dimensions with the extra dimension reducer
        extra_dimension_reducer = (
            FunctionTransformer(None) if extra_dimension_reducer is None
            else extra_dimension_reducer)
        data_dim_reduced = extra_dimension_reducer.fit_transform(
            data_dim_reduced)

        # Create a grid of equally spaced points
        grid_points, meshgrids = bounding_grid(
            data_dim_reduced, n_points_each_dim=n_points_each_dim)

        meshgrids_long_format = np.array([ary.flatten()
                                          for ary in meshgrids]).T
        n_features = len(meshgrids)
        n_points_each_dim = meshgrids[0].shape
        n_grid_points = np.prod(n_points_each_dim)
        assert meshgrids_long_format.shape == (n_grid_points, n_features)

        # Invert the dimension reduction and predict using the model:
        # TODO: This doesn't work with models with > 1 time lag.
        meshgrids_long_format_dim_increased = (
            extra_dimension_reducer.inverse_transform(meshgrids_long_format))

        predictions_long_format = self.target_transformer.inverse_transform(
            self.dimension_reducing_model.predict_from_reduced_dimensions(
                meshgrids_long_format_dim_increased))

        # Dimension-reduce back to a small number of dimensions:
        predictions_long_format_dim_reduced = (
            extra_dimension_reducer.transform(
                self.dimension_reducing_model.reduce_dimensions(
                    predictions_long_format)))

        predictions_shape_grids = [
            predictions_long_format_dim_reduced[:, i].reshape(
                *n_points_each_dim)
            for i in range(n_features)]

        # Difference the target and data if needed to produce the velocities:
        if self.model_predicts_change:
            velocities = predictions_shape_grids
        else:
            meshgrids_preds = zip(meshgrids, predictions_shape_grids)
            velocities = ([pred - grid for grid, pred in meshgrids_preds])

        # Optionally select only those in the convex hull of the
        # dimension-reduced data:
        if mask_velocity_to_convex_hull_of_data:
            velocities = mask_arrays_with_convex_hull(
                velocities, grid_points, ConvexHull(data_dim_reduced))

        return grid_points, velocities

    def plot_predictions_of_model_in_2D(
            self,
            kind='stream',
            extra_dimension_reducer=None, dimensions_to_keep=(0, 1),
            aggregator='mean',
            n_points_each_dim=30,
            show_colorbar=True,
            color_values='speed', colorbar_label='speed',
            ax=None, save_fig=None,
            mask_velocity_to_convex_hull_of_data=True,
            axis_labels_dict=None,
            subplots_kws=None,
            plot_kws=None):
        """Create a quiver plot of predictions of the model on a grid.

        Parameters
        ----------
        kind : {'quiver', 'stream'}
            Whether to make a quiver plot or a stream plot. See
            `matplotlib.pyplot.quiver` and `matplotlib.pyplot.streamplot`.

        extra_dimension_reducer : dimension reducer with `fit`, `transform`,
        `fit_transform`, and `inverse_transform` methods
            The dimension reducer to use to reduce dimensions enough to make
            a grid of points. It is fit to `self.X.loc[:, 1]` and then used
            to transform `self.X.loc[:, 1]` and `self.y`.

        dimensions_to_keep : tuple of int's length 2
            Which dimensions (features) to plot. Each entry in the tuple is an
            int between 0 and `n_features - 1` (inclusive), where `n_features`
            is the number of columns in `self.X`.

        aggregator : {'mean', 'median', or callable}, default: 'mean'
            How to aggregate over axes of the tensor. If callable, it must take
            as input the tensor and a keyword argument `axis` that is given a
            tuple of the indices of the axes to aggregate over.

        n_points_each_dim : scalar or tuple of length data.shape[1]
            The number of grid points to use each dimension. If this parameter
            is a scalar, then it is duplicated for every column of `data`.

        color_values : string or 2D numpy array
            Data for the colors of the arrows in the streamplot. If
            `color_values` is 'speed', then color_values is the magnitude of
            the velocity.

        colorbar_label : str, optional, default: 'speed'
            The label of the color bar

        ax : matplotlib axis, optional, default: None
            The axis on which to draw the plot. If None, then an axis is
            created.

        axis_labels_dict : None or dict
            A dictionary mapping dimension indices to strings, such as
            {0: 'component 0', 1: 'component 1'}.
            If None, then 'dimension i' is used for i = 0, 1, ....

        subplots_kws : None or dict, default: None
            Keyword arguments to pass to plt.subplots

        plot_kws : None or dict, default: None
            Keyword arguments to pass to matplotlib's streamplot or quiver

        Returns
        -------
        fig, ax : matplotlib Figure, Axis
        """
        if len(dimensions_to_keep) != 2:
            raise ValueError(
                'The number of dimensions to keep (i.e., the length of '
                '{}) must be 2.'.format(dimensions_to_keep))
        self._check_dimensions(dimensions_to_keep, extra_dimension_reducer)

        mask_to_convex_hull = mask_velocity_to_convex_hull_of_data
        grid_points, velocities = (
            self.compute_low_dimensional_predictions_on_a_grid(
                n_points_each_dim=n_points_each_dim,
                extra_dimension_reducer=extra_dimension_reducer,
                mask_velocity_to_convex_hull_of_data=mask_to_convex_hull))
        grid_points, velocities = (
            aggregate_dimensions_of_grid_points_and_velocities(
                grid_points, velocities, dimensions_to_keep,
                aggregator=aggregator))
        plotting_function = (vis_model.quiver_plot if kind == 'quiver'
                             else vis_model.stream_plot)
        return plotting_function(
            *grid_points, *[v.T for v in velocities],
            **make_axis_labels(axis_labels_dict, dimensions_to_keep),
            show_colorbar=show_colorbar,
            color_values=color_values, colorbar_label=colorbar_label, ax=ax,
            save_fig=save_fig, subplots_kws=subplots_kws, plot_kws=plot_kws)

    def _check_dimensions(self, dimensions_to_keep, extra_dimension_reducer):
        """Verify that each entry in `dimensions_to_keep` is between 0 and
        the number of dimensions that will be plotted.

        A heuristic is used to guess the dimensions produced by
        `extra_dimension_reducer`. For now it just uses its `n_components`
        attribute if it has one.
        """
        if isinstance(self.n_reduced_dimensions_in_model, int):
            n_dim = min(self.n_reduced_dimensions_after_preprocessing,
                        self.n_reduced_dimensions_in_model)
        else:
            n_dim = self.n_reduced_dimensions_after_preprocessing
        if hasattr(extra_dimension_reducer, 'n_components'):
            n_dim = min(n_dim, extra_dimension_reducer.n_components)
        for dim in dimensions_to_keep:
            if not isinstance(dim, int):
                raise ValueError('`dimensions_to_keep` = {} should contain '
                                 'only integers.'.format(dimensions_to_keep))
            elif dim < 0 or dim >= n_dim:
                raise ValueError(
                    'Each integer in `dimensions_to_keep` should be >= 0 and '
                    '< {}; got {}.'.format(n_dim, dimensions_to_keep))

    def plot_predictions_of_model_in_2D_all_pairs(
            self, kind='stream', extra_dimension_reducer=None,
            max_dimension_index=2, figsize=(12, 4), nrows=1,
            n_points_each_dim=30, axis_labels_dict=None,
            mask_velocity_to_convex_hull_of_data=True, save_fig=None):
        """Make a grid of plots of predictions for all pairs of dimensions up
        to a certain maximum dimension (`max_dimension_index`).

        For example, `max_dimension_index=2` means plot dimensions (0, 1),
        (0, 2), and (1, 2).
        """
        mask_to_convex_hull = mask_velocity_to_convex_hull_of_data
        grid_points, velocities = (
            self.compute_low_dimensional_predictions_on_a_grid(
                n_points_each_dim=n_points_each_dim,
                extra_dimension_reducer=extra_dimension_reducer,
                mask_velocity_to_convex_hull_of_data=mask_to_convex_hull))
        max_dimension_index = min(self.n_reduced_dimensions_in_model,
                                  max_dimension_index + 1)
        pairs_components = list(itertools.combinations(
            range(max_dimension_index), 2))
        plotting_function = (vis_model.quiver_plot if kind == 'quiver'
                             else vis_model.stream_plot)
        fig, ax = plt.subplots(
            nrows=nrows, ncols=int(np.ceil(len(pairs_components) / nrows)),
            figsize=figsize)
        if hasattr(ax, 'flatten'):
            axes_flat = ax.flatten()
        else:
            axes_flat = [ax]
        for col in range(0, len(pairs_components)):
            grid_pts_reduced, velocities_reduced = (
                aggregate_dimensions_of_grid_points_and_velocities(
                    grid_points, velocities, pairs_components[col]))
            plotting_function(
                *grid_pts_reduced, *[v.T for v in velocities_reduced],
                ax=axes_flat[col],
                colorbar_label=('speed' if col == len(pairs_components) - 1
                                else ''),
                **make_axis_labels(axis_labels_dict, pairs_components[col],
                                   label='dimension'))
        plt.tight_layout()
        maybe_save_fig(fig, save_fig)
        return fig, ax

    def iterated_predictions(
            self, items=None, num_time_steps=100,
            index_of_initial_condition=-1, extra_dimension_reducer=None,
            as_dataframes=False, as_combined_dataframe=False,
            reduce_dimensions_with_model_dimension_reducer=True,
            stop_predicting_at_norm=1e8,
            clip_at_percentiles=(0, 100)):
        """Compute iterated predictions of certain items in the panel.

        Parameters
        ----------
        items : list of strings (items in the panel) or `None`, default: `None`
            The items to select from the panel and to make predictions. If
            `None`, then use all items in the panel.

        num_time_steps : int or the string 'length_trajectory'
            The number of time steps to predict into the future. If
            `num_time_steps` is 'length_trajectory', then `num_time_steps` is
            set to the length of the trajectory of that time.

        index_of_initial_condition : int, optional, default: -1
            The index of the item's trajectory to use as initial conditon. If
            -1, then the initial condition is the last observation; if 0, then
            the intial condition is the first observation.

        extra_dimension_reducer : dimension reducer with `fit`, `transform`,
        `fit_transform`, and `inverse_transform` methods
            A dimension reducer to use to reduce dimensions even more after
            the model internally reduces dimensions.

        as_dataframes : bool, optional, default: False
            Whether to make each trajectory a DataFrame.
            (If `as_combined_dataframe` is also True, then all the predictions
            are stacked into one dataframe.)

        as_combined_dataframe : bool, optional, default: False
            If True, then return just one dataframe containing all the
            predictions, with items and times as the MultiIndex index and
            minor_axis as the columns.

        reduce_dimensions_with_model_dimension_reducer : bool, default: True
            Whether to reduce dimensions using the model's internal dimension
            reducer.

        stop_predicting_at_norm : float or int, default: 1e8
            Stop predicting as soon as the 2-norm of the prediction exceeds
            this threshold.

        clip_at_percentiles : tuple of two int's
            For each item, after the predictions are computed, the predictions
            are clipped at these two percentiles of the flattened data.

        Returns
        -------
        items_to_trajectories : dict mapping strings to arrays of shape
        [n_time_steps, n_features]
            Dictionary mapping the items to their trajectories. The first entry
            is the initial condition, specified by the parameter
            `index_of_initial_condition`, and then `num_time_steps - 1` steps
            into the future are predicted.

            The number of features is
                `self.n_reduced_dimensions_after_preprocessing`
            if
            `reduce_dimensions_with_model_dimension_reducer` is False and
                `n_reduced_dimensions_in_model`
            if `reduce_dimensions_with_model_dimension_reducer` is True.

        TODO:
            This does not yet handle models with multiple time lags.
            Need to check that `index_of_initial_condition` leaves enough
            samples in the history to be able to make predictions.
        """
        if self.num_lags != 1:
            raise NotImplementedError('This method only handles models with '
                                      '1 time lag.')
        self.check_that_best_model_exists()
        self.check_that_best_model_has_been_split_into_reducer_predictor()

        if reduce_dimensions_with_model_dimension_reducer:
            _dim_red_model = self.dimension_reducing_model

            def _predict(x):
                return self.target_transformer.inverse_transform(
                    _dim_red_model.predict_and_reduce_from_reduced_dimensions(
                        x))
        else:
            def _predict(x):
                return self.target_transformer.inverse_transform(
                    self.best_model.predict(x))

        items_to_trajectories = {}
        if items is None:
            items = self.panel.items

        for item in items:
            item_df = (self.panel_dim_reduced.loc[item]
                           .dropna(how='all').fillna(0))
            initial_condition = item_df.iloc[index_of_initial_condition].values
            if reduce_dimensions_with_model_dimension_reducer:
                initial_condition = (
                    self.dimension_reducing_model.reduce_dimensions(
                        np.atleast_2d(initial_condition)))[0]

            if num_time_steps in ['length_trajectory', 'length of trajectory']:
                n_steps_to_predict = len(item_df)
            else:
                n_steps_to_predict = num_time_steps

            trajectory = np.empty(
                (n_steps_to_predict, initial_condition.shape[0]))
            trajectory[0] = initial_condition

            for i in range(1, n_steps_to_predict):
                if np.linalg.norm(trajectory[i - 1]) > stop_predicting_at_norm:
                    msg = ('Terminating early the prediction of the trajectory'
                           ' of {} because the 2-norm exceeded the '
                           'threshold {}.')
                    warnings.warn(msg.format(item, stop_predicting_at_norm))
                    break
                trajectory[i] = _predict(trajectory[i - 1].reshape(1, -1))
                if self.model_predicts_change:
                    trajectory[i] += trajectory[i - 1]

            if clip_at_percentiles != (0, 100):
                trajectory = clip_array_at_percentiles(
                    trajectory, clip_at_percentiles)

            items_to_trajectories[item] = trajectory

        if extra_dimension_reducer:
            extra_dimension_reducer.fit(self.X.loc[:, 1])
            items_to_trajectories = {
                item: extra_dimension_reducer.transform(trajectory)
                for item, trajectory in items_to_trajectories.items()}

        if as_dataframes or as_combined_dataframe:
            for item, trajectory in items_to_trajectories.items():
                trajectory = pd.DataFrame(trajectory)
                trajectory.columns = self.y.columns
                initial_index = item_df.iloc[index_of_initial_condition].name
                trajectory.index = [
                    initial_index + i for i in range(n_steps_to_predict)]
                items_to_trajectories[item] = trajectory
            if as_combined_dataframe:
                dfs = []
                for item in items_to_trajectories:
                    df = items_to_trajectories[item]
                    df.index.name = self.panel.major_axis.name
                    df[self.panel.items.name] = item
                    dfs.append(df.reset_index())
                return (
                    pd.concat(dfs).set_index(['country_code', 'year'])
                    .sort_index().loc[:, 0])  # eliminate the `lag` level
        return items_to_trajectories

    def plot_trajectories_2d_3d(
            self, items_to_trajectories,
            extra_dimension_reducer=None, dimensions_to_keep=slice(None),
            label_predicted_items_at_index=None,
            ax=None, axis_labels_dict=None, labelpad=8, title=None,
            show_arrows=True, prediction_arrow_kws=None,
            show_trails=True, xlim=None, ylim=None, zlim=None, save_fig=None,
            prediction_plot_kws={'alpha': 1.0},
            plot_empirical_trajectories=False,
            label_empirical_items_at_time_index=None,
            empirical_plot_kws={'alpha': 0.5},
            empirical_arrow_kws=None,
            figsize=(5, 3)):
        """Plot 2D or 3D trajectories, with optional labels of trajectories and
        arrowheads.

        Parameters
        ----------
        items_to_trajectories : dict
            Maps strings to arrays of shape [n_time_steps, n_features]

        extra_dimension_reducer : a dimension reducer
            E.g., sklearn.decomposition.PCA, sklearn.decomposition.NMF

        dimensions_to_keep : tuple of 3 int's
            Which dimensions to plot. Each integer must be between 0 and
            the number of columns in the trajectories in
            `items_to_`trajectories`.

        label_predicted_items_at_index : None or int, optional, default: None
            If not None, then write the item name at the part of the trajectory
            given by this integer index. Use 0 for the initial condition or
            -1 for the final position in the trajectory.

        ax : None or matplotlib axis
            The axis on which to put the plot. If None, then create one.

        axis_labels_dict : None or dict
            A dictionary mapping dimension indices to strings, such as
            {0: 'component 0', 1: 'component 1'}.
            If None, then use 'dimension i' for the i-th axis.

        labelpad : int, default: 8
            Padding on the three axis labels

        title : {str, None}, default: None
            The title of the plot. If `None`, then no title is shown.

        prediction_arrow_kws, empirical_arrow_kws : dict
            Keyword arguments for the quiver plot showing an arrow at the end
            of the trajectory. If the trajectories have three columns, then
            use, e.g., {'length': 1, 'arrow_length_ratio': 1.}.

        show_trails : bool, default: True
            Whether to show the "trails" of the trajectories, i.e., to show the
            `plot`. If False, then set `label_predicted_items_at_index` to `-1`
            to show the labels moving around without any trails.

        show_arrows : bool, default: True
            Whether to show arrowheads at the ends of trajectories. They are
            shown on predicted trajectories if `show_arrows` is True and
            `show_trails` is True. They are shown on empirical trajectories
            if `show_arrows` is True and `plot_empirical_trajectories` is True.

        xlim, ylim, zlim : pairs of integers
            The values to use for the limits on the axes

        save_fig : None or string, default: None
            If not None, then save the figure to the path given by this string.

        prediction_plot_kws : dict
            Keyword arguments for the plot of the predictions

        plot_empirical_trajectories : False, 'all', or True
            If False, then do not show empirical trajectories.
            If 'all', then show the entire empirical trajectory.
            If True, then show the empirical trajectory for the same number of
            steps as in the trajectory for that country found in
            `items_to_trajectories`.

        label_empirical_items_at_time_index : None or int
            If not None, then write the item name at the part of the empirical
            trajectory given by this integer index. Use 0 for the initial
            condition or -1 for the final position in the trajectory.

        empirical_plot_kws : dict
            Keyword arguments for the plot of the empirical trajectory

        empirical_arrow_kws : dict
            Keyword arguments for the quiver showing the arrow at the end of
            the empirical trajectory.

        figsize : pair of floats or pair of ints
            The size of the figure
        """
        prediction_arrow_kws = convert_None_to_empty_dict_else_copy(
            prediction_arrow_kws)
        prediction_plot_kws = convert_None_to_empty_dict_else_copy(
            prediction_plot_kws)
        empirical_arrow_kws = convert_None_to_empty_dict_else_copy(
            empirical_arrow_kws)

        if extra_dimension_reducer:
            extra_dimension_reducer.fit(self.X.loc[:, 1])
            items_to_trajectories = items_to_trajectories = {
                item: extra_dimension_reducer.transform(trajectory)
                for item, trajectory in items_to_trajectories.items()}

        n_features = list(items_to_trajectories.values())[0].shape[1]

        if dimensions_to_keep == slice(None):
            dimensions_to_keep = tuple(range(min(3, n_features)))
        self._check_dimensions(dimensions_to_keep, extra_dimension_reducer)

        trajectories = {item: np.array(trajectory)[:, dimensions_to_keep]
                        for item, trajectory in items_to_trajectories.items()}

        n_features = list(trajectories.values())[0].shape[1]

        if ax is None:
            fig = plt.figure(figsize=figsize)
            if len(dimensions_to_keep) == 3:
                ax = fig.add_subplot(1, 1, 1, projection='3d')
            else:
                ax = fig.add_subplot(1, 1, 1)
        color_cycle = ax._get_lines.prop_cycler

        if xlim is not None:
            ax.set_xlim(xlim)
        if ylim is not None:
            ax.set_ylim(ylim)
        if n_features == 3 and zlim is not None:
            ax.set_zlim(zlim)

        axis_labels = make_axis_labels(axis_labels_dict, dimensions_to_keep)
        ax.set_xlabel(axis_labels['xlabel'], labelpad=labelpad)
        ax.set_ylabel(axis_labels['ylabel'], labelpad=labelpad)
        if n_features == 3:
            ax.set_zlabel(axis_labels['zlabel'], labelpad=labelpad)

        if title:
            ax.set_title(title)

        for item, traj in trajectories.items():
            color = next(color_cycle)['color']
            if show_trails:
                ax.plot(*traj.T, label=item, color=color,
                        **prediction_plot_kws)
            if label_predicted_items_at_index is not None:
                ax.text(
                    *np.array(traj)[label_predicted_items_at_index].T, item,
                    ha='center', va='center', color=color)
            if show_arrows and show_trails and len(traj) >= 2:
                penultimate = traj[-2]
                last_change = traj[-1] - traj[-2]
                ax.quiver(*penultimate, *last_change, color=color,
                          **prediction_arrow_kws)

            if plot_empirical_trajectories:
                empirical_trajectory = (
                    self.panel_dim_reduced.loc[item]
                        .dropna(how='all').fillna(0))
                if extra_dimension_reducer is not None:
                    empirical_trajectory = extra_dimension_reducer.transform(
                        empirical_trajectory)
                empirical_trajectory = (
                    np.array(empirical_trajectory)[:, dimensions_to_keep])

                if plot_empirical_trajectories is not 'all':
                    empirical_trajectory = empirical_trajectory[:len(traj)]
                ax.plot(*empirical_trajectory.T,
                        color=color, **empirical_plot_kws)
                if label_empirical_items_at_time_index is not None:
                    ax.text(
                        *empirical_trajectory[
                            label_empirical_items_at_time_index].T,
                        item, ha='center', va='center', color=color)
                if show_arrows and len(empirical_trajectory) >= 2:
                    penultimate = empirical_trajectory[-2]
                    last_change = (empirical_trajectory[-1] -
                                   empirical_trajectory[-2])
                    ax.quiver(*penultimate, *last_change,
                              color=color, **empirical_arrow_kws)

        maybe_save_fig(ax.get_figure(), save_fig)
        return ax.get_figure(), ax

    def sequence_plot_trajectories_2d_3d(
            self, items_to_trajectories, frames_folder='frames', alpha=.2,
            labelpad=8,
            label_predicted_items_at_index=-1,
            extra_dimension_reducer=None, dimensions_to_keep=slice(None),
            show_arrows=False, prediction_arrow_kws=None,
            show_trails=True,
            index_of_initial_condition=-1,
            prediction_plot_kws={'alpha': 1.0},
            plot_empirical_trajectories=False,
            label_empirical_items_at_time_index=None,
            empirical_plot_kws={'alpha': 0.5},
            empirical_arrow_kws=None,
            axes_limit_padding_fraction=.05):
        """Create a sequence of 3D scatter plots, and return their paths."""
        frame_path = os.path.join(self.animations_path, frames_folder)
        os.makedirs(frame_path, exist_ok=True)

        items_to_trajectories_2d_arrays = {
            item: np.array(trajectory)[:, dimensions_to_keep]
            for item, trajectory in items_to_trajectories.items()}

        first_trajectory = list(items_to_trajectories.values())[0]
        if isinstance(first_trajectory, pd.DataFrame):
            start_time = str(first_trajectory.iloc[0].name)
        else:
            start_time = self.panel.major_axis[index_of_initial_condition]

        trajectories_to_bound = (
            np.vstack(
                items_to_trajectories_2d_arrays.values()
            )[:, dimensions_to_keep])
        if plot_empirical_trajectories:
            empirical_trajectories_stacked = np.vstack(
                self.panel_dim_reduced
                    .loc[list(items_to_trajectories.keys())]
                    .values[:, dimensions_to_keep])
            trajectories_to_bound = np.vstack(
                [trajectories_to_bound, empirical_trajectories_stacked])
        axes_limits = tuple(zip(
            np.min(np.vstack(trajectories_to_bound), axis=0),
            np.max(np.vstack(trajectories_to_bound), axis=0)))
        pad = axes_limit_padding_fraction * (axes_limits[0][1] -
                                             axes_limits[0][0])
        xlim = tuple(np.array(axes_limits[0]) + np.array([-1, 1]) * pad)
        pad = axes_limit_padding_fraction * (axes_limits[1][1] -
                                             axes_limits[1][0])
        ylim = tuple(np.array(axes_limits[1]) + np.array([-1, 1]) * pad)
        if len(axes_limits) >= 3:
            zlim = axes_limits[2]
        else:
            zlim = None

        paths_of_figures = []

        for t in range(1, len(first_trajectory)):
            predicted_time = (start_time) + t
            title = ('Iterated prediction of time {time}'
                     ' starting from {start_time}').format(
                model=self._get_filename_of_attribute('model'),
                time=predicted_time,
                start_time=str(start_time))
            path_of_figure = os.path.join(
                frame_path, 'predictions_3d_{}.png'.format(predicted_time))

            self.plot_trajectories_2d_3d(
                items_to_trajectories={
                    item: traj[:t]
                    for item, traj in items_to_trajectories_2d_arrays.items()},
                extra_dimension_reducer=extra_dimension_reducer,
                dimensions_to_keep=dimensions_to_keep,
                label_predicted_items_at_index=label_predicted_items_at_index,
                ax=None,
                axis_labels_dict=None, labelpad=8,
                title=title,
                show_arrows=show_arrows,
                prediction_arrow_kws=prediction_arrow_kws,
                show_trails=show_trails,
                xlim=xlim, ylim=ylim, zlim=zlim, save_fig=path_of_figure,
                prediction_plot_kws=prediction_plot_kws,
                plot_empirical_trajectories=plot_empirical_trajectories,
                label_empirical_items_at_time_index=(
                    label_empirical_items_at_time_index),
                empirical_plot_kws=empirical_plot_kws,
                empirical_arrow_kws=empirical_arrow_kws)
            paths_of_figures.append(path_of_figure)
            plt.close()
        return paths_of_figures

    def create_gif(
            self, frame_paths, gif_filename='iterated_predictions.gif',
            fps=25, subrectangles=True):
        """Create a GIF from a list of paths to images."""
        os.makedirs(self.animations_path, exist_ok=True)
        gif_filepath = os.path.join(self.animations_path, gif_filename)
        images = []
        for image_filename in frame_paths:
            with open(image_filename, 'rb') as f:
                images.append(imageio.imread(f))
        imageio.mimsave(gif_filepath, images,
                        fps=fps, subrectangles=subrectangles)
        return gif_filepath

    def create_gif_of_iterated_predictions(
            self, items=None, num_time_steps=100,
            index_of_initial_condition=-1,
            extra_dimension_reducer=None,
            alpha=.2,
            labelpad=8, label_predicted_items_at_index=-1,
            show_arrows=False, show_trails=True,
            dimensions_to_keep=slice(None),
            prediction_plot_kws={'alpha': 1.}, prediction_arrow_kws=None,
            plot_empirical_trajectories=False,
            label_empirical_items_at_time_index=None,
            empirical_plot_kws={'alpha': 0.5},
            empirical_arrow_kws=None,
            fps=25, subrectangles=True,
            gif_filename='iterated_predictions.gif'):
        """Create a sequence of trajectories for certain items as a GIF and
        return the path to that GIF file.

        This is a helper method that calls the methods
        `iterated_predictions`, `sequence_plot_trajectories_2d_3d`,
        `create_gif`.
        """
        items_to_trajectories = self.iterated_predictions(
            items=items, num_time_steps=num_time_steps,
            extra_dimension_reducer=extra_dimension_reducer,
            index_of_initial_condition=index_of_initial_condition)

        scatter_file_paths = self.sequence_plot_trajectories_2d_3d(
            items_to_trajectories, alpha=alpha, labelpad=labelpad,
            label_predicted_items_at_index=label_predicted_items_at_index,
            show_arrows=show_arrows, show_trails=show_trails,
            prediction_arrow_kws=prediction_arrow_kws,
            dimensions_to_keep=dimensions_to_keep,
            index_of_initial_condition=index_of_initial_condition,
            prediction_plot_kws=prediction_plot_kws,
            plot_empirical_trajectories=plot_empirical_trajectories,
            label_empirical_items_at_time_index=(
                label_empirical_items_at_time_index),
            empirical_plot_kws=empirical_plot_kws,
            empirical_arrow_kws=empirical_arrow_kws)
        if not gif_filename.endswith('.gif'):
            gif_filename += '.gif'
        gif_path = self.create_gif(scatter_file_paths,
                                   gif_filename=gif_filename,
                                   fps=fps, subrectangles=subrectangles)
        return gif_path

    def rotate_and_zoom_3d_plot(
            self, fig, frames_folder='frames',
            init_elev=25., init_azim=321., init_dist=11.,
            filename='', subrectangles=True, fps=10):
        """Rotates and zooms in and out of a 3D figure; saves to files."""
        frame_path = os.path.join(self.animations_path, frames_folder)
        os.makedirs(frame_path, exist_ok=True)
        ax = fig.gca()

        # configure the initial viewing perspective
        ax.elev = init_elev
        ax.azim = init_azim
        ax.dist = init_dist

        paths = []
        n_frames = 300

        # zoom in to reveal the 3-D structure of the strange attractor
        for n in range(0, n_frames):
            if n <= n_frames * .18:
                ax.azim = ax.azim - 0.1  # begin by rotating very slowly
            elif n <= .29 * n_frames:
                ax.azim = ax.azim - 2
                ax.dist = ax.dist - 0.02
                ax.elev = ax.elev - 1  # quickly whip around to the other side
            elif n <= .45 * n_frames:
                ax.azim = ax.azim + 0.1
            elif n <= .54 * n_frames:
                ax.azim = ax.azim + 1
                ax.dist = ax.dist - 0.25
                ax.elev = ax.elev + .2  # zoom into the center
            elif n <= 0.6 * n_frames:
                ax.azim = ax.azim - 0.01
                ax.dist = ax.dist + 0.1
            elif n <= .79 * n_frames:
                ax.azim = ax.azim - 1
                ax.elev = ax.elev - 0.5
                ax.dist = ax.dist + 0.07  # pull back and pan up
            else:
                ax.azim = ax.azim - 0.1  # end by rotating very slowly

            path = os.path.join(frame_path,
                                '{}_frame_{:0=4}.png'.format(filename, n))
            fig.savefig(path)
            paths.append(path)
        return self.create_gif(
            paths,
            gif_filename='rotate_pan_zoom_{}_{}fps.gif'.format(filename, fps),
            fps=fps, subrectangles=subrectangles)

    def error_analyses(self, n_items_easiest_hardest_to_predict=10,
                       use_full_dimensions=True):
        """Do several kinds of analysis of the errors.

        Parameters
        ----------
        n_items_easiest_hardest_to_predict : int, default: 10
            The number of easiest/hardest to predict items to show.

        use_full_dimensions : bool, default: True
            Whether to analyze the residuals on all the dimensions of the data
            (i.e., invert the preprocessing dimension reduction
            `dim_reducer_preprocessing`).
        """
        n_top = n_items_easiest_hardest_to_predict
        print(self.error_analysis(
            n_items_easiest_hardest_to_predict=n_top,
            use_full_dimensions=use_full_dimensions))
        print(self.residual_histogram())
        print(self.squared_residual_histogram())

    def error_analysis(
            self, n_items_easiest_hardest_to_predict=10, alpha=.3,
            use_full_dimensions=True,
            force=False):
        """Analyze errors made by the model."""
        if hasattr(self, 'residuals') and not force:
            residuals = self.residuals.fillna(0)
        else:
            residuals = self.compute_residuals(
                use_full_dimensions=use_full_dimensions).fillna(0)
        squared_residuals = residuals.apply(lambda x: x**2)
        squared_residuals_panel = multiindex_to_panel(squared_residuals)

        fig, ax = plt.subplots(2, 2)

        for i, axis_to_keep in enumerate(['items', 'minor_axis']):
            axis_to_remove = (
                'minor_axis' if axis_to_keep == 'items' else 'items')
            axis_kw = {'axis': axis_to_remove}
            mse_over_time_mean = squared_residuals_panel.mean(**axis_kw)
            mse_over_time_std = squared_residuals_panel.std(**axis_kw)

            ylabel = 'MSE of {}\naveraged over {}'.format(
                axis_to_keep, axis_to_remove).replace('_', ' ')
            self.plot_mean_plus_minus_std(
                mse_over_time_mean.mean(axis=1),
                mse_over_time_std.mean(axis=1), ax=ax[i, 0],
                ylabel=ylabel)
            mse_over_time_mean.plot(ax=ax[i, 1], alpha=alpha)
            ax[i, 1].get_legend().set_visible(False)
            ax[i, 1].set_ylabel(ylabel)

            sorted_mse = mse_over_time_mean.mean().sort_values()
            n_top = n_items_easiest_hardest_to_predict
            template = (('-' * 5) +
                        ' Elements in {axis} with {kind} MSE ' + ('-' * 5))
            print(template.format(kind='smallest average', axis=axis_to_keep))
            print(sorted_mse.head(n_top))
            print()
            print(template.format(kind='largest average', axis=axis_to_keep))
            print(sorted_mse.tail(n_top))
            print()
            print(template.format(kind='largest max', axis=axis_to_keep))
            print((squared_residuals_panel.mean(**axis_kw).max(axis=0)
                   .sort_values().tail(n_top)))
            print()
        plt.subplots_adjust(wspace=.3, hspace=.3)
        return fig, ax

    def compute_residuals(self, use_full_dimensions=True):
        """Compute residuals of predictions of all the training data.

        Parameters
        ----------
        use_full_dimensions : bool, default: True
            Whether to analyze the residuals on all the dimensions of the data
            (i.e., invert the preprocessing dimension reduction
            `dim_reducer_preprocessing`).

        Returns
        -------
        residuals : pandas DataFrame
            The difference between the prediction and the actual value for
            every entry in the target. The index and columns are identical to
            that of `self.y_full_dim` except that the `lag` level of the column
            is removed.
        """
        self.check_that_best_model_exists()
        predictions = pd.DataFrame(
            self.target_transformer.inverse_transform(
                self.best_model.predict(np.array(self.X))),
            index=self.y.index, columns=self.y.columns).loc[:, 0]

        if use_full_dimensions:
            predictions = (
                self.dim_reducer_preprocessing.inverse_transform(predictions))
            target = self.target_transformer.inverse_transform(self.y_full_dim)
            columns = self.y_full_dim.columns
        else:
            target = self.target_transformer.inverse_transform(self.y)
            columns = self.y.columns

        self.residuals = pd.DataFrame(
            np.array(predictions) - np.array(target),
            index=self.y.index, columns=columns).loc[:, 0]
        return self.residuals

    def plot_mean_plus_minus_std(
            self, mean_series, std_series, ylabel='', subplots_kws=None,
            plot_kws=None, ax=None):
        subplots_kws = convert_None_to_empty_dict_else_copy(subplots_kws)
        plot_kws = convert_None_to_empty_dict_else_copy(plot_kws)
        fig, ax = create_fig_ax(ax, **subplots_kws)
        mean_series.plot(ax=ax, **plot_kws)
        ax.fill_between(
            mean_series.index, mean_series + std_series,
            mean_series - std_series, alpha=.2)
        ax.set_ylabel(ylabel)
        return fig, ax

    def residual_histogram(self, bins=50, subplots_kws=None):
        subplots_kws = convert_None_to_empty_dict_else_copy(subplots_kws)
        residuals_flattened = self.compute_residuals().values.flatten()
        fig, ax = plt.subplots(ncols=2, **subplots_kws)
        ax[0].hist(residuals_flattened, bins=bins)
        ax[1].hist(residuals_flattened, bins=bins)
        ax[1].set_yscale('log')
        ax[0].set_ylabel('count')
        ax[0].set_xlabel('residual')
        ax[1].set_xlabel('residual')
        return fig, ax

    def squared_residual_histogram(self, bins=50, subplots_kws=None):
        subplots_kws = convert_None_to_empty_dict_else_copy(subplots_kws)
        residuals_flattened = self.compute_residuals().values.flatten()
        fig, ax = plt.subplots(ncols=2, **subplots_kws)
        for axis in ax:
            axis.hist(residuals_flattened**2, bins=bins)
            axis.set_yscale('log')
            axis.set_ylabel('count')
            axis.set_xlabel('squared residual')
        ax[1].set_xscale('log')
        return fig, ax

    def scatter_residuals_against(self, data_dict, axis='items'):
        """Plot mean squared error (averaged at the given axis) versus data
        about the elements of that axis, given as a dictionary of data such as:
            {'data_description':
                {'item0': 24289.1415161326,
                 'item1': 569.94072879329997,
                 'item2': 3886.4793543251999,
                 ...}}
        """
        keys = data_dict.keys()
        fig, ax = plt.subplots(nrows=len(keys), ncols=2)
        ax = np.atleast_2d(ax)
        residuals = self.compute_residuals()
        squared_residuals = residuals.apply(lambda x: x**2)
        squared_residuals_panel = multiindex_to_panel(squared_residuals)
        for i, key in enumerate(keys):
            data_to_plot = data_dict[key]
            ax[i, 0].set_xlabel(key)
            ax[i, 1].set_xlabel(key)
            ax[i, 0].set_ylabel('MSE')
            ax[i, 1].set_ylabel('MSE')
            if axis == 'items':
                squared_residuals_to_plot = (
                    squared_residuals_panel.mean('minor_axis').mean(0))
            else:
                squared_residuals_to_plot = (
                    squared_residuals_panel.mean('items').mean(0))
            squared_residuals_to_plot = (squared_residuals_to_plot
                                         .rename(columns={0: 'MSE'}))
            plot_this = np.array(
                [(data_to_plot.get(k, np.nan),
                  squared_residuals_to_plot.get(k, np.nan))
                 for k in attrgetter(axis)(squared_residuals_panel)])
            ax[i, 0].scatter(*plot_this.T)
            ax[i, 1].set_xscale('log')
            ax[i, 1].scatter(*plot_this.T)
        return fig, ax

    # def density_mean_square_error_2d(
    #         self, use_full_dimensions=False, n_points_x=100, n_points_y=100,
    #         xlabel='first dimension', ylabel='second dimension',
    #         log_color=False):
    #     residuals = self.compute_residuals(use_full_dimensions=False)
    #     mean_square_errors = np.mean(np.square(residuals), axis=1)

    #     if self.X.shape[1] < 2:
    #         raise ValueError('The data must have at least 2 features.')

    #     return density_list_plot_2d(
    #         *self.X.values[:, :2].T, err,
    #         error_label='mean square\nerror',
    #         n_points_x=n_points_x, n_points_y=n_points_y,
    #         xlabel=xlabel, ylabel=ylabel, log_color=log_color)


class SKLearnPanelModel(_BaseDimensionReducingPanelModel):
    """Methods specific to models that are sklearn regressors."""

    # @classmethod
    # def init_from_fitted_model()

    @property
    def best_model_filename(self):
        return 'best_model.pkl'

    def save_best_model(self):
        """Pickle the best model (in the hyperparameter search) and the
        sklearn version."""
        joblib.dump(self.best_model, self.path_to_best_model)
        super().save_best_model()

    def load_best_model(self):
        """Load the best model from a file.

        The best model is stored in the attribute `best_model`."""
        assert os.path.exists(self.path_to_best_model)
        print('Loading the best model found in path', self.path_to_best_model)
        self.best_model = joblib.load(self.path_to_best_model)
        super().load_best_model()

    def split_best_model_into_dimension_reducer_and_predictor(
            self, step_of_dimension_reducer=None):
        """Split the model into a dimension reducer and predictor.

        If the model is a Pipeline and a string is given for
        `step_of_dimension_reducer`, then the model is split at the step
        specified by `step_of_dimension_reducer`; otherwise, it is assumed that
        the model does not reduce dimensions as an intermediate step.

        At the end, the number of reduced dimensions in the model is computed.
        """
        self.check_that_best_model_exists()

        is_pipeline = isinstance(self.best_model, Pipeline)
        if is_pipeline and step_of_dimension_reducer is not None:
            self.dimension_reducing_model = (
                DimensionReducingPipeline(self.best_model,
                                          step_of_dimension_reducer))
        else:
            self.dimension_reducing_model = (
                ModelThatDoesNotReduceDimensions(self.best_model))
        self._compute_n_reduced_dimensions_in_model()


class SINDyPanelModel(SKLearnPanelModel):
    """Add features specific to SINDy models."""

    def __init__(
            self, panel, model_with_hyperparameter_search,
            model_predicts_change,
            validate_on_full_dimensions=False, num_lags=1,
            dim_reducer_preprocessing=None,
            fit_preprocessing_dimension_reduction_to_train=True,
            fill_missing_value=0.0,
            metric=sklearn.metrics.mean_squared_error,
            metric_greater_is_better=False,
            results_directory='results',
            feature_expander_step_name='feature_expander',
            overwrite_existing_results=False,
            already_fitted_model_in_directory=None):
        """SINDy model of a panel dataset.

        SINDy-specific inputs
        ---------------------
        feature_expander_step_name : str, default: 'feature_expander'
            The name of the feature expanding step of the Pipeline.
        """
        SINDyPanelModel.__doc__ += SKLearnPanelModel.__doc__
        super(self.__class__, self).__init__(
            panel, model_with_hyperparameter_search, model_predicts_change,
            validate_on_full_dimensions=validate_on_full_dimensions,
            num_lags=num_lags,
            dim_reducer_preprocessing=dim_reducer_preprocessing,
            fit_preprocessing_dimension_reduction_to_train=(
                fit_preprocessing_dimension_reduction_to_train),
            fill_missing_value=fill_missing_value,
            metric=metric,
            metric_greater_is_better=metric_greater_is_better,
            results_directory=results_directory,
            overwrite_existing_results=overwrite_existing_results,
            already_fitted_model_in_directory=(
                already_fitted_model_in_directory))
        # SINDy-specific instance variables
        self.check_that_named_step_is_in_pipeline(feature_expander_step_name)
        self.feature_expander_step_name = feature_expander_step_name

    def split_best_model_into_dimension_reducer_and_predictor(self):
        """Split the model into a dimension reducer and predictor.

        For a SINDy model, there is no internal dimension reducer; just a
        preprocessing one (`self.dim_reducer_preprocessing`)."""
        self.check_that_best_model_exists()
        self.dimension_reducing_model = (
            ModelThatDoesNotReduceDimensions(self.best_model))
        self._compute_n_reduced_dimensions_in_model()

    def fit_model_to_entire_dataset_and_save_best_model(
            self, warnings_action='default', **fit_kws):
        """Fit the model to the entire dataset and save the best model.

        After fitting the model, the best model is split into a dimension
        reducer and predictor.
        """
        fit_time = super().fit_model_to_entire_dataset_and_save_best_model(
            warnings_action=warnings_action, **fit_kws)
        self.split_best_model_into_dimension_reducer_and_predictor()
        return fit_time

    def load_best_model(self):
        super().load_best_model()
        self.split_best_model_into_dimension_reducer_and_predictor()

    def check_that_named_step_is_in_pipeline(self, step_name):
        err = ValueError(
            '{} is not one of the steps in the pipeline'.format(step_name))
        if hasattr(self.model, 'estimator'):
            if step_name not in self.model.estimator.named_steps:
                raise err
        else:
            if step_name not in self.model.named_steps:
                raise err

    @property
    def sympy_expressions_of_reduced_dimension_labels(self):
        return [sym.var('x' + str(x))
                for x in range(self.n_reduced_dimensions_in_model)]

    def analyze(self, **subplots_kws):
        self.print_equations()
        fig, ax = plt.subplots(nrows=2, **subplots_kws)
        self.plot_coefficients(ax=ax[0])
        self.correlations_between_expanded_features(ax=ax[1])
        fig.subplots_adjust(hspace=.3)
        return fig, ax

    def correlations_between_expanded_features(self, ax=None, **subplots_kws):
        """Plot correlations between features of the model at a certain step
        in a pipeline.
        """
        self.check_that_best_model_exists()
        data_expanded, feature_names = (
            self.compute_expanded_features_and_their_symbolic_expressions())
        fig, ax = create_fig_ax(ax, **subplots_kws)
        cmap = shifted_color_map(mpl.cm.BrBG, data=data_expanded)
        mat_plot = ax.matshow(
            np.corrcoef(data_expanded, rowvar=False), cmap=cmap)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='4%', pad=0.04)
        cbar = fig.colorbar(mappable=mat_plot, cax=cax, orientation='vertical')
        cbar.ax.set_ylabel('Pearson correlation coefficient')
        ax.set_xticklabels([''] + feature_names, rotation=90)
        ax.set_yticklabels([''] + feature_names)
        ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(1))
        ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(1))
        return fig, ax

    def compute_expanded_features_and_their_symbolic_expressions(
            self, symbolic=True):
        self.check_that_best_model_exists()
        feature_expander, __ = split_pipeline_at_step(
            self.best_model, self.feature_expander_step_name)
        data_expanded = feature_expander.fit_transform(self.X)
        feature_names = feature_expander.steps[-1][1].get_feature_names(
            symbolic=symbolic,
            input_features=self.sympy_expressions_of_reduced_dimension_labels)
        # TODO: This does not yet handle a multi-step feature expander.
        # Need to propagate the feature expressions through the pipeline.
        return data_expanded, feature_names

    def print_equations(self, latex=False, num_sig_digits=3):
        """Print the equations of the model.

        The feature expanding step is the second to last step in the pipeline.
        This step must return SymPy expressions in a method with signature
        `get_feature_names(symbolic=True)`.
        `print_equations` takes the dot product of the output of the final step
        with the symbolic feature names of the feature expander.

        Parameters
        ----------
        latex : bool, default: False
            Whether to return the expressions in LaTeX format

        num_sig_digits : int, default: 3
            Number of significant digits to include in the numerical values

        Prints
        ------
        equations : str
            The equations as a string with new line characters separating
            equations
        """
        self.check_that_best_model_exists()
        __, feature_names = (
            self.compute_expanded_features_and_their_symbolic_expressions())

        regressor_step = self.best_model.steps[-1][1]
        equations_rhs = np.dot(regressor_step.coef_, feature_names)

        def formatter(x):
            return sym.latex(sym.N(x, n=num_sig_digits)) if latex else x

        print('Equations inferred by {}:'.format(
            self._get_filename_of_attribute('model')))
        print()

        if self.model_predicts_change:
            if latex:
                change_or_next = '\Delta'  # noqa
            else:
                change_or_next = 'change in'
        else:
            change_or_next = 'next'

        equations = ''
        for i, eq in enumerate(equations_rhs):
            equations += '{next} {var} {align}= {eq}{newline}'.format(
                next=change_or_next, var=formatter(feature_names[i + 1]),
                align='&' if latex else '',
                eq=formatter(eq),
                newline=('\\\\\n' if latex and i < len(equations_rhs) - 1
                         else ''))
        print(equations)

    def plot_coefficients(self, label_zeros=True, ax=None, **subplots_kws):
        """Plot coefficients of a SINDy model.

        The regressor step must be the last step in the pipeline.
        This step in the pipeline must have a 'coef_' attribute."""
        self.check_that_best_model_exists()
        __, feature_names = (
            self.compute_expanded_features_and_their_symbolic_expressions())

        regressor = self.best_model.steps[-1][1]
        fig, ax = create_fig_ax(ax, **subplots_kws)
        cmap = shifted_color_map(mpl.cm.PuOr_r,
                                 data=np.vstack(regressor.coef_))
        mat_plot = ax.matshow(np.vstack(regressor.coef_), cmap=cmap)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='4%', pad=0.04)
        cbar = fig.colorbar(mappable=mat_plot, cax=cax, orientation='vertical')
        cbar.ax.set_ylabel('coefficient')

        prefix = 'change in ' if self.model_predicts_change else 'next '
        ax.set_yticklabels(
            [''] +
            [prefix + str(l)
             for l in self.sympy_expressions_of_reduced_dimension_labels])
        ax.set_xticklabels([''] + feature_names, rotation=90)
        ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(1))
        ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(1))

        if label_zeros:
            for i in range(len(regressor.coef_)):
                for j in range(len(regressor.coef_[i])):
                    if regressor.coef_[i, j] == 0:
                        ax.text(j, i, 0, ha='center', va='center')
        return fig, ax


def clip_array_at_percentiles(arr, percentiles):
    clip_lower = np.percentile(arr, percentiles[0])
    clip_upper = np.percentile(arr, percentiles[1])
    return np.clip(arr, clip_lower, clip_upper)
