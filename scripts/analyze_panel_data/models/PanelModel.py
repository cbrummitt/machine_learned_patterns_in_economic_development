# Authors: Charlie Brummitt <brummitt@gmail.com>
#          Andres Gomez-Lievano <andres_gomez@hks.harvard.edu>
#          Mali Akmanalp <mali_akmanalp@hks.harvard.edu>
"""
The SymbolicFeatures module expands data into polynomial features and into
arbitrary symbolic expressions.
"""
import itertools
import os
import pickle
import warnings
from operator import attrgetter

import imageio
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.metrics
import sympy as sym
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.spatial import ConvexHull
from sklearn.exceptions import NotFittedError
from sklearn.utils import check_array

import analyze_panel_data.model_selection.split_data_target as split_data_target  # noqa
from analyze_panel_data.model_selection.utils import cross_val_score_with_times

from ..utils.convert_panel_dataframe import (multiindex_to_panel,
                                             panel_to_multiindex)
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


class PanelModel(object):
    def __init__(
            self, panel, model, model_predicts_change, cv_outer,
            validation_objective=sklearn.metrics.mean_squared_error,
            metrics=regression_metrics,
            results_folder='results', num_lags=1):
        """A panel dataset and model of it, with methods to fit and analyze the
        model.

        Parameters
        ----------
        panel : pandas Panel

        model : an estimator with fit and predict methods

        model_predicts_change : bool
            Whether the model predicts the change between one time step and the
            next or predicts the value at the next time step.

        cv_outer : cross-validation splitter
            An object that has a `split` method that yields a pair of
            train and test indicies: (train_indices, test_indices)

        validation_objective : callable
            A function that is given y_true, y_predicted and returns a score.

        metrics : list of metrics
            What metrics to compute when evaluating the performance on a test
            set.

        results_folder : name of folder in which to put results
            The results will be put into 'results_folder/{self.filename}/'
            where `self.filename` is created by combining the `filename`
            attributes of the `panel`, dimension reducer `dim_reducer`,
            `model`, `inner_cv`, and `outer_cv`.
        """
        self.panel = panel
        self.model = model
        self.model_predicts_change = model_predicts_change
        self.cv_outer = cv_outer
        self.num_lags = num_lags
        self.set_names_and_filenames()
        self.create_results_paths(results_folder)
        self.compute_data_target()
        self.set_up_times_in_outer_cv_and_model_cv_if_needed()

    def set_names_and_filenames(self):
        self.model_name = getattr(self.model, 'name', 'model')
        self.model_filename = getattr(self.model, 'filename', 'model')
        self.panel_filename = getattr(self.panel, 'filename', 'panel')
        self.cv_outer_filename = getattr(self.cv_outer, 'filename', 'cv_outer')
        self.filename = (
            '{panel_filename}__{model_filename}__'
            'outer_cv_{cv_outer_filename}').format(**self.__dict__)

    def create_results_paths(self, results_folder):
        self.results_path = os.path.join(results_folder, self.filename)
        os.makedirs(self.results_path, exist_ok=True)
        self.animations_path = os.path.join(self.results_path, 'animations')
        os.makedirs(self.animations_path, exist_ok=True)

    def compute_data_target(self):
        split = split_data_target.split_multiindex_dataframe_into_data_target

        def split_panel(panel, num_lags, target_is_difference):
            return (
                panel
                .pipe(panel_to_multiindex).dropna(how='all', axis=0).fillna(0)
                .pipe(split, num_lags, lag_label='lag',
                      target_is_difference=target_is_difference))
        lagged, unlagged = split_panel(self.panel,
                                       num_lags=self.num_lags,
                                       target_is_difference=False)
        __, unlagged_differenced = split_panel(self.panel,
                                               num_lags=self.num_lags,
                                               target_is_difference=True)
        self.X = self.data = lagged
        self.y = self.target = (unlagged_differenced
                                if self.model_predicts_change
                                else unlagged)
        return

    def set_up_times_in_outer_cv_and_model_cv_if_needed(self):
        """If cv_outer was given `level_of_index_for_time_values`
        then assign their `times` attribute to be the level
        `level_of_index_for_time_values` of the index of `self.X`. If
        `self.model` has a `cv` attribute, then set the `times` attribute
        of `self.model.cv` in the same way.

        This is needed because we cannot pass `times` to `split` without
        forking sklearn, and we cannot do this assignment of the cv's times
        in its `split` method for Keras because Keras cannot take pandas
        DataFrames as input to `fit` methods.
        """
        def set_times_attribute_if_needed(cv):
            need_to_set_time_attribute_of_cv = (
                hasattr(cv, 'level_of_index_for_time_values') and
                cv.times is None and
                hasattr(self.X, 'index') and hasattr(self.X.index, 'levels'))

            if need_to_set_time_attribute_of_cv:
                level_name = cv.level_of_index_for_time_values
                if level_name not in self.X.index.names:
                    raise ValueError(
                        'The level name {} is not a level in '
                        'the index of self.X; it should be an element of '
                        '{}'.format(level_name, self.X.index.names))
                cv.times = self.X.index.get_level_values(level_name)

        set_times_attribute_if_needed(self.cv_outer)
        if hasattr(self.model, 'cv'):
            set_times_attribute_if_needed(self.model.cv)

    def pickle_self(self):
        with open(self.filename, 'wb') as f:
            pickle.dump(self, f)

    def compute_cross_val_score(self, warnings_action='default', force=False,
                                pickle_result=True,
                                pickle_filename='cross_val_score.pkl',
                                verbose=1, **kwargs):
        """Compute the nested cross-validation scores of the model."""
        pickle_path = os.path.join(
            self.results_path, pickle_filename)
        if os.path.exists(pickle_path) and not force and pickle:
            if verbose:
                msg = ("Already computed 'cross_val_score' for \n\t{name}."
                       "Loading it from the path\n\t{path}")
                print(msg.format(name=self.filename, path=pickle_path))
            with open(pickle_path, 'rb') as f:
                scores = pickle.load(f)
        else:
            with warnings.catch_warnings():
                warnings.simplefilter(warnings_action)
                scores = cross_val_score_with_times(
                    self.model, X=np.array(self.X), y=np.array(self.y),
                    cv=self.cv_outer, times=self.X.index.get_level_values(1),
                    **kwargs)
            if pickle_result:
                with open(pickle_path, 'wb') as f:
                    pickle.dump(scores, f)
        self.cross_val_scores = scores
        self.mean_cross_val_score = np.mean(scores)
        return scores

    def fit_to_entire_dataset(
            self, warnings_action='default', force=False,
            pickle_filename='model_fit_to_entire_dataset.pkl', **kwargs):
        """Fit the model to the entire, dimension-reduced panel dataset."""
        pickle_path = os.path.join(self.results_path, pickle_filename)
        if os.path.exists(pickle_path) and not force:
            with open(pickle_path, 'rb') as f:
                self.model = pickle.load(f)
        else:
            with warnings.catch_warnings():
                warnings.simplefilter(warnings_action)
                self.model.fit(np.array(self.X), np.array(self.y),
                               **kwargs)
            with open(pickle_path, 'wb') as f:
                pickle.dump(self.model, f)
        return

    def print_parameters_at_boundary_of_parameter_grid(self):
        """Print which parameters of the parameter grid is at its boundary.

        This is only relevant for models that are a `GridSearchCV`, meaning
        that they have attributes `param_grid` and `best_params_`.
        """
        if not hasattr(self.model, 'best_params_'):
            raise NotFittedError("The model has not yet been fit or it does"
                                 " not have a 'best_params_' attribute.")
        else:
            at_least_one_param_at_boundary = False
            for param in self.model.best_params_:
                param_grid = self.model.param_grid[param]
                best_param_value = self.model.best_params_[param]
                if (len(param_grid) >= 2 and
                        list(param_grid).index(best_param_value)
                        in [0, len(param_grid) - 1]):
                    at_least_one_param_at_boundary = True
                    msg = ('{param} = {value} is at the boundary of its'
                           'parameter grid {param_grid}')
                    print(msg.format(param=param, param_grid=param_grid,
                                     value=best_param_value))
            if not at_least_one_param_at_boundary:
                print('All parameters are in the interior of their grid.')

    def print_equations(self):
        """Print the equations of the model. Only works for SINDy (i.e., models
        with a sklearn.pipelien.Pipeline containing a symbolic feature expander
        (such as `SymbolicPolynomialFeatures`) and then a regressor)."""
        fitted_model = self.model.best_estimator_
        feature_names = (
            fitted_model.steps[0][1].get_feature_names(symbolic=True))

        reg = fitted_model.named_steps['regressor']
        equations_rhs = np.dot(reg.coef_, feature_names)

        print('Equations inferred by {}:'.format(self.model_name))
        print()
        for i, eq in enumerate(equations_rhs):
            change_or_next = (
                'change in' if self.model_predicts_change else 'next')
            print('{next} x{i} = {eq}'.format(next=change_or_next, i=i, eq=eq))

    def plot_coefficients(self, figsize=(12, 8), label_zeros=False):
        """Plot coefficients of a SINDy model.

        The model (`self.model`) must have a `best_estimator_` attribute, which
        must have a 'regressor' step in its pipeline, which must have a 'coef_'
        attribute."""
        if not hasattr(self.model, 'best_estimator_'):
            raise NotFittedError(
                ("The model {} has not yet been fitted. call "
                 "'fit_to_entire_dataset' first.").format(self.model_name))

        fitted_model = self.model.best_estimator_

        def wrap_parens_if_needed(expression):
            if ' ' in expression:
                return '({})'.format(expression.replace(' ', ''))
            else:
                return expression
        input_features = sym.var(
            [wrap_parens_if_needed(x)
             for x in self.panel.minor_axis.values])
        feature_names = (
            fitted_model.steps[0][1].get_feature_names(
                symbolic=True, input_features=input_features))

        reg = fitted_model.named_steps['regressor']

        fig, ax = plt.subplots(figsize=figsize)
        cmap = shifted_color_map(mpl.cm.PuOr_r, data=np.vstack(reg.coef_))
        mat_plot = ax.matshow(np.vstack(reg.coef_), cmap=cmap)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='4%', pad=0.04)
        cbar = fig.colorbar(mappable=mat_plot, cax=cax, orientation='vertical')
        cbar.ax.set_ylabel('coefficient')

        prefix = 'change in ' if self.model_predicts_change else 'next '
        ax.set_yticklabels(
            [''] + [prefix + l for l in self.panel.minor_axis.values])
        ax.set_xticklabels([''] + feature_names, rotation=90)
        ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(1))
        ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(1))

        if label_zeros:
            for i in range(len(reg.coef_)):
                for j in range(len(reg.coef_[i])):
                    if reg.coef_[i, j] == 0:
                        ax.text(j, i, 0, ha='center', va='center')
        return fig, ax

    def correlations_between_features(self):
        """Plot correlations between features of the model.

        Only works for models that are a pipeline with a feature expander as
        the first step."""
        if (hasattr(self.model, 'best_estimator_') and
                hasattr(self.model.best_estimator_, 'steps')):
            feature_expander = self.model.best_estimator_.steps[0][1]
        else:
            return 'The model does not have a `steps` attribute.'
        data_expanded = feature_expander.fit_transform(self.data)
        feature_names = feature_expander.get_feature_names()

        fig, ax = plt.subplots()
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

    def quiver_plot_of_predictions(
            self, dim_reducer=None, dimensions_to_keep=(0, 1),
            aggregator='mean',
            n_points_each_dim=30,
            color_values='speed', colorbar_label='speed',
            ax=None, save_fig=None,
            mask_velocity_to_convex_hull_of_data=True,
            axis_labels_dict=None,
            **subplots_kws):
        """Create a quiver plot of predictions of the model on a grid.

        Parameters
        ----------
        dim_reducer : dimension reducer with `fit`, `transform`,
        `fit_transform`, and `inverse_transform` methods
            The dimension reducer to use to reduce dimensions enough to make
            a grid of points. It is fit to `self.data.loc[:, 1]` and then used
            to transform `self.data.loc[:, 1]` and `self.target`.

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

        subplots_kws : keyword arguments to pass to plt.subplots, default: None

        Returns
        -------
        fig, ax : matplotlib Figure, Axis
        """
        mask_to_convex_hull = mask_velocity_to_convex_hull_of_data
        grid_points, velocities = self.compute_predictions_on_grid(
            n_points_each_dim=n_points_each_dim,
            dim_reducer=dim_reducer,
            mask_velocity_to_convex_hull_of_data=mask_to_convex_hull)
        grid_points, velocities = (
            aggregate_dimensions_of_grid_points_and_velocities(
                grid_points, velocities, dimensions_to_keep,
                aggregator=aggregator))
        return vis_model.quiver_plot(
            *grid_points, *[v.T for v in velocities],
            **make_axis_labels(axis_labels_dict, dimensions_to_keep),
            color_values=color_values,
            colorbar_label=colorbar_label, ax=ax,
            save_fig=save_fig,
            **subplots_kws)

    def streamplots_of_all_pairs(
            self, dim_reducer=None, n_components=2, figsize=(12, 4), nrows=1,
            n_points_each_dim=30, axis_labels_dict=None,
            mask_velocity_to_convex_hull_of_data=True):
        mask_to_convex_hull = mask_velocity_to_convex_hull_of_data
        grid_points, velocities = self.compute_predictions_on_grid(
            dim_reducer, n_points_each_dim=n_points_each_dim,
            mask_velocity_to_convex_hull_of_data=mask_to_convex_hull)
        pairs_components = list(itertools.combinations(range(n_components), 2))
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
            vis_model.stream_plot(
                *grid_pts_reduced, *[v.T for v in velocities_reduced],
                ax=axes_flat[col],
                colorbar_label=('speed' if col == len(pairs_components) - 1
                                else ''),
                **make_axis_labels(axis_labels_dict, pairs_components[col],
                                   label='principal component'))
        plt.tight_layout()
        return fig, ax

    def compute_predictions_on_grid(
            self, dim_reducer, n_points_each_dim=30,
            mask_velocity_to_convex_hull_of_data=True, aggregator='mean'):
        """Compute predictions on a grid of points, potentially
        dimension-reduced.

        Parameters
        ----------
        dim_reducer : dimension reducer with `fit`, `transform`,
        `fit_transform`, and `inverse_transform` methods
            The dimension reducer to use to reduce dimensions enough to make
            a grid of points. It is fit to `self.data.loc[:, 1]` and then used
            to transform `self.data.loc[:, 1]` and `self.target`.

        n_points_each_dim : scalar or tuple of length data.shape[1]
            The number of grid points to use each dimension. If this parameter
            is a scalar, then it is duplicated for every column of `data`.

        Returns
        -------
        grid_points : list of 1D arrays of shape given by `n_points_each_dim`
            1D list s of locatiaons of the grid points in the (reduced)
            dimensions.
            If `n_points_each_dim` is a scalar, then each element of this
            list is an array with `n_points_each_dim` many numbers. Otherwise,
            the shape of `grid_points[i]` is `n_points_each_dim[i],`.

        velocities : list of ND arrays of shape specified by n_points_each_dim
            The predicted velocities at each grid point.
        """
        # Need to grab the value 1 time step ago, so use `data.loc[:, 1]`
        if dim_reducer is not None:
            data_dim_reduced = dim_reducer.fit_transform(self.data.loc[:, 1])
            # target_dim_reduced = dim_reducer.transform(self.target)
        else:
            data_dim_reduced = self.data.loc[:, 1]
            # target_dim_reduced = self.target

        grid_points, meshgrids = bounding_grid(
            data_dim_reduced, n_points_each_dim=n_points_each_dim)

        meshgrids_long_format = np.array([ary.flatten()
                                          for ary in meshgrids]).T
        n_features = len(meshgrids)
        n_points_each_dim = meshgrids[0].shape
        n_grid_points = np.prod(n_points_each_dim)
        assert meshgrids_long_format.shape == (n_grid_points, n_features)

        # Invert the dimension reduction and predict using the model:
        meshgrids_long_format_dim_increased = dim_reducer.inverse_transform(
            meshgrids_long_format)
        predictions_long_format = self.model.predict(
            meshgrids_long_format_dim_increased)

        # Dimension-reduce back to a small number of dimensions:
        predictions_long_format_dim_reduced = dim_reducer.transform(
            predictions_long_format)

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

    def iterated_predictions(
            self, items=None, num_time_steps=100,
            index_of_initial_condition=-1, dim_reducer=None,
            as_dataframe=False):
        """Compute iterated predictions of certain items in the panel.

        Parameters
        ----------
        items : list of strings (items in the panel) or None, default: None
            The items to select from the panel and to make predictions. If
            None, then use all items in the panel.

        num_time_steps : int or 'length_trajectory'
            The number of time steps to predict into the future. If
            `num_time_steps` is 'length_trajectory', then `num_time_steps` is
            set to the length of the trajectory of that time.

        index_of_initial_condition : int, optional, default: -1
            The index of the item's trajectory to use as initial conditon. If
            -1, then the initial condition is the last observation; if 0, then
            the intial condition is the first observation.

        dim_reducer : None or dimension reducer, optional, default: None
            If not None, then `dim_reducer` must be a dimension reducer
            such as `sklearn.decomposition.PCA`. This dimension reducer
            is fit to `self.data.loc[:, 1]` (the time-series with one lag)
            and transforms the trajectories.

        as_dataframe : bool, optional, default: False
            Whether to make each trajectory a DataFrame

        Returns
        -------
        items_to_trajectories : dict mapping strings to arrays of shape
        [n_time_steps, n_features]
            Dictionary mapping the items to their trajectories.

        TODO:
            This does not yet handle models with multiple time lags.
            Need to check that `index_of_initial_condition` leaves enough
            samples in the history to be able to make predictions.
        """
        items_to_trajectories = {}
        if items is None:
            items = self.panel.items

        for item in items:
            item_df = self.panel.loc[item].dropna(how='all').fillna(0)
            initial_condition = item_df.iloc[index_of_initial_condition].values

            if num_time_steps in ['length_trajectory', 'length of trajectory']:
                n_steps_to_predict = len(item_df)
            else:
                n_steps_to_predict = num_time_steps

            trajectory = np.empty(
                (n_steps_to_predict, initial_condition.shape[0]))
            trajectory[0] = initial_condition

            for i in range(1, n_steps_to_predict):
                trajectory[i] = self.model.predict(
                    trajectory[i - 1].reshape(1, -1))
                if self.model_predicts_change:
                    trajectory[i] += trajectory[i - 1]
            if as_dataframe:
                trajectory = pd.DataFrame(trajectory)
                trajectory.columns = self.y.columns
                initial_index = item_df.iloc[index_of_initial_condition].name
                trajectory.index = [
                    initial_index + i for i in range(n_steps_to_predict)]
            items_to_trajectories[item] = trajectory

        if dim_reducer:
            dim_reducer.fit(self.data.loc[:, 1])
            items_to_trajectories = reduce_dimensions_of_items_to_trajectories(
                items_to_trajectories, dim_reducer)
        return items_to_trajectories

    def plot_trajectories_2d_3d(
            self, items_to_trajectories,
            dim_reducer=None, dimensions_to_keep=slice(None),
            label_predicted_items_at_index=None,
            axis=None, axis_labels_dict=None, labelpad=8, title='model_name',
            show_arrows=True, prediction_arrow_kws=None,
            show_trails=True, xlim=None, ylim=None, zlim=None, save_fig=None,
            prediction_plot_kws={'alpha': 1.0},
            plot_empirical_trajectories=False,
            label_empirical_items_at_time_index=None,
            empirical_plot_kws={'alpha': 0.5},
            empirical_arrow_kws=None):
        """Plot 2D or 3D trajectories, with optional labels of trajectories and
        arrowheads.

        Parameters
        ----------
        items_to_trajectories : dict
            Maps strings to arrays of shape [n_time_steps, n_features]

        dim_reducer : a dimension reducer
            E.g., sklearn.decomposition.PCA, sklearn.decomposition.NMF

        dimensions_to_keep : tuple of 3 int's
            Which dimensions to plot. Each integer must be between 0 and
            the number of columns in the trajectories in
            `items_to_`trajectories`.

        label_predicted_items_at_index : None or int, optional, default: None
            If not None, then write the item name at the part of the trajectory
            given by this integer index. Use 0 for the initial condition or
            -1 for the final position in the trajectory.

        axis : None or matplotlib axis
            The axis on which to put the plot. If None, then create one.

        axis_labels_dict : None or dict
            A dictionary mapping dimension indices to strings, such as
            {0: 'component 0', 1: 'component 1'}.
            If None, then use 'dimension i' for the i-th axis.

        labelpad : int, default: 8
            Padding on the three axis labels

        title : str, default: 'model_name'
            If title is 'model_name', then `self.model_name` is used as the
            title of the axis. Otherwise the value of `title` is used.

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
        """
        prediction_arrow_kws = convert_None_to_empty_dict_else_copy(
            prediction_arrow_kws)
        prediction_plot_kws = convert_None_to_empty_dict_else_copy(
            prediction_plot_kws)
        empirical_arrow_kws = convert_None_to_empty_dict_else_copy(
            empirical_arrow_kws)

        if dim_reducer:
            dim_reducer.fit(self.data.loc[:, 1])
            items_to_trajectories = reduce_dimensions_of_items_to_trajectories(
                items_to_trajectories, dim_reducer)

        n_features = list(items_to_trajectories.values())[0].shape[1]

        if dimensions_to_keep == slice(None):
            dimensions_to_keep = tuple(range(min(3, n_features)))

        trajectories = {item: check_array(trajectory)[:, dimensions_to_keep]
                        for item, trajectory in items_to_trajectories.items()}

        if axis is None:
            fig = plt.figure()
            if n_features == 3:
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

        if title is 'model_name':
            title = self.model_name
        if title:
            ax.set_title(title)

        for item, traj in trajectories.items():
            color = next(color_cycle)['color']
            if show_trails:
                ax.plot(*traj.T, label=item, color=color,
                        **prediction_plot_kws)
            if label_predicted_items_at_index is not None:
                ax.text(
                    *check_array(traj)[label_predicted_items_at_index].T, item,
                    ha='center', va='center', color=color)
            if show_arrows and show_trails and len(traj) >= 2:
                penultimate = traj[-2]
                last_change = traj[-1] - traj[-2]
                ax.quiver(*penultimate, *last_change, color=color,
                          **prediction_arrow_kws)

            if plot_empirical_trajectories:
                empirical_trajectory = (
                    self.panel.loc[item].dropna(how='all').fillna(0))
                if dim_reducer is not None:
                    empirical_data = dim_reducer.transform(
                        empirical_trajectory)
                empirical_trajectory = (
                    np.array(empirical_trajectory)[:, dimensions_to_keep])

                if plot_empirical_trajectories is not 'all':
                    empirical_data = empirical_data[:len(traj)]
                ax.plot(*empirical_data.T, color=color, **empirical_plot_kws)
                if label_empirical_items_at_time_index is not None:
                    ax.text(
                        *empirical_data[label_empirical_items_at_time_index].T,
                        item, ha='center', va='center', color=color)
                if show_arrows and len(empirical_data) >= 2:
                    penultimate = empirical_data[-2]
                    last_change = empirical_data[-1] - empirical_data[-2]
                    ax.quiver(*penultimate, *last_change,
                              color=color, **empirical_arrow_kws)

        maybe_save_fig(ax.get_figure(), save_fig)
        return fig, ax

    def sequence_plot_trajectories_2d_3d(
            self, items_to_trajectories, frames_folder='frames', alpha=.2,
            labelpad=8,
            label_predicted_items_at_index=-1,
            title='model_name',
            dim_reducer=None, dimensions_to_keep=slice(None),
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
            item: check_array(trajectory)[:, dimensions_to_keep]
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
                self.panel
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
            title = ('{model}\nIterated prediction of time {time}'
                     ' starting from {start_time}').format(
                model=self.model_name, time=predicted_time,
                start_time=str(start_time))
            path_of_figure = os.path.join(
                frame_path, 'predictions_3d_{}.png'.format(predicted_time))
            self.plot_trajectories_2d_3d(
                {item: traj[:t]
                 for item, traj in items_to_trajectories_2d_arrays.items()},
                title=title,
                show_arrows=show_arrows, show_trails=show_trails,
                prediction_arrow_kws=prediction_arrow_kws,
                label_predicted_items_at_index=label_predicted_items_at_index,
                xlim=xlim, ylim=ylim, zlim=zlim, save_fig=path_of_figure,
                dimensions_to_keep=dimensions_to_keep,
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
            dim_reducer=None,
            alpha=.2,
            labelpad=8, label_predicted_items_at_index=-1, title='model_name',
            show_arrows=False, show_trails=True,
            dimensions_to_keep=slice(None),
            prediction_plot_kws={'alpha': 1.}, prediction_arrow_kws=None,
            plot_empirical_trajectories=False,
            label_empirical_items_at_time_index=None,
            empirical_plot_kws={'alpha': 0.5},
            empirical_arrow_kws=None,
            fps=25, subrectangles=True,
            gif_filename='iteratedpredictions.gif'):
        """Create a sequence of trajectories for certain items as a GIF and
        return the path to that GIF file.

        This is a helper method that calls the methods
        `iterated_predictions`, `sequence_plot_trajectories_2d_3d`,
        `create_gif`.
        """
        items_to_trajectories = self.iterated_predictions(
            items=items, num_time_steps=num_time_steps,
            dim_reducer=dim_reducer,
            index_of_initial_condition=index_of_initial_condition)

        scatter_file_paths = self.sequence_plot_trajectories_2d_3d(
            items_to_trajectories, alpha=alpha, labelpad=labelpad,
            label_predicted_items_at_index=label_predicted_items_at_index,
            title=title, show_arrows=show_arrows, show_trails=show_trails,
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
                                'rotate_pan_zoom_{:0=4}.png'.format(n))
            fig.savefig(path)
            paths.append(path)
        return self.create_gif(
            paths,
            gif_filename='rotate_pan_zoom_{}_{}fps.gif'.format(filename, fps),
            fps=fps, subrectangles=subrectangles)

    def error_analyses(self, n_items_easiest_hardest_to_predict=10):
        """Do several kinds of analysis of the errors."""
        n_top = n_items_easiest_hardest_to_predict
        print(self.error_analysis(
            n_items_easiest_hardest_to_predict=n_top))
        print(self.residual_histogram())
        print(self.squared_residual_histogram())

    def error_analysis(
            self, n_items_easiest_hardest_to_predict=10, alpha=.3,
            force=False):
        """Analyze errors made by the model."""
        if hasattr(self, 'residuals') and not force:
            residuals = self.residuals.fillna(0)
        else:
            residuals = self.compute_residuals().fillna(0)
        squared_residuals = residuals.apply(lambda x: x**2)
        squared_residuals_panel = multiindex_to_panel(squared_residuals)

        fig, ax = plt.subplots(2, 2)

        for i, axis_to_keep in enumerate(['items', 'minor_axis']):
            axis_to_remove = (
                'minor_axis' if axis_to_keep == 'items' else 'items')
            axis_kw = {'axis': axis_to_remove}
            mse_over_time_mean = squared_residuals_panel.mean(**axis_kw)
            mse_over_time_std = squared_residuals_panel.std(**axis_kw)
            # mse_over_time_max = squared_residuals_panel.max(**axis_kw)

            ylabel = 'MSE of {}\naveraged over {}'.format(
                axis_to_keep, axis_to_remove)
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

    def compute_residuals(self):
        """Compute residuals of predictions of all the training data."""
        predictions = self.model.predict(self.data.values)
        residuals = pd.DataFrame(predictions - self.target.values)
        residuals.index = self.target.index
        residuals.columns = self.target.columns
        self.residuals = residuals.loc[:, 0]
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


def reduce_and_select_dimensions(
        dataframes, dim_reducer, dimensions_to_keep):
    """Reduce dimensions of dataframes, and then select dimensions."""
    if dim_reducer:
        dim_reducer.fit(dataframes[0])
        dataframes = (dim_reducer.transform(df) for df in dataframes)

    for dim in dimensions_to_keep:
        for df in dataframes:
            assert 0 <= dim <= df.shape[1]
    return tuple(check_array(df)[:, dimensions_to_keep] for df in dataframes)


def reduce_dimensions_of_items_to_trajectories(
        items_to_trajectories, dim_reducer):
    """Reduce dimensions of a dict {item: trajectory} using an (already fitted)
    dimension reducer."""
    if dim_reducer is None:
        return items_to_trajectories

    return {item: dim_reducer.transform(trajectory)
            for item, trajectory in items_to_trajectories.items()}
