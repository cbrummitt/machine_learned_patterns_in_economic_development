# inferred_model.py contains functions to visualize
# timeseries, models for predicting them, and their comparison.
#
# Authors:
# Charlie Brummitt <charles_brummitt@hms.harvard.edu>, Github: cbrummitt
# Andres Gomez <Andres_Gomez@hks.harvard.edu>
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Ellipse
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.kernel_ridge import KernelRidge
from sklearn.utils import check_array
from sklearn.metrics import mean_squared_error
from .utils import create_fig_ax, maybe_save_fig
from .utils import convert_None_to_empty_dict_else_copy
from .utils import make_color_cycler
from scipy.spatial import ConvexHull
import itertools
from palettable.colorbrewer.qualitative import Dark2_3
import scipy


country_color_dict = dict(zip([
    'ARG', 'AUS', 'BRA', 'CAN', 'CHN', 'COL', 'ESP', 'FRA', 'GBR', 'GRC',
    'HND', 'IDN', 'IND', 'JPN', 'KEN', 'KOR', 'MDG', 'MEX', 'RUS', 'THA',
    'USA', 'BEL', 'NLD'],
    make_color_cycler()))


def make_data_for_streamplot(
        data, model, n_points_each_dim, model_predicts_change=True,
        mask_velocity_to_convex_hull_of_data=False,
        dim_reducer=None,
        dimensions_to_keep=(0, 1), aggregator='mean'):
    """Make data for a stream plot or quiver plot by creating a grid of equally
    spaced points that contains the data, predicting at those grid points using
    the model, and returning those grid points as 1D vectors and the
    predictions as tensors.

    Parameters
    ----------
    data : array-like, shape [n_samples, n_features]
        The data used to train the model, where `model` was fit with
        `model.fit(data, target)`.

    model : fitted estimator with a predict method

    n_points_each_dim : int or array of ints of length equal to data.ndim
        The number of grid points to use in each dimension

    model_predicts_change : bool, optional, default: True
        Whether the model predicts the change in the system from one time step
        to the next (model_predicts_change == True) or predicts the value of
        the system at the next time step.

    mask_velocity_to_convex_hull_of_data : bool, optional, default: False
        If True, then mask the `velocities` arrays by excluding points outside
        the convex hull of `data`.

    dimensions_to_keep : int, optional, default: 2
        Which dimensions to keep in the tensors of grid points and
        predictions. The other dimensions are aggregated over.

    aggregator : {'mean', 'median'} or user-defined function
        What function to use to aggregate over the higher-order dimensions in
        order to reduce the dimensions of the tensor to keep only those
        dimensions in the tuple `dimensions_to_keep`.
        The function must take as input a tensor (of shape specified by
        `n_points_each_dim`) and an `axis` argument that, when given a tuple
        of integers for which axes to aggregate over, reduces the dimension of
        the tensor by applying the function to those higher-order axes.

    Returns
    -------
    grid_points_1d_for_each_dim : `len(dimensions_to_keep)` many arrays of
    length specified by n_points_each_dim
        The locations of the grid points of the grid at which the model
        predicts the change in the system.

    predictions : arrays of shape specified by the dimensions in the tuple
    dimensions_to_keep with dimensions specified by `n_points_each_dim`
        The predictions at the grid, with the higher-order dimensions reduced
        according to the `aggregator` function.

    Examples
    --------
    >>> grid_points, predictions = make_data_for_streamplot(data, model, 20)
    >>> plt.streamplot(*grid_points, *predictions)
    """
    data = check_array(data)
    grid_points, grids = bounding_grid(data, n_points_each_dim)

    velocities = predict_velocities_on_grid(
        model, grids, model_predicts_change=model_predicts_change)

    if mask_velocity_to_convex_hull_of_data:
        velocities = mask_arrays_with_convex_hull(
            velocities, grid_points, ConvexHull(data))

    def reduce_dimensions_by_aggregating_over_axes(tensors):
        return reduce_dimensions_of_grids(
            tensors,
            dimensions_to_keep=dimensions_to_keep, aggregator=aggregator)

    grids = reduce_dimensions_by_aggregating_over_axes(grids)
    velocities = reduce_dimensions_by_aggregating_over_axes(velocities)

    grid_points_1d_for_each_dim = [grid_points[i] for i in dimensions_to_keep]
    # Must transpose the velocities for streamplot and quiver plot
    velocities = [v.T for v in velocities]

    return grid_points_1d_for_each_dim, velocities


def mask_arrays_with_convex_hull(arrays, grid_points, convex_hull):
    """Mask the arrays to retain only those in the given convex hull.

    Parameters
    ----------
    arrays : list of tensors of shapes n1, n2, n3, ...

    grid_points : list of 1D arrays of length n1, n2, n3, ...
        The arrays are data measured on the Cartesian product of the
        `grid_points`

    convex_hull : ConvexHull of the data

    Returns
    -------
    arrays_masked : NumPy masked arrays of shapes n1, n2, n3, ...
        The given `arrays` with points outside `convex_hull` masked.
    """
    mask = np.zeros(arrays[0].shape, dtype='bool').flatten()
    for i, point in enumerate(itertools.product(*grid_points)):
        mask[i] = not point_in_hull(point, convex_hull)
    mask = mask.reshape(arrays[0].shape)
    return [np.ma.masked_array(data=arr, mask=mask) for arr in arrays]


def point_in_hull(point, hull, tolerance=1e-12):
    return all(
        (np.dot(eq[:-1], point) + eq[-1] <= tolerance)
        for eq in hull.equations)


def example_make_data_for_streamplot(n_samples=10, n_features=2, p=4):
    fake_data = np.random.randn(n_samples, n_features)
    fake_target = np.random.randn(n_samples, n_features)
    krr = KernelRidge()
    krr.fit(fake_data, fake_target)

    grid_points, predictions = make_data_for_streamplot(
        fake_data, krr, p, model_predicts_change=False)


def bounding_grid(data, n_points_each_dim=30, max_num_points=1e7):
    """Creates an evenly spaced grid that contains the data. It returns a
    numpy meshgrid with indexing 'ij'.

    Parameters
    ----------
    data : array-like, shape [num_samples, num_features] with num_features >= 2
        The data to create a grid around.

    n_points_each_dim : scalar or tuple of length data.shape[1]
        The number of grid points to use each dimension. If this parameter
        is a scalar, then it is duplicated for every column of `data`.

    max_num_points : int
        The maximum number of points allowed in the grid; above this value,
        a ValueError is raised.

    Returns
    -------
    grid_points : list of vectors of lengths given by n_points_each_dim
        The grid points in dimensions 0, 1, ..., data.shape[1]

    meshgrid : tensors of grid locations
        The output of np.meshgrid. data.shape[1] many tensors are returned.

    Raises
    ------
    ValueError if `n_points_each_dim` is a tuple or list of length different
    from the number of columns in `data`

    ValueError if the number of points in the grid is larger than
    `max_num_points`

    Examples
    --------
    >>> bounding_grid(np.arange(6).reshape(3, 2), 3)
    [array([[ 0.,  0.,  0.],
            [ 2.,  2.,  2.],
            [ 4.,  4.,  4.]]),
     array([[ 1.,  3.,  5.],
            [ 1.,  3.,  5.],
            [ 1.,  3.,  5.]])]
    """
    dim_n_points_each_dim = len(np.atleast_1d(n_points_each_dim))
    if dim_n_points_each_dim > 1 and dim_n_points_each_dim != data.shape[1]:
        raise ValueError('The number of points requested for each dimension '
                         'is a vector of length {}, but the data has {} col'
                         'umns'.format(dim_n_points_each_dim, data.shape[1]))
    min_of_columns = np.floor(np.min(data, axis=0))
    max_of_columns = np.ceil(np.max(data, axis=0))

    n_points_each_dim = duplicate_to_desired_length_if_scalar(
        n_points_each_dim, data.shape[1])

    n_points_total = np.prod(n_points_each_dim)
    if n_points_total > max_num_points:
        raise ValueError(
            'The bounding grid has {} many points, which exceeds the maximum'
            '`max_num_points`={}'.format(n_points_total, max_num_points))

    start_stop_num = zip(min_of_columns, max_of_columns, n_points_each_dim)
    grid_points = [np.linspace(lo, hi, num) for lo, hi, num in start_stop_num]
    meshgrid = np.meshgrid(*grid_points, indexing='ij')
    return grid_points, meshgrid


def duplicate_to_desired_length_if_scalar(x, desired_length):
    x = np.atleast_1d(x)
    if len(x) == 1 and desired_length > 1:
        x = np.repeat(x, desired_length)
    return x


def predict_velocities_on_grid(model, meshgrids, model_predicts_change=True):
    """Predict the velocities of a system at a mesh of grid points, and return
    the grid and the predicted velocities, which can then be provided as input
    to streamplot and quiver.
    """
    meshgrids_long_format = np.array([ary.flatten() for ary in meshgrids]).T

    n_features = len(meshgrids)
    n_points_each_dim = meshgrids[0].shape
    n_grid_points = np.prod(n_points_each_dim)
    assert meshgrids_long_format.shape == (n_grid_points, n_features)

    predictions_long_format = model.predict(
        meshgrids_long_format)
    assert predictions_long_format.shape == (n_grid_points, n_features)

    predictions_shape_grids = [
        predictions_long_format[:, i].reshape(*n_points_each_dim)
        for i in range(n_features)]
    assert len(predictions_shape_grids) == n_features
    assert predictions_shape_grids[0].shape == meshgrids[0].shape

    if model_predicts_change:
        velocities = predictions_shape_grids
    else:
        meshgrids_preds = zip(meshgrids, predictions_shape_grids)
        velocities = ([pred - grid for grid, pred in meshgrids_preds])
    return velocities


def aggregate_dimensions_of_grid_points_and_velocities(
        grid_points, velocities, dimensions_to_keep, aggregator='mean'):
    """Select dimensions and aggregate over other dimensions.

    Parameters
    ----------
    grid_points : list of 1D arrays

    velocities : list of ND arrays of shape given by the lengths of the
    elements of grid_points

    dimensions_to_keep : tuple of int's length 2
        Which dimensions (features) to keep. Each entry in the tuple is an
        int between 0 and n_features - 1 (inclusive).

    aggregator : {'mean', 'median', or callable}, default: 'mean'
        How to aggregate over axes of the tensor. If callable, it must take
        as input the tensor and a keyword argument `axis` that is given a
        tuple of the indices of the axes to aggregate over.
    """
    grid_points = [grid_points[dim] for dim in dimensions_to_keep]
    velocities = reduce_dimensions_of_grids(
        velocities, dimensions_to_keep, aggregator=aggregator)
    return grid_points, velocities


def reduce_dimensions_of_grids(
        grids, dimensions_to_keep=(0, 1), aggregator='mean'):
    """Reduce the dimensions of tensors by aggregating over some dimensions.

    Inputs
    ------
    grids : a list of arrays of the same shape

    dimensions_to_keep : tuple of int's between 0 and the number of dimensions
    of the tensors in `grids`
        Which dimensions to keep. The other dimensions are aggregated over.

    aggregator : {'mean', 'median', or callable}, default: 'mean'
        How to aggregate over axes of the tensor. If callable, it must take
        as input the tensor and a keyword argument `axis` that is given a tuple
        of the indices of the axes to aggregate over.
    """
    for i in range(len(grids)):
        for j in range(i + 1, len(grids)):
            assert grids[i].shape == grids[j].shape

    axes_to_aggregate = tuple(
        dim for dim in range(grids[0].ndim) if dim not in dimensions_to_keep)
    aggregation_function = {
        'mean': np.mean, 'median': np.median}.get(aggregator, aggregator)
    return [aggregation_function(grids[i], axis=axes_to_aggregate)
            for i in dimensions_to_keep]


def quiver_plot_empirical(
        data, target, dimensions_to_keep=(0, 1),
        color_values='speed', colorbar_label='speed',
        ax=None, save_fig=None,
        xlabel='first component', ylabel='second component', **subplots_kws):
    """Create a quiver plot of a dynamical system.

    Parameters
    ----------
    data : array-like, shape [n_samples, n_features]
        Starting points of the arrows. If n_features > 3, then only the first
        two features are plotted.

    target : array-like, shape [n_samples, n_features]
        Ending points of the arrows. If n_features > 3, then only the first
        two features are plotted.

    dimensions_to_keep : tuple of int's length 2
        Which dimensions (features) to plot. Each entry in the tuple is an
        int between 0 and n_features - 1 (inclusive).

    color_values : string or 2D numpy array
        Data for the colors of the arrows in the streamplot. If color_values is
        'speed', then color_values is the magnitude of the velocity.

    colorbar_label : str, optional, default: 'speed'
        The label of the color bar

    ax : matplotlib axis, optional, default: None
        The axis on which to draw the plot. If None, then an axis is created.

    xlabel, ylabel : str, optional,
    default: 'first component', 'second component'
        The labels of the axes

    subplots_kws : keyword arguments to pass to plt.subplots, default: None

    Returns
    -------
    fig, ax : matplotlib Figure, Axis
    """
    assert len(dimensions_to_keep) == 2
    for dim in dimensions_to_keep:
        assert 0 <= dim <= data.shape[1]

    data = check_array(data)[:, dimensions_to_keep]
    target = check_array(target)[:, dimensions_to_keep]
    X, Y, U, V = make_X_Y_U_V_for_quiver_plot(data, target)
    return quiver_plot(X, Y, U, V, color_values=color_values,
                       colorbar_label=colorbar_label, ax=ax, save_fig=save_fig,
                       xlabel=xlabel, ylabel=ylabel, **subplots_kws)


def make_X_Y_U_V_for_quiver_plot(data, target, target_is_difference=False):
    data, target = check_array(data), check_array(target)
    X, Y = data.T
    if target_is_difference:
        U, V = target.T
    else:
        U, V = (target - data).T
    return X, Y, U, V


def quiver_plot(
        X, Y, U, V,
        show_colorbar=True, color_values='speed', colorbar_label='speed',
        xlabel='first component', ylabel='second component',
        ax=None, save_fig=None,
        subplots_kws=None,
        plot_kws={'cmap': mpl.cm.viridis,
                  'angles': 'xy', 'scale': 1, 'scale_units': 'xy'}):
    """Create a quiver plot of a dynamical system with units given by the
    velocities.

    Parameters
    ----------
    X, Y : vectors of length equal to the number of samples
        Horizontal and vertical locations of the tails of the arrows

    U, V : vectors of length equal to the number of samples
        The velocities at the corresponding points in `X`, `Y`

    show_colorbar : bool, default: True
        Whether to show a colorbar.

    color_values : string or 2D numpy array
        Data for the colors of the arrows in the streamplot. If color_values is
        'speed', then color_values is the magnitude of the velocity.

    colorbar_label : str, optional, default: 'speed'
        The label of the color bar

    xlabel, ylabel : str, optional, default: 'first component',
    'second component'
        The labels of the axes

    ax : matplotlib axis, optional, default: None
        The axis on which to draw the plot. If None, then an axis is created.

    save_fig : str or None, default: None
        If not None, then save the figure to the path specified by this string.

    subplots_kws : None or dict
        Keyword arguments to pass to plt.subplots

    plot_kws : None or dict
        Keyword arguments to pass to plt.quiver

    Returns
    -------
    fig, ax : matplotlib Figure, Axis
    """
    subplots_kws = convert_None_to_empty_dict_else_copy(subplots_kws)
    quiver_kws = convert_None_to_empty_dict_else_copy(plot_kws)

    fig, ax = create_fig_ax(ax, **subplots_kws)
    if color_values == 'speed':
        color_values = np.sqrt(U**2 + V**2)
    quiv = ax.quiver(
        X, Y, U, V, color_values, **quiver_kws)
    if show_colorbar:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='4%', pad=0.04)
        cbar = fig.colorbar(mappable=quiv, cax=cax, orientation='vertical')
        cbar.ax.set_ylabel(colorbar_label)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    maybe_save_fig(fig, save_fig)
    return fig, ax


def stream_plot(
        X, Y, U, V,
        show_colorbar=True, color_values='speed', colorbar_label='speed',
        subplots_kws=None, plot_kws=None,
        ax=None, save_fig=None,
        xlabel='first component', ylabel='second component'):
    """Create a stream plot of a dynamical system with units given by the
    velocities.

    Parameters
    ----------
    X, Y : vectors of length equal to the number of samples
        Horizontal and vertical locations of the tails of the arrows

    U, V : vectors of length equal to the number of samples
        The velocities at the corresponding points in `X`, `Y`. These can be
        masked arrays; their missing values will be filled with zero.

    show_colorbar : bool, default: True
        Whether to show a colorbar.

    color_values : string or 2D numpy array
        Data for the colors of the arrows in the streamplot. If color_values is
        'speed', then color_values is the magnitude of the velocity.

    colorbar_label : str, optional, default: 'speed'
        The label of the color bar

    xlabel, ylabel : str, optional, default: 'first component',
    'second component'
        The labels of the axes

    ax : matplotlib axis, optional, default: None
        The axis on which to draw the plot. If None, then an axis is created.

    save_fig : str or None, default: None
        If not None, then save the figure to the path specified by this string.

    subplots_kws : keyword arguments to pass to plt.subplots

    plot_kws : keyword arguments to pass to plt.subplots

    Returns
    -------
    fig, ax : matplotlib Figure, Axis
    """
    subplots_kws = convert_None_to_empty_dict_else_copy(subplots_kws)
    plot_kws = {
        **{'cmap': mpl.cm.viridis},
        **convert_None_to_empty_dict_else_copy(plot_kws)}

    U = fill_mask_if_needed(U, fill_value=0)
    V = fill_mask_if_needed(V, fill_value=0)

    fig, ax = create_fig_ax(ax, **subplots_kws)
    if color_values == 'speed':
        color_values = np.sqrt(U**2 + V**2)
    strm = ax.streamplot(X, Y, U, V, color=color_values, **plot_kws)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if show_colorbar:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='4%', pad=0.04)
        cbar = fig.colorbar(mappable=strm.lines, cax=cax,
                            orientation='vertical')
        cbar.ax.set_ylabel(colorbar_label)
    maybe_save_fig(fig, save_fig)
    return fig, ax


def fill_mask_if_needed(x, fill_value=0):
    if hasattr(x, 'filled'):
        return x.filled(fill_value=fill_value)
    else:
        return x


def plot_ellipse(ax, position, dimensions, label='kernel scale:'):
    width, height = duplicate_to_desired_length_if_scalar(dimensions, 2)
    ellipse = Ellipse(position, width, height,
                      color='y', alpha=.5, clip_on=True)
    ax.add_artist(ellipse)
    ax.text(position[0] - width / 4., position[1] + height / 4.,
            '{lab}\n{w:.2f} x {h:.2f}'.format(lab=label, w=width, h=height))


def select_trajectories(panel, items=None):
    """Select trajectories of certain items in a panel.

    Parameters
    ----------
    panel : pandas Panel
        A panel of trajectories, with time on the major axis, items denoting
        different trajectories, and features on the minor axis.

    items : iterable of strings (each string an item of the `panel`) or None
        The items of the panel to select. If None, then use all items in the
        panel.

    Returns
    -------
    items_to_trajectories : dict
        Dictionary mapping the items to their trajectories.
    """
    items_to_trajectories = {}
    if items is None:
        items = panel.items
    for item in items:
        items_to_trajectories[item] = panel.loc[item].dropna(how='all').values
    return items_to_trajectories


def iterated_predictions(
        panel, model, items=None, model_predicts_change=True,
        num_time_steps=100, index_of_initial_condition=-1):
    """Compute iterated predictions of certain items in a panel.

    Parameters
    ----------
    panel : pandas Panel
        A panel of trajectories, with time on the major axis, items denoting
        different trajectories, and features on the minor axis.

    items : list of strings (items in the panel) or None, default: None
        The items to select from the panel and to make predictions

    model : a fitted estimator that can predict the next year

    model_predicts_change : bool, optional, default: True
        Whether the model predicts the change in the system from one time step
        to the next (model_predicts_change == True) or predicts the value of
        the system at the next time step.

    num_time_steps : int or 'length_trajectory'
        The number of time steps to predict into the future. If num_time_steps
        is 'length_trajectory', then num_time_steps is set to the length of
        the trajectory of that time.

    index_of_initial_condition : int, optional, default: -1
        The index of the item's trajectory to use as initial conditon. If -1,
        then the initial condition is the last observation; if 0, then the
        intial condition is the first observation.

    Returns
    -------
    items_to_trajectories : dict mapping strings to arrays of shape
    [n_time_steps, n_features]
        Dictionary mapping the items to their trajectories.
    """
    items_to_trajectories = {}
    if items is None:
        items = panel.items

    for item in items:
        item_df = panel.loc[item].dropna(how='all')
        initial_condition = item_df.iloc[index_of_initial_condition].values
        if num_time_steps == 'length_trajectory':
            n_steps_to_predict = len(item_df)
        else:
            n_steps_to_predict = num_time_steps
        trajectory = np.empty((n_steps_to_predict, initial_condition.shape[0]))
        trajectory[0] = initial_condition
        for i in range(1, n_steps_to_predict):
            trajectory[i] = model.predict(trajectory[i - 1].reshape(1, -1))
            if model_predicts_change:
                trajectory[i] += trajectory[i - 1]
        items_to_trajectories[item] = trajectory
    return items_to_trajectories


def plot_item_trajectories(
        items_to_trajectories,
        items_to_colors=None,
        dimensions_to_keep=(0, 1),
        ax=None, text_labels=False, shift_text_labels=None,
        arrow_start_index=-2, subplots_kws=None, plot_kws=None,
        arrow_kws={'head_length': 0.5, 'head_width': 0.5},
        text_kws={'fontsize': 10, 'alpha': 1,
                  'bbox': {'alpha': 0.2, 'pad': 1}},
        use_plot_alpha_for_arrowhead=False,
        xlabel=None, ylabel=None,
        save_fig=None):
    """Plot iterated predictions of certain items in a panel.

    Parameters
    ----------
    items_to_trajectories : dict mapping strings to arrays of shape
    [n_time_steps, n_features], or pandas Panel
        Dictionary mapping the items to their trajectories. If a pandas Panel,
        then it is converted to a dictionary mapping items to trajectories.
        If `n_features` is > 2, then only the first two features are plotted.

    items_to_colors : dict mapping items to colors, or None
        The keys of this dictionary are which items' trajectories should be
        plotted. If None, then default colors are used.

    ax : matplotlib axis, optional, default: None
        The axis on which to draw the plot. If None, then an axis is created.

    text_labels : bool, optional, default: False
        Whether to show text labels of the items

    shift_text_labels : dict, optional, default: None
        Provide a dictionary to shift the text labels. By default the labels
        appear at the third-to-last point in the trajectory.

    arrow_start_index : int, optional, default: -3
        The tail of the arrow is located at the trajectory sliced at this
        index. It must be between 0 and the second-to-last index; if not, then
        the second-to-last index is used.

    subplots_kws, plot_kws, arrow_kws, text_kws : dict, optional
        Keyword arguments passed to `subplots`, `plot`, `arrow`, `text`,
        respectively.

    use_plot_alpha_for_arrowhead : bool, default: False
        If True, and if no alpha is specified in `arrow_kws` and 'alpha' is
        specified in `plot_kws`, then set the opacity 'alpha' of the arrowhead
        to the value of alpha in `plot_kws`.

    xlabel, ylabel : None or str
        If not None, then use these as the labels of the x and y axes

    save_fig : str or None, default: None
        If not None, then save the figure to the path specified by this string.
    """
    items_to_colors = convert_None_to_empty_dict_else_copy(items_to_colors)
    subplots_kws = convert_None_to_empty_dict_else_copy(subplots_kws)
    plot_kws = convert_None_to_empty_dict_else_copy(plot_kws)
    arrow_kws = convert_None_to_empty_dict_else_copy(arrow_kws)
    text_kws = convert_None_to_empty_dict_else_copy(text_kws)
    shift_text_labels = convert_None_to_empty_dict_else_copy(shift_text_labels)

    if hasattr(items_to_trajectories, 'major_axis'):  # if it's a panel
        items_to_trajectories = select_trajectories(
            items_to_trajectories)

    color_cycler = make_color_cycler()
    fig, ax = create_fig_ax(ax, **subplots_kws)

    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)

    for item in items_to_trajectories:
        color = items_to_colors.get(item, next(color_cycler))

        trajectory = items_to_trajectories[item][:, dimensions_to_keep]
        plot_keyword_args = plot_kws.copy()
        plot_keyword_args.update(color=color)
        ax.plot(*trajectory.T, **plot_keyword_args)

        try:
            trajectory[arrow_start_index], trajectory[arrow_start_index + 1]
        except IndexError:
            arrow_start_index = len(trajectory) - 1
        if arrow_start_index == -1:
            print('trajectry for item ', item, ' is empty')
        else:
            x, y = trajectory[arrow_start_index]
            dx, dy = (trajectory[arrow_start_index + 1] -
                      trajectory[arrow_start_index])
            if use_plot_alpha_for_arrowhead:
                if 'alpha' not in arrow_kws and 'alpha' in plot_kws:
                    arrow_kws['alpha'] = plot_kws['alpha']
            ax.arrow(x, y, dx, dy, color=color, **arrow_kws)

            if text_labels:
                text_kws.update(color=color)
                text_kws['bbox'].update(color=color)
                x_text, y_text = (
                    trajectory[arrow_start_index] +
                    shift_text_labels.get(item, np.array([0, 0])))
                ax.text(x_text, y_text, item, **text_kws)

    maybe_save_fig(fig, save_fig)
    return fig, ax


def plot_model_predictions_with_trajectories(
        panel, model, model_predicts_change=True,
        items_to_colors=country_color_dict,
        dimensions_to_keep=(0, 1),
        n_steps_to_predict=50,
        arrow_kws={'head_length': 1, 'head_width': 1},
        predictions_plot_kws={'alpha': .9, 'linestyle': '--', 'linewidth': 5,
                              'label': 'iterated 1-year prediction'},
        empirical_plot_kws={'alpha': .3, 'linestyle': '-', 'linewidth': 3,
                            'label': 'empirical data'},
        axis_labels=None, save_fig=None):
    """Plot the model's predictions of the trajectories starting from the
    initial conditions of the items, and plot the predictions into the future
    starting from the last state of the items.
    """
    fig, ax = plt.subplots(nrows=2, figsize=(15, 15))
    items_to_trajectories = select_trajectories(panel, items_to_colors)

    labels_xyz_axes = compute_axes_labels(axis_labels, dimensions_to_keep)

    # Plot actual trajectories on both axes
    for axis in ax:
        arrow_kws_actual_trajectory = arrow_kws.copy()
        if 'alpha' in empirical_plot_kws:
            arrow_kws_actual_trajectory['alpha'] = empirical_plot_kws['alpha']
        plot_item_trajectories(
            items_to_trajectories,
            items_to_colors=items_to_colors,
            dimensions_to_keep=dimensions_to_keep,
            ax=axis,
            text_labels=True,
            arrow_kws=arrow_kws_actual_trajectory,
            plot_kws=empirical_plot_kws,
            **labels_xyz_axes)

    # Plot iterated predictions from the initial year for each country
    iterated_pred = iterated_predictions(
        panel, model,
        items=items_to_colors.keys(), num_time_steps='length_trajectory',
        model_predicts_change=model_predicts_change,
        index_of_initial_condition=0)

    plot_item_trajectories(
        iterated_pred, items_to_colors=items_to_colors,
        dimensions_to_keep=dimensions_to_keep,
        ax=ax[0], arrow_kws=arrow_kws,
        plot_kws=predictions_plot_kws)
    title = 'iterated 1-year predictions starting from the initial condition'
    ax[0].set_title(title)

    # Plot iterated predictions from the last year for each country
    iterated_pred = iterated_predictions(
        panel,
        model=model,
        items=items_to_trajectories.keys(),
        num_time_steps=n_steps_to_predict,
        model_predicts_change=model_predicts_change,
        index_of_initial_condition=-1)

    plot_item_trajectories(
        iterated_pred,
        items_to_colors=items_to_colors,
        ax=ax[1], dimensions_to_keep=dimensions_to_keep,
        arrow_kws=arrow_kws, plot_kws=predictions_plot_kws)
    title = ('iterated 1-year predictions of the next {n_steps} years '
             'starting from the last year').format(n_steps=n_steps_to_predict)
    ax[1].set_title(title)

    pred_patch = mpl.patches.Patch(color='k', **predictions_plot_kws)
    data_patch = mpl.patches.Patch(color='k', **empirical_plot_kws)

    fig.legend(
        handles=[pred_patch, data_patch],
        labels=['iterated 1-year prediction', 'empirical data'],
        loc='upper center', ncol=2, borderaxespad=0., handlelength=6)
    fig.tight_layout()

    maybe_save_fig(fig, save_fig)
    return fig


def plot_iterated_predictions_for_every_item(
        panel, model, model_predicts_change=True, n_columns=6,
        figsize=(16, 40), size_of_each_plot=None,
        prediction_color=Dark2_3.mpl_colors[0],
        empirical_color=Dark2_3.mpl_colors[1],
        dimensions_to_keep=(0, 1),
        save_fig=None):
    """Plot the iterated prediction and empirical trajectory for every item.

    The prediction and empirical trajectories are plotted on separate axes, one
    axis for each time.
    """
    items_to_empirical_trajectories = select_trajectories(panel)

    iterated_pred = iterated_predictions(
        panel, model, items_to_empirical_trajectories.keys(),
        num_time_steps='length_trajectory',
        model_predicts_change=model_predicts_change,
        index_of_initial_condition=0)

    mse_of_final_predictions = sorted([
        (item, mean_squared_error(items_to_empirical_trajectories[item][-1],
                                  iterated_pred[item][-1]))
        for item in items_to_empirical_trajectories],
        key=lambda item_err: item_err[1])

    n_rows = int(np.ceil(len(items_to_empirical_trajectories) / n_columns))
    if size_of_each_plot:
        figsize = (n_columns * size_of_each_plot[0],
                   n_rows * size_of_each_plot[1])
    else:
        figsize = figsize
    fig, axes = plt.subplots(ncols=n_columns, nrows=n_rows,
                             sharex=True, sharey=True, figsize=figsize)
    axes_flat = axes.flatten()

    predictions_plot_kws = {
        'alpha': 1, 'linestyle': '-', 'linewidth': 2,
        'color': prediction_color}
    empirical_plot_kws = {
        'alpha': .4, 'linestyle': '-', 'linewidth': 2,
        'color': empirical_color}

    pred_patch = mpl.patches.Patch(**predictions_plot_kws)
    data_patch = mpl.patches.Patch(**empirical_plot_kws)

    for i, (item, mse_final_pred) in enumerate(mse_of_final_predictions):
        plot_item_trajectories(
            {item: iterated_pred[item]}, ax=axes_flat[i],
            plot_kws=predictions_plot_kws, arrow_start_index=-2,
            dimensions_to_keep=dimensions_to_keep,
            items_to_colors={item: prediction_color})
        plot_item_trajectories(
            {item: items_to_empirical_trajectories[item]}, ax=axes_flat[i],
            plot_kws=empirical_plot_kws, arrow_start_index=-2,
            items_to_colors={item: empirical_color},
            dimensions_to_keep=dimensions_to_keep,
            use_plot_alpha_for_arrowhead=False)
        axes_flat[i].text(.95, .95, item,
                          size=16, transform=axes_flat[i].transAxes,
                          horizontalalignment='right',
                          verticalalignment='top')

    fig.legend(
        handles=[pred_patch, data_patch],
        labels=['iterated 1-year prediction', 'empirical data'],
        loc='upper center', ncol=2, borderaxespad=0.)
    fig.tight_layout()
    maybe_save_fig(fig, save_fig)
    return fig


def quiver_stream_grid(
        model, data, target, panel, model_name,
        dimensions_to_keep=(0, 1), axis_labels=None,
        n_points_each_dim=30,
        model_predicts_change=True, mask_velocity_to_convex_hull_of_data=True,
        quiver_empirical_kws=None,
        quiver_kws={'cmap': mpl.cm.viridis,
                    'angles': 'xy', 'scale_units': 'xy'},
        stream_kws={'density': 1, 'cmap': mpl.cm.viridis},
        stream_with_data_kws={'density': 0.5, 'cmap': mpl.cm.Greys},
        item_trajectories_kws=None,
        ellipse=None,
        save_fig=None):
    """Create quiver and stream plots of the time-series data and the model.

    Parameters
    ----------
    model : estimator with fit and predict methods that has been fit to the
    timeseries data

    data, target : array-like, shape [n_samples, n_features]
        `data` is the lagged times-series with lags >= 1. `target` is the
        time-series without any lags. To see how these lags are computed, see
        the function `split_panel_into_data_and_target_dataframes` in the
        module `panelyze.model_selection.split_data_target`.

    panel : pandas Panel
        The items are time-series; major_axis is time; minor_axis are features.

    model_name : str
        The name of the model to show in the plot label

    dimensions_to_keep : tuple of int's, each between 0 and n_features
        Which dimensions of the tensors of the meshgrid (i.e., which features
        of `data` and `target`) to show.

    axis_labels : None, or string, or list of strings of length equal to
    `len(dimensions_to_keep)`, or dict with keys being dimensions_to_keep and
    values being strings, default: None
        If None, then label the axes as 'dimension 0', 'dimension 1', etc.
        If a string such as 'component', then label the axes 'component 0',
        'component 1', etc.
        If a tuple of length equal to `len(dimensions_to_keep)`, then use those
        strings as the axes' labels.
        If a dict with keys given by `dimensions_to_keep`, then the labels
        are those values with '(dimension i)' (i=0, ...) written after each
        label.

    n_points_each_dim : int or tuple of int's of length equal to
    `len(dimensions_to_keep)`
        The number of grid points to use in each dimension for the quiver
        and stream plots

    model_predicts_change : bool, default: True
        Whether the model predicts the change in the time-series or the next
        value of the time-series.

    quiver_kws, stream_kws, stream_with_data_kws, item_trajectories_kws : dict
        Keyword arguments for the 2nd, 3rd, and 4th plots


    ellipse : None or dict {'position': (x, y), 'radius': r}
        If not None, then plot an ellipse in the bottom-left plot with given
        radius and at the given position. For example,
        ellipse={'position': (30, 16),
                 'radius': 1 / model.best_params_['gamma']**(-0.5))}

    save_fig : str or None, default: None
        If not None, then save the figure to the path specified by this string.

    Returns
    -------
    fig, axes : Figure and 4 axes
        A 2 by 2 grid of plots
    """
    item_trajectories_kws = convert_None_to_empty_dict_else_copy(
        item_trajectories_kws)
    quiver_empirical_kws = convert_None_to_empty_dict_else_copy(
        quiver_empirical_kws)

    fig, axes = plt.subplots(2, 2, figsize=(10, 6))
    mask_velocity = mask_velocity_to_convex_hull_of_data
    grid_points, velocities = make_data_for_streamplot(
        data, model, n_points_each_dim,
        model_predicts_change=model_predicts_change,
        dimensions_to_keep=dimensions_to_keep,
        mask_velocity_to_convex_hull_of_data=mask_velocity)

    labels_xyz_axes = compute_axes_labels(axis_labels, dimensions_to_keep)

    quiver_plot_empirical(data, target, dimensions_to_keep=dimensions_to_keep,
                          ax=axes[0, 0],
                          **quiver_empirical_kws, **labels_xyz_axes)
    quiver_plot(
        *itertools.chain(np.meshgrid(*grid_points, indexing='ij'), velocities),
        ax=axes[0, 1], quiver_kws=quiver_kws, **labels_xyz_axes)

    stream_plot(*itertools.chain(grid_points, velocities),
                stream_kws=stream_kws, ax=axes[1, 0], **labels_xyz_axes)
    if ellipse:
        plot_ellipse(axes[1, 0], ellipse['position'], ellipse['radius'])
    stream_plot(*itertools.chain(grid_points, velocities),
                stream_kws=stream_with_data_kws, ax=axes[1, 1],
                **labels_xyz_axes)
    plot_item_trajectories(
        panel[panel.items.isin(country_color_dict)],
        ax=axes[1, 1], dimensions_to_keep=dimensions_to_keep,
        text_labels=True,
        **item_trajectories_kws)
    title = 'data and predictions for {}'.format(model_name)
    fig.suptitle(title, y=1.02)
    fig.tight_layout()
    maybe_save_fig(fig, save_fig)
    return fig, axes


def make_axis_labels(axis_labels_dict, dimensions_to_keep, label='dimension'):
    """Make a dictionary of axis label names {'xlabel': 'dimension 0', ...}.

    Parameters
    ----------
    axis_labels_dict : None or dict
        A dictionary mapping dimension indices to strings, such as
        {0: 'component 0', 1: 'component 1'}.
        If None, then use '{label} i' where '{label}' is replaced by the
        keyword argument `label`. If any dimension in `dimensions_to_keep` is
        not a key of `axis_labels_dict`, then the default label '{label} i'
        is used.

    dimensions_to_keep : list of int's
        The integers of which dimensions are being selected.

    label : str, default: 'dimension'
        The string to use to label axes when axis_labels_dict is None.

    Returns
    -------
    axis_label_names_to_axis_labels : dict
        Dictionary with keys in ['xlabel', 'ylabel', 'zlabel'], and values are
        strings.
    """
    axis_label_names = ['xlabel', 'ylabel', 'zlabel']

    if axis_labels_dict is None:
        axis_labels_dict = {}
    axis_labels = [axis_labels_dict.get(dim, '{} {}'.format(label, dim))
                   for dim in dimensions_to_keep]
    return dict(zip(axis_label_names, axis_labels))


def compute_axes_labels(axis_labels, dimensions_to_keep, label='dimension'):
    axis_label_names = ['xlabel', 'ylabel', 'zlabel']

    if isinstance(axis_labels, str):
        label = axis_labels

    default_axis_labels = ['{label} {dim}'.format(label=label, dim=dim)
                           for dim in dimensions_to_keep]

    if axis_labels is None or isinstance(axis_labels, str):
        axis_labels = default_axis_labels
    elif isinstance(axis_labels, dict):
        axis_labels = [axis_labels[dim] for dim in dimensions_to_keep]
        axis_labels = ['{} ({})'.format(l, d)
                       for l, d in zip(axis_labels, default_axis_labels)]
    elif len(axis_labels) == len(dimensions_to_keep):
        pass
    else:
        msg = ('Could not compute axes labels from {al} with '
               '`dimensions_to_keep` = {dtk}')
        raise ValueError(msg.format(al=axis_labels, dtk=dimensions_to_keep))

    return dict(zip(axis_label_names, axis_labels))


def density_list_plot_2d(
        x, y, z, convex_hull_only=True, error_label='error',
        n_points_x=100, n_points_y=100,
        xlabel='score on the first principal component',
        ylabel='score on the second principal component',
        log_color=False, subplots_kws=None, ax=None, save_fig=None):
    """Return a density plot of z as a color, interpolated with an RBF kernel.

    Parameters
    ----------
    x, y, z : arrays of the same length

    convex_hull_only : bool, default: True
        Whether to only plot values in the convex hull of the `x` and `y` data

    error_label : str, default: 'error'
        The label on the colorbar

    n_points_x, n_points_y : int, default: 100
        The number of bins to use in the x and y dimensions

    xlabel, ylabel : str
        Axis labels

    log_color : bool, default: True
        Whether to logarithmically transform the color scale

    ax : matplotlib axis, optional, default: None
        The axis on which to draw the plot. If None, then an axis is created.

    save_fig : str or None, default: None
        If not None, then save the figure to the path specified by this string.

    subplots_kws : keyword arguments to pass to plt.subplots, default: None
    """
    # Set up a regular grid of interpolation points
    xi = np.linspace(x.min(), x.max(), n_points_x)
    yi = np.linspace(y.min(), y.max(), n_points_y)
    xgrid, ygrid = np.meshgrid(xi, yi)

    # Interpolate
    rbf = scipy.interpolate.Rbf(x, y, z, function='linear')
    zi = rbf(xgrid, ygrid)

    if convex_hull_only:
        zi = mask_arrays_with_convex_hull(
            [zi], [xi, yi],
            scipy.spatial.ConvexHull(
                np.vstack((x, y)).T))[0]

    subplots_kws = convert_None_to_empty_dict_else_copy(subplots_kws)
    fig, ax = create_fig_ax(ax, **subplots_kws)
    axim = ax.imshow(
        zi, vmin=z.min(), vmax=z.max(), origin='lower',
        extent=[x.min(), x.max(), y.min(), y.max()],
        norm=(mpl.colors.LogNorm(vmin=zi.min(), vmax=zi.max()) if log_color
              else None))

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    cax, kwargs = mpl.colorbar.make_axes(
        [ax], location='bottom', fraction=0.05, pad=.1)
    cax.text(-.15, 0.1, error_label, ha='center', va='center')
    cb = fig.colorbar(axim, cax=cax, orientation='horizontal', extend='max')

    maybe_save_fig(fig, save_fig)
    return fig, ax
