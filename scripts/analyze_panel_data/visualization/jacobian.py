# jacobian.py contains functions for visualizing the Jacobian
# of an inferred model
import numpy as np
import pandas as pd


def convert_Jacobian_to_long_format(coef, columns):
    """Return the Jacobian as a pandas DataFrame in long format.

    Inputs
    ------
    coef : numpy array (n_variables, n_variables)
        The Jacobian matrix. The rows correspond to the model's
        output. The columns are the model's input.

    columns : pandas Index
        The names of the variables

    Returns
    -------
    coef_long : pandas DataFrame
        Each entry in `coef` becomes a row in `coef_long`.
    """
    coef_df = pd.DataFrame(coef)
    coef_df.index = columns.copy()
    coef_df.columns = columns.copy()

    if columns.names is not None:
        coef_df.index.names = [col + '_predicted' for col in columns.names]
        coef_df.columns.names = [col + '_predictor' for col in columns.names]
    elif columns.name is not None:
        coef_df.index.name = [col + '_predicted' for col in columns.name]
        coef_df.columns.name = [col + '_predictor' for col in columns.name]
    coef_long = (coef_df.stack().stack()
                 .reset_index().rename(columns={0: 'coef'}))
    return coef_long


def scale_coefficients(coefficients, scale):
    """Return the coefficients rescaled to the original units.

    `scale` is a 1D array of floats that were the numbers by which the raw
    data was dividied by.
    """
    return coefficients * np.array([1 / scale]) * scale.reshape(-1, 1)


# def test_scale_coefficients():
#     X = np.array([[2, 20], [6, 40]])
#     Y = np.emtpy_like(X)
#     Y[:, 0] = 1 * X[:, 0] + 2 * X[:, 1]
#     Y[:, 1] = 3 * X[:, 0] + 4 * X[:, 1]
#     scaler = RobustScaler()
#     scaler.fit(X)
#     X_scaled = scaler.transform(X)
#     Y_scaled = scaler.transform(Y)
#     return scale_coefficients()


def compute_edges_from_adjacency_matrix(
        matrix, weight_threshold=0.0,
        map_index_to_str=None, allow_self_loops=False):
    """Return a list of edges for use in networkx.

    Inputs
    ------
    matrix : a 2D numpy array
        The adjacency matrix

    weight_threshold : float
        Weights with absolute value less than or equal to this number are
        not included in the list of edges.

    map_index_to_str : dict or None, optional, default: None
        If a dict, then the indices of the matrix are renamed using this
        dictionary.

    allow_self_loops : bool, default: False
        Whether to allow edges between the same vertex (i.e., edge `(i, i)`).

    Returns
    -------
    edges : list of tuples (vertex_i, vertex_j, {'weight': weight})
        `vertex_i` and `vertex_j` are the names of the nodes.
        If `map_index_to_str` is None, then `vertex_i` is the index i; else
        it is `map_index_to_str[i]`.
        `weight` is the weight of the edge, which si `matrix[i, j]`.
    """
    def include_edge(i, j):
        if not allow_self_loops and i == j:
            return False
        elif (weight_threshold is not None and
                np.abs(matrix[i, j]) <= weight_threshold):
            return False
        else:
            return True

    def rename_node(i):
        if map_index_to_str is not None:
            return map_index_to_str[i]
        else:
            return i
    return [(rename_node(i), rename_node(j), {'weight': matrix[i, j]})
            for i in range(matrix.shape[0]) for j in range(matrix.shape[1])
            if include_edge(i, j)]
