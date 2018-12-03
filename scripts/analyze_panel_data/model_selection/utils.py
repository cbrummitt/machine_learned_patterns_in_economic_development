"""Utilities for selecting models.
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone, is_classifier
from sklearn.externals.joblib import Parallel, delayed
from sklearn.metrics.scorer import check_scoring
from sklearn.model_selection._split import check_cv
from sklearn.model_selection._validation import _fit_and_score
from sklearn.utils import indexable


def const_searches_per_hyperparam(searches_per_param=5, max_searches=100):
    """Return a function that maps the number of hyperparameters to a constant
    multiple per hyperparameter, capped at a certain maximum."""
    def n_iter(n_hyperparameters):
        return max(1,
                   min(max_searches, searches_per_param * n_hyperparameters))
    return n_iter


def five_searches_per_hyperparam(n_hyperparameters):
    return const_searches_per_hyperparam(5, 100)


def ten_searches_per_hyperparam(n_hyperparameters):
    return const_searches_per_hyperparam(10, 100)


def cross_val_score_with_times(
        estimator, X, y=None, times=None, scoring=None, cv=None,
        n_jobs=1, verbose=0, fit_params=None,
        pre_dispatch='2*n_jobs'):
    """Evaluate a score by cross-validation

    Read more in the :ref:`User Guide <cross_validation>`.

    Parameters
    ----------
    estimator : estimator object implementing 'fit'
        The object to use to fit the data.

    X : array-like
        The data to fit. Can be, for example a list, or an array at least 2d.

    y : array-like, optional, default: None
        The target variable to try to predict in the case of
        supervised learning.

    times : array-like, with shape (n_samples,), optional
        Time labels for the samples used while splitting the dataset into
        train/test set.

    scoring : string, callable or None, optional, default: None
        A string (see model evaluation documentation) or
        a scorer callable object / function with signature
        ``scorer(estimator, X, y)``.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 3-fold cross validation,
          - integer, to specify the number of folds in a `(Stratified)KFold`,
          - An object to be used as a cross-validation generator.
          - An iterable yielding train, test splits.

        For integer/None inputs, if the estimator is a classifier and ``y`` is
        either binary or multiclass, :class:`StratifiedKFold` is used. In all
        other cases, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validation strategies that can be used here.

    n_jobs : integer, optional
        The number of CPUs to use to do the computation. -1 means
        'all CPUs'.

    verbose : integer, optional
        The verbosity level.

    fit_params : dict, optional
        Parameters to pass to the fit method of the estimator.

    pre_dispatch : int, or string, optional
        Controls the number of jobs that get dispatched during parallel
        execution. Reducing this number can be useful to avoid an
        explosion of memory consumption when more jobs get dispatched
        than CPUs can process. This parameter can be:

            - None, in which case all the jobs are immediately
              created and spawned. Use this for lightweight and
              fast-running jobs, to avoid delays due to on-demand
              spawning of the jobs

            - An int, giving the exact number of total jobs that are
              spawned

            - A string, giving an expression as a function of n_jobs,
              as in '2*n_jobs'

    Returns
    -------
    scores : array of float, shape=(len(list(cv)),)
        Array of scores of the estimator for each run of the cross validation.

    Examples
    --------
    >>> from sklearn import datasets, linear_model
    >>> from sklearn.model_selection import cross_val_score
    >>> diabetes = datasets.load_diabetes()
    >>> X = diabetes.data[:150]
    >>> y = diabetes.target[:150]
    >>> lasso = linear_model.Lasso()
    >>> print(cross_val_score(lasso, X, y))  # doctest: +ELLIPSIS
    [ 0.33150734  0.08022311  0.03531764]

    See Also
    ---------
    :func:`sklearn.metrics.make_scorer`:
        Make a scorer from a performance metric or loss function.

    """
    X, y, times = indexable(X, y, times)

    cv = check_cv(cv, y, classifier=is_classifier(estimator))
    cv_iter = list(cv.split(X, y, times))
    scorer = check_scoring(estimator, scoring=scoring)
    # We clone the estimator to make sure that all the folds are
    # independent, and that it is pickle-able.
    parallel = Parallel(n_jobs=n_jobs, verbose=verbose,
                        pre_dispatch=pre_dispatch)
    scores = parallel(delayed(_fit_and_score)(clone(estimator), X, y, scorer,
                                              train, test, verbose, None,
                                              fit_params)
                      for train, test in cv_iter)
    return np.array(scores)[:, 0]


def study_marginals_of_cv_results(
        cv_results, params_to_take_log_of=None,
        drop_variables_that_are_logged=False,
        cutoff_for_plotting=8,
        ignore_results_with_mean_test_score_above=1e6):
    """Print plots and tables analyzing the marginals of a hyperparameter
    search.

    Parameters
    ----------
    cv_results : dict or DataFrame
        The `cv_results_` attribute of a hyperaparameter search in sklearn, or
        a DataFrame wrapped around such a dictionary.

    params_to_take_log_of` : string or list of strings or None
        Which parameters to take the logarithm (base 10) of.
        If None, then no parameters are logged.

    drop_variables_that_are_logged : bool, default: False
        Whther to show results about the parameters that are logged without
        apply the logarithm to those parameters.

    cutoff_for_plotting : int, default: 8
        Cutoff number of distinct values for a hyperparameter below
        which a table is shown and above or equal to which a plot is shown.

    ignore_results_with_mean_test_score_above : int or float, default: 1e6
        Remove results with mean test score above this amount
    """
    cv_results_ = pd.DataFrame(cv_results).copy()

    mask = (cv_results_.mean_test_score.apply(np.abs) <
            ignore_results_with_mean_test_score_above)

    cv_results_ = cv_results_[mask]

    if params_to_take_log_of:
        if isinstance(params_to_take_log_of, str):
            params_to_take_log_of = [params_to_take_log_of]
        for param in params_to_take_log_of:
            if param in cv_results_.columns:
                new_param = param + '_log10'
                cv_results_[new_param] = np.log10(
                    list(cv_results_.loc[:, param]))
                if drop_variables_that_are_logged:
                    cv_results_ = cv_results_.drop(param, axis=1)
            else:
                raise ValueError(('Could not take the log of the parameter '
                                  '{} because it was not found in the columns'
                                  .format(param)))

    param_columns = [col for col in cv_results_.columns if col[:6] == 'param_']
    for col in param_columns:
        df = cv_results_.groupby(col)['mean_test_score']
        try:
            if len(df) < cutoff_for_plotting:
                print(df.agg(['mean', 'median', 'max', 'std', 'size']))
            else:
                df.agg('mean').plot(label='mean')
                df.agg('median').plot(label='median')
                df.agg('max').plot(label='max')
                plt.xlabel(col.replace('_', ' '))
                plt.ylabel('mean test score')
                plt.legend()
                plt.show()
                plt.close()
        except Exception as err:
            print('Got error on column {}:\n\t{}'.format(col, err))


def read_all_cv_results(results_dir):
    """Read and combine all cv results found in the directory `results_dir`.

    The path structure is assumed to be:
        results_dir
            filename1
                description.txt
                best_model
                    cv_results.csv
            filename1
                description.txt
                best_model
                    cv_results.csv
            filename1
                description.txt
                best_model
                    cv_results.csv
            ...

    Four new columns are added to the DataFrame:
        1. 'filename' is the child directory (e.g., `filename1` in the example
        above)
        2. 'results_directory' is the input `results_dir`
        3. 'is_best_model_in_its_hyperparam_search' is `True` if that
        hyperparameter choice was the best one in its hyperparameter search,
        else `False`.
        4. 'description' is the contents of the file 'description.txt' found
        in `results_directory`
    """
    if not os.path.exists(results_dir):
        raise ValueError(
            'The results directory {} does not exist.'.format(
                results_dir))
    all_filenames = [
        p for p in os.listdir(results_dir)
        if (p[0] != '.' and os.path.join(results_dir, p, 'best_model') and
            'fit_time.txt' in os.listdir(
                os.path.join(results_dir, p, 'best_model')))]

    cv_results = pd.concat([
        read_cv_results(results_dir, f)
        for f in all_filenames
    ]).sort_values('mean_test_score', ascending=False).reset_index(drop=True)

    cv_results['rank_test_score'] = range(1, 1 + len(cv_results))
    return cv_results


def read_cv_results(results_directory, filename):
    """Return the cross validation results as a DataFrame.

    The cv results are found in the path
        results_directory
            filename
                description.txt
                best_model
                    cv_results.csv
    """
    path = os.path.join(results_directory, filename,
                        'best_model', 'cv_results.csv')
    df = pd.read_csv(path, index_col=0)

    description_path = os.path.join(results_directory, filename,
                                    'description.txt')
    if os.path.exists(description_path):
        with open(description_path, 'r') as f:
            description = f.read()
    else:
        description = np.nan
    df['description'] = description
    df['filename'] = filename
    df['results_directory'] = results_directory
    df['is_best_model_in_its_hyperparam_search'] = df.rank_test_score == 1
    return df
