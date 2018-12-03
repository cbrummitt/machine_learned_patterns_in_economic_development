import numpy as np
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline


def create_random_search(estimator, distributions, cv,
                         n_iter=10, verbose=1, scoring=None, name=None,
                         fit_params=None, n_jobs=1, error_score=np.nan):
    """Make a random hyperparameter search from an estimator.

    If the distributions are all lists, then a GridSearchCV is returned.
    Otherwise a RandomizedSearchCV is returned.

    Inputs
    ------
    estimator : sklearn Pipeline or sklearn estimator or KerasRegressor
        The model to find good hyperparameters for.

    distributions : list of hyperparameter distribution objects that inherit
    from `_BaseHyperparameterRandomVariable`
        A list of objects containing the hyperparameter name,
        step name, and random variable.
        See `panelyze.model_selection.random_variables`.

    cv : cross validator

    n_iter : int or function mapping number of hyperparameters to number of
    searches
        The number of hyperparameter choices to try

    verbose : int
        Verbosity of `sklearn.model_selection.RandomizedSearchCV`

    scoring : None, string, or callable
        Scoring parameter passed to
        `sklearn.model_selection.RandomizedSearchCV`

    name : None or string, default: None
        The name of the model to use in the filename attribute of the random
        search. If None, then the class names of the last step is used (if
        the model is a pipeline) or else the name of the class of the estimator
        is used.

    fit_params : None or dict, default: None
        Parameters to pass to the fit method. `fit_params` is an argument of
        `GridSearchCV` and of `RandomizedSearchCV`.

    n_jobs : int, default: 1
        Number of jobs to run in parallel.

    error_score : 'raise' or numeric, default: `np.nan`
        Value to assign to the score if an error occurs in estimator fitting.
        If set to ‘raise’, the error is raised. If a numeric value is given,
        FitFailedWarning is raised. This parameter does not affect the refit
        step, which will always raise the error.

    Returns
    -------
    hyperparam_search : RandomizedSearchCV or GridSearchCV object
        The hyperparameter search. It also has an attribute `filename` with a
        string that describes the model and its hyperparameter ranges.
        If the distributions are all lists, then a GridSearchCV is returned.
        Otherwise a RandomizedSearchCV is returned.
    """
    param_distributions = {}
    for dist in distributions:
        if isinstance(estimator, Pipeline):
            assert dist.step_name in estimator.named_steps
        param_distributions.update(dist.param_distribution)
    if all(dist.is_a_list_of_choices for dist in distributions):
        hyperparam_search = GridSearchCV(
            estimator=estimator,
            param_grid=param_distributions, n_jobs=n_jobs,
            error_score=error_score, return_train_score=True,
            verbose=verbose, scoring=scoring, cv=cv, fit_params=fit_params)
    else:
        if not isinstance(n_iter, int):
            n_iter = n_iter(len(param_distributions))
        hyperparam_search = RandomizedSearchCV(
            estimator=estimator,
            param_distributions=param_distributions,
            n_iter=n_iter, n_jobs=n_jobs,
            error_score=error_score, return_train_score=True,
            verbose=verbose, scoring=scoring, cv=cv, fit_params=fit_params)
    if name is None:
        if isinstance(estimator, Pipeline):
            filename = estimator.steps[-1][1].__class__.__name__
        else:
            filename = estimator.__class__.__name__
    else:
        filename = name
    filename += '__' + '__'.join([dist.filename for dist in distributions])
    if hasattr(cv, 'filename'):
        filename += '__{}'.format(cv.filename)
    hyperparam_search.filename = filename
    return hyperparam_search
