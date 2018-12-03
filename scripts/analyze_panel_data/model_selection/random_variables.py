from abc import ABCMeta, abstractproperty

from scipy.stats import uniform as sp_uniform

__all__ = [
    'ExponentiateDistribution', 'loguniform_dist', 'LogUniformDistribution',
    'ContinuousUniformDistribution', 'IntegerUniformDistribution',
    'DiscreteChoice'
]


class ExponentiateDistribution():
    """Exponentiate a distribution using a certain base number.

    This class simply overrides the `rvs` attribute.

    Parameters
    ----------
    dist : a distribution with `rvs` attribute

    base : float
    """

    def __init__(self, dist, base=1):
        self.dist = dist
        self.base = base

    def __str__(self):
        return '{}**{}'.format(self.base, self.dist)

    def rvs(self, *args, **kwargs):
        """Return base raised to the random variate."""
        return self.base**self.dist.rvs(*args, **kwargs)


def loguniform_dist(low, high, base=10):
    """Return a random variable that is `base` raised to a Uniform(low, high).
    """
    return ExponentiateDistribution(sp_uniform(low, high - low), base=base)


def uniform_dist(low, high):
    """Return a random variable uniformly distributed between `low` and `high`.
    """
    return sp_uniform(low, high - low)


class _BaseHyperparameterRandomVariable(metaclass=ABCMeta):

    def __init__(self, step_name, variable_name, distribution):
        """Create a random variable for a hyperparameter search.

        Inputs
        ------
        step_name : str
            The name of the step in the pipeline.

        variable_name : str or None
            If None, then the hyperparameter is what step to apply rather than
            a variable for a particular step.

        distribution : object or list
            A random variable with `rvs` method, or a list of choices.
        """
        self.step_name = step_name
        self.variable_name = variable_name
        self._distribution = distribution

    @abstractproperty
    def filename(self):
        pass

    @property
    def distribution(self):
        return self._distribution

    @property
    def step_param(self):
        """Return a string '{step}__{variable}' for use in hyperparameter
        search.
        """
        if self.variable_name is None:
            return self.step_name
        elif self.step_name is None:
            return self.variable_name
        else:
            return '{step}__{var}'.format(
                step=self.step_name, var=self.variable_name)

    @abstractproperty
    def is_a_list_of_choices(self):
        """Return whether the distribution is a list of choices."""
        pass

    @property
    def param_distribution(self):
        """Return a dictionary {'{step}__{variable}': distribution}."""
        return {self.step_param: self.distribution}


class LogUniformDistribution(_BaseHyperparameterRandomVariable):
    """Exponentiation of a uniform distribution"""

    def __init__(self, low, high, step_name, variable_name, base=10):
        """Random variable whose log in the given `base` is uniformly
        distributed between `low` and `high`.

        Inputs
        ------
        low, high : float

        base : float or int, default: 10

        step_name, variable_name : str
            The name of the step in the sklearn pipeline and the name of the
            variable.
        """
        super().__init__(
            step_name, variable_name, loguniform_dist(low, high, base=base))
        self.low = min(low, high)
        self.high = max(low, high)
        self.base = base

    @property
    def filename(self):
        return 'log{base}{variable_name}=uniform_{low}to{high}'.format(
            **self.__dict__)

    @property
    def is_a_list_of_choices(self):
        return False


class ContinuousUniformDistribution(_BaseHyperparameterRandomVariable):
    """Continuous uniform distribution"""

    def __init__(self, low, high, step_name, variable_name):
        """Random variable uniformly distributed between `low` and `high`.

        Inputs
        ------
        low, high : float

        step_name, variable_name : str
            The name of the step in the sklearn pipeline and the name of the
            variable.
        """
        super().__init__(step_name, variable_name, sp_uniform(low, high - low))
        self.low = min(low, high)
        self.high = max(low, high)

    @property
    def filename(self):
        return '{variable_name}_uniform_{low}to{high}'.format(**self.__dict__)

    @property
    def is_a_list_of_choices(self):
        return False


class IntegerUniformDistribution(_BaseHyperparameterRandomVariable):
    """Continuous uniform distribution"""

    def __init__(self, low, high, step_name, variable_name):
        """Random variable uniformly distributed between `low` and `high`.

        Inputs
        ------
        low, high : int

        step_name, variable_name : str
            The name of the step in the sklearn pipeline and the name of the
            variable.
        """
        super().__init__(step_name, variable_name, list(range(low, high + 1)))
        self.low = min(low, high)
        self.high = max(low, high)

    @property
    def filename(self):
        return '{variable_name}=randint_{low}to{high}'.format(**self.__dict__)

    @property
    def is_a_list_of_choices(self):
        return True


class DiscreteChoice(_BaseHyperparameterRandomVariable):
    """Discrete set of choices, sampled uniformly.

    The distribution is simply a list because scikit-learn's hyperparameter
    search objects accept lists or random variables.
    """

    def __init__(self, choices, step_name, variable_name, choice_names=None,
                 max_length_filename=200):
        """Random variable uniformly distributed between `low` and `high`.

        Inputs
        ------
        choices : iterable

        choice_names :  None or iterable of strings of the same length as
        `choices`, optional, default: `None`
            The names of the choices. If None, then `str` is called on the
            choices.

        max_length_filename : int
            The maximum length of the filename. If joining the choices
            with '_' results in a filename with more than `max_length_filename`
            many charactes, then the filename simply reports the number of
            choices.

        step_name, variable_name : str
            The name of the step in the sklearn pipeline and the name of the
            variable.
        """
        super().__init__(step_name, variable_name, list(choices))
        self.max_length_filename = max_length_filename
        self.choices = list(choices)
        if choice_names is None:
            self.choice_names = [str(choice) for choice in self.choices]
        else:
            self.choice_names = choice_names
        if choice_names is not None:
            for choice_name in self.choice_names:
                assert isinstance(choice_name, str)

    @property
    def filename(self):
        concat_choices = '_'.join(self.choice_names)
        identifer = (self.step_name if self.variable_name is None
                     else self.variable_name)
        if len(concat_choices) < self.max_length_filename:
            return identifer + '=' + concat_choices
        else:
            return '{identifer}_{n_choices}choices'.format(
                identifer=identifer, n_choices=len(self.choices))

    @property
    def is_a_list_of_choices(self):
        return True
