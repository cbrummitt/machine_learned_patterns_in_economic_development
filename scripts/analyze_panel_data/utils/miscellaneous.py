# Miscellaneous functions
import hashlib

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer


def hash_string(s):
    """Return a SHA-256 hash of a given string."""
    return hashlib.sha256(s.encode('utf-8')).hexdigest()


def split_pipeline_at_step(pipe, step):
    """Split a pipeline at a named step and return two pipelines.

    The given step is included in the first pipeline.

    Parameters
    ----------
    pipe : sklearn.pipeline.Pipeline

    step : str
        The name of the step at which to split the pipeline into two. This
        step is included in the first pipeline returned.

    Returns
    -------
    pipe_beginning, pipe_ending : sklearn.pipeline.Pipeline's
        The part of the pipeline up to the dimension reducing part
        (inclusive) and after the dimension reducing part.

    Raises
    ------
    ValueError
        If `step` is not one of the steps or if `step` is the last step.

    Example
    -------
    >>> pipe = Pipeline([('a', None), ('b', None), ('c', None)])
    >>> DimensionReducingPipeline.split_pipeline_at_step(pipe, 'b')
    (Pipeline(steps=[('a', None), ('b', None)]),
     Pipeline(steps=[('c', None)]))
    """
    steps = pipe.steps
    step_names = [name for name, transform in steps]
    if step not in step_names:
        msg = 'The step {step} must be one of the step names {names}.'
        raise ValueError(msg.format(step=step, names=step_names))
    index_step = step_names.index(step)

    beginning_steps = steps[:index_step + 1]
    pipe_beginning = (
        Pipeline(beginning_steps) if len(beginning_steps) > 0
        else FunctionTransformer(None))
    if index_step == len(steps) - 1:
        raise ValueError("Cannot split at the last step '{}'. Split at a "
                         "step earlier in the pipeline.".format(step))
    pipe_ending = Pipeline(steps[index_step + 1:])
    return pipe_beginning, pipe_ending
