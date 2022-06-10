from typing import Optional

import hydra
from hydra import initialize, compose
from omegaconf import DictConfig
from sklearn.pipeline import Pipeline


def make_pipeline(steps_config: DictConfig) -> Pipeline:
    """Creates a pipeline with all the preprocessing steps specified in `steps_config`, ordered in a sequential manner

    Args:
        steps_config (DictConfig): the config containing the instructions for
                                    creating the feature selectors or transformers

    Returns:
        [sklearn.pipeline.Pipeline]: a pipeline with all the preprocessing steps, in a sequential manner
    """
    steps = []

    for step_config in steps_config:
        # retrieve the name and parameter dictionary of the current steps
        step_name, step_params = list(step_config.items())[0]

        # instantiate the pipeline step, and append to the list of steps
        pipeline_step = (step_name, hydra.utils.instantiate(step_params))
        steps.append(pipeline_step)

    return Pipeline(steps, memory='./cache')


def get_preprocessing_pipeline(overrides: Optional[list[str]] = None) -> Pipeline:
    if overrides is None:
        overrides = []

    with initialize(version_base=None, config_path='configs'):
        config = compose(config_name="config", overrides=overrides)

        return hydra.utils.instantiate(
            config.preprocessing_pipeline, _recursive_=False
        )
