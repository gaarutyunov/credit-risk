import abc
import pathlib
from os import PathLike
from typing import Optional, List, Iterable

import hydra
import pandas as pd
from hydra import initialize, compose
from omegaconf import DictConfig, OmegaConf
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.pipeline import Pipeline

import utils


def make_preprocessing_pipeline(steps_config: DictConfig) -> Pipeline:
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
        pipeline_step = (step_name, hydra.utils.instantiate(step_params, _recursive_=True))
        steps.append(pipeline_step)

    return LabelInferPipeline(steps, memory='./.cache/preprocessing')


def make_classifier_pipeline(steps_config: DictConfig) -> Pipeline:
    """Creates a pipeline with all the classifier steps specified in `steps_config`, ordered in a sequential manner

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
        pipeline_step = (step_name, hydra.utils.instantiate(step_params, _recursive_=True))
        steps.append(pipeline_step)

    return Pipeline(steps, memory='./.cache/classifier')


def get_preprocessing_pipeline(name: str = 'cat_boot', overrides: Optional[List[str]] = None, debug: bool = False) -> Pipeline:
    if overrides is None:
        overrides = []

    with initialize(version_base=None, config_path='configs'):
        config = compose(config_name=name + '_config', overrides=overrides)

        if debug:
            print(OmegaConf.to_yaml(config))

        return hydra.utils.instantiate(
            config.preprocessing_pipeline, _recursive_=False
        )


def get_classifier_pipeline(name: str = 'cat_boot', overrides: Optional[List[str]] = None, debug: bool = False) -> Pipeline:
    if overrides is None:
        overrides = []

    with initialize(version_base=None, config_path='configs'):
        config = compose(config_name=name + '_config', overrides=overrides)

        if debug:
            print(OmegaConf.to_yaml(config))

        return hydra.utils.instantiate(
            config.classifier_pipeline, _recursive_=False
        )


class EmptyFit(TransformerMixin, abc.ABC):
    def fit(self, X, y=None, **fit_params):
        return self

    @abc.abstractmethod
    def transform(self, X):
        pass


class LabelTransformer(TransformerMixin):
    label: pd.Series

    def fit(self, X, y=None, **fit_params):
        self.label = X['loan_status'].replace(('Fully Paid',
                                               'Charged Off',
                                               'Does not meet the credit policy. Status:Fully Paid',
                                               'Does not meet the credit policy. Status:Charged Off',
                                               'Default'
                                               ),
                                              (0, 1, 0, 1, 1))
        self.label.drop(index=self.label[~self.label.isin([0, 1])].index, inplace=True)
        self.label = pd.to_numeric(self.label)

        return self

    def transform(self, X):
        X['label'] = self.label
        X.drop(columns=['loan_status'], inplace=True)
        X.drop(index=X[~X.label.isin([0, 1])].index, inplace=True)
        X['label'] = pd.to_numeric(X['label'])

        return X


class DataReader(TransformerMixin, abc.ABC):
    X: pd.DataFrame

    def __init__(self, file: str, columns: List[str]) -> None:
        self.file = pathlib.Path(file)
        self.cols = columns

    def fit(self, X, y=None, **fit_params):
        self.X = self._read(self.file)

        return self

    def transform(self, X):
        return self.X

    @abc.abstractmethod
    def _read(self, path: PathLike) -> pd.DataFrame:
        pass


class CSVReader(DataReader):
    def _read(self, path: PathLike) -> pd.DataFrame:
        return utils.load_csv_compressed(self.file, usecols=self.cols)


class ReaderPipeline(Pipeline):
    reader: DataReader

    def __init__(self, steps, *, memory=None, verbose=False):
        super().__init__(steps[1:], memory=memory, verbose=verbose)
        self.reader = steps[0][1]

    def _read(self, X, y=None, **fit_params):
        self.reader.fit(X, y, **fit_params)


class LabelInferPipeline(ReaderPipeline):
    label_transformer: TransformerMixin

    def __init__(self, steps, *, memory=None, verbose=False):
        super().__init__(steps[:1] + steps[2:], memory=memory, verbose=verbose)
        self.label_transformer = steps[1][1]

    def fit(self, X, y=None, **fit_params):
        self._read(X, y, **fit_params)
        self.reader.X = self.label_transformer.fit_transform(self.reader.X)

        return super().fit(self.reader.X.drop(columns=['label']), self.reader.X['label'], **fit_params)

    def fit_transform(self, X, y=None, **fit_params):
        return self.fit(X, y, **fit_params).transform(X)


class ApplyToColumns(TransformerMixin, BaseEstimator):
    def __init__(self, inner: TransformerMixin, columns: Iterable[str]) -> None:
        self.inner = inner
        self.columns = columns

    def fit(self, X, y=None, **fit_params):
        self.columns = list(self.columns)
        self.inner.fit(X[self.columns], y, **fit_params)

        return self

    def transform(self, X):
        X.loc[:, self.columns] = self.inner.transform(X[self.columns])

        return X
