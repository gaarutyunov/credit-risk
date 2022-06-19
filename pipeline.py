import abc
import pathlib
from os import PathLike
from typing import Optional, List, Iterable, Callable, Any

import hydra
import pandas as pd
from catboost import CatBoostClassifier
from hydra import initialize, compose
from omegaconf import OmegaConf
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.pipeline import Pipeline
import numpy as np
import utils

PipelineCtr = Callable[[Any], Pipeline]


def get_pipeline(
    name: str = "cat_boot",
    group: str = "preprocessing",
    overrides: Optional[List[str]] = None,
    debug: bool = False,
) -> Pipeline:
    if overrides is None:
        overrides = []

    with initialize(version_base=None, config_path="configs"):
        config = compose(config_name=name + "_config", overrides=overrides)

        if debug:
            print(OmegaConf.to_yaml(config[group + "_pipeline"]))

        return hydra.utils.instantiate(config[group + "_pipeline"], _recursive_=True)


class EmptyFit(TransformerMixin, abc.ABC):
    def fit(self, X, y=None, **fit_params):
        return self

    @abc.abstractmethod
    def transform(self, X):
        pass


class LabelTransformer(TransformerMixin):
    label: pd.Series

    def fit(self, X, y=None, **fit_params):
        self.label = X["loan_status"].replace(
            (
                "Fully Paid",
                "Charged Off",
                "Does not meet the credit policy. Status:Fully Paid",
                "Does not meet the credit policy. Status:Charged Off",
                "Default",
            ),
            (0, 1, 0, 1, 1),
        )
        self.label.drop(index=self.label[~self.label.isin([0, 1])].index, inplace=True)
        self.label = pd.to_numeric(self.label)

        return self

    def transform(self, X):
        X["label"] = self.label
        X.drop(columns=["loan_status"], inplace=True)
        X.drop(index=X[~X.label.isin([0, 1])].index, inplace=True)
        X["label"] = pd.to_numeric(X["label"])

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


class BaseReaderPipeline(Pipeline):
    reader: DataReader

    def __init__(self, steps, *, memory=None, verbose=False):
        super().__init__(steps[1:], memory=memory, verbose=verbose)
        self.reader = steps[0][1]

    def _read(self, X, y=None, **fit_params):
        self.reader.fit(X, y, **fit_params)

    def fit(self, X, y=None, **fit_params):
        return super().fit(self.reader.X, y, **fit_params)

    def transform(self, X):
        return super().transform(self.reader.X)

    def fit_transform(self, X, y=None, **fit_params):
        return self.fit(X, y, **fit_params).transform(X)


class ReaderPipeline(BaseReaderPipeline):
    def fit(self, X, y=None, **fit_params):
        self._read(X, y, **fit_params)
        return super().fit(self.reader.X, y, **fit_params)

    def transform(self, X):
        return super().transform(self.reader.X)

    def fit_transform(self, X, y=None, **fit_params):
        return self.fit(X, y, **fit_params).transform(X)


class LabelInferPipeline(BaseReaderPipeline):
    label_transformer: TransformerMixin

    def __init__(self, steps, *, memory=None, verbose=False):
        super().__init__(steps[:1] + steps[2:], memory=memory, verbose=verbose)
        self.label_transformer = steps[1][1]

    def fit(self, X, y=None, **fit_params):
        self._read(X, y, **fit_params)
        self.reader.X = self.label_transformer.fit_transform(self.reader.X)
        y = self.reader.X["label"]
        self.reader.X.drop(columns=["label"], inplace=True)

        return super().fit(self.reader.X, y, **fit_params)

    def transform(self, X):
        return super().transform(self.reader.X)

    def fit_transform(self, X, y=None, **fit_params):
        return self.fit(X, y, **fit_params).transform(X)


class ApplyToColumns(TransformerMixin, BaseEstimator):
    def __init__(self, inner: TransformerMixin, columns: Iterable[str]) -> None:
        self.inner = inner
        self.columns = columns

    def fit(self, X, y=None, **fit_params):
        self.inner.fit(X[self.columns], y, **fit_params)

        return self

    def transform(self, X):
        X.loc[:, self.columns] = self.inner.transform(X[self.columns])

        return X


class CatBoostLoader(CatBoostClassifier):
    def __init__(
        self,
        iterations=None,
        learning_rate=None,
        depth=None,
        l2_leaf_reg=None,
        model_size_reg=None,
        rsm=None,
        loss_function=None,
        border_count=None,
        feature_border_type=None,
        per_float_feature_quantization=None,
        input_borders=None,
        output_borders=None,
        fold_permutation_block=None,
        od_pval=None,
        od_wait=None,
        od_type=None,
        nan_mode=None,
        counter_calc_method=None,
        leaf_estimation_iterations=None,
        leaf_estimation_method=None,
        thread_count=None,
        random_seed=None,
        use_best_model=None,
        best_model_min_trees=None,
        verbose=None,
        silent=None,
        logging_level=None,
        metric_period=None,
        ctr_leaf_count_limit=None,
        store_all_simple_ctr=None,
        max_ctr_complexity=None,
        has_time=None,
        allow_const_label=None,
        target_border=None,
        classes_count=None,
        class_weights=None,
        auto_class_weights=None,
        class_names=None,
        one_hot_max_size=None,
        random_strength=None,
        name=None,
        ignored_features=None,
        train_dir=None,
        custom_loss=None,
        custom_metric=None,
        eval_metric=None,
        bagging_temperature=None,
        save_snapshot=None,
        snapshot_file=None,
        snapshot_interval=None,
        fold_len_multiplier=None,
        used_ram_limit=None,
        gpu_ram_part=None,
        pinned_memory_size=None,
        allow_writing_files=None,
        final_ctr_computation_mode=None,
        approx_on_full_history=None,
        boosting_type=None,
        simple_ctr=None,
        combinations_ctr=None,
        per_feature_ctr=None,
        ctr_description=None,
        ctr_target_border_count=None,
        task_type=None,
        device_config=None,
        devices=None,
        bootstrap_type=None,
        subsample=None,
        mvs_reg=None,
        sampling_unit=None,
        sampling_frequency=None,
        dev_score_calc_obj_block_size=None,
        dev_efb_max_buckets=None,
        sparse_features_conflict_fraction=None,
        max_depth=None,
        n_estimators=None,
        num_boost_round=None,
        num_trees=None,
        colsample_bylevel=None,
        random_state=None,
        reg_lambda=None,
        objective=None,
        eta=None,
        max_bin=None,
        scale_pos_weight=None,
        gpu_cat_features_storage=None,
        data_partition=None,
        metadata=None,
        early_stopping_rounds=None,
        cat_features=None,
        grow_policy=None,
        min_data_in_leaf=None,
        min_child_samples=None,
        max_leaves=None,
        num_leaves=None,
        score_function=None,
        leaf_estimation_backtracking=None,
        ctr_history_unit=None,
        monotone_constraints=None,
        feature_weights=None,
        penalties_coefficient=None,
        first_feature_use_penalties=None,
        per_object_feature_penalties=None,
        model_shrink_rate=None,
        model_shrink_mode=None,
        langevin=None,
        diffusion_temperature=None,
        posterior_sampling=None,
        boost_from_average=None,
        text_features=None,
        tokenizers=None,
        dictionaries=None,
        feature_calcers=None,
        text_processing=None,
        embedding_features=None,
        callback=None,
        load="models/cat_boost",
    ):
        super().__init__(
            iterations,
            learning_rate,
            depth,
            l2_leaf_reg,
            model_size_reg,
            rsm,
            loss_function,
            border_count,
            feature_border_type,
            per_float_feature_quantization,
            input_borders,
            output_borders,
            fold_permutation_block,
            od_pval,
            od_wait,
            od_type,
            nan_mode,
            counter_calc_method,
            leaf_estimation_iterations,
            leaf_estimation_method,
            thread_count,
            random_seed,
            use_best_model,
            best_model_min_trees,
            verbose,
            silent,
            logging_level,
            metric_period,
            ctr_leaf_count_limit,
            store_all_simple_ctr,
            max_ctr_complexity,
            has_time,
            allow_const_label,
            target_border,
            classes_count,
            class_weights,
            auto_class_weights,
            class_names,
            one_hot_max_size,
            random_strength,
            name,
            ignored_features,
            train_dir,
            custom_loss,
            custom_metric,
            eval_metric,
            bagging_temperature,
            save_snapshot,
            snapshot_file,
            snapshot_interval,
            fold_len_multiplier,
            used_ram_limit,
            gpu_ram_part,
            pinned_memory_size,
            allow_writing_files,
            final_ctr_computation_mode,
            approx_on_full_history,
            boosting_type,
            simple_ctr,
            combinations_ctr,
            per_feature_ctr,
            ctr_description,
            ctr_target_border_count,
            task_type,
            device_config,
            devices,
            bootstrap_type,
            subsample,
            mvs_reg,
            sampling_unit,
            sampling_frequency,
            dev_score_calc_obj_block_size,
            dev_efb_max_buckets,
            sparse_features_conflict_fraction,
            max_depth,
            n_estimators,
            num_boost_round,
            num_trees,
            colsample_bylevel,
            random_state,
            reg_lambda,
            objective,
            eta,
            max_bin,
            scale_pos_weight,
            gpu_cat_features_storage,
            data_partition,
            metadata,
            early_stopping_rounds,
            cat_features,
            grow_policy,
            min_data_in_leaf,
            min_child_samples,
            max_leaves,
            num_leaves,
            score_function,
            leaf_estimation_backtracking,
            ctr_history_unit,
            monotone_constraints,
            feature_weights,
            penalties_coefficient,
            first_feature_use_penalties,
            per_object_feature_penalties,
            model_shrink_rate,
            model_shrink_mode,
            langevin,
            diffusion_temperature,
            posterior_sampling,
            boost_from_average,
            text_features,
            tokenizers,
            dictionaries,
            feature_calcers,
            text_processing,
            embedding_features,
            callback,
        )
        self.load_model(load)


FILL_MAX_COLS = [
    "bc_open_to_buy",
    "mths_since_last_delinq",
    "mths_since_last_major_derog",
    "mths_since_last_record",
    "mths_since_rcnt_il",
    "mths_since_recent_bc",
    "mths_since_recent_bc_dlq",
    "mths_since_recent_inq",
    "mths_since_recent_revol_delinq",
    "pct_tl_nvr_dlq",
    "sec_app_mths_since_last_major_derog",
]

FILL_MEAN_COLS = [
    "annual_inc",
    "dti",
    "sec_app_fico_range_low",
    "sec_app_fico_range_high",
    "total_rev_hi_lim",
    "total_bc_limit",
    "total_il_high_credit_limit",
    "tot_hi_cred_lim",
    "il_util",
    "sec_app_revol_util",
    "bc_util",
]

PURPOSE_MAP = {
    "home_improvement": "consumer_expenses",
    "car": "consumer_expenses",
    "vacation": "consumer_expenses",
    "wedding": "consumer_expenses",
}


class CustomColumnTransformer(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        us_urbanization_xlsx_path: str,
        fill_max_cols: Optional[List[str]] = None,
        fill_mean_cols: Optional[List[str]] = None,
        purpose_map=None,
    ):
        if purpose_map is None:
            purpose_map = PURPOSE_MAP
        if fill_mean_cols is None:
            fill_mean_cols = FILL_MEAN_COLS
        if fill_max_cols is None:
            fill_max_cols = FILL_MAX_COLS
        self.fill_max_cols = fill_max_cols

        self.fill_mean_cols = fill_mean_cols

        self.fill_min_cols = None

        self.fill_dict = {}

        self.purpose_map = purpose_map

        self.us_urbanization_xlsx_path = us_urbanization_xlsx_path
        self.us_state_to_urbanization = pd.read_excel(
            us_urbanization_xlsx_path, index_col=0
        )["Urbanization"]

    def fit(self, X, y=None):
        self.fit_or_transform(X, y, mode="fit")
        return self

    def transform(self, X, y=None):
        return self.fit_or_transform(X, y, mode="transform")

    def fit_or_transform(self, X, y=None, mode="fit"):
        assert mode in ["fit", "transform"]

        X = self.transform_verification_status(X, "verification_status")
        X = self.simple_impute(X, mode)
        X = self.transform_emp_title(X, "emp_title", mode)
        X = self.transform_purpose(X, "purpose")
        X = self.transform_addr_state(X, "addr_state")

        return X

    @staticmethod
    def transform_verification_status(X, col):
        X[col] = X[col].map({"Not Verified": False, "Source Verified": True})
        X[col] = X[col].astype(bool)
        return X

    def transform_emp_title(self, X, col, mode):
        # Аббревиатура и расшифровка
        if mode == "fit":
            self.abbr_to_occupation = {}

            occupations = X["emp_title"].value_counts(ascending=True).index

            for occ in occupations:
                occ_split = occ.split()

                if len(occ_split) < 2:
                    continue

                abbr = "".join([s[0] for s in occ_split])
                self.abbr_to_occupation[abbr] = occ

        X[col] = [self.abbr_to_occupation.get(occ, occ) for occ in X["emp_title"]]

        if mode == "fit":
            self.selected_jobs = X["emp_title"].value_counts().index[:20]

        not_selected_jobs_mask = ~X["emp_title"].isin(self.selected_jobs)
        X["emp_title"][not_selected_jobs_mask] = "other"

        return X

    def transform_purpose(self, X, col):
        X[col] = [self.purpose_map.get(purp, purp) for purp in X[col]]
        return X

    def transform_addr_state(self, X, col):
        X["addr_urbanization"] = X[col].map(self.us_state_to_urbanization)
        X.drop(columns=col, inplace=True)
        return X

    def simple_impute(self, X, mode):

        if mode == "fit":
            self.fill_max_cols = list(set(self.fill_max_cols) & set(X.columns))
            self.fill_mean_cols = list(set(self.fill_mean_cols) & set(X.columns))
            self.fill_min_cols = list(
                set(X.columns) - set(self.fill_max_cols) - set(self.fill_mean_cols)
            )

        for cols, kind in zip(
            [self.fill_min_cols, self.fill_mean_cols, self.fill_max_cols],
            ["min", "mean", "max"],
        ):

            if mode == "fit":
                f = getattr(np, kind)
                self.fill_dict[kind] = f(X[cols], axis=0)

            fillna_vals = self.fill_dict[kind]
            X[cols] = X[cols].fillna(fillna_vals)

        return X
