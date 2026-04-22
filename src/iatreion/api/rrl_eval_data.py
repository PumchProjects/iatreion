import pandas as pd

from iatreion.configs import DiscreteRrlConfig, RrlEvalConfig
from iatreion.exceptions import IatreionException
from iatreion.models import DiscreteRrlModel
from iatreion.preprocessors import get_preprocessors


def get_callbacks(config: RrlEvalConfig) -> tuple[DiscreteRrlConfig, list[object]]:
    process_config, rrl_config = config.make_configs()
    preprocessors = get_preprocessors(process_config)
    try:
        callbacks = [
            preprocessor.get_stem_to_name_callback() for preprocessor in preprocessors
        ]
    except IatreionException as error:
        raise IatreionException(
            'Failed to get the callback for displaying "$data_name" rules.',
            **error.mapping,
        ) from error
    return rrl_config, callbacks


def build_model(config: RrlEvalConfig) -> DiscreteRrlModel:
    rrl_config, callbacks = get_callbacks(config)
    return DiscreteRrlModel(rrl_config, callbacks)


def get_data_model(
    config: RrlEvalConfig,
) -> tuple[
    list[pd.DataFrame],
    list[pd.DataFrame],
    pd.DataFrame | None,
    DiscreteRrlModel,
]:
    process_config, rrl_config = config.make_configs()
    preprocessors = get_preprocessors(process_config)
    data = [preprocessor.get_data_outer() for preprocessor in preprocessors]
    callbacks = [
        preprocessor.get_stem_to_name_callback() for preprocessor in preprocessors
    ]
    additional_data = process_config._final_indices
    group_names = preprocessors[0].get_group_names() if config.mode == 'eval' else None
    model = DiscreteRrlModel(rrl_config, callbacks)
    return data, additional_data, group_names, model
