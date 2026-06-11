MODEL_LOG_NAMES = {
    'DiscreteRrlModel': 'rrl-parser',
    'LimiXModel': 'limix',
    'RandomForestModel': 'random-forest',
    'RrlModel': 'rrl',
    'TabPFNModel': 'tabpfn',
    'XgboostModel': 'xgboost',
}


def model_name_for(model_cls: type[object]) -> str:
    name = model_cls.__name__
    if name in MODEL_LOG_NAMES:
        return MODEL_LOG_NAMES[name]
    name = name.removesuffix('Model')
    return name[:1].lower() + name[1:]
