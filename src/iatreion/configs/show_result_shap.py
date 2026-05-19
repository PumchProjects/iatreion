from dataclasses import dataclass

from cyclopts import Parameter

from .show_result_interpretability import ShowInterpretabilityConfig


@Parameter(name='*')
@dataclass(kw_only=True)
class ShowShapConfig(ShowInterpretabilityConfig):
    shap_output_index: int | None = None
    'Output/class index for SHAP plots. For binary tasks, the default is the positive class.'

    shap_sample_index: int = 0
    'Sample index within the concatenated SHAP samples, used for waterfall plot.'

    shap_feature: str | None = None
    'Feature name used for SHAP dependence plot.'

    shap_color_feature: str | None = None
    'Optional feature name used to color the SHAP dependence plot.'

    def __post_init__(self) -> None:
        super().__post_init__()
