from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from category_encoders.woe import WOEEncoder
from category_encoders.cat_boost import CatBoostEncoder
from category_encoders.leave_one_out import LeaveOneOutEncoder
from category_encoders.target_encoder import TargetEncoder
from category_encoders.count import CountEncoder
from src.config import FEATURES

ENCODER_DICT = {
    "woe": WOEEncoder(handle_missing=0),
    "catboost": CatBoostEncoder(return_df=False),
    "loo": LeaveOneOutEncoder(return_df=False),
    "target": TargetEncoder(return_df=False),
    "count": CountEncoder(
        handle_unknown=0,
        combine_min_nan_groups=False,
        min_group_size=10
    ),
}

def create_encoder(encoder):
    function_transformer = FunctionTransformer(
        lambda x: x[FEATURES]
    )
    pipeline = Pipeline(steps=[
        ("function_transformer", function_transformer),
        ("encoder", ENCODER_DICT[encoder]),
    ])
    return pipeline