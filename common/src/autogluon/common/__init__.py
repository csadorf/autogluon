from .dataset import TabularDataset
from .features.feature_metadata import FeatureMetadata
from .utils.cuml_accel_utils import is_cuml_accel_available
from .utils.log_utils import _add_stream_handler
from .utils.log_utils import fix_logging_if_kaggle as __fix_logging_if_kaggle
from .version import __version__

# Fixes logger in Kaggle to show logs in notebook.
__fix_logging_if_kaggle()

_add_stream_handler()

# Pre-check cuml.accel availability to trigger early import
is_cuml_accel_available()
