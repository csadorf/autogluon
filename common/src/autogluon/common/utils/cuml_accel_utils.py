"""
Utility module for managing cuML GPU acceleration for scikit-learn models.

This module provides functions to activate cuml.accel on a per-module basis,
allowing AutoGluon to transparently use GPU-accelerated sklearn implementations
when num_gpus >= 1 is specified.
"""

import logging

logger = logging.getLogger(__name__)

# Global state to track activation
_activated_modules = set()
_accelerator = {}  # Dict mapping module_name -> Accelerator instance


def is_cuml_accel_available() -> bool:
    """
    Check if cuml.accel is available for import.

    Returns
    -------
    bool
        True if cuml.accel can be imported, False otherwise.
    """
    try:
        import cuml.accel  # noqa: F401
        return True
    except ImportError:
        return False


def is_module_accelerated(module_name: str) -> bool:
    """
    Check if a specific sklearn module has already been accelerated.

    Parameters
    ----------
    module_name : str
        The sklearn module name (e.g., "sklearn.ensemble")

    Returns
    -------
    bool
        True if the module is already accelerated, False otherwise.
    """
    return module_name in _activated_modules


def activate_cuml_accel_for_module(module_name: str) -> bool:
    """
    Activate cuML GPU acceleration for a specific sklearn module.

    This function registers a single sklearn module with cuml.accel, enabling
    GPU acceleration for that module's estimators. It tracks which modules
    have been activated to avoid duplicate registration.

    Parameters
    ----------
    module_name : str
        The sklearn module to accelerate (e.g., "sklearn.ensemble", "sklearn.neighbors")

    Returns
    -------
    bool
        True if activation succeeded (or was already active), False if it failed.

    Examples
    --------
    >>> activate_cuml_accel_for_module("sklearn.ensemble")
    True
    >>> # Now sklearn.ensemble imports will use cuML implementations
    >>> from sklearn.ensemble import RandomForestClassifier
    """
    global _accelerator, _activated_modules

    # Check if already activated
    if module_name in _activated_modules:
        logger.log(15, f"cuML acceleration already active for {module_name}")
        return True

    # Check if cuml.accel is available
    if not is_cuml_accel_available():
        logger.log(15, f"cuml.accel not available, cannot accelerate {module_name}")
        return False

    try:
        # Import cuml.accel components
        from cuml.accel.accelerator import Accelerator
        from cuml.accel.core import _exclude_from_acceleration

        # Create a dedicated accelerator instance for this module
        accelerator = Accelerator(exclude=_exclude_from_acceleration)
        logger.log(15, f"Created cuML Accelerator instance for {module_name}")

        # Map sklearn module to cuml wrapper module
        # Example: "sklearn.ensemble" -> "cuml.accel._wrappers.sklearn.ensemble"
        wrapper_module = module_name.replace("sklearn", "cuml.accel._wrappers.sklearn")

        # Register and install the module
        accelerator.register(module_name, wrapper_module)
        accelerator.install()

        # Store the accelerator instance and track successful activation
        _accelerator[module_name] = accelerator
        _activated_modules.add(module_name)
        logger.log(15, f"Successfully activated cuML acceleration for {module_name}")

        return True

    except Exception as e:
        logger.log(15, f"Failed to activate cuML acceleration for {module_name}: {e}")
        return False


def get_activated_modules() -> set:
    """
    Get the set of sklearn modules that have been activated.

    Returns
    -------
    set
        Set of module names that have been activated.
    """
    return _activated_modules.copy()

