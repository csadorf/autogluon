#!/usr/bin/env python3
"""
Demo: Training AutoGluon models with GPU acceleration (num_gpus=1)
"""

import click
import pandas as pd
from sklearn.datasets import make_classification
from autogluon.tabular import TabularPredictor


def _run_autogluon_demo(model: str, num_gpus: int):
    # Create test dataset
    X, y = make_classification(n_samples=300, n_features=8, random_state=42)
    df = pd.DataFrame(X, columns=[f'f{i}' for i in range(8)])
    df['target'] = y

    match model:
        case "rf":
            predictor = TabularPredictor(label='target', path='/tmp/demo_rf').fit(
                train_data=df,
                time_limit=15,
                hyperparameters={'RF': {}},
                num_bag_folds=0,
                num_stack_levels=0,
                ag_args_fit={'num_gpus': num_gpus},
                verbosity=2
            )
        case "knn":
            predictor = TabularPredictor(label="target", path="/tmp/demo_knn").fit(
                train_data=df,
                time_limit=15,
                hyperparameters={'KNN': {}},
                num_bag_folds=0,
                num_stack_levels=0,
                ag_args_fit={'num_gpus': num_gpus},
                verbosity=2
            )
        case "lr":
            predictor = TabularPredictor(label='target', path='/tmp/demo_lr').fit(
                train_data=df,
                time_limit=15,
                hyperparameters={'LR': {}},
                num_bag_folds=0,
                num_stack_levels=0,
                ag_args_fit={'num_gpus': num_gpus},
                verbosity=2
            )
        case _:
            raise ValueError(f"Invalid model: {model}")

    predictor.leaderboard(df)

    print("\nDemo complete!")


@click.command()
@click.argument("model", type=click.Choice(["rf", "knn", "lr"]))
@click.option("--num-gpus", type=int, default=0, help="Number of GPUs to use for training")
@click.option("--profile", is_flag=True, help="Profile the training")
def main(model: str, num_gpus: int, profile: bool):
    if profile:
        # importing late here to avoid otherwise contaminating the test with an
        # explicit import of cuml.accel at the user-level
        import cuml.accel
        with cuml.accel.profile():
            _run_autogluon_demo(model, num_gpus)
    else:
        _run_autogluon_demo(model, num_gpus)


if __name__ == "__main__":
    main()
