#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Optional, Tuple

import numpy as np
from ax.core.search_space import SearchSpaceDigest
from ax.core.types import TCandidateMetadata
from ax.models.numpy_base import NumpyModel
from ax.utils.common.docutils import copy_doc
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor


class RandomForest(NumpyModel):
    """A Random Forest model.

    Uses a parametric bootstrap to handle uncertainty in Y.

    Can be used to fit data, make predictions, and do cross validation; however
    gen is not implemented and so this model cannot generate new points.

    Args:
        max_features: Maximum number of features at each split. With one-hot
            encoding, this should be set to None. Defaults to "sqrt", which is
            Breiman's version of Random Forest.
        num_trees: Number of trees.
    """

    def __init__(
        self, max_features: Optional[str] = "sqrt", num_trees: int = 500
    ) -> None:
        self.max_features = max_features
        self.num_trees = num_trees
        self.models: List[RandomForestRegressor] = []

    @copy_doc(NumpyModel.fit)
    def fit(
        self,
        Xs: List[np.ndarray],
        Ys: List[np.ndarray],
        Yvars: List[np.ndarray],
        search_space_digest: SearchSpaceDigest,
        metric_names: List[str],
        candidate_metadata: Optional[List[List[TCandidateMetadata]]] = None,
    ) -> None:
        for i, X in enumerate(Xs):
            self.models.append(
                _get_rf(
                    X=X,
                    Y=Ys[i],
                    Yvar=Yvars[i],
                    num_trees=self.num_trees,
                    max_features=self.max_features,
                )
            )

    @copy_doc(NumpyModel.predict)
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return _rf_predict(self.models, X)

    @copy_doc(NumpyModel.cross_validate)
    def cross_validate(
        self,
        Xs_train: List[np.ndarray],
        Ys_train: List[np.ndarray],
        Yvars_train: List[np.ndarray],
        X_test: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        cv_models: List[RandomForestRegressor] = []
        for i, X in enumerate(Xs_train):
            cv_models.append(
                _get_rf(
                    X=X,
                    Y=Ys_train[i],
                    Yvar=Yvars_train[i],
                    num_trees=self.num_trees,
                    max_features=self.max_features,
                )
            )
        return _rf_predict(cv_models, X_test)


def _get_rf(
    X: np.ndarray,
    Y: np.ndarray,
    Yvar: np.ndarray,
    num_trees: int,
    max_features: Optional[str],
) -> RandomForestRegressor:
    """Fit a Random Forest model.

    Args:
        X: X
        Y: Y
        Yvar: Variance for Y
        num_trees: Number of trees
        max_features: Max features specifier

    Returns: Fitted Random Forest.
    """
    r = RandomForestRegressor(
        n_estimators=num_trees, max_features=max_features, bootstrap=True
    )
    # pyre-fixme[16]: `RandomForestRegressor` has no attribute `estimators_`.
    r.estimators_ = [DecisionTreeRegressor() for i in range(r.n_estimators)]
    for estimator in r.estimators_:
        # Parametric bootstrap
        y = np.random.normal(loc=Y[:, 0], scale=np.sqrt(Yvar[:, 0]))
        estimator.fit(X, y)
    return r


def _rf_predict(
    models: List[RandomForestRegressor], X: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Make predictions with Random Forest models.

    Args:
        models: List of models for each outcome
        X: X to predict

    Returns: mean and covariance estimates
    """
    f = np.zeros((X.shape[0], len(models)))
    cov = np.zeros((X.shape[0], len(models), len(models)))
    for i, m in enumerate(models):
        # pyre-fixme[16]: `RandomForestRegressor` has no attribute `estimators_`.
        preds = np.vstack([tree.predict(X) for tree in m.estimators_])
        f[:, i] = preds.mean(0)
        cov[:, i, i] = preds.var(0)
    return f, cov

# function above returns mean and var.
from typing import Any, Callable, Dict, List, Optional, Tuple
from ax.models.types import TConfig
from ax.core.types import TCandidateMetadata, TGenMetadata
def gen(
        self,
        n: int,
        bounds: List[Tuple[float, float]],
        objective_weights: np.ndarray,
        outcome_constraints: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        linear_constraints: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        fixed_features: Optional[Dict[int, float]] = None,
        pending_observations: Optional[List[np.ndarray]] = None,
        model_gen_options: Optional[TConfig] = None,
        rounding_func: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    ) -> Tuple[
        np.ndarray, np.ndarray, TGenMetadata, Optional[List[TCandidateMetadata]]
    ]:
        """
        Generate new candidates.

        Args:
            n: Number of candidates to generate.
            bounds: A list of (lower, upper) tuples for each column of X.
            objective_weights: The objective is to maximize a weighted sum of
                the columns of f(x). These are the weights.
            outcome_constraints: A tuple of (A, b). For k outcome constraints
                and m outputs at f(x), A is (k x m) and b is (k x 1) such that
                A f(x) <= b.
            linear_constraints: A tuple of (A, b). For k linear constraints on
                d-dimensional x, A is (k x d) and b is (k x 1) such that
                A x <= b.
            fixed_features: A map {feature_index: value} for features that
                should be fixed to a particular value during generation.
            pending_observations:  A list of m (k_i x d) feature arrays X
                for m outcomes and k_i pending observations for outcome i.
            model_gen_options: A config dictionary that can contain
                model-specific options.
            rounding_func: A function that rounds an optimization result (xbest)
                appropriately (i.e., according to `round-trip` transformations)

        Returns:
            4-element tuple containing

            - (n x d) tensor of generated points.
            - n-tensor of weights for each point.
            - Generation metadata
            - Dictionary of model-specific metadata for the given
                generation candidates
        """
        pass