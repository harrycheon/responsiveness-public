from typing import Optional, Union

import pandas as pd
import numpy as np
from .action_set import ActionSet

class ReachableSet(object):
    """
    Class to represent or manipulate a reachable set over a discrete feature space
    """

    _TOLERANCE = 1e-16
    _METADATA_KEYS = ["complete"]

    def __init__(
        self,
        action_set: ActionSet,
        x: Optional[np.ndarray] = None,
        complete: bool = False,
        values: Optional[np.ndarray] = None,
        initialize_from_actions: bool = False,
        **kwargs,
    ):
        """
        :param action_set: action set
        :param x: source point
        :param complete: indicator if complete
        :param values: array-like -- set of d-dimensional feature vectors or actions to add
        :param initialize_from_actions: set to True if values that we pass are actions
        """

        self._action_set = action_set
        self._complete = complete

        if x is None:
            if initialize_from_actions:
                raise ValueError(
                    "Cannot initialize from actions without the initial point."
                )
            if values is None or len(values) == 0:
                raise ValueError(
                    "Need to provide values if the initial point is not given."
                )
            else:
                x = values[0].flatten()

        self._x = x
        self._X = np.array(x).reshape((1, self.d))
        if self.discrete:
            self.has_feature_vector = lambda x: np.all(self._X == x, axis=1).any()
        else:
            self.has_feature_vector = (
                lambda x: np.isclose(self._X, x, atol=self._TOLERANCE).all(axis=1).any()
            )

        if values is not None:
            self.add(values=values, actions=initialize_from_actions, **kwargs)

    @property
    def action_set(self):
        """return action set"""
        return self._action_set

    @property
    def discrete(self) -> bool:
        """returns True if fixed point"""
        return self._action_set.discrete

    @property
    def x(self):
        """return source point"""
        return self._x

    @property
    def d(self):
        """returns number of dimensions"""
        return len(self._x)

    @property
    def X(self):
        """returns reachable feature vectors"""
        return self._X

    @property
    def actions(self) -> np.ndarray:
        """returns action vectors, computed on the fly"""
        return np.subtract(self.X, self.x)

    @property
    def complete(self) -> bool:
        """returns True if reachable set contains all reachable points"""
        return self._complete

    @property
    def fixed(self) -> bool:
        """returns True if fixed point"""
        return len(self) == 1 and self._complete

    def __getitem__(self, i: int) -> np.ndarray:
        return self._X[i]

    def __check_rep__(self) -> bool:
        """returns True if class invariants hold"""
        assert self.X.shape[0] == len(self)
        assert self.has_feature_vector(self.x)
        return True

    def __contains__(self, item: Union[np.ndarray, list]):
        """returns True if reachable set contains all reachable points"""
        if isinstance(item, list):
            out = np.all(self.X == item, axis=1).any()
        elif isinstance(item, np.ndarray) and item.ndim == 1:
            out = self.has_feature_vector(item)
        elif isinstance(item, np.ndarray) and item.ndim == 2:
            out = np.all([self.has_feature_vector(x) for x in item])
        else:
            out = False
        return out

    def __is_comparable_to__(self, other):
        out = (
            isinstance(other, ReachableSet)
            and self.action_set.names == other.action_set.names
        )
        return out

    def __eq__(self, other):
        out = (
            self.__is_comparable_to__(other)
            and np.array_equal(self.x, other.x)
            and len(self) == len(other)
            and self.__contains__(other.X)
        )
        return out

    def recourse_scores(self, feature_idx, predictor, scaler, target=1):
        """
        computes 1D (marginal) recourse information for a feature. can divide the output
        to compute recourse importance score

        :param feature_idx: index of the feature
        :param predictor: prediction handle that can can take ReachableSet.X
        :param target: desired outcome (default: 1)
        :return: number of marginal actions, number of marginal actions that lead to target
        """
        marginal_actions = self.get_marginal_action(feature_idx)

        n_marginal_actions = marginal_actions.shape[0]
        
        # feature not actionable for this instance
        if n_marginal_actions == 0:
            return 0, 0   

        reformat = lambda x: x.reshape(1, -1) if x.ndim == 1 else x
        rescale = lambda x: scaler.transform(reformat(x)) if scaler is not None else reformat(x)

        # scaling (temporary)
        marginal_x = self._x + marginal_actions
        marginal_preds = predictor(rescale(marginal_x))

        return n_marginal_actions, np.count_nonzero(marginal_preds == target)

    def get_marginal_action(self, j):
        """
        """
        actions = self.actions
        # feat j is not actionable
        if j not in self.action_set.actionable_features:
            return np.array([])
        
        # no action has feat a[j] != 0
        change_j = actions[actions[:,j] != 0]

        if change_j.shape[0] == 0:
            return np.array([])
        
        # check if there is 1-D action
        s = change_j[np.abs(change_j).clip(0,1).sum(axis=1) == 1]
        if s.shape[0] > 0:
            return s

        ass_f = set(self.action_set.constraints.get_associated_features(j))

        # check if there isn't any potential action either
        mask = np.ones_like(self._x, dtype=bool)
        mask[list(ass_f)] = False

        rest_0_idx = (change_j[:, mask] == 0).all(axis=1)
        rem_act = change_j[rest_0_idx]

        if rem_act.shape[0] == 0:
            return np.array([])
        
        marg_acts = []
        feat_sets = []
        for a in rem_act:
            non_zero = set(np.where(a != 0)[0])
            if len(non_zero) > 0 and j in non_zero and non_zero.issubset(ass_f):
                feat_sets.append(non_zero)
                marg_acts.append(a)

        set_size = np.array(list(map(len, feat_sets)))
        min_size = np.where(set_size == set_size.min())[0]

        return np.vstack(marg_acts)[min_size]

    def scores(
        self,
        point_mask=None,
        feature_mask=None,
        max_score=None,
        weigh_changes=False,
        invert=False #,ignore_downstream_effects = False,
    ):
        """
        computes reachability scores across features.
            r[j] = {% of points in reachable_set that can be reached by changes in feature j}
        :param point_mask: boolean array to select subset of points (e.g., points with recourse)
        :param feature_mask: boolean array to select subset of features (e.g., mutable features)
        :param max_score: normalization factor -- use to normalize across subpopulations
        :param invert: returns a mutability score -- {% of points in reachable set that remain the same}
        :return:
        """
        if feature_mask is None:
            R = self._X
            x = self._x
        else:
            R = self._X[:, feature_mask]
            x = self._x[feature_mask]

        if R.shape[0] == 1 or (point_mask is not None and not any(point_mask)):
            return np.zeros_like(x)

        if point_mask is not None:
            R = R[point_mask, :]

        # score computation
        if invert:
            changes = np.equal(R, x)
        else:
            changes = np.not_equal(R, x)

        if weigh_changes:
            weights = np.sum(changes, axis=1)
            keep = weights > 0
            if not any(keep):
                return np.zeros_like(x)
            scores = np.dot(1.0 / weights[keep], changes[keep, :])
        else:
            scores = np.sum(changes, axis=0)

        if max_score is not None:
            scores = scores / max_score
        return scores

    def describe(self, predictor=None, target=None):
        """
        describes predictions by features
            r[j] = {% of points in reachable_set that can be reached by changes in feature j}
        :param predictor: prediction handle that can can take ReachableSet.X as input e.g.,
               lambda x: clf.predict(x)
        :param target: target prediction
        :param max_score: normalization factor -- use to normalize across subpopulations
        :param invert: returns a mutability score -- {% of points in reachable set that remain the same}
        :return:
        """
        if predictor is None:
            df = pd.DataFrame(
                index=self.action_set.names,
                data={
                    "x": self._x,
                    "n_total": np.not_equal(self._X, self._x).sum(axis=0),
                },
            )
        else:
            assert target is not None
            changes = np.not_equal(self._X, self._x)
            idx = np.equal(predictor(self._X), target)
            df = pd.DataFrame(
                index=self.action_set.names,
                data={
                    "x": self._x,
                    "n_total": np.sum(changes, axis=0),
                    "n_target": np.sum(changes[idx], axis=0),
                },
            )
        return df

    def add(
        self,
        values: np.ndarray,
        actions: bool = False,
        check_distinct: bool = True,
        check_exists: bool = True,
    ):
        """
        :param values: array-like -- feature vectors / to add
        :param actions: set to True if `values` are actions rather than feature vectors
        :param check_distinct: set to False if `values` are distinct
        :param check_exists: set to False if `values` do not exist in current reachable set
        :return:
        """
        if isinstance(values, list):
            values = np.vstack(values)

        assert values.ndim == 2
        assert values.shape[0] > 0
        assert values.shape[1] == self.d

        if check_distinct:
            values = np.unique(values, axis=0)

        if actions:
            values = self._x + values

        if check_exists:
            keep_idx = [not self.has_feature_vector(x) for x in values]
            values = values[keep_idx]

        self._X = np.append(self._X, values=values, axis=0)
        out = values.shape[0]
        return out

    def find(self, clf, target):
        """
        :param clf: classifier with a predict function
        :param target: float/int that attains a target class, or array-like or target classes
        :return: first reachable point that attains a target prediction from the classifier
        """
        if not isinstance(target, (list, tuple)):
            target = np.float(target)

        out = (False, None)
        for x in self.X:
            if clf.predict(x) in target:
                out = (True, x)
                break
        return out

    def __len__(self) -> int:
        """returns number of points in the reachable set, including the original point"""
        return self._X.shape[0]

    def __repr__(self) -> str:
        return f"<ReachableSet<n={len(self)}, complete={self._complete}>"

    def extract(self, other):
        """extract points from another reachable set"""
        raise NotImplementedError()

    def __add__(self, other):
        """add two reachable sets together"""
        raise NotImplementedError()

    def _get_metadata(self) -> pd.Series:
        metadata = pd.Series(dict(complete=self._complete))
        assert metadata.index == self._METADATA_KEYS
        return metadata
