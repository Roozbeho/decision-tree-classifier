from typing import Tuple, Type
import pandas as pd
import numpy as np


class TreeEstimator:
    """
    A class to perform tree-based operations for decision tree algorithms.

    """

    def _split_data(self, x, y, threshold: int, feature_index: int):
        """Split the data based on a given threshold and feature index."""
        left_indexes = x[:, feature_index] < threshold
        return x[left_indexes], x[~left_indexes], y[left_indexes], y[~left_indexes]

    def _calculate_gini(self, splited_y: np.array) -> float:
        """Calculate the Gini impurity of a splitted data."""
        ـ, counts = np.unique(splited_y, return_counts=True)
        if len(counts) == 1:
            return 0
        return 1 - np.sum((counts / counts.sum()) ** 2)

    def _calculate_entropy(self, splited_y: np.array) -> float:
        """Calculate the entropy of a splitted data."""
        ـ, counts = np.unique(splited_y, return_counts=True)
        if len(counts) == 1:
            return 0
        probabilities = counts / counts.sum()
        return -np.sum(probabilities * np.log2(probabilities))

    def Gini(self, x: np.ndarray, y: np.ndarray, feature: int) -> Tuple[float, float]:
        """Calculate the best Gini impurity and threshold for a given feature."""
        unique_x = sorted(np.unique(x[:, feature]))
        best_gini, best_threshold = -np.inf, None
        parent_gini = self._calculate_gini(y)

        for i in range(len(unique_x) - 1):
            threshold = np.mean([unique_x[i], unique_x[i + 1]])
            left_x, right_x, left_y, right_y = self._split_data(x, y, threshold, feature)
            
            left_gini = self._calculate_gini(left_y)
            right_gini = self._calculate_gini(right_y)

            gini_impurity = ((len(left_y) / len(y)) * left_gini) + (
                (len(right_y) / len(y) * right_gini)
            )
            gini_gain = parent_gini - gini_impurity

            if gini_gain > best_gini:
                best_gini = gini_gain
                best_threshold = threshold

        return best_gini, best_threshold

    def Entropy(self, x: np.ndarray, y: np.ndarray, feature: int) -> Tuple[float, float]:
        """Calculate the best entropy and threshold for a given feature."""
        unique_x = sorted(np.unique(x[:, feature]))
        best_entropy, best_threshold = -np.inf, None
        parent_entropy = self._calculate_entropy(y)

        for i in range(len(unique_x) - 1):
            threshold = np.mean([unique_x[i], unique_x[i + 1]])
            left_x, right_x, left_y, right_y = self._split_data(x, y, threshold, feature)

            left_entropy = self._calculate_entropy(left_y)
            right_entropy = self._calculate_entropy(right_y)

            info_gain = (
                parent_entropy
                - (len(left_y) / len(y) * left_entropy)
                - (len(right_y) / len(y) * right_entropy)
            )

            if info_gain > best_entropy:
                best_entropy = info_gain
                best_threshold = threshold

        return best_entropy, best_threshold


class BaseDecisionTree(TreeEstimator):
    """
    A class to perform tree-based operations for decision tree algorithms.

    Attributes:
    -----------
    CRITERIA : tuple
        A tuple containing the available criteria for splitting the data.
    """

    CRITERIA = ("gini", "entropy")

    def __init__(self, criteria: str, max_depth=np.inf, minimum_samples_per_split=2):
        assert (criteria in self.CRITERIA), f"Choose one of the {self.CRITERIA} as criteria"
        self.criteria_func = self.Gini if criteria == "gini" else self.Entropy

        assert isinstance(max_depth, (int, float)), "max_depth must be intger"
        assert isinstance(minimum_samples_per_split, (int, float)), "minimum_samples_per_split must be intger"

        self.max_depth = max_depth
        self.minimum_samples_per_split = minimum_samples_per_split

        self.threshold = 0
        self.predicted_class = None
        self.split_feature_index = None
        self.impurity = None
        self.left = None
        self.right = None
        self.is_leaf = True

    def find_best_split(self, x: np.ndarray, y: np.ndarray) -> Tuple[float, int, float]:
        """Find the best feature to split the data based on the given criteria."""
        best_score, best_feature_index, best_threshold = -np.inf, None, None

        for feature_index in range(x.shape[1]):
            score, threshold = self.criteria_func(x, y, feature_index)
            if score > best_score:
                best_score, best_feature_index, best_threshold = (
                    score,
                    feature_index,
                    threshold,
                )

        return best_score, best_feature_index, best_threshold

    def _fit(self, x: np.ndarray, y: np.ndarray, depth=0):
        """Recursively fits the decision tree to the training data."""
        maximum_sample_per_class = lambda x: x[0][np.where(x[1] == max(x[1]))]
        predicted_class = maximum_sample_per_class(
            np.unique(y, return_counts=True)
        ).tolist()[0]
        self.predicted_class = predicted_class

        if (
            depth < self.max_depth
            and len(set(y)) > 1
            and len(y) >= self.minimum_samples_per_split
        ):
            best_score, best_feature_index, best_threshold = self.find_best_split(x, y)

            if best_score > 0:
                self.split_feature_index = best_feature_index
                self.impurity = best_score
                self.threshold = best_threshold
                self.is_leaf = False

                left_x, right_x, left_y, right_y = self._split_data(
                    x, y, best_threshold, best_feature_index
                )

                self.left = BaseDecisionTree(
                    self.criteria_func.__name__.lower(),
                    self.max_depth,
                    self.minimum_samples_per_split,
                )
                self.left._fit(left_x, left_y, depth + 1)

                self.right = BaseDecisionTree(
                    self.criteria_func.__name__.lower(),
                    self.max_depth,
                    self.minimum_samples_per_split,
                )
                self.right._fit(right_x, right_y, depth + 1)
        return self

    def __str__(self) -> str:
        return str(self.predicted_class)


class ClasificationTree(BaseDecisionTree):
    """
    A class to perform classification using decision tree algorithm.

    Parameters:
    -----------
    criteria : str, optional
        The criteria to split the data. It can be either 'gini' or 'entropy'.
        Default is 'gini'.
    max_depth : int or float, optional
        The maximum depth of the decision tree. If it's np.inf, the tree will grow until all leaves are pure.
        Default is np.inf.
    minimum_samples_per_split : int or float, optional
        The minimum number of samples required to split an internal node.
        Default is 2.
    """

    def __init__(
        self,
        criteria="gini",
        max_depth=np.inf,
        minimum_samples_per_split=2,
    ):
        super().__init__(criteria, max_depth, minimum_samples_per_split)

    def fit(self, x: pd.Series, y: pd.Series):
        """Fits the decision tree to the training data."""
        if x.shape[0] != y.shape[0]:
            raise ValueError('inputs have inconsistent number of dimensions')
        x, y = x.to_numpy(), y.to_numpy()
        self.tree = self._fit(x, y)

    def predict(self, x: pd.DataFrame) -> np.array:
        """Predicts the target values for the given test data."""
        return np.array([self._predict(row) for row in x.to_numpy()], dtype="int64")

    def _predict(self, test_val: np.array) -> str:
        node = self.tree

        while not node.is_leaf:
            if test_val[node.split_feature_index] < node.threshold:
                node = node.left
            else:
                node = node.right

        return str(node)

    def display_tree_structure(self):
        """Prints the structure of the decision tree."""

        def _display(node: Type[BaseDecisionTree], depth=0) -> None:
            print(("|" + (" " * 1)) * depth, end="")
            if node.is_leaf:
                print("|" + "-" * (1), "class = " + str(node.predicted_class))
            else:
                print("|" + "-" * (1), "feature <= " + str(round(node.impurity, 2)))
            if node.left:
                _display(node.left, depth + 1)
            if node.right:
                _display(node.right, depth + 1)

        _display(self.tree)

    def score(self, y: pd.Series, prediction: np.array) -> float:
        """Calculates the accuracy of the decision tree model."""
        y = y.to_numpy()
        if y.shape[0] != prediction.shape[0]:
            raise ValueError("y and predisction must have the same length")
        return np.mean(y == prediction)


