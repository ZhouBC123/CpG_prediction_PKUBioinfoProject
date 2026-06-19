from __future__ import annotations

import numpy as np


class MarkovBinaryClassifier:
    """Class-conditional first-order Markov baseline for sequence classification."""

    def __init__(self, num_symbols: int = 5, alpha: float = 1.0):
        self.num_symbols = num_symbols
        self.alpha = alpha
        self.log_class_prior_: np.ndarray | None = None
        self.log_start_: np.ndarray | None = None
        self.log_transition_: np.ndarray | None = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> "MarkovBinaryClassifier":
        x = np.asarray(x, dtype=np.int64)
        y = np.asarray(y, dtype=np.int64)
        if x.ndim != 2 or y.ndim != 1:
            raise ValueError("expected x=(samples,length), y=(samples,)")

        class_counts = np.bincount(y, minlength=2).astype(np.float64) + self.alpha
        start_counts = np.full((2, self.num_symbols), self.alpha, dtype=np.float64)
        transition_counts = np.full((2, self.num_symbols, self.num_symbols), self.alpha, dtype=np.float64)

        for cls in (0, 1):
            cls_x = x[y == cls]
            if cls_x.size == 0:
                continue
            np.add.at(start_counts[cls], cls_x[:, 0], 1)
            prev = cls_x[:, :-1].ravel()
            nxt = cls_x[:, 1:].ravel()
            np.add.at(transition_counts[cls], (prev, nxt), 1)

        self.log_class_prior_ = np.log(class_counts / class_counts.sum())
        self.log_start_ = np.log(start_counts / start_counts.sum(axis=1, keepdims=True))
        self.log_transition_ = np.log(transition_counts / transition_counts.sum(axis=2, keepdims=True))
        return self

    def _check_fitted(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.log_class_prior_ is None or self.log_start_ is None or self.log_transition_ is None:
            raise RuntimeError("model is not fitted")
        return self.log_class_prior_, self.log_start_, self.log_transition_

    def decision_function(self, x: np.ndarray) -> np.ndarray:
        priors, starts, transitions = self._check_fitted()
        x = np.asarray(x, dtype=np.int64)
        scores = np.zeros((x.shape[0], 2), dtype=np.float64)
        for cls in (0, 1):
            score = np.full(x.shape[0], priors[cls], dtype=np.float64)
            score += starts[cls, x[:, 0]]
            score += transitions[cls, x[:, :-1], x[:, 1:]].sum(axis=1)
            scores[:, cls] = score
        return scores

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        scores = self.decision_function(x)
        scores = scores - scores.max(axis=1, keepdims=True)
        probs = np.exp(scores)
        return probs / probs.sum(axis=1, keepdims=True)

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.decision_function(x).argmax(axis=1).astype(np.uint8)


class SupervisedHMMCpGSegmenter:
    """Two-state supervised HMM for base-level CpG island segmentation."""

    def __init__(self, num_symbols: int = 5, alpha: float = 1.0):
        self.num_symbols = num_symbols
        self.alpha = alpha
        self.log_start_: np.ndarray | None = None
        self.log_transition_: np.ndarray | None = None
        self.log_emission_: np.ndarray | None = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> "SupervisedHMMCpGSegmenter":
        x = np.asarray(x, dtype=np.int64)
        y = np.asarray(y, dtype=np.int64)
        if x.ndim != 2 or y.shape != x.shape:
            raise ValueError("expected x=(samples,length), y=(samples,length)")

        start_counts = np.full(2, self.alpha, dtype=np.float64)
        transition_counts = np.full((2, 2), self.alpha, dtype=np.float64)
        emission_counts = np.full((2, self.num_symbols), self.alpha, dtype=np.float64)

        np.add.at(start_counts, y[:, 0], 1)
        np.add.at(transition_counts, (y[:, :-1].ravel(), y[:, 1:].ravel()), 1)
        np.add.at(emission_counts, (y.ravel(), x.ravel()), 1)

        self.log_start_ = np.log(start_counts / start_counts.sum())
        self.log_transition_ = np.log(transition_counts / transition_counts.sum(axis=1, keepdims=True))
        self.log_emission_ = np.log(emission_counts / emission_counts.sum(axis=1, keepdims=True))
        return self

    def _check_fitted(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.log_start_ is None or self.log_transition_ is None or self.log_emission_ is None:
            raise RuntimeError("model is not fitted")
        return self.log_start_, self.log_transition_, self.log_emission_

    def predict(self, x: np.ndarray) -> np.ndarray:
        log_start, log_transition, log_emission = self._check_fitted()
        x = np.asarray(x, dtype=np.int64)
        predictions = np.zeros_like(x, dtype=np.uint8)

        for i, seq in enumerate(x):
            length = seq.shape[0]
            dp = np.empty((length, 2), dtype=np.float64)
            back = np.zeros((length, 2), dtype=np.uint8)
            dp[0] = log_start + log_emission[:, seq[0]]

            for t in range(1, length):
                scores = dp[t - 1][:, None] + log_transition
                back[t] = scores.argmax(axis=0)
                dp[t] = scores.max(axis=0) + log_emission[:, seq[t]]

            state = int(dp[-1].argmax())
            predictions[i, -1] = state
            for t in range(length - 1, 0, -1):
                state = int(back[t, state])
                predictions[i, t - 1] = state

        return predictions
