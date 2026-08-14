from typing import Iterable, List

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


AGENT_DESCRIPTIONS = [
    "understands semantic meaning, implication, context, and hidden meaning",
    "detects sentiment, emotion, sarcasm, subjective attitude, and implicit feeling",
    "detects persuasion, misleading intention, exaggeration, and click-inducing purpose",
    "detects sensational words, punctuation, keywords, and surface-level patterns",
    "detects mismatch, contradiction, inconsistency between title, content, and image description",
]


class TfidfRelevanceEstimator:
    """TF-IDF cosine relevance between tasks/samples and fixed agent descriptions."""

    def __init__(self, agent_descriptions: List[str] = None):
        self.agent_descriptions = agent_descriptions or AGENT_DESCRIPTIONS
        # Character n-grams keep the demo dependency-light and still work for
        # mixed English/Chinese toy text without tokenization packages.
        self.vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            lowercase=True,
            min_df=1,
        )
        self.agent_matrix = None

    def fit(self, task_descriptions: Iterable[str], sample_texts: Iterable[str]):
        corpus = list(self.agent_descriptions) + list(task_descriptions) + list(sample_texts)
        self.vectorizer.fit(corpus)
        self.agent_matrix = self.vectorizer.transform(self.agent_descriptions)
        return self

    @staticmethod
    def _stabilize(scores: np.ndarray) -> np.ndarray:
        scores = np.maximum(scores, 0.0)
        row_min = scores.min(axis=1, keepdims=True)
        row_max = scores.max(axis=1, keepdims=True)
        scaled = (scores - row_min) / (row_max - row_min + 1e-8)
        return (0.05 + 0.95 * scaled).astype(np.float32)

    def relevance(self, texts: Iterable[str]) -> np.ndarray:
        if self.agent_matrix is None:
            raise RuntimeError("TfidfRelevanceEstimator must be fitted before use.")
        matrix = self.vectorizer.transform(list(texts))
        scores = cosine_similarity(matrix, self.agent_matrix)
        return self._stabilize(scores)

    def task_relevance(self, task_descriptions: Iterable[str]) -> np.ndarray:
        return self.relevance(task_descriptions)

    def sample_relevance(self, sample_texts: Iterable[str]) -> np.ndarray:
        return self.relevance(sample_texts)
