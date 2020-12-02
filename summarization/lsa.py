import math
from collections import namedtuple
from operator import attrgetter
from warnings import warn

import nltk
import numpy
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from numpy.linalg import svd as singular_value_decomposition

nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)


SentenceInfo = namedtuple(
    "SentenceInfo",
    (
        "sentence",
        "order",
        "rating",
    ),
)


class ItemsCount:
    def __init__(self, value):
        self._value = value

    def __call__(self, sequence):
        if isinstance(
            self._value,
            (
                bytes,
                str,
            ),
        ):
            if self._value.endswith("%"):
                total_count = len(sequence)
                percentage = int(self._value[:-1])
                # at least one sentence should be chosen
                count = max(1, total_count * percentage // 100)
                return sequence[:count]
            else:
                return sequence[: int(self._value)]
        elif isinstance(self._value, (int, float)):
            return sequence[: int(self._value)]
        else:
            ValueError("Unsuported value of items count '%s'." % self._value)


class LsaSummarizer:
    MIN_DIMENSIONS = 3
    REDUCTION_RATIO = 1 / 1

    _stop_words = list(stopwords.words("english"))

    @staticmethod
    def normalize_word(word):
        return word.lower()

    @staticmethod
    def _get_best_sentences(sentences, count, rating, *args, **kwargs):
        rate = rating
        if isinstance(rating, dict):
            assert not args and not kwargs
            rate = lambda s: rating[s]

        infos = (
            SentenceInfo(s, o, rate(s, *args, **kwargs))
            for o, s in enumerate(sentences)
        )

        # sort sentences by rating in descending order
        infos = sorted(infos, key=attrgetter("rating"), reverse=True)
        # get `count` first best rated sentences
        if not isinstance(count, ItemsCount):
            count = ItemsCount(count)
        infos = count(infos)
        # sort sentences by their order in document
        infos = sorted(infos, key=attrgetter("order"))

        return tuple(i.sentence for i in infos)

    @property
    def stop_words(self):
        return self._stop_words

    @stop_words.setter
    def stop_words(self, words):
        self._stop_words = words

    def __call__(self, document, sentences_count):

        dictionary = self._create_dictionary(document)

        if not dictionary:
            return ()

        sentences = sent_tokenize(document)

        matrix = self._create_matrix(document, dictionary)
        matrix = self._compute_term_frequency(matrix)
        u, sigma, v = singular_value_decomposition(matrix, full_matrices=False)

        ranks = iter(self._compute_ranks(sigma, v))
        return self._get_best_sentences(
            sentences, sentences_count, lambda s: next(ranks)
        )

    def _create_dictionary(self, document):
        """Creates mapping key = word, value = row index"""

        words = word_tokenize(document)
        words = tuple(words)

        words = map(self.normalize_word, words)

        unique_words = frozenset(w for w in words if w not in self._stop_words)

        return dict((w, i) for i, w in enumerate(unique_words))

    def _create_matrix(self, document, dictionary):
        """
        Creates matrix of shape where cells
        contains number of occurences of words (rows) in senteces (cols).
        """
        sentences = sent_tokenize(document)
        words_count = len(dictionary)
        sentences_count = len(sentences)
        if words_count < sentences_count:
            message = (
                "Number of words (%d) is lower than number of sentences (%d). "
                "LSA algorithm may not work properly."
            )
            warn(message % (words_count, sentences_count))

        matrix = numpy.zeros((words_count, sentences_count))
        for col, sentence in enumerate(sentences):
            words = word_tokenize(sentence)
            for word in words:
                # only valid words is counted (not stop-words, ...)
                if word in dictionary:
                    row = dictionary[word]
                    matrix[row, col] += 1

        return matrix

    def _compute_term_frequency(self, matrix, smooth=0.4):
        """
        Computes TF metrics for each sentence (column) in the given matrix and  normalize
        the tf weights of all terms occurring in a document by the maximum tf in that document
        according to ntf_{t,d} = a + (1-a)\frac{tf_{t,d}}{tf_{max}(d)^{'}}.

        The smoothing term $a$ damps the contribution of the second term - which may be viewed
        as a scaling down of tf by the largest tf value in $d$
        """
        assert 0.0 <= smooth < 1.0

        max_word_frequencies = numpy.max(matrix, axis=0)
        rows, cols = matrix.shape
        for row in range(rows):
            for col in range(cols):
                max_word_frequency = max_word_frequencies[col]
                if max_word_frequency != 0:
                    frequency = matrix[row, col] / max_word_frequency
                    matrix[row, col] = smooth + (1.0 - smooth) * frequency

        return matrix

    def _compute_ranks(self, sigma, v_matrix):
        assert len(sigma) == v_matrix.shape[0]

        dimensions = max(
            LsaSummarizer.MIN_DIMENSIONS,
            int(len(sigma) * LsaSummarizer.REDUCTION_RATIO),
        )
        powered_sigma = tuple(
            s ** 2 if i < dimensions else 0.0 for i, s in enumerate(sigma)
        )

        ranks = []

        for column_vector in v_matrix.T:
            rank = sum(s * v ** 2 for s, v in zip(powered_sigma, column_vector))
            ranks.append(math.sqrt(rank))

        return ranks
