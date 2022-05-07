from __future__ import division
from gensim.corpora.dictionary import Dictionary
from gensim.models.coherencemodel import CoherenceModel
from gensim.models import KeyedVectors
import gensim.downloader as api
from scipy.spatial.distance import cosine
from bisect import bisect_left
from collections import namedtuple
import abc
import math

import numpy as np
import itertools

"""Rank-biased overlap, a ragged sorted list similarity measure.

See http://doi.acm.org/10.1145/1852102.1852106 for details. All functions
directly corresponding to concepts from the paper are named so that they can be
clearly cross-identified.

The definition of overlap has been modified to account for ties. Without this,
results for lists with tied items were being inflated. The modification itself
is not mentioned in the paper but seems to be reasonable, see function
``overlap()``. Places in the code which diverge from the spec in the paper
because of this are highlighted with comments.

The two main functions for performing an RBO analysis are ``rbo()`` and
``rbo_dict()``; see their respective docstrings for how to use them.

The following doctest just checks that equivalent specifications of a
problem yield the same result using both functions:

    >>> lst1 = [{"c", "a"}, "b", "d"]
    >>> lst2 = ["a", {"c", "b"}, "d"]
    >>> ans_rbo = _round(rbo(lst1, lst2, p=.9))
    >>> dct1 = dict(a=1, b=2, c=1, d=3)
    >>> dct2 = dict(a=1, b=2, c=2, d=3)
    >>> ans_rbo_dict = _round(rbo_dict(dct1, dct2, p=.9, sort_ascending=True))
    >>> ans_rbo == ans_rbo_dict
    True

"""

RBO = namedtuple("RBO", "min res ext")
RBO.__doc__ += ": Result of full RBO analysis"
RBO.min.__doc__ = "Lower bound estimate"
RBO.res.__doc__ = "Residual corresponding to min; min + res is an upper bound estimate"
RBO.ext.__doc__ = "Extrapolated point estimate"


def _round(obj):
    if isinstance(obj, RBO):
        return RBO(_round(obj.min), _round(obj.res), _round(obj.ext))
    else:
        return round(obj, 3)


def set_at_depth(lst, depth):
    ans = set()
    for v in lst[:depth]:
        if isinstance(v, set):
            ans.update(v)
        else:
            ans.add(v)
    return ans


def raw_overlap(list1, list2, depth):
    """Overlap as defined in the article.

    """
    set1, set2 = set_at_depth(list1, depth), set_at_depth(list2, depth)
    return len(set1.intersection(set2)), len(set1), len(set2)


def overlap(list1, list2, depth):
    """Overlap which accounts for possible ties.

    This isn't mentioned in the paper but should be used in the ``rbo*()``
    functions below, otherwise overlap at a given depth might be > depth which
    inflates the result.

    There are no guidelines in the paper as to what's a good way to calculate
    this, but a good guess is agreement scaled by the minimum between the
    requested depth and the lengths of the considered lists (overlap shouldn't
    be larger than the number of ranks in the shorter list, otherwise results
    are conspicuously wrong when the lists are of unequal lengths -- rbo_ext is
    not between rbo_min and rbo_min + rbo_res.

    >>> overlap("abcd", "abcd", 3)
    3.0

    >>> overlap("abcd", "abcd", 5)
    4.0

    >>> overlap(["a", {"b", "c"}, "d"], ["a", {"b", "c"}, "d"], 2)
    2.0

    >>> overlap(["a", {"b", "c"}, "d"], ["a", {"b", "c"}, "d"], 3)
    3.0

    """
    return agreement(list1, list2, depth) * min(depth, len(list1), len(list2))
    # NOTE: comment the preceding and uncomment the following line if you want
    # to stick to the algorithm as defined by the paper
    # return raw_overlap(list1, list2, depth)[0]


def agreement(list1, list2, depth):
    """Proportion of shared values between two sorted lists at given depth.

    >>> _round(agreement("abcde", "abdcf", 1))
    1.0
    >>> _round(agreement("abcde", "abdcf", 3))
    0.667
    >>> _round(agreement("abcde", "abdcf", 4))
    1.0
    >>> _round(agreement("abcde", "abdcf", 5))
    0.8
    >>> _round(agreement([{1, 2}, 3], [1, {2, 3}], 1))
    0.667
    >>> _round(agreement([{1, 2}, 3], [1, {2, 3}], 2))
    1.0

    """
    len_intersection, len_set1, len_set2 = raw_overlap(list1, list2, depth)
    return 2 * len_intersection / (len_set1 + len_set2)


def cumulative_agreement(list1, list2, depth):
    return (agreement(list1, list2, d) for d in range(1, depth + 1))


def average_overlap(list1, list2, depth=None):
    """Calculate average overlap between ``list1`` and ``list2``.

    >>> _round(average_overlap("abcdefg", "zcavwxy", 1))
    0.0
    >>> _round(average_overlap("abcdefg", "zcavwxy", 2))
    0.0
    >>> _round(average_overlap("abcdefg", "zcavwxy", 3))
    0.222
    >>> _round(average_overlap("abcdefg", "zcavwxy", 4))
    0.292
    >>> _round(average_overlap("abcdefg", "zcavwxy", 5))
    0.313
    >>> _round(average_overlap("abcdefg", "zcavwxy", 6))
    0.317
    >>> _round(average_overlap("abcdefg", "zcavwxy", 7))
    0.312

    """
    depth = min(len(list1), len(list2)) if depth is None else depth
    return sum(cumulative_agreement(list1, list2, depth)) / depth


def rbo_at_k(list1, list2, p, depth=None):
    # ``p**d`` here instead of ``p**(d - 1)`` because enumerate starts at
    # 0
    depth = min(len(list1), len(list2)) if depth is None else depth
    d_a = enumerate(cumulative_agreement(list1, list2, depth))
    return (1 - p) * sum(p ** d * a for (d, a) in d_a)


def rbo_min(list1, list2, p, depth=None):
    """Tight lower bound on RBO.

    See equation (11) in paper.

    >>> _round(rbo_min("abcdefg", "abcdefg", .9))
    0.767
    >>> _round(rbo_min("abcdefgh", "abcdefg", .9))
    0.767

    """
    depth = min(len(list1), len(list2)) if depth is None else depth
    x_k = overlap(list1, list2, depth)
    log_term = x_k * math.log(1 - p)
    sum_term = sum(
        p ** d / d * (overlap(list1, list2, d) - x_k) for d in range(1, depth + 1)
    )
    return (1 - p) / p * (sum_term - log_term)


def rbo_res(list1, list2, p):
    """Upper bound on residual overlap beyond evaluated depth.

    See equation (30) in paper.

    NOTE: The doctests weren't verified against manual computations but seem
    plausible. In particular, for identical lists, ``rbo_min()`` and
    ``rbo_res()`` should add up to 1, which is the case.

    >>> _round(rbo_res("abcdefg", "abcdefg", .9))
    0.233
    >>> _round(rbo_res("abcdefg", "abcdefghijklmnopqrstuvwxyz", .9))
    0.239

    """
    S, L = sorted((list1, list2), key=len)
    s, l = len(S), len(L)
    x_l = overlap(list1, list2, l)
    # since overlap(...) can be fractional in the general case of ties and f
    # must be an integer --> math.ceil()
    f = int(math.ceil(l + s - x_l))
    # upper bound of range() is non-inclusive, therefore + 1 is needed
    term1 = s * sum(p ** d / d for d in range(s + 1, f + 1))
    term2 = l * sum(p ** d / d for d in range(l + 1, f + 1))
    term3 = x_l * (math.log(1 / (1 - p)) - sum(p ** d / d for d in range(1, f + 1)))
    return p ** s + p ** l - p ** f - (1 - p) / p * (term1 + term2 + term3)


def rbo_ext(list1, list2, p):
    """RBO point estimate based on extrapolating observed overlap.

    See equation (32) in paper.

    NOTE: The doctests weren't verified against manual computations but seem
    plausible.

    >>> _round(rbo_ext("abcdefg", "abcdefg", .9))
    1.0
    >>> _round(rbo_ext("abcdefg", "bacdefg", .9))
    0.9

    """
    S, L = sorted((list1, list2), key=len)
    s, l = len(S), len(L)
    x_l = overlap(list1, list2, l)
    x_s = overlap(list1, list2, s)
    # the paper says overlap(..., d) / d, but it should be replaced by
    # agreement(..., d) defined as per equation (28) so that ties are handled
    # properly (otherwise values > 1 will be returned)
    # sum1 = sum(p**d * overlap(list1, list2, d)[0] / d for d in range(1, l + 1))
    sum1 = sum(p ** d * agreement(list1, list2, d) for d in range(1, l + 1))
    sum2 = sum(p ** d * x_s * (d - s) / s / d for d in range(s + 1, l + 1))
    term1 = (1 - p) / p * (sum1 + sum2)
    term2 = p ** l * ((x_l - x_s) / l + x_s / s)
    return term1 + term2


def rbo(list1, list2, p):
    """Complete RBO analysis (lower bound, residual, point estimate).

    ``list`` arguments should be already correctly sorted iterables and each
    item should either be an atomic value or a set of values tied for that
    rank. ``p`` is the probability of looking for overlap at rank k + 1 after
    having examined rank k.

    >>> lst1 = [{"c", "a"}, "b", "d"]
    >>> lst2 = ["a", {"c", "b"}, "d"]
    >>> _round(rbo(lst1, lst2, p=.9))
    RBO(min=0.489, res=0.477, ext=0.967)

    """
    if not 0 <= p <= 1:
        raise ValueError("The ``p`` parameter must be between 0 and 1.")
    args = (list1, list2, p)
    return RBO(rbo_min(*args), rbo_res(*args), rbo_ext(*args))


def sort_dict(dct, *, ascending=False):
    """Sort keys in ``dct`` according to their corresponding values.

    Sorts in descending order by default, because the values are
    typically scores, i.e. the higher the better. Specify
    ``ascending=True`` if the values are ranks, or some sort of score
    where lower values are better.

    Ties are handled by creating sets of tied keys at the given position
    in the sorted list.

    >>> dct = dict(a=1, b=2, c=1, d=3)
    >>> list(sort_dict(dct)) == ['d', 'b', {'a', 'c'}]
    True
    >>> list(sort_dict(dct, ascending=True)) == [{'a', 'c'}, 'b', 'd']
    True

    """
    scores = []
    items = []
    # items should be unique, scores don't have to
    for item, score in dct.items():
        if not ascending:
            score *= -1
        i = bisect_left(scores, score)
        if i == len(scores):
            scores.append(score)
            items.append(item)
        elif scores[i] == score:
            existing_item = items[i]
            if isinstance(existing_item, set):
                existing_item.add(item)
            else:
                items[i] = {existing_item, item}
        else:
            scores.insert(i, score)
            items.insert(i, item)
    return items


def rbo_dict(dict1, dict2, p, *, sort_ascending=False):
    """Wrapper around ``rbo()`` for dict input.

    Each dict maps items to be sorted to the score according to which
    they should be sorted. The RBO analysis is then performed on the
    resulting sorted lists.

    The sort is descending by default, because scores are typically the
    higher the better, but this can be overridden by specifying
    ``sort_ascending=True``.

    >>> dct1 = dict(a=1, b=2, c=1, d=3)
    >>> dct2 = dict(a=1, b=2, c=2, d=3)
    >>> _round(rbo_dict(dct1, dct2, p=.9, sort_ascending=True))
    RBO(min=0.489, res=0.477, ext=0.967)

    """
    list1, list2 = (
        sort_dict(dict1, ascending=sort_ascending),
        sort_dict(dict2, ascending=sort_ascending),
    )
    return rbo(list1, list2, p)


if __name__ in ("__main__", "__console__"):
    import doctest

    doctest.testmod()


class Measure:
    def __init__(self):
        pass

    def score(self):
        pass


class TopicDiversity(Measure):
    def __init__(self, topics):
        super().__init__()
        self.topics = topics

    def score(self, topk=25):
        """
        :param topk: topk words on which the topic diversity will be computed
        :return:
        """
        if topk > len(self.topics[0]):
            raise Exception('Words in topics are less than topk')
        else:
            unique_words = set()
            for t in self.topics:
                unique_words = unique_words.union(set(t[:topk]))
            td = len(unique_words) / (topk * len(self.topics))
            return td


class Coherence(abc.ABC):
    """
    :param topics: a list of lists of the top-k words
    :param texts: (list of lists of strings) represents the corpus on which the empirical frequencies of words are computed
    """
    def __init__(self, topics, texts):
        self.topics = topics
        self.texts = texts
        self.dictionary = Dictionary(self.texts)

    @abc.abstractmethod
    def score(self):
        pass


class CoherenceNPMI(Coherence):
    def __init__(self, topics, texts):
        super().__init__(topics, texts)

    def score(self, topk=10, per_topic=False):
        """
        :param topk: how many most likely words to consider in the evaluation
        :param per_topic: if True, returns the coherence value for each topic (default: False)
        :return: NPMI coherence
        """
        if topk > len(self.topics[0]):
            raise Exception('Words in topics are less than topk')
        else:
            npmi = CoherenceModel(topics=self.topics, texts=self.texts, dictionary=self.dictionary,
                                  coherence='c_npmi', topn=topk)
            if per_topic:
                return npmi.get_coherence_per_topic()
            else:
                return npmi.get_coherence()

class CoherenceUMASS(Coherence):
    def __init__(self, topics, texts):
        super().__init__(topics, texts)

    def score(self, topk=10, per_topic=False):
        """
        :param topk: how many most likely words to consider in the evaluation
        :param per_topic: if True, returns the coherence value for each topic (default: False)
        :return: UMass coherence
        """
        if topk > len(self.topics[0]):
            raise Exception('Words in topics are less than topk')
        else:
            umass = CoherenceModel(topics=self.topics, texts=self.texts, dictionary=self.dictionary,
                                   coherence='u_mass', topn=topk)
            if per_topic:
                return umass.get_coherence_per_topic()
            else:
                return umass.get_coherence()

class CoherenceUCI(Coherence):
    def __init__(self, topics, texts):
        super().__init__(topics, texts)

    def score(self, topk=10, per_topic=False):
        """
        :param topk: how many most likely words to consider in the evaluation
        :param per_topic: if True, returns the coherence value for each topic (default: False)
        :return: UCI coherence
        """
        if topk > len(self.topics[0]):
            raise Exception('Words in topics are less than topk')
        else:
            uci = CoherenceModel(topics=self.topics, texts=self.texts, dictionary=self.dictionary,
                                 coherence='c_uci', topn=topk)
            if per_topic:
                return uci.get_coherence_per_topic()
            else:
                return uci.get_coherence()

class CoherenceCV(Coherence):
    def __init__(self, topics, texts):
        super().__init__(topics, texts)

    def score(self, topk=10, per_topic=False):
        """
        :param topk: how many most likely words to consider in the evaluation
        :param per_topic: if True, returns the coherence value for each topic (default: False)
        :return: C_V coherence
        """
        if topk > len(self.topics[0]):
            raise Exception('Words in topics are less than topk')
        else:
            cv = CoherenceModel(topics=self.topics, texts=self.texts, dictionary=self.dictionary,
                                coherence='c_v', topn=topk)
            if per_topic:
                return cv.get_coherence_per_topic()
            else:
                return cv.get_coherence()


class CoherenceWordEmbeddings(Measure):
    def __init__(self, topics, word2vec_path=None, binary=False):
        """
        :param topics: a list of lists of the top-n most likely words
        :param word2vec_path: if word2vec_file is specified, it retrieves the word embeddings file (in word2vec format) to
         compute similarities between words, otherwise 'word2vec-google-news-300' is downloaded
        :param binary: if the word2vec file is binary
        """
        super().__init__()
        self.topics = topics
        self.binary = binary
        if word2vec_path is None:
            self.wv = api.load('word2vec-google-news-300')
        else:
            self.wv = KeyedVectors.load_word2vec_format(word2vec_path, binary=binary)

    def score(self, topk=10, binary= False):
        """
        :param topk: how many most likely words to consider in the evaluation
        :return: topic coherence computed on the word embeddings similarities
        """
        if topk > len(self.topics[0]):
            raise Exception('Words in topics are less than topk')
        else:
            arrays = []
            for index, topic in enumerate(self.topics):
                if len(topic) > 0:
                    local_simi = []
                    for word1, word2 in itertools.combinations(topic[0:topk], 2):
                        if word1 in self.wv.vocab and word2 in self.wv.vocab:
                            local_simi.append(self.wv.similarity(word1, word2))
                    arrays.append(np.mean(local_simi))
            return np.mean(arrays)


class InvertedRBO(Measure):
    def __init__(self, topics):
        """
        :param topics: a list of lists of words
        """
        super().__init__()
        self.topics = topics

    def score(self, topk = 10, weight=0.9):
        """
        :param weight: p (float), default 1.0: Weight of each agreement at depth d:
        p**(d-1). When set to 1.0, there is no weight, the rbo returns to average overlap.
        :return: rank_biased_overlap over the topics
        """
        if topk > len(self.topics[0]):
            raise Exception('Words in topics are less than topk')
        else:
            collect = []
            for list1, list2 in itertools.combinations(self.topics, 2):
                rbo_val = rbo.rbo(list1[:topk], list2[:topk], p=weight)[2]
                collect.append(rbo_val)
            return 1 - np.mean(collect)


class Matches(Measure):
    def __init__(self, doc_distribution_original_language, doc_distribution_unseen_language):
        """
         :param doc_distribution_original_language: numpy array of the topical distribution of
         the documents in the original language (dim: num docs x num topics)
         :param doc_distribution_unseen_language: numpy array of the topical distribution of the
          documents in an unseen language (dim: num docs x num topics)
         """
        super().__init__()
        self.orig_lang_docs = doc_distribution_original_language
        self.unseen_lang_docs = doc_distribution_unseen_language
        if len(self.orig_lang_docs) != len(self.unseen_lang_docs):
            raise Exception('Distributions of the comparable documents must have the same length')

    def score(self):
        """
        :return: proportion of matches between the predicted topic in the original language and
        the predicted topic in the unseen language of the document distributions
        """
        matches = 0
        for d1, d2 in zip(self.orig_lang_docs, self.unseen_lang_docs):
            if np.argmax(d1) == np.argmax(d2):
                matches = matches + 1
        return matches/len(self.unseen_lang_docs)


class KLDivergence(Measure):
    def __init__(self, doc_distribution_original_language, doc_distribution_unseen_language):
        """
         :param doc_distribution_original_language: numpy array of the topical distribution of
         the documents in the original language (dim: num docs x num topics)
         :param doc_distribution_unseen_language: numpy array of the topical distribution of the
          documents in an unseen language (dim: num docs x num topics)
         """
        super().__init__()
        self.orig_lang_docs = doc_distribution_original_language
        self.unseen_lang_docs = doc_distribution_unseen_language
        if len(self.orig_lang_docs) != len(self.unseen_lang_docs):
            raise Exception('Distributions of the comparable documents must have the same length')

    def score(self):
        """
        :return: average kullback leibler divergence between the distributions
        """
        kl_mean = 0
        for d1, d2 in zip(self.orig_lang_docs, self.unseen_lang_docs):
            kl_mean = kl_mean + kl_div(d1, d2)
        return kl_mean/len(self.unseen_lang_docs)


def kl_div(a, b):
    a = np.asarray(a, dtype=np.float)
    b = np.asarray(b, dtype=np.float)
    return np.sum(np.where(a != 0, a * np.log(a / b), 0))


class CentroidDistance(Measure):
    def __init__(self, doc_distribution_original_language, doc_distribution_unseen_language, topics, word2vec_path=None,
                 binary=True, topk=10):
        """
         :param doc_distribution_original_language: numpy array of the topical distribution of the
         documents in the original language (dim: num docs x num topics)
         :param doc_distribution_unseen_language: numpy array of the topical distribution of the
         documents in an unseen language (dim: num docs x num topics)
         :param topics: a list of lists of the top-n most likely words
         :param word2vec_path: if word2vec_file is specified, it retrieves the word embeddings
         file (in word2vec format) to compute similarities between words, otherwise
         'word2vec-google-news-300' is downloaded
         :param binary: if the word2vec file is binary
         :param topk: max number of topical words
         """
        super().__init__()
        self.topics = [t[:topk] for t in topics]
        self.orig_lang_docs = doc_distribution_original_language
        self.unseen_lang_docs = doc_distribution_unseen_language
        if len(self.orig_lang_docs) != len(self.unseen_lang_docs):
            raise Exception('Distributions of the comparable documents must have the same length')

        if word2vec_path is None:
            self.wv = api.load('word2vec-google-news-300')
        else:
            self.wv = KeyedVectors.load_word2vec_format(word2vec_path, binary=binary)

    def score(self):
        """
        :return: average centroid distance between the words of the most likely topic of the
        document distributions
        """
        cd = 0
        for d1, d2 in zip(self.orig_lang_docs, self.unseen_lang_docs):
            top_words_orig = self.topics[np.argmax(d1)]
            top_words_unseen = self.topics[np.argmax(d2)]

            centroid_lang = self.get_centroid(top_words_orig)
            centroid_en = self.get_centroid(top_words_unseen)

            cd += (1 - cosine(centroid_lang, centroid_en))
        return cd/len(self.unseen_lang_docs)

    def get_centroid(self, word_list):
        vector_list = []
        for word in word_list:
            if word in self.wv.vocab:
                vector_list.append(self.wv.get_vector(word))
        vec = sum(vector_list)
        return vec / np.linalg.norm(vec)
