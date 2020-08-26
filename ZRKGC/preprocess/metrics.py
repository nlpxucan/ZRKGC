# -*- coding: utf-8 -*- 
import sys
from collections import Counter
import argparse
from nltk import ngrams
import re
import os
import nltk
import numpy as np
# from stop_words import get_stop_words
# from nltk.translate.bleu_score import corpus_bleu
# from nltk.translate.bleu_score import sentence_bleu
from nltk.translate import bleu_score as nltkbleu

def distinct(hypothesis):
    '''
    compute distinct metric
    :param hypothesis: list of str
    :return:
    '''
    unigram_counter, bigram_counter = Counter(), Counter()
    for hypo in hypothesis:
        tokens = hypo.split()
        unigram_counter.update(tokens)
        bigram_counter.update(ngrams(tokens, 2))

    distinct_1 = len(unigram_counter) / sum(unigram_counter.values()) #越大越好 不同的词组
    distinct_2 = len(bigram_counter) / sum(bigram_counter.values())
    return distinct_1, distinct_2


re_art = re.compile(r'\b(a|an|the)\b')
re_punc = re.compile(r'[!"#$%&()*+,-./:;<=>?@\[\]\\^`{|}~_\']')


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re_art.sub(' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        return re_punc.sub(' ', text)  # convert punctuation to spaces

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def _prec_recall_f1_score(pred_items, gold_items):
    """
    Compute precision, recall and f1 given a set of gold and prediction items.
    :param pred_items: iterable of predicted values
    :param gold_items: iterable of gold values
    :return: tuple (p, r, f1) for precision, recall, f1
    """
    common = Counter(gold_items) & Counter(pred_items)

    num_same = sum(common.values())

    if num_same == 0:
        return 0, 0, 0
    precision = 1.0 * num_same / len(pred_items)
    recall = 1.0 * num_same / len(gold_items)
    f1 = (2 * precision * recall) / (precision + recall)
    return precision, recall, f1


def _f1_score(guess, answers):
    """Return the max F1 score between the guess and *any* answer."""
    if guess is None or answers is None:
        return 0
    g_tokens = normalize_answer(guess).split()
    scores = [
        _prec_recall_f1_score(g_tokens, normalize_answer(a).split()) for a in answers
    ]
    return max(f1 for _,_,f1 in scores),max(pre for pre,_,_ in scores),max(rec for _,rec,_ in scores)


def f_one(hypothesis, references):
    '''
    calculate f1 metric
    :param hypothesis: list of str
    :param references: list of str
    :return:
    '''
    f1 = []
    pre = []
    rec = []
    for hyp, ref in zip(hypothesis, references):
        res = _f1_score(hyp, [ref])
        f1.append(res[0])
        pre.append(res[1])
        rec.append(res[2])
    return np.mean(f1),np.mean(pre),np.mean(rec)


def _bleu1(guess, answers):
    """Compute approximate BLEU score between guess and a set of answers."""
    if nltkbleu is None:
        # bleu library not installed, just return a default value
        return None
    # Warning: BLEU calculation *should* include proper tokenization and
    # punctuation etc. We're using the normalize_answer for everything though,
    # so we're over-estimating our BLEU scores.  Also note that NLTK's bleu is
    # going to be slower than fairseq's (which is written in C), but fairseq's
    # requires that everything be in arrays of ints (i.e. as tensors). NLTK's
    # works with strings, which is better suited for this module.
    return nltkbleu.sentence_bleu(
        [normalize_answer(a).split(" ") for a in answers],
        normalize_answer(guess).split(" "),
        weights=(1.0/1.0, ),
        smoothing_function=nltkbleu.SmoothingFunction(epsilon=1e-12).method1,
    )

def _bleu2(guess, answers):
    """Compute approximate BLEU score between guess and a set of answers."""
    if nltkbleu is None:
        # bleu library not installed, just return a default value
        return None
    # Warning: BLEU calculation *should* include proper tokenization and
    # punctuation etc. We're using the normalize_answer for everything though,
    # so we're over-estimating our BLEU scores.  Also note that NLTK's bleu is
    # going to be slower than fairseq's (which is written in C), but fairseq's
    # requires that everything be in arrays of ints (i.e. as tensors). NLTK's
    # works with strings, which is better suited for this module.
    return nltkbleu.sentence_bleu(
        [normalize_answer(a).split(" ") for a in answers],
        normalize_answer(guess).split(" "),
        weights=(1.0/2.0, 1.0/2.0),
        smoothing_function=nltkbleu.SmoothingFunction(epsilon=1e-12).method1,
    )

def _bleu3(guess, answers):
    """Compute approximate BLEU score between guess and a set of answers."""
    if nltkbleu is None:
        # bleu library not installed, just return a default value
        return None
    # Warning: BLEU calculation *should* include proper tokenization and
    # punctuation etc. We're using the normalize_answer for everything though,
    # so we're over-estimating our BLEU scores.  Also note that NLTK's bleu is
    # going to be slower than fairseq's (which is written in C), but fairseq's
    # requires that everything be in arrays of ints (i.e. as tensors). NLTK's
    # works with strings, which is better suited for this module.
    return nltkbleu.sentence_bleu(
        [normalize_answer(a).split(" ") for a in answers],
        normalize_answer(guess).split(" "),
        weights=(1.0/3.0, 1.0/3.0, 1.0/3.0),
        smoothing_function=nltkbleu.SmoothingFunction(epsilon=1e-12).method1,
    )

def _bleu4(guess, answers):
    """Compute approximate BLEU score between guess and a set of answers."""
    if nltkbleu is None:
        # bleu library not installed, just return a default value
        return None
    # Warning: BLEU calculation *should* include proper tokenization and
    # punctuation etc. We're using the normalize_answer for everything though,
    # so we're over-estimating our BLEU scores.  Also note that NLTK's bleu is
    # going to be slower than fairseq's (which is written in C), but fairseq's
    # requires that everything be in arrays of ints (i.e. as tensors). NLTK's
    # works with strings, which is better suited for this module.
    return nltkbleu.sentence_bleu(
        [normalize_answer(a).split(" ") for a in answers],
        normalize_answer(guess).split(" "),
        weights=(1.0/4.0, 1.0/4.0, 1.0/4.0, 1.0/4.0),
        smoothing_function=nltkbleu.SmoothingFunction(epsilon=1e-12).method1,
    )

def bleu(hypothesis,  references):
    bleu_scores = []
    for hyp, ref in zip(hypothesis, references):
        b1 = _bleu1(hyp, [ref])
        b2 = _bleu2(hyp, [ref])
        b3 = _bleu3(hyp, [ref])
        bleu_scores.append([b1, b2, b3])
    bleu_scores = np.mean(bleu_scores, axis=0) # [bleu1, bleu2, bleu3]
    return tuple(bleu_scores)

def bleu_corpus(hypothesis, references):
    from nltk.translate.bleu_score import corpus_bleu
    hypothesis = hypothesis.copy()
    references = references.copy()
    hypothesis = [hyp.split() for hyp in hypothesis]
    references = [[ref.split()] for ref in references]
    # hypothesis = [normalize_answer(hyp).split(" ") for hyp in hypothesis]
    # references = [[normalize_answer(ref).split(" ")] for ref in references]
    b1 = corpus_bleu(references, hypothesis, weights=(1.0/1.0,), smoothing_function=nltkbleu.SmoothingFunction(epsilon=1e-12).method1)
    b2 = corpus_bleu(references, hypothesis, weights=(1.0/2.0, 1.0/2.0), smoothing_function=nltkbleu.SmoothingFunction(epsilon=1e-12).method1)
    b3 = corpus_bleu(references, hypothesis, weights=(1.0/3.0, 1.0/3.0, 1.0/3.0), smoothing_function=nltkbleu.SmoothingFunction(epsilon=1e-12).method1)
    return (b1, b2, b3)

def bleu_metric(hypothesis, references):
    return bleu_corpus(hypothesis, references)


def knowledge_metric(responses, knowledges):
    '''
    calculate knowledge metric
    :param responses: list of str
    :param knowledges: list of list of str
    :return:
    '''
    stop_words = get_stop_words('en')
    p_scores,  r_scores, f_scores = [], [], []
    for hyp, know in zip(responses, knowledges):
        # hyp_tokens = set([w for w in hyp.split() if w not in stop_words])
        # know = ' '.join(know)
        # know_tokens = set([w for w in know.split() if w not in stop_words])
        #
        # if len(hyp_tokens & know_tokens) == 0:
        #     _p, _r, _f1 = .0, .0, .0
        # else:
        #     _p = len(hyp_tokens & know_tokens) / len(hyp_tokens)
        #     _r = len(hyp_tokens & know_tokens) / len(know_tokens)
        #     _f1 = 2 * (_p * _r) / (_p + _r)

        # hyp_tokens = list(set([w for w in hyp.split() if w not in stop_words]))
        hyp_tokens = [w for w in hyp.split() if w not in stop_words]
        know = ' '.join(know)
        know_tokens = [w for w in know.split() if w not in stop_words]
        _p, _r, _f1 = _prec_recall_f1_score(hyp_tokens, know_tokens)
        p_scores.append(_p)
        r_scores.append(_r)
        f_scores.append(_f1)

    return np.mean(r_scores), np.mean(p_scores),  np.mean(f_scores)

def knowledge_metric_new(responses, knowledges):
    '''
    calculate knowledge metric
    :param responses: list of str
    :param knowledges: list of list of str
    :return:
    '''
    # stop_words = get_stop_words('en')
    # p_scores,  r_scores, f_scores = [], [], []
    # for hyp, know in zip(responses, knowledges):
    #     hyp_tokens = set([w for w in hyp.split() if w not in stop_words])
    #     know = ' '.join(know)
    #     know_tokens = set([w for w in know.split() if w not in stop_words])
    #
    #     if len(hyp_tokens & know_tokens) == 0:
    #         _p, _r, _f1 = .0, .0, .0
    #     else:
    #         _p = len(hyp_tokens & know_tokens) / len(hyp_tokens)
    #         _r = len(hyp_tokens & know_tokens) / len(know_tokens)
    #         _f1 = 2 * (_p * _r) / (_p + _r)
    #
    #     p_scores.append(_p)
    #     r_scores.append(_r)
    #     f_scores.append(_f1)
    #
    # return np.mean(r_scores), np.mean(p_scores),  np.mean(f_scores)

    stop_words = get_stop_words('en')
    p_scores, r_scores, f_scores = [], [], []
    for hyp, know in zip(responses, knowledges):
        hyp_tokens = set([w for w in hyp.split() if w not in stop_words])
        know = ' '.join(know)
        know_tokens = set([w for w in know.split() if w not in stop_words])

        if len(hyp_tokens & know_tokens) == 0:
            _p, _r, _f1 = .0, .0, .0
        else:
            _p = len(hyp_tokens & know_tokens) / len(hyp_tokens)
            _r = len(hyp_tokens & know_tokens) / len(know_tokens)
            _f1 = 2 * (_p * _r) / (_p + _r)

        p_scores.append(_p)
        r_scores.append(_r)
        f_scores.append(_f1)

    return np.mean(r_scores), np.mean(p_scores), np.mean(f_scores)
