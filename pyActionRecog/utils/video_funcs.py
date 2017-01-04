"""
This module provides our implementation of different functions to do video-level classification and stream fusion
"""
import numpy as np
from metrics import softmax


def default_aggregation_func(score_arr, normalization=True, crop_agg=None):
    """
    This is the default function for make video-level prediction
    :param score_arr: a 3-dim array with (frame, crop, class) layout
    :return:
    """
    crop_agg = np.mean if crop_agg is None else crop_agg
    if normalization:
        return softmax(crop_agg(score_arr, axis=1).mean(axis=0))
    else:
        return crop_agg(score_arr, axis=1).mean(axis=0)


def top_k_aggregation_func(score_arr, k, normalization=True, crop_agg=None):
    crop_agg = np.mean if crop_agg is None else crop_agg
    if normalization:
        return softmax(np.sort(crop_agg(score_arr, axis=1), axis=0)[-k:, :].mean(axis=0))
    else:
        return np.sort(crop_agg(score_arr, axis=1), axis=0)[-k:, :].mean(axis=0)


def sliding_window_aggregation_func(score, spans=[1, 2, 4, 8, 16], overlap=0.2, norm=True, fps=1):
    """
    This is the aggregation function used for ActivityNet Challenge 2016
    :param score:
    :param spans:
    :param overlap:
    :param norm:
    :param fps:
    :return:
    """
    frm_max = score.max(axis=1)
    slide_score = []

    def top_k_pool(scores, k):
        return np.sort(scores, axis=0)[-k:, :].mean(axis=0)

    for t_span in spans:
        span = t_span * fps
        step = int(np.ceil(span * (1-overlap)))
        local_agg = [frm_max[i: i+span].max(axis=0) for i in xrange(0, frm_max.shape[0], step)]
        k = max(15, len(local_agg)/4)
        slide_score.append(top_k_pool(np.array(local_agg), k))

    out_score = np.mean(slide_score, axis=0)

    if norm:
        return softmax(out_score)
    else:
        return out_score


def default_fusion_func(major_score, other_scores, fusion_weights, norm=True):
    assert len(other_scores) == len(fusion_weights)
    out_score = major_score
    for s, w in zip(other_scores, fusion_weights):
        out_score += s * w

    if norm:
        return softmax(out_score)
    else:
        return out_score
