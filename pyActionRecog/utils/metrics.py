"""
This module provides some utils for calculating metrics
"""
import numpy as np
from sklearn.metrics import average_precision_score, confusion_matrix


def softmax(raw_score, T=1):
    exp_s = np.exp((raw_score - raw_score.max(axis=-1)[..., None])*T)
    sum_s = exp_s.sum(axis=-1)
    return exp_s / sum_s[..., None]


def top_k_acc(lb_set, scores, k=3):
    idx = np.argsort(scores)[-k:]
    return len(lb_set.intersection(idx)), len(lb_set)


def top_k_hit(lb_set, scores, k=3):
    idx = np.argsort(scores)[-k:]
    return len(lb_set.intersection(idx)) > 0, 1


def top_3_accuracy(score_dict, video_list):
    return top_k_accuracy(score_dict, video_list, 3)


def top_k_accuracy(score_dict, video_list, k):
    video_labels = [set([i.num_label for i in v.instances]) for v in video_list]

    video_top_k_acc = np.array(
        [top_k_hit(lb, score_dict[v.id], k=k) for v, lb in zip(video_list, video_labels)
         if v.id in score_dict])

    tmp = video_top_k_acc.sum(axis=0).astype(float)
    top_k_acc = tmp[0] / tmp[1]

    return top_k_acc


def video_mean_ap(score_dict, video_list):
    avail_video_labels = [set([i.num_label for i in v.instances]) for v in video_list if
                          v.id in score_dict]
    pred_array = np.array([score_dict[v.id] for v in video_list if v.id in score_dict])
    gt_array = np.zeros(pred_array.shape)

    for i in xrange(pred_array.shape[0]):
        gt_array[i, list(avail_video_labels[i])] = 1
    mean_ap = average_precision_score(gt_array, pred_array, average='macro')
    return mean_ap


def mean_class_accuracy(scores, labels):
    pred = np.argmax(scores, axis=1)
    cf = confusion_matrix(labels, pred).astype(float)

    cls_cnt = cf.sum(axis=1)
    cls_hit = np.diag(cf)

    return np.mean(cls_hit/cls_cnt)