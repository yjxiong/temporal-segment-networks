import argparse
import sys
import numpy as np
sys.path.append('.')

from pyActionRecog.utils.video_funcs import default_aggregation_func
from pyActionRecog.utils.metrics import mean_class_accuracy

parser = argparse.ArgumentParser()
parser.add_argument('score_files', nargs='+', type=str)
parser.add_argument('--score_weights', nargs='+', type=float, default=None)
parser.add_argument('--crop_agg', type=str, choices=['max', 'mean'], default='mean')
args = parser.parse_args()

score_npz_files = [np.load(x) for x in args.score_files]

if args.score_weights is None:
    score_weights = [1] * len(score_npz_files)
else:
    score_weights = args.score_weights
    if len(score_weights) != len(score_npz_files):
        raise ValueError("Only {} weight specifed for a total of {} score files"
                         .format(len(score_weights), len(score_npz_files)))

score_list = [x['scores'][:, 0] for x in score_npz_files]
label_list = [x['labels'] for x in score_npz_files]

# label verification

# score_aggregation
agg_score_list = []
for score_vec in score_list:
    agg_score_vec = [default_aggregation_func(x, normalization=False, crop_agg=getattr(np, args.crop_agg)) for x in score_vec]
    agg_score_list.append(np.array(agg_score_vec))

final_scores = np.zeros_like(agg_score_list[0])
for i, agg_score in enumerate(agg_score_list):
    final_scores += agg_score * score_weights[i]

# accuracy
acc = mean_class_accuracy(final_scores, label_list[0])
print 'Final accuracy {:02f}%'.format(acc * 100)