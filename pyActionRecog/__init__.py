from benchmark_db import *


split_parsers = dict()
split_parsers['ucf101'] = parse_ucf_splits

def parse_split_file(dataset):
    sp = split_parsers[dataset]
    return sp()

