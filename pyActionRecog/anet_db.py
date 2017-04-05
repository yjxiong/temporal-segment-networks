from utils import *


class Instance(object):
    """
    Representing an instance of activity in the videos
    """

    def __init__(self, idx, anno, vid_id, vid_info, name_num_mapping):
        self._starting, self._ending = anno['segment'][0], anno['segment'][1]
        self._str_label = anno['label']
        self._total_duration = vid_info['duration']
        self._idx = idx
        self._vid_id = vid_id
        self._file_path = None

        if name_num_mapping:
            self._num_label = name_num_mapping[self._str_label]

    @property
    def time_span(self):
        return self._starting, self._ending

    @property
    def covering_ratio(self):
        return self._starting / float(self._total_duration), self._ending / float(self._total_duration)

    @property
    def num_label(self):
        return self._num_label

    @property
    def label(self):
        return self._str_label

    @property
    def name(self):
        return '{}_{}'.format(self._vid_id, self._idx)

    @property
    def path(self):
        if self._file_path is None:
            raise ValueError("This instance is not associated to a file on disk. Maybe the file is missing?")
        return self._file_path

    @path.setter
    def path(self, path):
        self._file_path = path


class Video(object):
    """
    This class represents one video in the activity-net db
    """
    def __init__(self, key, info, name_idx_mapping=None):
        self._id = key
        self._info_dict = info
        self._instances = [Instance(i, x, self._id, self._info_dict, name_idx_mapping)
                           for i, x in enumerate(self._info_dict['annotations'])]
        self._file_path = None

    @property
    def id(self):
        return self._id

    @property
    def url(self):
        return self._info_dict['url']

    @property
    def instances(self):
        return self._instances

    @property
    def duration(self):
        return self._info_dict['duration']

    @property
    def subset(self):
        return self._info_dict['subset']

    @property
    def instance(self):
        return self._instances

    @property
    def path(self):
        if self._file_path is None:
            raise ValueError("This video is not associated to a file on disk. Maybe the file is missing?")
        return self._file_path

    @path.setter
    def path(self, path):
        self._file_path = path


class ANetDB(object):
    """
    This class is the abstraction of the activity-net db
    """

    _CONSTRUCTOR_LOCK = object()

    def __init__(self, token):
        """
        Disabled constructor
        :param token:
        :return:
        """
        if token is not self._CONSTRUCTOR_LOCK:
            raise ValueError("Use get_db to construct an instance, do not directly use the constructor")

    @classmethod
    def get_db(cls, version="1.2"):
        """
        Build the internal representation of Activity Net databases
        We use the alphabetic order to transfer the label string to its numerical index in learning
        :param version:
        :return:
        """
        if version not in ["1.2","1.3"]:
            raise ValueError("Unsupported database version {}".format(version))

        import os
        raw_db_file = os.path.join("data/activitynet_splits",
                                   "activity_net.v{}.min.json".format(version.replace('.', '-')))

        import json
        db_data = json.load(open(raw_db_file))

        me = cls(cls._CONSTRUCTOR_LOCK)
        me.version = version
        me.prepare_data(db_data)

        return me

    def prepare_data(self, raw_db):
        self._version = raw_db['version']

        # deal with taxonomy
        self._taxonomy = raw_db['taxonomy']
        self._parse_taxonomy()

        self._database = raw_db['database']
        self._video_dict = {k: Video(k, v, self._name_idx_table) for k,v in self._database.items()}

        # split testing/training/validation set
        self._testing_dict = {k: v for k, v in self._video_dict.items() if v.subset == 'testing'}
        self._training_dict = {k: v for k, v in self._video_dict.items() if v.subset == 'training'}
        self._validation_dict = {k: v for k, v in self._video_dict.items() if v.subset == 'validation'}

        self._training_inst_dict = {i.name: i for v in self._training_dict.values() for i in v.instances}
        self._validation_inst_dict = {i.name: i for v in self._validation_dict.values() for i in v.instances}

    def get_subset_videos(self, subset_name):
        if subset_name == 'training':
            return self._training_dict.values()
        elif subset_name == 'validation':
            return self._validation_dict.values()
        elif subset_name == 'testing':
            return self._testing_dict.values()
        else:
            raise ValueError("Unknown subset {}".format(subset_name))

    def get_subset_instance(self, subset_name):
        if subset_name == 'training':
            return self._training_inst_dict.values()
        elif subset_name == 'validation':
            return self._validation_inst_dict.values()
        else:
            raise ValueError("Unknown subset {}".format(subset_name))

    def get_ordered_label_list(self):
        return [self._idx_name_table[x] for x in sorted(self._idx_name_table.keys())]

    def _parse_taxonomy(self):
        """
        This function just parse the taxonomy file
        It gives alphabetical ordered indices to the classes in competition
        :return:
        """
        name_dict = {x['nodeName']: x for x in self._taxonomy}
        parents = set()
        for x in self._taxonomy:
            parents.add(x['parentName'])

        # leaf nodes are those without any child
        leaf_nodes = [name_dict[x] for x
                      in list(set(name_dict.keys()).difference(parents))]
        sorted_lead_nodes = sorted(leaf_nodes, key=lambda l: l['nodeName'])
        self._idx_name_table = {i: e['nodeName'] for i, e in enumerate(sorted_lead_nodes)}
        self._name_idx_table = {e['nodeName']: i for i, e in enumerate(sorted_lead_nodes)}
        self._name_table = {x['nodeName']: x for x in sorted_lead_nodes}


if __name__ == '__main__':
    db = ANetDB.get_db("1.3")
