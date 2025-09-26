import os
import warnings
from typing import List, Union

import numpy as np
import blosc2

from batchgenerators.utilities.file_and_folder_operations import join, load_pickle, load_json

class Blosc2Reader:
    def __init__(self, folder: str, identifiers: List[str] = None, labels_dict: dict = None):
        blosc2.set_nthreads(1)
        
        if identifiers is None:
            identifiers = self.get_identifiers(folder)
        identifiers.sort()
        if labels_dict is None:
            labels_dict = load_json(join(folder, os.pardir, os.pardir,'dataset.json'))
        labels_dict = labels_dict['labels']
        
        self.source_folder = folder
        self.identifiers = identifiers
        self.labels = self.get_labels(labels_dict)
        self.ignore_label = self.determine_ignore_label(labels_dict)
        self.has_ignore_label = self.ignore_label is not None
             
    def __getitem__(self, identifier):
        return self.load_case(identifier)

    def load_case(self, identifier):
        dparams = {'nthreads': 1}
        data_b2nd_file = join(self.source_folder, identifier + '.b2nd')

        mmap_kwargs = {} if os.name == "nt" else {'mmap_mode': 'r'}
        data = blosc2.open(urlpath=data_b2nd_file, mode='r', dparams=dparams, **mmap_kwargs)
        data = np.asarray(data[:])

        seg_b2nd_file = join(self.source_folder, identifier + '_seg.b2nd')
        seg = blosc2.open(urlpath=seg_b2nd_file, mode='r', dparams=dparams, **mmap_kwargs)
        seg  = np.asarray(seg[:])

        properties = load_pickle(join(self.source_folder, identifier + '.pkl'))
        return data, seg, properties

    @staticmethod
    def get_identifiers(folder: str) -> List[str]:
        case_identifiers = [i[:-5] for i in os.listdir(folder) if i.endswith(".b2nd") and not i.endswith("_seg.b2nd")]
        return case_identifiers
    
    @staticmethod
    def get_labels(label_dict: dict) -> List[int]:
        all_labels = []
        for k, r in label_dict.items():
            if k == 'ignore':
                continue
            else:
                all_labels.append(int(r))
        all_labels = list(np.unique(all_labels))
        all_labels.sort()
        return all_labels
    
    @staticmethod
    def determine_ignore_label(label_dict: dict) -> Union[None, int]:
        ignore_label = label_dict.get('ignore')
        if ignore_label is not None:
            assert isinstance(ignore_label, int), f'Ignore label has to be an integer. It cannot be a region ' \
                                                    f'(list/tuple). Got {type(ignore_label)}.'
        return ignore_label