from typing import List

import numpy as np

from batchgenerators.utilities.file_and_folder_operations import join, isfile, load_json, save_json
from sklearn.model_selection import KFold

from b2nd_reader import Blosc2Reader

class DataSplit:
    def __init__(self, fold: int, folder: str, folder_base: str):
        self.source_folder = folder
        self.source_folder_base = folder_base
        self.fold = fold
        
    def generate_crossval_split(self, train_identifiers: List[str], seed=12345, n_splits=5) -> List[dict[str, List[str]]]:
        splits = []
        kfold = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        for i, (train_idx, test_idx) in enumerate(kfold.split(train_identifiers)):
            train_keys = np.array(train_identifiers)[train_idx]
            test_keys = np.array(train_identifiers)[test_idx]
            splits.append({})
            splits[-1]['train'] = list(train_keys)
            splits[-1]['val'] = list(test_keys)
        return splits
         
    def do_split(self):
        if self.fold == "all":
            # if fold==all then we use all images for training and validation
            case_identifiers = Blosc2Reader.get_identifiers(self.source_folder)
            tr_keys = case_identifiers
            val_keys = tr_keys
        else:
            splits_file = join(self.source_folder_base, "splits.json")
            dataset = Blosc2Reader(self.source_folder, identifiers=None)
            
            # if the split file does not exist we need to create it
            if not isfile(splits_file):
                print("Creating new 5-fold cross-validation split...")
                all_keys_sorted = list(np.sort(list(dataset.identifiers)))
                splits = self.generate_crossval_split(all_keys_sorted, seed=12345, n_splits=5)
                save_json(splits, splits_file)
            else:
                print("Using splits from existing split file:", splits_file)
                splits = load_json(splits_file)
                print(f"The split file contains {len(splits)} splits.")

            print("Desired fold for training: %d" % self.fold)
            if self.fold < len(splits):
                tr_keys = splits[self.fold]['train']
                val_keys = splits[self.fold]['val']
                print("This split has %d training and %d validation cases."
                                       % (len(tr_keys), len(val_keys)))
            else:
                print("INFO: You requested fold %d for training but splits "
                                       "contain only %d folds. I am now creating a "
                                       "random (but seeded) 80:20 split!" % (self.fold, len(splits)))
                # if we request a fold that is not in the split file, create a random 80:20 split
                rnd = np.random.RandomState(seed=12345 + self.fold)
                keys = np.sort(list(dataset.identifiers))
                idx_tr = rnd.choice(len(keys), int(len(keys) * 0.8), replace=False)
                idx_val = [i for i in range(len(keys)) if i not in idx_tr]
                tr_keys = [keys[i] for i in idx_tr]
                val_keys = [keys[i] for i in idx_val]
                self.print_to_log_file("This random 80:20 split has %d training and %d validation cases."
                                       % (len(tr_keys), len(val_keys)))
            if any([i in val_keys for i in tr_keys]):
                print('WARNING: Some validation cases are also in the training set. Please check the '
                                       'splits.json or ignore if this is intentional.')
        return tr_keys, val_keys