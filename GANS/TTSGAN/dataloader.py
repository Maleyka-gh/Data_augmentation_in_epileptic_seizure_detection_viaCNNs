import os
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import random
import seizures as seizure_collection


# data augmentation for later use
# import tsaug

class Mond_data_loader(Dataset):
    def __init__(self,
                 is_normalize=False,  # If true then channel wise normalize to mean=0 std=1
                 data_path=None,
                 data_mode='Train'
                 ):

        self.data_path = data_path
        self.data_mode = data_mode
        self.is_normalize = is_normalize

        train, valid, test = self.create_sets()
        train = train + valid

        train_feat, train_label, _ = self.load_all_numpy_data(train, self.data_path)
        test_feat, test_label, _ = self.load_all_numpy_data(test, self.data_path)

        train_list = []
        test_list = []

        for feature, values in train_feat.items():
            train_list.append(values)
        for feature, values in test_feat.items():
            test_list.append(values)

        train_data = np.concatenate(train_list, axis=2)
        test_data = np.concatenate(test_list, axis=2)

        train_data = np.transpose(train_data, (0, 2, 1))
        self.train_data = train_data.reshape(train_data.shape[0], train_data.shape[1], 1, train_data.shape[2])

        test_data = np.transpose(test_data, (0, 2, 1))
        self.test_data = test_data.reshape(test_data.shape[0], test_data.shape[1], 1, test_data.shape[2])

        if self.is_normalize:
            self.train_data = self.normalization(self.train_data)
            self.test_data = self.normalization(self.test_data)

        self.train_label = np.reshape(np.array(train_label), (len(train_label), 1))
        self.test_label = np.reshape(np.array(test_label), (len(test_label), 1))

        print(f'train_data shape is {self.train_data.shape}, test_data shape is {self.test_data.shape}')
        print(f'train label shape is {self.train_label.shape}, test data shape is {self.test_label.shape}')

    def __call__(self, *args, **kwargs):
        return (self.train_data, self.train_label), (self.test_data, self.test_label)

    def _normalize(self, epoch):
        """ A helper method for the normalization method.
            Returns
                result: a normalized epoch
        """
        e = 1e-10
        result = (epoch - epoch.mean(axis=0)) / ((np.sqrt(epoch.var(axis=0))) + e)
        return result

    def _min_max_normalize(self, epoch):

        result = (epoch - min(epoch)) / (max(epoch) - min(epoch))
        return result

    def normalization(self, epochs):
        """ Normalizes each epoch e s.t mean(e) = 0 and var(e) = 1
            Args:
                epochs - Numpy structure of epochs
            Returns:
                epochs_n - mne data structure of normalized epochs (mean=0, var=1)
        """
        for i in range(epochs.shape[0]):
            for j in range(epochs.shape[1]):
                epochs[i, j, 0, :] = self._normalize(epochs[i, j, 0, :])
        #                 epochs[i,j,0,:] = self._min_max_normalize(epochs[i,j,0,:])

        return epochs

    def load_all_numpy_data(self, pat_set, path):
        """
        This method loads all computed numpy arrays. It creates two np arrays (features and labels) and fills them with the
        concatenated numpy arrays.
        :param pat_set: List of required patients
        :return: 3 np arrays (feats, labels, timestamps)
        """
        features = dict()
        labels = None
        times = dict()
        first_iter = True  # label, feature
        for file in os.listdir(path):
            if file[0:6] in pat_set:
                if "times" in file:
                    curr_times = np.array(pd.read_pickle(os.path.join(path, file)))
                    times[file[0:6]] = curr_times
                    continue
                curr_np = np.load(os.path.join(path, file))
                if "label" in file:
                    if first_iter:
                        labels = curr_np
                        first_iter = False
                    else:
                        labels = np.concatenate([labels, curr_np])
                else:
                    name = file[7:-10]
                    if name not in features:
                        features[name] = curr_np
                    else:
                        features[name] = np.concatenate([features[name], curr_np])
        return features, labels, times

    def create_sets(self, min_train=0.7):
        """
        Creates train, valid and test set for learning. min_train of all 110 seizures are in the train set.
        Test set is hard coded and contains 15 seizures of 5 patients. train and valid set are filled randomly per patient.
        :param min_train: minimum of percentage for num of seizures in train set.
        :return: 3 lists of patient names.
        """
        all_seizures = seizure_collection.get_seizures_to_use()  # 110 seizures in 46 patients
        num_of_seizures = {pat: len(seizures) for pat, seizures in all_seizures.items()}
        total_seizures = sum(num_of_seizures.values())

        hard_coded_test = self.hard_coded_test_set()
        train_set = {}
        num_of_seizures = {key: value for key, value in num_of_seizures.items() if key not in hard_coded_test}
        while True:  # pythonic do-while loop
            random_key = random.choice(list(num_of_seizures.keys()))
            train_set[random_key] = num_of_seizures[random_key]
            num_of_seizures.pop(random_key)
            if sum(train_set.values()) >= min_train * total_seizures:  # break condition
                return list(train_set.keys()), list(num_of_seizures.keys()), list(hard_coded_test.keys())

    def __len__(self):

        if self.data_mode == 'Train':
            return len(self.train_label)
        else:
            return len(self.test_label)

    def __getitem__(self, idx):
        if self.data_mode == 'Train':
            return self.train_data[idx], self.train_label[idx]
        else:
            return self.test_data[idx], self.test_label[idx]

    @staticmethod
    def hard_coded_test_set():
        """
        Hard coded test set as a dictionary. Key: Patient name. Value: Num of seizures of patient.
        :return: dictionary of test set patients.
        """
        return {'BN_011': 3, 'BN_012': 1, 'BN_031': 1, 'BN_103': 2, 'BN_107': 1, 'BN_160': 4, 'BN_166': 1, 'BN_167': 9,
                'BN_017': 1}