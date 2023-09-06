import os
import pathlib
from tensorflow.keras.utils import to_categorical
from datetime import datetime
import config
import numpy as np
import pandas as pd


class Preparator:
    def __init__(self, result, x_train, y_train, x_valid, y_valid, times_valid, path, k, cw=None, name="", info=""):
        self.original_path=path
        self.k=k
        self.result = result
        self.name = name
        self.times=times_valid
        self.path = self.create_save_dir(path, k)
        self.set_information(info)
        self.num_classes = np.unique(y_train, return_counts=True)[0].shape[0]
        self.result['Num_Classes'] = self.num_classes
        # data preparation

        self.train_x = self.train_y = self.valid_x = self.valid_y = self.valid_times = self.train_y_orig = \
            self.valid_y_orig = None

        '''
        self.train_x_acc_x = self.train_x_acc_y = self.train_x_acc_z = self.train_x_hr = self.train_x_temp = 
            self.valid_x_acc_x = self.valid_x_acc_y = self.valid_x_acc_z = self.valid_x_hr = self.valid_x_temp = None
        '''
        self.train_x_acc_x = self.train_x_acc_y = self.train_x_acc_z  = \
            self.valid_x_acc_x = self.valid_x_acc_y = self.valid_x_acc_z= None

        self.train_x_single = self.valid_x_single = None

        self.assign_data(x_train, y_train, x_valid, y_valid, times_valid)
        self.valid_time_non_seizure = self.compute_complete_seizure_times(times=times_valid, labels=y_valid)

        # dimension computing
        self.num_input_train = self.height_train = self.width_train = self.num_input_valid = self.height_valid = \
            self.width_valid = None

        self.num_input_train_acc_x = self.height_train_acc_x = self.width_train_acc_x = self.num_input_train_acc_y = \
            self.height_train_acc_y = self.width_train_acc_y = self.num_input_train_acc_z = self.height_train_acc_z = \
            self.width_train_acc_z = None

            # self.num_input_train_hr = self.height_train_hr = self.width_train_hr = \
            # self.num_input_train_temp = self.height_train_temp = self.width_train_temp = None]


        self.num_input_valid_acc_x = \
            self.height_valid_acc_x = self.width_valid_acc_x = self.num_input_valid_acc_y = self.height_valid_acc_y = \
            self.width_valid_acc_y = self.num_input_valid_acc_z = self.height_valid_acc_z = self.width_valid_acc_z = None

            # self.num_input_valid_hr = self.height_valid_hr = self.width_valid_hr = self.num_input_valid_temp = \
            # self.height_valid_temp = self.width_valid_temp = None

        self.num_input_train_single = self.height_train_single = self.width_train_single = self.num_input_valid_single \
            = self.height_valid_single = self.width_valid_single = None

        self.compute_dimensions()

        self.class_weight = self.compute_class_weights(cw)

    def create_save_dir(self, path=None, k=None):
        """
        This method creates a directory of the current time stamp. The directory is used
        for storing the model, images and so on.
        :return: path of directory
        """
        if not path:
            time_stamp = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
            path = os.path.join(config.MODEL, time_stamp, self.name)
            if k:
                path = os.path.join(path, str(k))
            os.makedirs(path)
        else:
            path = os.path.join(path, self.name, str(k))
            os.makedirs(path)
        return path

    def set_information(self, info):
        """
        Writes information to a text file. Checks if text file already exists before.
        :param info: string of information
        """
        p = str(pathlib.Path(self.path).parent)
        if not os.path.isfile(p + "/information.txt"):
            with open(p + '/information.txt', 'w') as f:
                f.write(info)

    def assign_data(self, x_train, y_train, x_valid, y_valid, times_valid):
        """
        In this method the input is reshaped and assigned
        """
        # feats
        self.train_x_single = self.assign_data_for_single_input(x_train)
        self.valid_x_single = self.assign_data_for_single_input(x_valid)

        self.train_x = x_train
        self.valid_x = x_valid

        self.train_x_acc_x = self.train_x['acc_x']  # noqa
        self.train_x_acc_x = self.train_x_acc_x.reshape(self.train_x_acc_x.shape + (1,))

        self.train_x_acc_y = self.train_x['acc_y']
        self.train_x_acc_y = self.train_x_acc_y.reshape(self.train_x_acc_y.shape + (1,))

        self.train_x_acc_z = self.train_x['acc_z']
        self.train_x_acc_z = self.train_x_acc_z.reshape(self.train_x_acc_z.shape + (1,))

        # self.train_x_hr = self.train_x['heart_rate']
        # self.train_x_hr = self.train_x_hr.reshape(self.train_x_hr.shape + (1,))
        #
        # self.train_x_temp = self.train_x['temperature']
        # self.train_x_temp = self.train_x_temp.reshape(self.train_x_temp.shape + (1,))

        self.valid_x_acc_x = self.valid_x['acc_x']  # noqa
        self.valid_x_acc_x = self.valid_x_acc_x.reshape(self.valid_x_acc_x.shape + (1,))

        self.valid_x_acc_y = self.valid_x['acc_y']
        self.valid_x_acc_y = self.valid_x_acc_y.reshape(self.valid_x_acc_y.shape + (1,))

        self.valid_x_acc_z = self.valid_x['acc_z']
        self.valid_x_acc_z = self.valid_x_acc_z.reshape(self.valid_x_acc_z.shape + (1,))

        # self.valid_x_hr = self.valid_x['heart_rate']
        # self.valid_x_hr = self.valid_x_hr.reshape(self.valid_x_hr.shape + (1,))
        #
        # self.valid_x_temp = self.valid_x['temperature']
        # self.valid_x_temp = self.valid_x_temp.reshape(self.valid_x_temp.shape + (1,))

        # labels
        self.train_y = to_categorical(y_train, self.num_classes)
        self.valid_y = to_categorical(y_valid, self.num_classes)
        self.train_y_orig = y_train
        self.valid_y_orig = y_valid
        # times
        self.valid_times = times_valid

    def assign_data_for_single_input(self, data):
        """data is dict"""
        shortest = min([a.shape[1] for key, a in data.items()])  # length of shortest ndarray
        neu_dict = dict()
        for key, val in data.items():
            new_shape = (val.shape[0], shortest, int(val.shape[1]/shortest), val.shape[2])
            neu_dict[key] = np.mean(val.reshape(new_shape), axis=2)
        # merge dict to one ndarray:
        return np.concatenate(list(neu_dict.values()), axis=2)

    def compute_dimensions(self):
        """
        Computes the dimensions of train and valid set and saves them to the vars.
        """
        self.num_input_train_single, self.height_train_single, self.width_train_single = self.train_x_single.shape[0:3]
        self.num_input_valid_single, self.height_valid_single, self.width_valid_single = self.valid_x_single.shape[0:3]

        self.num_input_train_acc_x, self.height_train_acc_x, self.width_train_acc_x = self.train_x['acc_x'].shape
        self.num_input_train_acc_y, self.height_train_acc_y, self.width_train_acc_y = self.train_x['acc_y'].shape
        self.num_input_train_acc_z, self.height_train_acc_z, self.width_train_acc_z = self.train_x['acc_z'].shape
        # self.num_input_train_hr, self.height_train_hr, self.width_train_hr = self.train_x['heart_rate'].shape
        # self.num_input_train_temp, self.height_train_temp, self.width_train_temp = self.train_x['temperature'].shape

        self.num_input_valid_acc_x, self.height_valid_acc_x, self.width_valid_acc_x = self.valid_x['acc_x'].shape
        self.num_input_valid_acc_y, self.height_valid_acc_y, self.width_valid_acc_y = self.valid_x['acc_y'].shape
        self.num_input_valid_acc_z, self.height_valid_acc_z, self.width_valid_acc_z = self.valid_x['acc_z'].shape
        # self.num_input_valid_hr, self.height_valid_hr, self.width_valid_hr = self.valid_x['heart_rate'].shape
        # self.num_input_valid_temp, self.height_valid_temp, self.width_valid_temp = self.valid_x['temperature'].shape

        # ASSIGN TO RESULT DICTIONARY!
        self.result['Shape_train_SINGLE_Batch-Rows-Width'] = self.train_x_single.shape
        self.result['Shape_train_SINGLE_Batch-Rows-Width'] = self.valid_x_single.shape

        self.result['Shape_train_ACC_X_Batch-Rows-Width'] = self.train_x['acc_x'].shape
        self.result['Shape_train_ACC_Y_Batch-Rows-Width'] = self.train_x['acc_y'].shape
        self.result['Shape_train_ACC_Z_Batch-Rows-Width'] = self.train_x['acc_z'].shape
        # self.result['Shape_train_HR_Batch-Rows-Width'] = self.train_x['heart_rate'].shape
        # self.result['Shape_train_TEMP_Batch-Rows-Width'] = self.train_x['temperature'].shape
        self.result['Shape_valid_ACC_X_Batch-Rows-Width'] = self.valid_x['acc_x'].shape
        self.result['Shape_valid_ACC_Y_Batch-Rows-Width'] = self.valid_x['acc_y'].shape
        self.result['Shape_valid_ACC_Z_Batch-Rows-Width'] = self.valid_x['acc_z'].shape
        # self.result['Shape_valid_HR_Batch-Rows-Width'] = self.valid_x['heart_rate'].shape
        # self.result['Shape_valid_TEMP_Batch-Rows-Width'] = self.valid_x['temperature'].shape

    def compute_class_weights(self, class_weight):
        """
        Returns the class weights used in model.fit for imbalanced data.
        :param class_weight: class_weight
        :return: class weight dictionary
        """
        try:
            if class_weight is None:
                self.result['class_weight'] = "None"
                return None
            else:
                self.result['class_weight'] = str(class_weight)
                return class_weight
        except:  # noqa
            self.result['class_weight'] = str(class_weight)
            return class_weight

    @staticmethod
    def compute_complete_seizure_times(times, labels):
        """
        This method computes the complete time of valid_times by sum up all the times in valid times. Only timestamps
        where label is zero are considered.
        :param labels: 1d numpy array of labels. -> preparation is needed to put them into a dict format.
        :param times: Dict with key for every patient. The value per patient is 3column np.array. One column is
        measurement-begin, one for end and one for the center of this period.
        :return: summed up timedelta of valid_times (label=0)
        """
        dict_labels = Preparator.distribute_seizures_per_patient(times, labels)
        timedelta = pd.Timedelta(seconds=0)

        for key, value in times.items():
            # remove the seizure rows in value; only non-seizure time should be summed up.
            temp_indices_to_delete = np.argwhere(dict_labels[key] == 1)
            value = np.delete(value, temp_indices_to_delete, axis=0)

            temp_row_sum = value[:, 1] - value[:, 0]  # sum of end-begin for each row
            temp_sum_nanoseconds = np.sum(temp_row_sum)  # sum up all rows in nanoseconds
            sum_timedelta = pd.Timedelta(value=temp_sum_nanoseconds, unit="ns")  # convert in timedelta format
            timedelta = timedelta + sum_timedelta
        return timedelta

    @staticmethod
    def distribute_seizures_per_patient(times, labels):
        """
        This method distributes all labels in the 1d np array labels into a dictionary with patients as keys. The keys
        and the shape is from times. At the end the result is a dictionary of np.arrays of labels. The keys are
        patients. The values are the np.arrays.
        :param times: only used for the shape and key names of the result
        :param labels: 1d np.array of concatenated labels
        :return: labels in dict of the same basic structure like times.
        """
        start = 0
        result_dict = dict()
        for key, value in times.items():
            shape = value.shape
            labels_temp = labels[start:start+shape[0]]
            result_dict[key] = labels_temp
            start = shape[0]
        return result_dict
