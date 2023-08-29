#Library Import
import os
import shutil
import pandas as pd
import numpy as np
import pathlib
import json
from datetime import datetime


#Files Import
import config
from collections import defaultdict
import PreProcessingClasses.seizures as seizure_collection



class Raw_data_preprocesor:

    def __init__(self , raw_data_path = config.RAW_DATA_PATH , # Raw Data PAth
                 feat_to_use_list = config.feat_to_use, # feat to use list
                 k = config.k, # allowed percent of missing values in window
                 win_len = config.window, # window Length
                 label_additional_time = config.label_additional_time, # additional Time for label
                 overlap=config.overlap,  #Overlap  0% or 50%
                 save=False,  # if True, will save the dataframes and learn df ,else will save  only numpy step
                 raw_df_save_path=config.RAW_DF_FOLDER,#saving path for raw_df_sub_dirs
                 learn_df_save_path=config.LEARN_DF,#saving path for dataframes_learm
                 ):

        #Global Variables
        self.raw_data_path = raw_data_path
        self.feat_to_use_list = feat_to_use_list
        self.k = k
        self.win_len = win_len #use as an integer
        self.overlap = overlap  # Overlap

        #Window Settings
        self.window_length = pd.Timedelta(seconds=win_len) #use as date-time format
        self.label_add_time = pd.Timedelta(seconds=label_additional_time) #additional label time in datetime format
        self.label_offset = self.window_length + self.label_add_time # label offset

        #Saving Configurations
        self.save_flag = save   #save flag for whether to save dataframes_learn/learnDf or not
        self.raw_df_save_path = raw_df_save_path
        self.learn_df_save_path = learn_df_save_path

        saving_numpy_path = self.create_numpy_path() # creating numpy path  where we save numpy arrays
        self.save_path_numpy = saving_numpy_path
        if saving_numpy_path == -1:
            print("Program Stoped ")
        else:
            self.run_preprocessor(saving_numpy_path) #run all the preprocessing patient by patient
            print('Saving Information')
            self.save_information()
            print('Finished')

    def save_information(self):
        info_list = list()
        info_list.append('The K value is : ' + str(self.k))
        info_list.append('The Overlap value is : ' + str(self.overlap))
        info_list.append('The Window length is : ' + str(self.window_length))
        info_list.append('The label additional time is : ' + str(self.label_add_time))
        s, n = self.count_windows(self.save_path_numpy)
        info_list.append('The total seizure windows : ' + s)
        info_list.append('The total seizure windows : ' + n)


        with open(self.save_path_numpy + '/' + 'readme.txt', 'w') as f:
            f.write('\n'.join(info_list))

    def count_windows(self,path):
        total_seiz_window = 0
        total_non_seiz_window = 0
        for files in sorted(os.listdir(path)):
            if 'label' in files:
                l = np.load(os.path.join(path,files))
                total_seiz_window = total_seiz_window + sum(l)
                total_non_seiz_window = total_non_seiz_window + len(l) - total_seiz_window
        return str(total_seiz_window), str(total_non_seiz_window)


    def run_preprocessor(self,saving_numpy_path):
        list_of_pats = sorted(self.create_list_of_pats()) # creating list of patient
        for patient in list_of_pats:
            pat_dict = self.Create_raw_df_sub_dirs(self.raw_data_path,patient)
            if pat_dict == -1:   # if any data , feat is missing or any problem occurs in running code , then don't move to next step, but move to next patient
                continue
            else:
                learn_data = self.LEARN_DF(pat_dict)
                if self.save_flag:
                    print("Saving DataFrames for Patient : ", patient)
                    self.save_raw_df_folder(pat_dict,self.raw_df_save_path)
                    self.save_learn_df(learn_data,self.learn_df_save_path)
                if True:
                    print("Computing Numpy Data for Patient : ", patient)
                    self.compute_numpy_data(learn_data,saving_numpy_path)  # last step is saved under this path : Numpy_learn_win_len_xOverlap
        return True

    # Last preprocessing step / creating numpy arrays
    def create_numpy_path(self):
        name = 'Numpy_Learn_' + str(self.win_len) + 's' + str(self.overlap) +'Overlap'
        if name in os.listdir(config.PreProcessResultPath):
            warning = self.generate_warning(name)

            if warning: # if response == 'y': in warning function , it continues deleting already existing folder and running the program to create a new folder
                shutil.rmtree(os.path.join(config.PreProcessResultPath, name))
                os.mkdir(os.path.join(config.PreProcessResultPath, name))
                return os.path.join(config.PreProcessResultPath, name)
            else:
                return -1
        else:
            os.mkdir(os.path.join(config.PreProcessResultPath, name))
            return os.path.join(config.PreProcessResultPath, name)



    def generate_warning(self,name):
        print('The Following Folder Already Exist \t ' , name )
        print('\n Rename previous one if you want to keep it, otherwise  it will be deleted')
        response = input('Do You want to continue running the program [y/n]')
        if response == 'y':
            return True
        return False


    def compute_numpy_data(self,learn_data,saving_numpy_path):
        for pat , features in learn_data.items():
            if pat in seizure_collection.get_seizures_to_use().keys():
                df = self.merge_to_one(features,axis=1)
                feats, labels, timestamps = self.compute_df_snippets(df,pat)
                if feats and labels.size != 0 and not timestamps.empty:
                    for key, val in feats.items():
                        np.save(os.path.join(saving_numpy_path, pat + "_" + key + "_feats"), val)
                    np.save(os.path.join(saving_numpy_path, pat + "_label"), labels)
                    timestamps.to_pickle(os.path.join(saving_numpy_path, pat + "_times"))



    def compute_next_start(self,time, indices):
        """
        Find the next date time (so its either =time or later than time) which is really in indices.
        :param time: starting time
        :param indices: list of time points
        :return: returns the next fitting time inside indices.
        """
        try:
            time = indices[np.searchsorted(indices, time)]
        except IndexError:
            a = indices[-1]
            return a
        return time

    def set_label_for_ts(self,ts, pat):
        """
        This methods returns the label for a time slot.
        :param ts: time series (with start and end of an interval)
        :param pat: name of patient
        :param window_settings: offset is stored in here
        :return: label (0 or 1)
        """
        label_list = seizure_collection.get_seizures_to_use()[pat]
        for label in label_list:
            begin = label['start'] - self.label_add_time
            end = label['end'] + self.label_add_time
            if (begin < ts.loc[0, 'begin'] < end) or (begin < ts.loc[0, 'end'] < end):
                return 1
        return 0

    def put_data_in_np_format(self,data, pat):
        """
        This methods puts the data in the required np - format. First a dataframe with all features in data is created.
        After that the timestamps for that window are determined (start, end, center). The label is determined and the data
        is converted in a numpy array and stored in a dictionary with key for every feature.
        :param data: List of dataframes. (one for each feature)
        :param pat: name of patient (for receiving correct label)
        :param window_settings: because of offset for labels
        :return: dict of feats, timestamp for this measurement, labels
        """
        df_complete = pd.concat(data, axis=1)
        timestamp = pd.DataFrame(data=[[df_complete.index[0], df_complete.index[-1], df_complete.index[0] +
                                        (df_complete.index[-1] - df_complete.index[0]) / 2]], columns=['begin', 'end',
                                                                                                       'center'])
        label = self.set_label_for_ts(timestamp, pat)
        result_data = dict()
        for df in data:
            curr_df = df.dropna()
            result = curr_df.to_numpy()
            result_data[df.columns[0]] = result
        return result_data, timestamp, label

    def correct_size_of_df(self,df, target_value):
        """
        sometimes interpolated Dataframes are too small or too large. Here the last entries are either removed or doubled,
        so that we receive the expected size (target_value)
        :param df: interpolated Dataframe
        :param target_value: expected num of entries in df
        :return: Boolean if the result is okay or not, and adjusted df.
        """
        # we need to cut off the dataframe tail
        while df.shape[0] > target_value:
            df = df.iloc[:-1]
        # we have to append the last value again. but not too often!
        i = 0
        while df.shape[0] < target_value:
            new_data = df.values[-1]
            new_index = (df.index[-1] - df.index[-2]) + df.index[-1]
            temp_df = pd.DataFrame(data=new_data, index=[new_index], columns=df.columns.to_list())
            df = df.append(temp_df)
            i = i + 1
            if i > 5:
                return False, df
        return True, df

    def interpolate(self,df, freq, expected_values, method="linear"):
        """
        This method interpolates missing values. First the dataframe is resampled, so that the missing timestamps are
        created. After that the values for these new timestamps are interpolated and pasted. Some errors in the size of the
        new dataframe are still possible: The correct size is created by cropping of the last value or duplicate the last
        value of the dataframe.
        :param df: Dataframe with the data to interpolate
        :param freq: Frequency; so we can compute the num of missing values
        :param expected_values: Num of samples we expect after interpolation
        :param method: Method of interpolation
        :return: Interpolated DataFrame.
        """
        df = pd.DataFrame(df)
        df = df.resample(rule=pd.Timedelta(seconds=1 / freq)).mean()
        # todo think about handling of a max value of nans one behind one another. Limit in next call is possible.
        df = df.interpolate(method=method)  # , limit=5)
        is_ok, df = self.correct_size_of_df(df, expected_values)
        return is_ok, df

    def is_num_values_ok(self,real_num, target_num, k=0.15):
        """
        Checks if the available number/real_num is okay: So if the num is not smaller than
        k=0.15 (15%) of target_num.
        :param real_num: A value which has to be checked
        :param target_num: The comparison value
        :param k: Puffer around target_num
        :return: boolean if real num is in the area below target_num
        """
        threshold = target_num - (k * target_num)
        if real_num >= threshold:
            return True
        return False


    def frequency_of(self,feat):
        """
        Returns frequency of a feature based on its name.
        :param feat: name of integer
        :return: int freq
        """
        if "acc" in feat:
            return 50
        else:
            return 1

    def get_freq_and_num_of(self,feat, window_length):
        """
        Computes the frequency of feat and the expected num of values considering the window_length and frequency.
        :param feat: Name of feature
        :param window_length: Length of window as pd.Timedelta
        :return: frequency and expected values.
        """
        freq = self.frequency_of(feat)
        expected_values = window_length.seconds * freq
        return freq, expected_values

    def compute_df_snippets(self,df, pat):
        """
        Computes dataframe snippets with length like stored in window_settings. This method iterates through df; for every
        column a truncated df is created; and checked if the num of values in the df are enough. If so, its stored. If not,
        the next section will be truncated. The num of entries in a dataframe column depends of its frequency.
        :param df: dataframe
        :param window_settings: here the window length is stored.
        :param pat: Name of patient
        :return: feats (np.array), labels (np.array), timestamps (pd.Df)
        """
        if self.overlap==0:
            normal_step = self.window_length  # used if the truncate is ok; so the next section is window_length far
        else:
            normal_step = self.window_length / 2   # Overlap of 50%
        small_step = pd.Timedelta(seconds=1)  # used if the truncate was not good; so we try 2 seconds later again.
        current_start = df.index[0]  # initial
        # results:
        elements = defaultdict(list)
        class_labels = list()
        times = list()
        while True:  # pythonic do-while
            # print(str(current_start) + "\t\t" + str(df.index[-1]))
            data = list()
            trunc_df = df.truncate(before=current_start, after=current_start + self.window_length)
            next_step = normal_step  # for computation of the next section
            for column in trunc_df:
                freq, expected_num = self.get_freq_and_num_of(column, self.window_length)
                curr_ser = trunc_df[column]

                # check if the num of values are enough
                if not self.is_num_values_ok(curr_ser.dropna().shape[0], expected_num,self.k):
                    data = list()  # empty list if one feature is not ok
                    next_step = small_step  # the next step is smaller, because the truncate was not successful
                    break  # this truncate does not contain enough values; so we don't need the other columns to check

                is_ok, curr_ser = self.interpolate(curr_ser.dropna(), freq, expected_num)  # interpolate missing values

                # too many values interpolated
                if not is_ok:
                    data = list()
                    next_step = small_step
                    break

                data.append(curr_ser)

            if data:
                data, time, label = self.put_data_in_np_format(data, pat)
                class_labels.append(label)
                times.append(time)
                for k, v in data.items():
                    elements[k].append(v)

            current_start = self.compute_next_start(current_start + next_step, df.index)  # compute next start point

            if current_start == df.index[-1]:  # if next start point is the last entry , return the result
                for k, v in elements.items():
                    elements[k] = np.stack(v, axis=0)  # noqa
                class_labels = np.array(class_labels)
                if times:
                    times = pd.concat(times, ignore_index=True)
                else:
                    times = pd.DataFrame(times)
                return elements, class_labels, times

    # dataframes_learn / LEARN_DF
    def save_learn_df(self,learn_data,path):
        for pat,feats in learn_data.items():
            os.mkdir(path + "/" + pat)
            for feat,dataframe in feats.items():
                dataframe.to_pickle(path + "/" + pat + "/" + feat)



    def mean_values_of_duplicate_indices(self, dict_of_df):
        """Compute the mean of all values with duplicated indices."""
        for key, value in dict_of_df.items():
            if any(value.index.duplicated(keep=False)):
                dict_of_df[key] = value.groupby(value.index).mean()
        return dict_of_df

    def merge_to_one(self,dict_of_df, axis):
        """Receives a dictionary with x dataframes inside. These dataframes are merged and sorted by index.
        If there are  a dataframe has duplicated indices, the mean of their values will be computed and used."""
        dict_of_df = self.mean_values_of_duplicate_indices(dict_of_df)
        df = pd.concat(dict_of_df.values(), ignore_index=False, axis=axis)
        df = df.sort_index()
        df = df.astype(float)
        return df

    def is_epitect(self,device):
        """
        Checks is device is in-ear-sensor
        :param device: device name
        :return: boolean
        """
        return device == "epi_o2" or device == "epi_green"

    def is_feat_used(self,feat):
        """
        Checks if the feat is a feat to be used
        :param feat: name of feature
        :return: boolean
        """
        feat_to_use = self.feat_to_use_list
        if feat in feat_to_use:
            return True
        else:
            return False

    def LEARN_DF(self,pat_dict):
        new_dict = dict()
        for pat, dfrms in pat_dict.items(): #this pat dictionary contains pat name, inside there are three dictionaries: dfs for all obs,meta and events
            new_dict[pat] = dict()     # empty dic with pat names as a key in order to save features later
            for file,value in dfrms.items(): # if file is meta, save value in meta, if it is event, move to other dataframes
                if file == "meta":
                    meta = value
                if file == "events":
                    continue
                if file == 'dfs':
                    for feat,dictionary in value.items(): # value.items() contains all dfs for observations like in raw_dataframes_sub_dirs
                        if self.is_feat_used(feat):
                            new_dict[pat][feat] = dict() #create another feature dic inside new_dict[pat]
                            for num,df in dictionary.items():
                                if self.is_feat_used(feat) and self.is_epitect(meta.at[num, 'device_id']): #we filter features which don't exist in feat_to_use list
                                    new_dict[pat][feat][num] = df  #saves only necessary features with obs.num as df (feats in feat_to_use list, this list  is in config.py)
                            new_dict[pat][feat] = self.merge_to_one(new_dict[pat][feat],axis=0)  # calculate mean for duplicates and concatenate all obs per particular feature
        return new_dict # same as dataframes_learn

    # raw_dataframes_sub_dirs / RAW_DF_FOLDER
    def save_raw_df_folder(self, data, path):

        for pat_folder, dfrms in data.items():
            os.mkdir(path + '/' + pat_folder)
            for file, value in dfrms.items():
                if file == 'meta' or file =='events':
                    value.to_pickle(path + '/' + pat_folder + '/' + file )
                else:
                    for feat, dat in value.items():
                        for num, df in dat.items():
                            df.to_pickle(path + '/' + pat_folder + '/' + num + '_' + feat)

    def create_list_of_pats(self):
        """ This methods creates a list of all available patients by iterating through MOND_DATA//Data"""

        l_temp = list()
        for filename in os.listdir(self.raw_data_path):
            if filename[0:2] == "BN":
                l_temp.append(filename)
        return l_temp


    def Create_raw_df_sub_dirs(self,path,pat_id):  # Raw_DF_FOLDER /dataframes_learn_sub_dirs
        pat_dict = dict()
        for root, dirs, files in os.walk(path):
            patient_id = 'unknown'

            try:
                if self.is_file_for_pat(root, files, path):  # if we have files for a patient!
                    patient_id = self.extract_patient_id(path, root)
                    if patient_id == pat_id:
                        data_of_patient = self.create_dict_for_patient()
                        print(patient_id)
                        pat_dict[patient_id] = dict()
                        # first only read meta-data:
                        data_of_patient = self.compute_and_integrate_meta_data(data_of_patient, files, root)
                        # now read all raw feature data:
                        data_of_patient = self.compute_and_integrate_feat_data(data_of_patient, files, root)
                        pat_dict[patient_id]['meta'],pat_dict[patient_id]['events'],pat_dict[patient_id]['dfs']  \
                            = self.split_meta_events_raw_data(data_of_patient)
                        for feat in self.feat_to_use_list:
                            if pat_dict[patient_id]['dfs'][feat]:
                                continue
                            else:
                                return -1
            except Exception:
                print('PROBLEM: ' + str(patient_id))
                del pat_dict
                return -1
        return pat_dict

    def is_file_for_pat(self, root, files, path):

        """Checks if the current root and its' files contain the data of a patient."""
        if root == path + "/DataFrames":
            return False  # The folder DataFrames only contains computed DataFrames
        if files:
            for f in files:
                _, file_extension = os.path.splitext(f)
                if file_extension == ".csv" or file_extension == ".json":
                    return True
            return False
        else:
            return False

    def extract_patient_id(self , path, root):
        """This method extracts the patient id '_BN_xxx' from path and root."""
        path_without_root = root[len(path) + 1:]
        path_without_root = pathlib.Path(path_without_root)
        return path_without_root.parts[0]

    def create_dict_for_patient(self):

        """This is the structure of the dictionary every patient gets.
        It contains all possible data from the set. But this does not mean that for every patient all this data is
        available."""

        return {"meta": dict(),
                "acc_x": dict(),
                "acc_y": dict(),
                "acc_z": dict(),
                "heart_rate": dict(),
                "ppg_ir": dict(),
                "ppg_red": dict(),
                "ppg_green": dict(),
                "quality": dict(),
                "rr_int": dict(),
                "temperature": dict(),
                "spo2": dict(),
                "ecg": dict()}

    def switch_events_in_datetime_format(self,event_list):
        """This method puts the beginning and ending of a list of events in the correct datetime format."""
        for i in range(len(event_list)):
            event_list[i]['event_start'] = datetime.strptime(event_list[i]['event_start'], '%Y-%m-%d %H:%M:%S')
            event_list[i]['event_end'] = datetime.strptime(event_list[i]['event_end'], '%Y-%m-%d %H:%M:%S')
        return event_list

    def read_time_stamps(self, data):

        """This method reads timestamp from data."""
        if data is None:
            return None
        try:
            return datetime.strptime(data, '%Y-%m-%d %H:%M:%S')
        except ValueError:  # string does not match the time-format
            if data != 'None':
                print(data)
            return None

    def read_meta(self,json_file):
        """Read the requested meta-data from the passed parameter json_file."""

        with open(json_file) as json_file:
            data = json.load(json_file)
            meta_dict = {'person_id': data['meta']['db']['person_id'],
                         'person_label': data['meta']['db']['person_label'],
                         'device_id': data['meta']['db']['device_model_number'],
                         'events': self.switch_events_in_datetime_format(data['meta']['events']),
                         'date_time_start': self.read_time_stamps(data['meta']['db']['date_time_start']),
                         'date_time_end': self.read_time_stamps(data['meta']['db']['date_time_end'])
                         }

        return meta_dict

    def extract_num_and_name(self,string):

        """Extract letters and numbers from a string. 001234abc!d becomes 001234 abc!d. _341jkl5432 becomes 3415432 _jkl.
        Additional: Preprocessing customized for the data in this project (Removal of leading underscore '_',
        Removal of file suffix. But if the name is 'spo2' the 2 will be added to the name"""

        num = ""
        name = ""
        for c in string:
            if c.isnumeric():
                if name == "_spo" and c == '2':
                    name = name + str(c)
                else:
                    num = num + c
            else:
                name = name + c
        name = os.path.splitext(os.path.basename(name))[0][1:]  # Removal of file suffix and leading underscore
        return num, name

    def compute_and_integrate_meta_data(self,dictionary, files, root):

        """Iterating through the list of files called files. If the file is a .json type extract the meta-data of this
        file and save the values into the dictionary at the correct place. The place is given by the num in the filename."""

        for f in files:
            _, file_extension = os.path.splitext(f)
            num, _ = self.extract_num_and_name(f)
            if file_extension == ".json":
                dictionary['meta'][num] = self.read_meta(root + "/" + f)
        return dictionary

    def receive_data_from_csv(self,file, name, starting):
        """load the data from a csv file."""
        data = pd.read_csv(file, delimiter=";", names=[name + "time_orig", name])
        data[name + "time_orig"] = pd.to_timedelta(data[name + "time_orig"], unit="seconds")
        data[name + "time_orig"] = data[name + "time_orig"] + starting
        data = data.set_index(name + "time_orig")
        return data

    def compute_and_integrate_feat_data(self, dictionary, files, root):
        """Iterating through the list of files called files. If the file is a .csv type extract the feat-data of this
        file and save the values into the dictionary at the correct place. The place is given by the num in the filename."""
        for f in files:
            _, file_extension = os.path.splitext(f)
            num, name = self.extract_num_and_name(f)
            if file_extension == ".csv":
                values = self.receive_data_from_csv(file=root + "/" + f, name=name,
                                               starting=dictionary['meta'][num]['date_time_start'])
                dictionary[name][num] = values
        return dictionary

    def split_meta_event(self,dictionary):
        """This methods returns two dataframes from the dictionary.
        The dictionary contains all the data.
        Meta-data and event data will be extracted from the dictionary."""
        meta = dictionary['meta']
        events = list()
        for key, value in meta.items():
            events = events + value['events']
            del meta[key]['events']
        meta_df = pd.DataFrame.from_dict(meta).T
        events_df = pd.DataFrame.from_dict(events)
        return meta_df, events_df

    def split_meta_events_raw_data(self , dictionary):
        """Dictionary is a dict with dicts inside. The inside dicts contain dataframes (feats) or other dicts (meta).
        The meta dict is split to an event-dataframe with the seizures and a new meta-dataframe (without events)."""
        # first split meta and event
        meta_df, events_df = self.split_meta_event(dictionary)
        # second merge dictionary
        del dictionary['meta']
        return meta_df, events_df, dictionary
