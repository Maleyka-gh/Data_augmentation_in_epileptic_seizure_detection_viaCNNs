import os
import json
from datetime import datetime
import config
import pandas as pd
import pathlib
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import plotly.express as px
# from new_code.Code.DataExploration.Plot import plot_data
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go






def mean_values_of_duplicate_indices(dict_of_df):
    """Compute the mean of all values with duplicated indices."""
    for key, value in dict_of_df.items():
        if any(value.index.duplicated(keep=False)):   # mark all duplicates  as True
            dict_of_df[key] = value.groupby(value.index).mean()
    return dict_of_df


def merge_to_one(dict_of_df, axis):
    """Receives a dictionary with x dataframes inside. These dataframes are merged and sorted by index.
    If their a dataframe has duplicated indices, the mean of their values will be computed and used."""
    dict_of_df = mean_values_of_duplicate_indices(dict_of_df)
    df = pd.concat(dict_of_df.values(), ignore_index=False, axis=axis)
    df = df.sort_index()
    df = df.astype(float)
    return df


def split_meta_event(dictionary):
    """This methods returns two dataframes from the dictionary.
    The dictionary contains all the data.
    Meta data and event data will be extracted from the dictionary."""
    meta = dictionary['meta']
    events = list()
    for key, value in meta.items():
        events = events + value['events']
        del meta[key]['events']
    meta_df = pd.DataFrame.from_dict(meta).T
    events_df = pd.DataFrame.from_dict(events)
    return meta_df, events_df


def split_meta_events_raw_data(dictionary):
    """Dictionary is a dict with dicts inside. The inside dicts contain dataframes (feats) or other dicts (meta).
    The meta dict is split to an event-dataframe with the seizures and a new meta-dataframe (without events)."""
    # first split meta and event
    meta_df, events_df = split_meta_event(dictionary)
    # second merge dictionary
    del dictionary['meta']
    return meta_df, events_df, dictionary


def receive_data_from_csv(file, name, starting):
    """load the data from a csv file."""
    data = pd.read_csv(file, delimiter=";", names=[name+"time_orig", name])
    data[name+"time_orig"] = pd.to_timedelta(data[name+"time_orig"], unit="seconds")
    data[name+"time_orig"] = data[name+"time_orig"] + starting
    data = data.set_index(name+"time_orig")
    return data


def compute_and_integrate_feat_data(dictionary, files, root):
    """Iterating through the list of files called files. If the file is a .csv type extract the feat-data of this
    file and save the values into the dictionary at the correct place. The place is given by the num in the filename."""
    for f in files:
        _, file_extension = os.path.splitext(f)
        num, name = extract_num_and_name(f)
        if file_extension == ".csv":
            values = receive_data_from_csv(file=root + "/" + f, name=name,
                                                     starting=dictionary['meta'][num]['date_time_start'])
            dictionary[name][num] = values
    return dictionary


def switch_events_in_datetime_format(event_list):
    """This method puts the beginning and ending of a list of events in the correct datetime format."""
    for i in range(len(event_list)):
        event_list[i]['event_start'] = datetime.strptime(event_list[i]['event_start'], '%Y-%m-%d %H:%M:%S')
        event_list[i]['event_end'] = datetime.strptime(event_list[i]['event_end'], '%Y-%m-%d %H:%M:%S')
    return event_list


def read_time_stamps(data):
    """This method reads timestamp from data."""
    if data is None:
        return None
    try:
        return datetime.strptime(data, '%Y-%m-%d %H:%M:%S')
    except ValueError:  # string does not match the time-format
        if data != 'None':
            print(data)
        return None


def read_meta(json_file):
    """Read the requested meta data from the passed parameter json_file."""
    with open(json_file) as json_file:
        data = json.load(json_file)
        meta_dict = {'person_id': data['meta']['db']['person_id'],
                     'person_label': data['meta']['db']['person_label'],
                     'device_id': data['meta']['db']['device_model_number'],
                     'events': switch_events_in_datetime_format(data['meta']['events']),
                     'date_time_start': read_time_stamps(data['meta']['db']['date_time_start']),
                     'date_time_end': read_time_stamps(data['meta']['db']['date_time_end'])
                     }
    return meta_dict


def extract_num_and_name(string):
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


def compute_and_integrate_meta_data(dictionary, files, root):
    """Iterating through the list of files called files. If the file is a .json type extract the meta-data of this
    file and save the values into the dictionary at the correct place. The place is given by the num in the filename."""
    for f in files:
        _, file_extension = os.path.splitext(f)
        num, _ = extract_num_and_name(f)
        if file_extension == ".json":
            dictionary['meta'][num] = read_meta(root + "/" + f)
    return dictionary


def create_dict_for_patient():
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


def is_already_computed(name, directory):
    """Checks if the id is already saved in the directory"""
    files = os.listdir(directory)
    for f in files:
        if name in f:
            return True
    return False


def extract_patient_id(path, root):
    """This method extracts the patient id '_BN_xxx' from path and root."""
    path_without_root = root[len(path) + 1:]
    path_without_root = pathlib.Path(path_without_root)
    return path_without_root.parts[0]


def is_file_for_pat(root, files, path):
    """Checks if the current root and its files contains the data of a patient."""
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


def start(path):
    """Iterates through folder and extract all patient information.
    Results are saved in the folder 'RAW_DF'."""

    for root, dirs, files in sorted(os.walk(path)):
        files=sorted(files)
        patient_id = 'unknown'
        # noinspection PyBroadException
        try:
            if is_file_for_pat(root, files, path):  # if we have files for a patient!
                patient_id = extract_patient_id(path, root)
                print(patient_id)
                # if not is_already_computed(patient_id, config.RAW_DF_FOLDER):
                if not is_already_computed(patient_id, config.RAW_DF_FOLDER):
                    os.mkdir(config.RAW_DF_FOLDER + "/" + patient_id)
                    data_of_patient = create_dict_for_patient()
                    # first only read meta-data:
                    data_of_patient = compute_and_integrate_meta_data(data_of_patient, files, root)
                    # now read all raw feature data:
                    data_of_patient = compute_and_integrate_feat_data(data_of_patient, files, root)
                    meta, events, dfs = split_meta_events_raw_data(data_of_patient)
                    for feat, dictionary in dfs.items():
                        for num, val in dictionary.items():
                            val.to_pickle(config.RAW_DF_FOLDER + "/" + patient_id + "/" + num + "_" + feat)
                            # val.to_pickle(config.RAW_DF+ "/" + patient_id + "_" + num + "_" + feat)
                    meta.to_pickle(config.RAW_DF_FOLDER + "/" + patient_id + "/meta")
                    # meta.to_pickle(config.RAW_DF + "/" + patient_id + "_meta")
                    events.to_pickle(config.RAW_DF_FOLDER + "/" + patient_id + "/events")
                    # events.to_pickle(config.RAW_DF + "/" + patient_id + "_events")
        except Exception:
            print('PROBLEM: ' + str(patient_id))


if __name__ == "__main__":
    # os.mkdir(config.RAW_DF)
    start(config.DATA_PATH)

    # print('Before \n ',pd.read_csv("C:/Users/seyidova/Desktop/Data/Data/BN_006/data/tmp/export_for_mond/00000039_acc_x.csv",sep=';',header=None).head(10))
    # df_before = pd.read_csv('C:/Users/seyidova/Desktop/Data/Data/BN_006/data/tmp/export_for_mond/00000039_acc_x.csv' ,sep=';' , names = ['Time', 'acc_x'], index_col = 0)
    #
    # print(df_before.head())

    # plt.figure(1)
    # plt.plot(df_before.index, df_before['acc_x'])
    # plt.title("acceleration in x direction for BN_006", fontsize=16)
    # plt.ylabel("acc_x")
    # plt.xlabel("Time")
    # plt.show()
    # end_time = "2017-03-08 18:32:44"
    # start_time = "2017-03-08 15:12:35"

    # values = {"dates":[ "2017-03-11 04:20:00" ,"2017-03-11 04:25:00"]}
    #
    # df1= pd.DataFrame(values)
    # df1['dates'] = pd.to_datetime(df1['dates'],format='%Y-%m-%d %H:%M:%S')
    # x=df1['dates'][0]
    # y=df1['dates'][1]

    # df = df.drop_duplicates(keep='first')
    # # import plotly.express as px


    # print(x)
    # print(df1)

    # df_after = pd.read_pickle(r'C:\Users\seyidova\Desktop\Data\raw_Dataframes_sub_Dirs\BN_011\00000159_acc_x')
    # #print(df_after[x:y])
    #
    # df_y = pd.read_pickle(r'C:\Users\seyidova\Desktop\Data\raw_Dataframes_sub_Dirs\BN_011\00000159_acc_y')
    # df_z = pd.read_pickle(r'C:\Users\seyidova\Desktop\Data\raw_Dataframes_sub_Dirs\BN_011\00000159_acc_z')
    # df_y = df_y[x:y]
    # df_z = df_z[x:y]

    '''comparison of normalized and orig.dataframe in x direction for BN011'''


    # test = df_after[x:y]
    #
    # fig = px.line(test, x=test.index, y='acc_x', title='acceleration in x direction for B011')
    # fig.update_xaxes(rangeslider_visible=True)
    # fig.show()
    #
    # print(test)
    #
    # scaler = StandardScaler()  # noqa
    # a = scaler.fit_transform(test)
    # print(a)
    #
    #
    # norm_df = pd.DataFrame(a, index=test.index ,columns=['acc_x_norm'])
    #
    # fig2 = px.line(norm_df, x=norm_df.index, y=norm_df['acc_x_norm'], title='acceleration in x direction for B011')
    # fig2.update_xaxes(rangeslider_visible=True)
    # fig2.show()
    #
    #
    #
    #
    # df = pd.concat([test , norm_df], axis=1)
    #
    #
    # print(df)
    # fig3 = px.line(df, x=df.index, y=df.columns,
    #               title='Check')
    # fig3.update_xaxes(rangeslider_visible=True)
    # fig3.show()

    # print(test.values)
    #
    # k = test.values.shape[0]
    # fig4 = make_subplots(rows=3, cols=1)

    # fig4.add_trace(go.scatter(x=list(np.reshape(test.index, (1,k))), y=list(np.reshape(test.values,(1,test.values.shape[0]))),row=1,col=1
    #                   ))
    # fig4.add_trace(go.scatter(x=np.reshape(df_y.index, (1,k)), y=np.reshape(df_y.values,(1,df_y.values.shape[0]))),
    #
    #                   row=2,col=1)
    # fig4.add_trace(go.scatter(x=np.reshape(df_z.index, (1,k)), y=np.reshape(df_z.values,(1,df_z.values.shape[0]))),
    #
    #                   row=3,col=1)

    # fig = make_subplots(rows=3, cols=1)
    #
    # fig.append_trace(go.Scatter(
    #     x=[3, 4, 5],
    #     y=[1000, 1100, 1200],
    # ), row=1, col=1)
    #
    # fig.append_trace(go.Scatter(
    #     x=[2, 3, 4],
    #     y=[100, 110, 120],
    # ), row=2, col=1)
    #
    # fig.append_trace(go.Scatter(
    #     x=[0, 1, 2],
    #     y=[10, 11, 12]
    # ), row=3, col=1)
    #
    # fig.update_layout(height=600, width=600, title_text="Stacked Subplots")
    # fig.show()

    # fig.update_xaxes(rangeslider_visible=True)
    # fig.show()
    #)
    # test.plot()
    # plt.show()

    #print(df_after.tail())

    # print(df_after.index)
    # df_before['']
    # print("After \n " ,  df_after.head())

    # print(type(df_after.index))

    # plot_data.Plot(df_after)

    # df_after.plot()
    # plt.show()

    # plt.figure(2)
    # plt.plot(df_after.index, df_after['acc_x'])
    # plt.title("acceleration in x direction for BN_006 (After)", fontsize=16)
    # plt.ylabel("acc_x")
    # plt.xlabel("Time")
    # plt.show()











