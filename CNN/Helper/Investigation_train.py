import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from Helper.visualizationMetrics import visualization
import seizures

train_invest_path = 'train_investigation'
train_data_folder = 'Data'
train_acc_data = 'train_acc.npy'
train_hrtemp_data = 'train_hr_temp.npy'
train_label = 'train_label.npy'
train_time = 'train_time.npy'

test_acc_data = 'test_acc.npy'
test_hrtemp = 'test_hr_temp.npy'
test_time = 'test_time.npy'
test_label = 'test_label.npy'


def trainset_investigation(train_x, train_y, train_times, path, k=0):
    save_path = path + '/' + train_invest_path + '_' + str(k)
    os.mkdir(save_path)

    # Save train data
    acc_np = np.concatenate([train_x['acc_x'], train_x['acc_y'], train_x['acc_z']], axis=2)
    try:
        hrtemp_np = np.concatenate([train_x['heart_rate'], train_x['temperature']], axis=2)
    except:
        print('no_hrtemp')
    time_np = np.concatenate(list(train_times.values()), axis=0)

    label_np = train_y

    print('Train_Seiz_Windows', sum(label_np))

    seiz_dict = seizures.get_seizures_to_use()

    seiz_df = create_seiz_df(seiz_dict)

    # print(train_x.keys())
    non_seiz_plot_path = save_path + '/' + 'Non_Seizure'
    os.mkdir(non_seiz_plot_path)
    # seizure plots
    seiz_plot_path = save_path + '/' + 'Seizure'
    os.mkdir(seiz_plot_path)
    try:
        acc_df, hr_df = convert_window_to_dataframe(acc_np, time_np, hrtemp_np)

        # Non seizure Plots
        df_temp = pd.DataFrame(data={'t_label': label_np,
                                     'start': time_np[:, 0],
                                     'end': time_np[:, 1]}, index=time_np[:, 0])
        df_temp = df_temp.sort_index()
        df_for90 = df_temp[df_temp['t_label'] == 0]  # Filtering For the non-seizure part

        # print('Here')

        df_90s_tn = df_for90.resample('90s').agg(compute_non_seiz).dropna()
        tn_list = list(df_90s_tn.values)

        smp_to_plot = 6
        count = 0
        print('plotting Non Seiz')

        print(len(tn_list))
        for tn_sample in tn_list:
            if count == smp_to_plot:
                break
            s = tn_sample[0]
            e = tn_sample[1]
            sample_acc = acc_df[s:e].resample('20ms').mean()
            sample_hr_temp = hr_df[s:e].resample('1s').mean()
            # print('sddsd')
            create_plot(save_path=non_seiz_plot_path,
                        sample=sample_acc,
                        sample_hr_temp=sample_hr_temp,
                        title=f'{count}_Train_Non_Seizure')
            count += 1

        _, seiz_index = seizure_indices(label_np)

        tp_list, _ = compute_tp_fn_seizure(seiz_index, label_np, time_np)

        print('Plotting Seizures')
        count = 0
        for tp_sample in tp_list:
            s = tp_sample[0]
            e = tp_sample[1]
            sample = acc_df[s:e].resample('20ms').mean()
            cat, patient = get_category_pat(s, e, seiz_df=seiz_df)
            sample_hr_temp = hr_df[s:e].resample('1s').mean()
            create_plot(save_path=seiz_plot_path,
                        sample=sample,
                        sample_hr_temp=sample_hr_temp,
                        title=f'{count}_{patient}_{cat}')
            count += 1

    except:
        # print('No Hrtemp')
        acc_df = convert_window_to_dataframe(acc_np, time_np)

        # Non seizure Plots
        df_temp = pd.DataFrame(data={'t_label': label_np,
                                     'start': time_np[:, 0],
                                     'end': time_np[:, 1]}, index=time_np[:, 0])
        df_temp = df_temp.sort_index()
        df_for90 = df_temp[df_temp['t_label'] == 0]  # Filtering For the non-seizure part

        df_90s_tn = df_for90.resample('90s').agg(compute_non_seiz).dropna()
        tn_list = list(df_90s_tn.values)

        print('plotting Non Seiz')

        smp_to_plot = 6
        count = 0
        for tn_sample in tn_list:
            if count == smp_to_plot:
                break
            s = tn_sample[0]
            e = tn_sample[1]
            sample_acc = acc_df[s:e].resample('20ms').mean()
            create_plot(non_seiz_plot_path, sample_acc, title=f'{count}_Train_Non_Seizure')
            count += 1

        _, seiz_index = seizure_indices(label_np)
        tp_list, _ = compute_tp_fn_seizure(seiz_index, label_np, time_np)
        count = 0
        print('Plotting Seizures')
        for tp_sample in tp_list:
            s = tp_sample[0]
            e = tp_sample[1]
            sample = acc_df[s:e].resample('20ms').mean()
            cat, patient = get_category_pat(s, e, seiz_df=seiz_df)
            create_plot(seiz_plot_path, sample, title=f'{count}_{patient}_{cat}')
            count += 1
    return True


def convert_window_to_dataframe(acc, times, hrtemp=[]):
    df_list = []
    df_list2 = []

    for s_idx in range(len(times)):
        start = times[s_idx, 0]
        end = times[s_idx, 1]
        df_time = pd.date_range(start=start, end=end, periods=500)
        data = acc[s_idx, :, :]
        df = pd.DataFrame(data=data, index=df_time, columns=['acc_x', 'acc_y', 'acc_z'])
        df_list.append(df)
    accxyz_df = pd.concat(df_list, axis=0)
    accxyz_df = accxyz_df.sort_index()
    # print(hrtemp)

    if len(hrtemp) != 0:
        # print('here')
        for s_idx in range(len(times)):
            start = times[s_idx, 0]
            end = times[s_idx, 1]
            df_time2 = pd.date_range(start=start, end=end, periods=10)
            data2 = hrtemp[s_idx, :, :]
            df2 = pd.DataFrame(data=data2, index=df_time2, columns=['hr', 'Temperature'])
            df_list2.append(df2)

        hrtemp_df = pd.concat(df_list2, axis=0)
        hrtemp_df = hrtemp_df.sort_index()

        return accxyz_df, hrtemp_df
    else:
        return accxyz_df


def compute_non_seiz(window90):
    if window90.size:
        if 0 in window90['t_label'].values:
            start = window90['start'].iloc[0]  # first value of column in df.
            end = window90['end'].iloc[-1]  # last value of column in df
            return [start, end]
        else:
            return np.nan
    else:
        return np.nan


def create_plot(save_path, sample, title, sample_hr_temp=[], time_window=[], win_label='Fp_window'):
    acc_x = sample["acc_x"]
    acc_y = sample["acc_y"]
    acc_z = sample["acc_z"]
    t = sample.index
    title = title

    f, (x, y, z) = plt.subplots(3, 1)
    f.set_figheight(15)
    f.set_figwidth(30)
    f.suptitle(title, fontsize=14)

    x.plot(t, acc_x, "b")
    x.set_title("Acc x")
    x.set_ylabel('Acc_x', fontsize=20)

    y.plot(t, acc_y, "r")
    y.set_title("Acc y")
    y.set_ylabel('Acc_y', fontsize=20)

    z.plot(t, acc_z, "g")
    z.set_title("Acc z")
    z.set_ylabel('Acc_z', fontsize=20)
    z.set_xlabel('Time', fontsize=20)

    # print(sample_hr_temp)
    if len(sample_hr_temp) != 0:
        # print('here_plot_herat')
        # Hear Rate And Temperature Plot
        hr_ = sample_hr_temp["hr"]
        temp_ = sample_hr_temp["Temperature"]
        time = sample_hr_temp.index

        f3, (h2, t2) = plt.subplots(2, 1)
        f3.set_figheight(10)
        f3.set_figwidth(20)
        f3.suptitle(title, fontsize=14)

        h2.plot(time, hr_, "m")
        h2.set_title("Heart_Rate")

        t2.plot(time, temp_, "g")
        t2.set_title("Temperature")

    for f_wind in time_window:
        start_time = pd.to_datetime(f_wind[0], infer_datetime_format=True)
        end_time = pd.to_datetime(f_wind[1], infer_datetime_format=True)

        text_pos = start_time + pd.Timedelta('5s')
        x.text(text_pos, acc_x.max(), win_label, fontsize=12)
        y.text(text_pos, acc_y.max(), win_label, fontsize=12)
        z.text(text_pos, acc_z.max(), win_label, fontsize=12)

        x.axvline(x=start_time, color='k')
        x.axvline(x=end_time, color='k')
        y.axvline(x=start_time, color='k')
        y.axvline(x=end_time, color='k')
        z.axvline(x=start_time, color='k')
        z.axvline(x=end_time, color='k')

        if len(sample_hr_temp) != 0:
            h2.text(text_pos, hr_.max(), win_label, fontsize=12)
            t2.text(text_pos, temp_.max(), win_label, fontsize=12)
            h2.axvline(x=start_time, color='k')
            h2.axvline(x=end_time, color='k')
            t2.axvline(x=start_time, color='k')
            t2.axvline(x=end_time, color='k')

    f.savefig(save_path + '/' + title + '_acc.png')

    if len(sample_hr_temp) != 0:
        f3.savefig(save_path + '/' + title + '_hr.png')


def seizure_indices(y):
    """
    This method computes the ranges (indices) of consecutive 0's and 1's in y
    :param y: np array 1d labels
    """
    # compute start and end for every value change in y
    changes = np.where(y[:-1] != y[1:])[0]  # after these values there are changes!
    ends = np.append(changes, y.shape[0])
    starts = changes + 1
    starts = np.insert(starts, 0, 0)

    # stack them into pairs
    result = np.stack([starts, ends], axis=-1)
    # split every 2nd value to second; the rest in first:
    first = np.copy(result[::2, :])
    second = np.copy(result[1::2, :])

    # order for returned values depends on first label in y.
    if y[0] == 0:
        return first, second
    else:
        return second, first


def compute_tp_window(pred, interval):
    tp_windows = []
    for i in range(len(pred)):
        if pred[i] == 1:
            win = interval[i]
            win = [win[0], win[1]]
            tp_windows.append(win)
    return tp_windows


def compute_tp_fn_seizure(seiz_index, pred, times):
    """
    Computes the tp and fn for
    :param gt: seizure ranges (indices) in 2d numpy array with 2columns: 'start_index' and 'end_index' for every
    seizure
    :param pred: prediction of model
    :return: true positive and false negative for every seizure
    """
    detected = []
    non_detected = []
    for seizure in seiz_index:
        pred_temp = pred[seizure[0]:seizure[1] + 1]
        if 1 in pred_temp:
            seiz_det_int = times[seizure[0]:seizure[1] + 1, :]
            tp_window = compute_tp_window(pred_temp, seiz_det_int)
            seiz_det_int = [seiz_det_int[0, 0], seiz_det_int[-1, 1], tp_window]

            detected.append(seiz_det_int)
        else:
            non_det_int = times[seizure[0]:seizure[1] + 1, :]
            non_det_int = [non_det_int[0, 0], non_det_int[-1, 1]]
            non_detected.append(non_det_int)
    return detected, non_detected


def plot_train_distribute(path, train_x, train_y, k=0, name='Train'):
    path = path + '/' + train_invest_path + '_' + str(k)

    acc_np = np.concatenate([train_x['acc_x'], train_x['acc_y'], train_x['acc_z']], axis=2)
    seiz_win, non_seiz_win = compute_seiz_non_seiz_window(acc_np, train_y)
    if name == 'Train':
        visualization(path, non_seiz_win[:1000, :, :], seiz_win[:1000, :, :], 'pca', 'Non_Seiz', 'Seiz', name=name)
        visualization(path, non_seiz_win[:1000, :, :], seiz_win[:1000, :, :], 'tsne', 'Non_Seiz', 'Seiz', name=name)
    else:
        visualization(path, non_seiz_win[:300, :, :], seiz_win[:300, :, :], 'pca', 'Non_Seiz', 'Seiz', name=name)
        visualization(path, non_seiz_win[:300, :, :], seiz_win[:300, :, :], 'tsne', 'Non_Seiz', 'Seiz', name=name)


def compute_seiz_non_seiz_window(acc_np, train_y):
    seiz_win = []
    non_seiz_win = []

    for sidx in range(len(train_y)):
        if train_y[sidx] == 1:
            seiz_win.append(acc_np[sidx, :, :])
        else:
            non_seiz_win.append(acc_np[sidx, :, :])

    return np.stack(seiz_win, axis=0), np.stack(non_seiz_win, axis=0)


def get_category_pat(start, end, seiz_df):
    for _, seiz in seiz_df.iterrows():
        begin_seiz = pd.to_datetime(seiz['start']) - pd.Timedelta(seconds=10)  #
        end_seiz = pd.to_datetime(seiz['end']) + pd.Timedelta(seconds=10)

        if (begin_seiz <= start) and (end <= end_seiz):
            return seiz['catg_new'], seiz['pat']
        elif (begin_seiz >= start) and (end > begin_seiz):
            return seiz['catg_new'], seiz['pat']
        elif (end_seiz >= start) and (end > end_seiz):
            return seiz['catg_new'], seiz['pat']


def create_seiz_df(seiz_dict):
    start = []
    end = []
    catg_new = []
    catg_old = []
    patient = []

    for pat, events in seiz_dict.items():
        for item in events:
            start.append(item['start'])
            end.append(item['end'])
            catg_old.append(item['seizure_old'])
            catg_new.append(item['seizure_new'])
            patient.append(pat)

    seiz_df = pd.DataFrame(index=end, data={'start': start, 'end': end, 'catg_old': catg_old, 'catg_new': catg_new,
                                            'pat': patient})
    seiz_df = seiz_df.sort_index()
    return seiz_df
