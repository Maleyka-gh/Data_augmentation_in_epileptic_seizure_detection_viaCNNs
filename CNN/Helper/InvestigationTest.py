import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from Helper.visualizationMetrics import visualization
import seizures



def test_investigate(path,
                     test_x,
                     times,
                     t_label,
                     p_label,
                     k=0
                     ):

    times = np.concatenate(list(times.values()), axis=0)

    path = path + '/' 'test_investigation' + '_' + str(k)
    os.mkdir(path)
    acc_xyz = np.concatenate([test_x['acc_x'], test_x['acc_y'], test_x['acc_z']], axis=2)
    try:
        hrtemp = np.concatenate([test_x['heart_rate'], test_x['temperature']], axis=2)
        acc_xyz_df, hrtemp_df = convert_window_to_dataframe(acc_xyz, times, hrtemp)
    except:
        hrtemp = []
        acc_xyz_df = convert_window_to_dataframe(acc_xyz, times, hrtemp)

    # Creating the Dataframe of label arrays to calculate 90s intervals for non seizures
    df_temp = pd.DataFrame(data={'p_label': p_label,
                                 't_label': t_label,
                                 'start': times[:, 0],
                                 'end': times[:, 1]}, index=times[:, 0])
    df_temp = df_temp.sort_index()

    # For Non seizures
    df_for90 = df_temp[df_temp['t_label'] == 0]
    df_90s_fp = df_for90.resample('90s').agg(compute_fp).dropna()
    df_90s_tn = df_for90.resample('90s').agg(compute_tn).dropna()
    tn_list = list(df_90s_tn.values)
    fp_list = list(df_90s_fp.values)

    print('Plotting False Positives')
    smp_to_plot = 10
    count = 0
    fp_path = path + '/' + 'Non_Seiz_As_Seiz'
    os.mkdir(fp_path)
    for fp_sample in fp_list:
        if count == smp_to_plot:
            break
        s = fp_sample[0]
        e = fp_sample[1]
        fp_win = fp_sample[2]
        sample = acc_xyz_df[s:e].resample('20ms').mean()
        if len(hrtemp)!=0:
            sample_hr_temp = hrtemp_df[s:e].resample('1s').mean()
        else:
            sample_hr_temp = []

        create_plot(save_path=fp_path,sample=sample, sample_hr_temp=sample_hr_temp,
                    title=f'{count}_False_Positives', time_window=fp_win, win_label='Fp_win')
        count += 1

    print('Plotting True Negatives')

    smp_to_plot = 10
    count = 0
    tn_path = path + '/' + 'Non_Seiz_As_Non_Seiz'
    os.mkdir(tn_path)
    for tn_sample in tn_list:
        if count == smp_to_plot:
            break
        s = tn_sample[0]
        e = tn_sample[1]
        sample = acc_xyz_df[s:e].resample('20ms').mean()
        if len(hrtemp)!=0:
            sample_hr_temp = hrtemp_df[s:e].resample('1s').mean()
        else:
            sample_hr_temp=[]
        create_plot(save_path=tn_path, sample=sample, sample_hr_temp=sample_hr_temp,
                    title=f'{count}_True_Negative', time_window=[])
        count += 1


    #Seizures

    seiz_dict = seizures.get_seizures_to_use()

    seiz_df = create_seiz_df(seiz_dict)

    non_seiz_index, seiz_index = seizure_indices(t_label)

    tp_list, fn_list = compute_tp_fn_seizure(seiz_index, p_label, times)

    count = 0
    print('Plotting False Negatives')

    fn_path  = path + '/' + 'Seiz_As_Non_Seiz'
    os.mkdir(fn_path)

    for fn_sample in fn_list:
        s = fn_sample[0]
        e = fn_sample[1]
        sample = acc_xyz_df[s:e].resample('20ms').mean()
        cat, patient = get_category_pat(s, e, seiz_df=seiz_df)
        if len(hrtemp)!=0:
            sample_hr_temp = hrtemp_df[s:e].resample('1s').mean()
        else:
            sample_hr_temp = []
        create_plot(save_path=fn_path, sample=sample, sample_hr_temp=sample_hr_temp,
                    title=f'{count}_FN_{patient}_{cat}', time_window=[])
        count += 1

    count = 0
    print('Plotting True Positives')

    tp_path  = path + '/' + 'Seiz_As_Seiz'
    os.mkdir(tp_path)
    for tp_sample in tp_list:
        s = tp_sample[0]
        e = tp_sample[1]
        tp_window = tp_sample[2]
        sample = acc_xyz_df[s:e].resample('20ms').mean()
        cat, patient = get_category_pat(s, e, seiz_df=seiz_df)
        if len(hrtemp)!=0:
            sample_hr_temp = hrtemp_df[s:e].resample('1s').mean()
        else:
            sample_hr_temp = []
        create_plot(save_path=tp_path, sample=sample, sample_hr_temp=sample_hr_temp,
                    title=f'{count}_TP_{patient}_{cat}', time_window=tp_window, win_label='Tp_win')
        count += 1

    try:
        tp_data, fn_data = get_tp_fn_window(seiz_index, p_label, acc_xyz)
        tp_data, fn_data = np.stack(tp_data, axis=0), np.stack(fn_data, axis=0)
    except:
        print('tp_fn data missing')

    try:
        tn_data, fp_data = get_tn_fp_window(non_seiz_index, p_label, acc_xyz)
        tn_data, fp_data = np.stack(tn_data, axis=0), np.stack(fp_data, axis=0)
    except:
        print('tn_fp data missing')


    try:
        visualization(path, tn_data[:len(tp_data),:, :], tp_data, 'pca', 'True_Negative', 'True_Positives',name='Tn_vs_Tp')
        visualization(path, tn_data[:len(tp_data),:, :], tp_data, 'tsne', 'True_Negative', 'True_Positives', name= 'Tn_vs_Tp')
    except:
        print('Issue Tn_vs_Tp')

    try:
        visualization(path, tn_data[:len(fn_data), :, :], fn_data, 'pca', 'True_Negatives', 'False_Negatives',name='Tn_vs_Fn')
        visualization(path, tn_data[:len(fn_data), :, :], fn_data, 'tsne', 'True_Negatives', 'False_Negatives',name='Tn_vs_Fn')
    except:
        print('Issue Tn_vs_Fn')

    try:
        visualization(path, tn_data[:200, :, :], fp_data[:200, :, :], 'pca', 'True_Negative', 'False_Positives', name='Tn_vs_Fp')
        visualization(path, tn_data[:200,:,:], fp_data[:200,:,:],'tsne','True_Negative','False_Positives',name='Tn_vs_Fp')
    except:
        print('Issue Tn_vs_Fp')
    try:
        visualization(path,tp_data, fn_data, 'pca', 'True_Positives', 'False_Negatives',name='Tp_vs_Fn')
        visualization(path,tp_data, fn_data, 'tsne', 'True_Positives', 'False_Negatives',name='Tp_vs_Fn')
    except:
        print('Issue Tp_vs_Fn')

    try:
        visualization(path,fp_data[:len(tp_data),:,:],tp_data,'pca','False_Positives','True_Positives',name='Fp_vs_Tp')
        visualization(path,fp_data[:len(tp_data),:,:],tp_data,'tsne','False_Positives','True_Positives',name='Fp_vs_Tp')
    except:
        print('Issue fp vs tp')

def create_plot(save_path, sample, title, sample_hr_temp=[], time_window=[],win_label='Fp_window'):
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

    if len(sample_hr_temp)!=0:
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

        if len(sample_hr_temp)!=0:
            h2.text(text_pos, hr_.max(), win_label, fontsize=12)
            t2.text(text_pos, temp_.max(), win_label, fontsize=12)
            h2.axvline(x=start_time, color='k')
            h2.axvline(x=end_time, color='k')
            t2.axvline(x=start_time, color='k')
            t2.axvline(x=end_time, color='k')

    f.savefig(save_path+'/' + title + '_acc.png')

    if len(sample_hr_temp)!=0:
        f3.savefig(save_path + '/' + title + '_hr.png')


# For finding which windows inside the 90s interval are False positives
def compute_fp_windows(window90):
    fp_window=[]
    for i in range(len(window90['p_label'].values)):
        if window90['p_label'].iloc[i]==1:
            start = window90['start'].iloc[i]
            end = window90['end'].iloc[i]
            fp_window.append([start,end])
    return fp_window


# For Finding the Non-seizure detected as the seizures 90s intervals
def compute_fp(window90):
    if window90.size:
#         print(window90.size)
        if 1 in window90['p_label'].values:
            start = window90['start'].iloc[0]  # first value of column in df.
            end = window90['end'].iloc[-1]    # last value of column in df
            fp_windows = compute_fp_windows(window90)
            return [start,end,fp_windows]
        else:
            return np.nan
    else: #if there is no single window exists/is present  within 90s range  , then return Nan
        return np.nan

# For Finding the Non-seizure detected as the Non-seizures 90s intervals
def compute_tn(window90):
    if window90.size:
        if 0 in window90['p_label'].values:
            start = window90['start'].iloc[0]  # first value of column in df.
            end = window90['end'].iloc[-1]    # last value of column in df
            return [start,end]
        else:
            return np.nan
    else:
        return np.nan







def convert_window_to_dataframe(acc, times, hrtemp=None):
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

    if len(hrtemp)!=0:
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


def compute_tp_window(pred,interval):
    tp_windows=[]
    for i in range(len(pred)):
        if pred[i]==1:
            win = interval[i]
            win = [win[0],win[1]]
            tp_windows.append(win)
    return tp_windows


def compute_tp_fn_seizure(seiz_index, pred ,times):
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
        pred_temp = pred[seizure[0]:seizure[1]+1]
        if 1 in pred_temp:
            seiz_det_int = times[seizure[0]:seizure[1]+1, :]
            if seiz_det_int.shape[0]==0:
                seiz_det_int= times[seizure[0],:]
            else:
                seiz_det_int = times[seizure[0]:seizure[1]+1, :]
            tp_window = compute_tp_window(pred_temp,seiz_det_int)
            if seiz_det_int.shape[0]==0:
                seiz_det_int= [seiz_det_int[0],seiz_det_int[1],tp_window]
            else:
                seiz_det_int = [seiz_det_int[0, 0], seiz_det_int[-1, 1], tp_window]
            detected.append(seiz_det_int)
        else:
            non_det_int = times[seizure[0]:seizure[1]+1, :]
            if non_det_int.shape[0]==0:
                non_det_int= times[seizure[0],:]
                non_det_int = [non_det_int[0], non_det_int[1]]
            else:
                non_det_int = times[seizure[0]:seizure[1]+1, :]
                non_det_int = [non_det_int[0, 0], non_det_int[-1, 1]]

            non_detected.append(non_det_int)
    return detected, non_detected


def get_tp_fn_window(seiz_index, p_label, data):
    tp_data = []
    fn_data = []
    for intervals in seiz_index:
        pred_temp = p_label[intervals[0]:intervals[1]+1]
        for index in range(len(pred_temp)):
            if pred_temp[index]==1:
                tp_data.append(data[index,:,:])
            else:
                fn_data.append(data[index,:,:])
    return tp_data,fn_data


def get_tn_fp_window(non_seiz_index, p_label, data):
    tn_data = []
    fp_data = []
    for intervals in non_seiz_index:
        pred_temp = p_label[intervals[0]:intervals[1]+1]
        for index in range(len(pred_temp)):
            if pred_temp[index]==0:
                tn_data.append(data[index,:,:])
            else:
                fp_data.append(data[index,:,:])
    return tn_data,fp_data



def get_category_pat(start, end, seiz_df):

    for _,seiz in seiz_df.iterrows():
        begin_seiz = pd.to_datetime(seiz['start']) - pd.Timedelta(seconds=10) #
        end_seiz = pd.to_datetime(seiz['end']) + pd.Timedelta(seconds=10)

        if (begin_seiz<=start) and (end <=end_seiz):
            return seiz['catg_new'] , seiz['pat']
        elif (begin_seiz>=start) and (end >begin_seiz):
            return seiz['catg_new'] , seiz['pat']
        elif (end_seiz>=start) and (end >end_seiz):
            return seiz['catg_new'] , seiz['pat']


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