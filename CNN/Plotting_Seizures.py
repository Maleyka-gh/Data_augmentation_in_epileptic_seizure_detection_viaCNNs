import os
import pandas as pd
import matplotlib.pyplot as plt
import config
from PreProcessingClasses import PreProcessHJ
from Data_Explore_Classes.SeizureCode import SeizureCode
from Data_Explore_Classes.DataExploration import DataExploration


def load_data(path, file):
    try:
        return pd.read_pickle(os.path.join(path, file))
    except:
        return pd.DataFrame(index=[pd.to_datetime(0)], columns=[file[16:]])


def extract_seizure(data, start, end):
    if data[start:end].empty:
        return data
    else:
        return data[start:end]


def plot_seizures(seizures, raw_df, plot_path):
    i = 0
    for _, seizure in seizures.iterrows():
        # print(seizure)
        i += 1  # counter for naming Unknown Seizure ids

        obs = seizure["observation_id"]
        obs_padd = PreProcessHJ.DataPreset.get_padded_id(obs)
        patient = seizure["participant_id"]
        seizure_id = seizure["seizure_id"]
        category = seizure["category"]
        start_time = seizure["start"]
        end_time = seizure["end"]

        # Reading The Acc Data
        x = load_data(raw_df, patient + '_' + obs_padd + '_' + 'acc_x')
        y = load_data(raw_df, patient + '_' + obs_padd + '_' + 'acc_y')
        z = load_data(raw_df, patient + '_' + obs_padd + '_' + 'acc_z')

        acc = pd.concat([x, y, z], axis=1)
        acc.index = pd.to_datetime(acc.index)

        # Reading Heart Rate and Quality
        heart_rate = load_data(raw_df, patient + '_' + obs_padd + '_' + 'heart_rate')
        quality = load_data(raw_df, patient + '_' + obs_padd + '_' + 'quality')

        # Reading the Temperature
        temperature = load_data(raw_df, patient + '_' + obs_padd + '_' + 'temperature')

        # 1 minute before and after
        one_min = pd.to_timedelta("60s")
        start_time_ext1 = str(-one_min + pd.to_datetime(start_time))
        end_time_ext1 = str(one_min + pd.to_datetime(end_time))


        window = extract_seizure(acc, start_time_ext1, end_time_ext1)
        window = window.resample("20ms").mean()

        acc_x = window["acc_x"]
        acc_y = window["acc_y"]
        acc_z = window["acc_z"]
        t = window.index
        title = "Patient " + patient + ", Anfall " + str(obs)

        f, (x, y, z) = plt.subplots(3, 1)
        f.set_figheight(10)
        f.set_figwidth(20)
        f.suptitle(title, fontsize=14)

        start_time = pd.to_datetime(start_time, infer_datetime_format=True)
        end_time = pd.to_datetime(end_time, infer_datetime_format=True)

        x.plot(t, acc_x, "b")
        x.set_title("Acc x")
        x.axvline(x=start_time, color='k')
        x.axvline(x=end_time, color='k')

        y.plot(t, acc_y, "r")
        y.set_title("Acc y")
        y.axvline(x=start_time, color='k')
        y.axvline(x=end_time, color='k')

        z.plot(t, acc_z, "g")
        z.set_title("Acc z")
        z.axvline(x=start_time, color='k')
        z.axvline(x=end_time, color='k')

        if category in os.listdir(plot_path):
            pass
        else:
            os.mkdir(os.path.join(plot_path, category))
# for some seizures seizure id is not available
        if 'Unknown' in seizure_id:
            seizure_id = 'u' + str(i) # for naming the plot 'u' in name will symbolize the unknown seizures


        plt.savefig(plot_path + '/' + category + "/" +patient+ "_" + str(obs_padd) +'_'+str(seizure_id)+ "_acc.png")

        # 5 minutes before and after
        five_mins = pd.to_timedelta("300s")
        start_time_ext = str(- five_mins + pd.to_datetime(start_time))
        end_time_ext = str(five_mins + pd.to_datetime(end_time))

        window_hr = extract_seizure(heart_rate, start_time_ext, end_time_ext)
        window_qual = extract_seizure(quality, start_time_ext, end_time_ext)
        window_temp = extract_seizure(temperature, start_time_ext, end_time_ext)

        window_hr = window_hr.resample("1s").mean()
        window_qual = window_qual.resample("1s").mean()
        window_temp = window_temp.resample("1s").mean()

        time_temp = window_temp.index
        time_hr = window_hr.index
        time_qual = window_qual.index

        hr_ = window_hr["heart_rate"]
        qual_ = window_qual["quality"]
        temp_ = window_temp["temperature"]

        f3, (h2, q2, t2) = plt.subplots(3, 1)
        f3.set_figheight(10)
        f3.set_figwidth(20)
        f3.suptitle(title, fontsize=14)

        h2.plot(time_hr, hr_, "m")
        h2.set_title("Heart_Rate")
        h2.axvline(x=start_time, color='k')
        h2.axvline(x=end_time, color='k')

        q2.plot(time_qual, qual_, "c")
        q2.set_title("Quality")
        q2.axhline(y=40, color='r')

        t2.plot(time_temp, temp_, "g")
        t2.set_title("Temperature")
        t2.axvline(x=start_time, color='k')
        t2.axvline(x=end_time, color='k')

        plt.savefig(plot_path + '/' + category + '/' + patient+'_'+obs_padd+"_"+str(seizure_id) + "_hr.png")


if __name__ == "__main__":
    seizure_object = SeizureCode(raw_df_path=config.Raw_df, seizure_excel_path=config.ExcelPath)
    motor_only_available = seizure_object(only_motor=True, check_num_feat=config.feat_to_use)
    DataExploration(seiz_record=motor_only_available, main_root=config.PlottingResultPath
                    , raw_df_path=config.Raw_df, raw_data_path=config.RAW_DATA_PATH
                    )
    seizure = PreProcessHJ.DataPreset.get_seizure_info(config.PlottingResultPath)
    plot_seizures(seizures=seizure, raw_df=config.Raw_df, plot_path=config.PlottingResultPath)

