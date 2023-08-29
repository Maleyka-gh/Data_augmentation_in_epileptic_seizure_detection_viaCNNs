import shutil
import os
import PreProcessingClasses.PreProcessYnnik as preprocs
import config


#Folder List - folders path are given in config.py
f_list = ['raw_Dataframes_sub_Dirs', 'dataframes_learn']

def check_folder_structure(path):
    list_of_exis_dirs = os.listdir(path)
    for folder in f_list:
        if folder in list_of_exis_dirs:
            shutil.rmtree(os.path.join(path,folder))
            os.mkdir(os.path.join(config.PreProcessResultPath, folder))
        else:
            os.mkdir(os.path.join(config.PreProcessResultPath,folder))


if __name__ == "__main__":

    is_save = input("Do you want to save the output at each step? [y/n]").lower()
    if is_save=='y':
        check_folder_structure(config.PreProcessResultPath)
        print('Running Code')
        preprocs.Raw_data_preprocesor(save=True)
    else:
        print('Not saving each step, but only last step of preprocessing')
        preprocs.Raw_data_preprocesor(save=False)