#!/usr/bin/env bash

import os

# import Nets_Orig_MultiInput.cnn_meisel_tuner
# import Nets_Orig_MultiInput.CNN_LSTM
# from Nets_Orig_MultiInput import cnn_meisel_tuner
# from Nets_Orig_MultiInput import cnn_meisel_old
# from Nets_Orig_MultiInput import cnn_meisel_tuner_acc
# import Nets_Orig_MultiInput.cnn_meisel
# import Nets_Orig_MultiInput.cnn_meisel_bfce
# import Nets_Orig_MultiInput.cnn_meisel_bfce_th_mov
# import Nets_Orig_MultiInput.cnn_meisel_th_mov
# from Nets_Orig_MultiInput import cnn_meisel_old_bfce_th_mov
import cnn_starter
from sklearn.preprocessing import StandardScaler
from Helper import Investigation_train



import config
import config_ignored
import numpy as np
if __name__ == "__main__":
    """First there are three sets created. Also the params for the windowing and features are determined.
    The data is loaded and then used for the CNN."""

    train, valid, test = cnn_starter.create_sets()
    train = train + valid

    # CHECK TENSORFLOW GPU USAGE
    print('###########################################################################################################')
    # os.environ["CUDA_VISIBLE_DEVICES"] = "-1" # switch off GPU
    import tensorflow as tf
    # tf.random.set_seed(42)
    print(tf.__version__)
    print(tf.test.is_built_with_cuda())
    print(tf.test.is_gpu_available(cuda_only=False, min_cuda_compute_capability=None))
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    print('###########################################################################################################')

    # path for results
    result_path = cnn_starter.create_path('NONENE')
    # result_path = cnn_starter.create_path('tts_gan_with_acc_only')

    # result dictionary:
    result = dict()
    result['train_set'] = train
    result['test_set'] = test

    #print('test' ,len(test))
    #print('train',len(train))
    # print('validation',validation)


    #train = ['BN_006','BN_011']

    train_x, train_y, train_times = cnn_starter.load_all_numpy_data(train,"/data/optimized/Numpy_Learn_10s0Overlap")
    # Investigation_train.trainset_investigation(train_x,train_y,train_times,result_path,k=0)
    # a_of_ind = np.array([j for j in range(len(train_y))if train_y[j]==1])
    #
    # print(train_times)
    # train_x['acc_x'] = np.delete(train_x['acc_x'],a_of_ind,axis=0)
    # train_x['acc_y'] = np.delete(train_x['acc_y'],a_of_ind,axis=0)
    # train_x['acc_z'] = np.delete(train_x['acc_z'],a_of_ind,axis=0)
    # train_y = np.delete(train_y,a_of_ind)

    #gan data & Epilepsygan data
    # epl_gan_path = '/data/Epilepsygan_data4cnn/epl_gan_70overlap_4cnn'
    # train_x_aug=dict()
    # for feat in os.listdir(epl_gan_path):
    #     if 'label' in feat:
    #         train_y_aug=np.load(os.path.join(epl_gan_path,feat))
    #     else:
    #         train_x_aug[feat[0:5]]=np.load(os.path.join(epl_gan_path,feat))
    # #
    # for k,v in train_x.items():
    #     train_x[k]=np.concatenate((train_x[k],train_x_aug[k]),axis =0)
    # train_y = np.concatenate((train_y,train_y_aug), axis=0)




    # augmented data
    # -------------Augnentation - Jittering, Permutation, Rotation , Rot(Perm)--change path for each technique

    # train_x_p, train_y_p, train_times_p = cnn_starter.load_all_numpy_data(train,"/data/augmentation/numpy_Permutation_n5_1x")
    # for k, v in train_x.items():
    #   train_x[k] = np.concatenate((train_x[k], train_x_p[k][:150,:,:]), axis=0)
    # train_y = np.concatenate((train_y,train_y_p[:150]), axis=0)
    #
    # print(train_x['acc_x'].shape)
    # print(train_x['acc_y'].shape)
    # print(train_y.shape)

    # train_x_p, train_y_p, train_times_p = cnn_starter.load_all_numpy_data(train,
    #                                                                       "/data/augmentation/numpy_Permutation_n5_1x")
    # for k, v in train_x.items():
    #     train_x[k] = np.concatenate((train_x[k], train_x_p[k]), axis=0)
    # train_y = np.concatenate((train_y, train_y_p), axis=0)

    # train_x_p, train_y_p, train_times_p = cnn_starter.load_all_numpy_data(train,
    #                                                                       "/data/augmentation/numpy_Rotation_4x_acc")
    # for k, v in train_x.items():
    #     train_x[k] = np.concatenate((train_x[k], train_x_p[k]), axis=0)
    # train_y = np.concatenate((train_y, train_y_p), axis=0)
    #
    # train_x_p, train_y_p, train_times_p = cnn_starter.load_all_numpy_data(train,
    #                                                                       "/data/augmentation/numpy_TimeWarp_0.2_2x")
    # for k, v in train_x.items():
    #     train_x[k] = np.concatenate((train_x[k], train_x_p[k]), axis=0)
    # train_y = np.concatenate((train_y, train_y_p), axis=0)
    #
    # train_x_p, train_y_p, train_times_p = cnn_starter.load_all_numpy_data(train,
    #                                                                       "/data/augmentation/numpy_MagWarp_0.2_4x")
    # for k, v in train_x.items():
    #     train_x[k] = np.concatenate((train_x[k], train_x_p[k]), axis=0)
    # train_y = np.concatenate((train_y, train_y_p), axis=0)
    #
    # train_x_p, train_y_p, train_times_p = cnn_starter.load_all_numpy_data(train,
    #                                                                       "/data/augmentation/numpy_Win_slice_4x")
    # for k, v in train_x.items():
    #     train_x[k] = np.concatenate((train_x[k], train_x_p[k]), axis=0)
    # train_y = np.concatenate((train_y, train_y_p), axis=0)





    # TTS GAN
    # tts_gan_path = "/data/ttsgan_data4cnn/tts_bestcos10%"
    # list_dat = []
    # for files in os.listdir(tts_gan_path):
    #     list_dat.append(files[0:6])
    #  # TTS_GAN
    # train_x_p, train_y_p, train_times_p = cnn_starter.load_all_numpy_data(list_dat,tts_gan_path)
    # for k, v in train_x.items():
    #     train_x[k] = np.concatenate((train_x[k], train_x_p[k]), axis=0)
    # train_y = np.concatenate((train_y, train_y_p), axis=0)





     # print number of train labels for seizure and non-seizure windows before undersampling
    # count=0
    # for i in train_y:
    #    if i==1:
    #         count+=1
    # print("train_y for seizure  before undersampling",count)
    # print('train_y for nonseizure before undersampling' , len(train_y)-count)
    # # print(train_y.shape)




    cw = None #compute_class_weight(train_y) No class weights for final evaluation


    test_x, test_y, test_times = cnn_starter.load_all_numpy_data(test,"/data/optimized/Numpy_Learn_10s0Overlap")


    # print number of test labels for seizure and non-seizure windows
    # count = 0
    # for i in test_y:
    #     if i == 1:
    #         count += 1
    # print("test_y for seizure ", count)
    # print('test_y for nonseizure', len(test_y) - count)






   #  count=0
   #  print(test)
   #  for i in range(len(test_y)):
   #     if test_y[i]==1:
   #          print(i)
   #          print(test_x["acc_x"][i])
   #          print(test_times[test[1]][i])
   #          count+=1



    # print("siezure window",count)
    # print('Non-seizure ' , len(test_y)-count)
    # print(test_y.shape)


   # #  c = 0
   # #  for i in test_y:
   # #      if i == 1:
   # #          c += 1
   # #  # print(c)
   #  #print(train)
   #  result['resampled'] = 'Original; acc has 50 Hz, rest 1 Hz.'
   #  # SAMPLING TRAIN DATA
   #  # print(train_y)
   #  # for i in range(len(train_y)):
   #  #     if train_y[i]==1:
   #  #         print(train_x['acc_x'][i])
   #  #         print(i)
   #  #         #print(train_times[train[0]][i])
   #  #         break
   #  #         # print(train_x['acc_x'][i])
   #  #         # #print(train_times)
   #  #         # print(train_times[train[0]][i])

    train_x, train_y, _ = cnn_starter.under_sample_data(train_x, train_y)# noqa

    # print number of train labels for seizure and non-seizure windows after undersampling

    print('train_y for seizure after undersampling :', np.sum(train_y))
    print('train_y for nonseizure after undersampling :', len(train_y)-np.sum(train_y))


    result['imbalance'] = 'under sampling'
   # test['BN_123', 'BN_046', 'BN_082', 'BN_141', 'BN_169']
   #  NORMALIZE DATA
    m_scaler = StandardScaler()  # noqa
    train_x, m_scaler = cnn_starter.normalize(train_x, m_scaler, fit=True)
    test_x, m_scaler = cnn_starter.normalize(test_x, m_scaler, fit=False)  # noqa


    # SHUFFLE TRAIN DATA
    train_x, train_y = cnn_starter.shuffle(train_x, train_y)
    # Investigation_train.plot_train_distribute(result_path,train_x,train_y,k=0,name='train')
    # Investigation_train.plot_train_distribute(result_path, test_x, test_y, k=0, name='test')




    # # Das beste Modell ist das "Standard" Modell Meisels. Also ohne Hyperparametertuning ohne Transferlernen und alles.
    Nets_Orig_MultiInput.cnn_meisel.CNN(result, train_x, train_y, test_x, test_y, test_times,
                                                   result_path, k=0, cw=cw, name='FINAL')
    #
    #
    # print('FINISHED :)')

