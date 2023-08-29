# import numpy as np
# import torch
import matplotlib.pyplot as plt

# y_pred = np.random.randint(0,4,size=(500,3))
# y_true = np.random.randint(0,4,size=(500,3))
#
# mse1 = np.mean((y_pred - y_true)**2)
# mse2=np.mean(np.mean((y_pred - y_true)**2,axis=0))
# print(mse1,mse2)

import torch

# Load the model from the .pth file
# model = torch.load("/data/TTS_GAN/logs/Running_2023_01_13_16_38_57/Model/checkpoint")
# model = torch.load("/data/TTS_GAN/logs/Running_2023_01_13_16_38_57/Model/checkpoint_bestcos_sim")
# model = torch.load("/data/TTS_GAN/logs/Running_2023_01_13_16_38_57/Model/checkpoint_bestmse")
# model = torch.load("/data/TTS_GAN/logs/Running_2022_12_14_13_23_43_bestmodel99%/Model/checkpoint")
# print('best_dtw_case: ',model["mse_list"][881],model["cos_sim_list"][881],model["dtw_list"][881])
# print('best_cossim_case: ',model["mse_list"][887],model["cos_sim_list"][887],model["dtw_list"][887])
# print('best_mse_case: ',model["mse_list"][57],model["cos_sim_list"][57],model["dtw_list"][57])

model=torch.load("/data/TTS_GAN/logs/0%_default_final_2023_03_08_22_04_32/Model/checkpoint")
# model=torch.load("/data/TTS_GAN/logs/b_s_8_bestmodel_parameters_2023_02_02_00_23_17/Model/checkpoint_bestdtw")

#print('best_mse_case: ',model["mse_list"][360],model["cos_sim_list"][360],model["dtw_list"][360])
#print('best_cossim_case: ',model["mse_list"][19],model["cos_sim_list"][19],model["dtw_list"][19])
#print('best_dtw_case: ',model["mse_list"][210],model["cos_sim_list"][210],model["dtw_list"][210])


# model['dtw_list'][-1]
# print(model['discriminator_loss'],model['generator_loss'])
# print(model['epoch'])
# dtw_list= model['dtw_list']
# mse_list=model['mse_list']
# cossim_list=model['cos_sim_list']
# plt.figure()
# plt.plot(dtw_list)
# plt.xlabel('epochs')
# plt.title('avg_dtw between real and synthetic samples at every 10th epoch')
# plt.ylabel('avg dtw')
# plt.axvline(x=211, color='red', linestyle='--')
# plt.savefig('/data/TTS_GAN/plots_mse_dtw_cossim_b_8/avg_dtw.png')


# plt.figure()
# plt.plot(mse_list)
# plt.xlabel('epochs')
# plt.title('avg_mse between real and synthetic samples at every 10th epoch')
# plt.ylabel('avg mse')
# plt.axvline(x=361, color='red', linestyle='--')
# plt.savefig('/data/TTS_GAN/plots_mse_dtw_cossim_b_8/avg_mse.png')



# plt.figure()
# plt.plot(cossim_list)
# plt.xlabel('epochs')
# plt.title('avg cossimilarity between real and synthetic samples at every 10th epoch')
# plt.ylabel('avg cos similarity')
# plt.axvline(x=20, color='red', linestyle='--')
# plt.savefig('/data/TTS_GAN/plots_mse_dtw_cossim_b_8/avg_cossim.png')
#
# # # The model can now be used for inference
# # k = [a.to('cpu').detach().numpy() for a in model["discriminator_loss"]]
#
# # print(k)
# # output = model(input)
# plt.plot(k[:1000])
# plt.show()