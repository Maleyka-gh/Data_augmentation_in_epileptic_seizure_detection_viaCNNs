import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

labels = ['Baseline','TTS-GAN(DTW)','TTS-GAN(DTW)','TTS-GAN(DTW)','TTS-GAN(DTW)','TTS_GAN(MSE)','TTS_GAN(MSE)','TTS_GAN(MSE)','TTS_GAN(MSE)','TTS_GAN(CS)','TTS_GAN(CS)','TTS_GAN(CS)','TTS_GAN(CS)','EPL_GAN','EPL_GAN','EPL_GAN','EPL_GAN']
Seizure_sensititvity = [0.8,0.8, 0.7, 0.6, 0.5,0.85, 0.8, 0.8, 0.7,0.85, 0.75, 0.8, 0.65,0.85, 0.8, 0.6, 0.65]
Sensitivity =  [0.6, 0.6133, 0.4533, 0.3733, 0.2666,0.5733, 0.5666, 0.5266, 0.44,0.6066, 0.54, 0.5133, 0.34,0.5933 , 0.5066 , 0.3666 , 0.34]
Specificity = [0.8057, 0.7832, 0.8548, 0.8659, 0.9679,0.7871, 0.8598, 0.8840, 0.9082,0.8244, 0.8516, 0.8711, 0.9335,0.8126 , 0.8653 , 0.9208 , 0.9297]
FAR = [453, 430, 328, 313, 124,414, 388, 341, 204,434, 387, 336, 180,416 , 330 , 240 , 196]
proportion = ['No', '10%','30%','50%','80%', '10%','30%','50%','80%', '10%','30%','50%','80%', '10%','30%','50%','80%']
# Define the colors for each label
colors_dict = {
    'Baseline': 'red',
    'TTS-GAN(DTW)': 'blue',
    'TTS_GAN(MSE)': 'green',
    'TTS_GAN(CS)': 'orange',
    'EPL_GAN': 'purple'
}

# Get the colors for each value
colors = [colors_dict[label] for label in labels]

# Create scatter plot
fig, ax = plt.subplots(figsize=(10, 6))
sc = ax.scatter(Seizure_sensititvity, FAR, c=colors,marker='s')

# Add labels and title
ax.set_xlabel('Sensitivity (Seizure)')
ax.set_ylabel('FA/24h')
ax.set_title('Relationship between Sensitivity (Seizure) and FA/24h')

# Add labels for each point
for i, label in enumerate(labels):
    text = f"({proportion[i]})"  # add proportion to label
    # print(label)
    if label == 'Baseline':
        ax.annotate(text, (Seizure_sensititvity[i], FAR[i]), xytext=(5, -5), textcoords='offset points', ha='left', va='bottom', fontsize=10)
    elif label=='TTS_GAN(CS)':
        ax.annotate(text, (Seizure_sensititvity[i], FAR[i]), xytext=(-5, 5), textcoords='offset points', ha='right', va='top', fontsize=10)
    else:
        ax.annotate(text, (Seizure_sensititvity[i], FAR[i]), xytext=(5, -5), textcoords='offset points', ha='left', va='bottom', fontsize=10)


# Add gridlines
ax.grid(linestyle='dashed', linewidth=0.5)
# Add legend
legend_elements = [plt.Line2D([0], [0], marker='s', color='w', label='Baseline', markerfacecolor='r', markersize=10),
                   plt.Line2D([0], [0], marker='s', color='w', label='TTS-GAN(DTW)', markerfacecolor='b', markersize=10),
                   plt.Line2D([0], [0], marker='s', color='w', label='TTS_GAN(MSE)', markerfacecolor='g', markersize=10),
                   plt.Line2D([0], [0], marker='s', color='w', label='TTS_GAN(CS)', markerfacecolor='orange', markersize=10),
                   plt.Line2D([0], [0], marker='s', color='w', label='EPL_GAN', markerfacecolor='purple', markersize=10)]

ax.legend(handles=legend_elements, loc='upper left', fontsize=10)


# Show plot
plt.show()
