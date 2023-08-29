import matplotlib.pyplot as plt
import numpy as np

# Create color map
cmap = plt.get_cmap('tab20')

# Create scatter plot
fig, ax = plt.subplots(figsize=(10, 6))
sc = ax.scatter(Seizure_sensititvity, FAR, c=np.arange(len(labels)), cmap=cmap)
proportion = ['No', '1x', '4x', '8x', '1x', '8x', '1x', '4x', '4x', '4x', '4x']


# Add labels and title
ax.set_xlabel('Sensitivity (Seizure)')
ax.set_ylabel('FA/24h')
ax.set_title('Relationship between Sensitivity (Seizure) and FA/24h')

# Add labels for each point
for i, label in enumerate(labels):
    text = f"{label} ({proportion[i]})"  # add proportion to label
    print(label)
    if label == 'Baseline' or label == 'RT' or label == 'W' :
        ax.annotate(text, (Seizure_sensititvity[i], FAR[i]), xytext=(-5, 5), textcoords='offset points', ha='right', va='top', fontsize=10)
    elif label=='R' or label=='T':
        ax.annotate(text, (Seizure_sensititvity[i], FAR[i]), xytext=(1, -5), textcoords='offset points', ha='right', va='top', fontsize=10)
    else:
        ax.annotate(text, (Seizure_sensititvity[i], FAR[i]), xytext=(5, -5), textcoords='offset points', ha='left', va='bottom', fontsize=10)

# Add color bar
cbar = plt.colorbar(sc, ticks=np.arange(len(labels)))
cbar.ax.set_yticklabels(labels)

# Add gridlines
ax.grid(linestyle='dashed', linewidth=0.5)

# Add red arrow pointing towards RP
rp_idx = labels.index('R')
ax.annotate("Best Case \nHigh seizure sensitivity\nat low FA/24h",
            xy=(Seizure_sensititvity[rp_idx], FAR[rp_idx]), xytext=(0.867,240), color='black',
            arrowprops=dict(facecolor='g', edgecolor='g', arrowstyle='simple,tail_width=0.1,head_width=0.5'))

# rp_idx = labels.index('T')
# ax.annotate("",
#             xy=(Seizure_sensititvity[rp_idx], FAR[rp_idx]), xytext=(0.9,450), color='black',
#             arrowprops=dict(facecolor='purple', edgecolor='purple', arrowstyle='simple,tail_width=0.1,head_width=0.5'))

# rp_idx = labels.index('M')
# ax.annotate("        Best Case 2\nHigh seizure sensitivity\n     at High FA/24H",
#             xy=(Seizure_sensititvity[rp_idx], FAR[rp_idx]), xytext=(0.86,400), color='black',
#             arrowprops=dict(facecolor='brown', edgecolor='brown', arrowstyle='simple,tail_width=0.1,head_width=0.5'))



# Show plot
plt.show()
