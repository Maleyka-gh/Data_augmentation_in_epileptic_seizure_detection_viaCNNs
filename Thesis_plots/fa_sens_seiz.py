# dfs = [J,R,P,T,M,W,PT,RT,RP,RPT]
import pandas as pd
labels = ['Baseline','J', 'R', 'P', 'T', 'M', 'W', 'PT', 'RT', 'RP', 'RPT']
Seizure_sensititvity = [0.9,0.8,0.9,0.9,0.9,0.85,0.9,0.9,0.9,0.9,0.85]
Sensitivity =  [0.696,0.576,0.728,0.68,0.736,0.664,0.696,0.696,0.728,0.664,0.696]
Specificity = [0.8085,0.9298,0.8579,0.879,0.828,0.914,0.825,0.8622,0.8312,0.9022,0.8923]
FAR = [382,181,351,317,358,242,363,343,337,259,298]
proportion = ['No', '2x', '1x', '4x', '2x', '4x', '2x', '2x', '2x', '4x', '2x']

df = pd.DataFrame({
            'Sen':Sensitivity,
            'Spec':Specificity,
            'SenSeiz': Seizure_sensititvity,
            'prop': proportion,
            'FAR': FAR}, index=labels
            )
import matplotlib.pyplot as plt
import numpy as np

# Create color map
cmap = plt.get_cmap('tab20')
proportion = ['No', '2x', '1x', '4x', '2x', '4x', '2x', '2x', '2x', '4x', '2x']



# Create scatter plot
fig, ax = plt.subplots(figsize=(10, 6))
sc = ax.scatter(Seizure_sensititvity, FAR, c=np.arange(len(labels)), cmap=cmap)

# Add labels and title
ax.set_xlabel('Sensitivity (Seizure)')
ax.set_ylabel('FA/24h')
ax.set_title('Relationship between Sensitivity (Seizure) and FA/24h')
ax.set_xlim(left=None, right=0.92)

# Add labels for each point
# Add labels for each point
for i, label in enumerate(labels):
    text = f"{label} ({proportion[i]})"  # add proportion to label
    if label == 'Baseline' or label == 'RT' or label == 'W' or label == 'R':
        ax.annotate(text, (Seizure_sensititvity[i], FAR[i]), xytext=(-5, 5), textcoords='offset points', ha='right', va='top', fontsize=10)
    else:
        ax.annotate(text, (Seizure_sensititvity[i], FAR[i]), xytext=(5, -5), textcoords='offset points', ha='left', va='bottom', fontsize=10)

# Add color bar
cbar = plt.colorbar(sc, ticks=np.arange(len(labels)))
cbar.ax.set_yticklabels(labels)

# Add gridlines
ax.grid(linestyle='dashed', linewidth=0.5)

# Add red arrow pointing towards RP
rp_idx = labels.index('RP')
ax.annotate("         Best Case \nHigh seizure sensitivity\n      at low FA/24h",
            xy=(Seizure_sensititvity[rp_idx], FAR[rp_idx]), xytext=(0.867,200), color='black',arrowprops=dict(facecolor='black',edgecolor='black', arrowstyle='simple,tail_width=0.1,head_width=0.5'))


# Show plot
plt.show()
