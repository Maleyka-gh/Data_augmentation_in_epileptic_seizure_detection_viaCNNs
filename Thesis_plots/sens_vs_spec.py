import matplotlib.pyplot as plt

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

# Set font size for better readability
plt.rcParams.update({'font.size': 12})

# Create subplots
fig, ax1 = plt.subplots(figsize=(10, 6))

# Plot sensitivity on ax1
ax1.plot(labels, Sensitivity, 'o--', label='Sensitivity', linewidth=1)
ax1.set_xlabel('Techniques')
ax1.set_ylabel('Sensitivity')
ax1.set_ylim([0.5, 1])
ax1.tick_params(axis='y', labelcolor='blue')

# Create a secondary y-axis for specificity
ax2 = ax1.twinx()

# Plot specificity on ax2
ax2.plot(labels, Specificity, 'o--', color='green', label='Specificity', linewidth=1)
ax2.set_ylabel('Specificity')
ax2.set_ylim([0.5, 1.0])
ax2.tick_params(axis='y', labelcolor='green')

# Add legend and title
fig.legend(loc='upper center', ncol=2, fontsize=12)
plt.title('Sensitivity and Specificity for each Technique', fontsize='xx-large')

# Add gridlines
ax1.grid(linestyle='dashed', linewidth=0.5)
ax2.grid(linestyle='dashed', linewidth=0.5)

# Add horizontal line for baseline
ax1.axhline(y=Sensitivity[0], color='black', linestyle='dashed', linewidth=1)
ax2.axhline(y=Specificity[0], color='black', linestyle='dashed', linewidth=1)

# Add vertical lines to separate baseline from other algorithms
for i, label in enumerate(labels):
    if i == 0:
        continue
    plt.axvline(i - 0.5, color='gray', linestyle='dashed', linewidth=0.5)

# Add text annotation for each point
for i, label in enumerate(labels):
    ax1.annotate(label, (i, Sensitivity[i]), textcoords="offset points", xytext=(0, 10), ha='center', fontsize=10)

# Add text annotation for each point
for i, label in enumerate(labels):
    ax1.annotate(label, (i, Specificity[i]), textcoords="offset points", xytext=(0, 10), ha='center', fontsize=10)

plt.show()
