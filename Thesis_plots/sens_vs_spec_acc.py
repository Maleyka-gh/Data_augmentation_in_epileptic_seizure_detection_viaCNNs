import pandas as pd
labels = ['Baseline','J', 'R', 'P', 'T', 'M', 'W', 'PT', 'RT', 'RP', 'RPT']
Seizure_sensititvity = [0.8, 0.7, 0.85, 0.85 , 0.95,0.8,0.85,0.85,0.85,0.85,0.8]
Sensitivity =  [0.6, 0.42,0.56, 0.5866, 0.5866,0.46,0.66,0.5733,0.6,0.64,0.6466]
Specificity =[0.8057, 0.9046, 0.9007,0.8355,0.6211, 0.9486,0.6437,0.668,0.859,0.88281,0.8265]
FAR = [453, 228, 293,422,551,213,540,553,297,337,345]

df = pd.DataFrame({
            'Sen':Sensitivity,
            'Spec':Specificity,
            'SenSeiz': Seizure_sensititvity,
            'FAR': FAR}, index=labels)


import matplotlib.pyplot as plt

# Set font size for better readability
plt.rcParams.update({'font.size': 12})

# Create subplots
fig, ax1 = plt.subplots(figsize=(10, 6))

# Plot sensitivity on ax1
ax1.plot(labels, Sensitivity, 'o--', label='Sensitivity', linewidth=1)
ax1.set_xlabel('Techniques')
ax1.set_ylabel('Sensitivity')
ax1.set_ylim([0.2, 1])
ax1.tick_params(axis='y', labelcolor='blue')

# Create a secondary y-axis for specificity
ax2 = ax1.twinx()

# Plot specificity on ax2
ax2.plot(labels, Specificity, 'o--', color='green', label='Specificity', linewidth=1)
ax2.set_ylabel('Specificity')
ax2.set_ylim([0.2, 1.0])
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
    if 'M' in label or 'T' in label:
      ax1.annotate(label, (i, Sensitivity[i]), textcoords="offset points", xytext=(0, -15), ha='center', fontsize=10)
    else:
      ax1.annotate(label, (i, Sensitivity[i]), textcoords="offset points", xytext=(0, 10), ha='center', fontsize=10)

# Add text annotation for each point
for i, label in enumerate(labels):
    if 'W' in label:
      ax1.annotate(label, (i, Specificity[i]), textcoords="offset points", xytext=(0, -15), ha='center', fontsize=10)
    else:
      ax1.annotate(label, (i, Specificity[i]), textcoords="offset points", xytext=(0, 10), ha='center', fontsize=10)

plt.show()
