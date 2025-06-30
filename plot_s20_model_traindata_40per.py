import numpy as np
import matplotlib.pyplot as plt


subjects = [f'S{i}' for i in range(1, 21)] + ['Average']

# 模型名稱（3 種）
models = ['Effortnet', 'CNN', 'STAnet']
colors = ['steelblue', 'brown', 'gray']  # 



data = np.array([
    [0.813, 0.692, 0.632],  # S1
    [0.8, 0.586, 0.536],  # S2
    [0.763, 0.556, 0.596],  # S3
    [0.8, 0.633, 0.673],  # S4
    [0.788, 0.685, 0.666],  # S5
    [0.8, 0.702, 0.682],  # S6
    [0.825,0.613, 0.653],  # S7
    [0.813, 0.582, 0.602],  # S8
    [0.788, 0.673, 0.653],  # S9
    [0.8, 0.654, 0.543],   # S10
    [0.8, 0.586, 0.536],  # S11
    [0.788,0.685, 0.633],  # S12
    [0.8, 0.654, 0.675],  # S13
    [0.813, 0.532, 0.593],  # S14
    [0.813, 0.701, 0.665],  # S15
    [0.813, 0.513, 0.652],  # S16
    [0.825, 0.684, 0.686],  # S17
    [0.788, 0.672, 0.543],  # S18
    [0.813, 0.651, 0.605],  # S19
    [0.788, 0.554, 0.653],  # S20
]) * 100

# Compute the mean
data_mean = np.mean(np.array([data]), axis=0)

# Ensure data_mean is in the same matrix format as data_1
data_mean = np.array(data_mean)

# Display the result
print(data_mean)

data = data_mean


mean_accuracy = np.mean(data, axis=0)
std_accuracy = np.std(data, axis=0)

data = np.vstack([data, mean_accuracy])

fig, ax = plt.subplots(figsize=(10, 3), dpi=200)

bar_width = 0.2  # 
x = np.arange(len(subjects))  # 


for i, (model, color) in enumerate(zip(models, colors)):
    ax.bar(x + i * bar_width, data[:, i], width=bar_width, label=model, color=color, alpha=0.8)


ax.errorbar(x[-1] + np.array([0, bar_width, 2 * bar_width]), mean_accuracy,
            yerr=std_accuracy, fmt='none', ecolor='black', capsize=2, capthick=1.5, elinewidth=1.2)


ax.axhline(y=80, color='black', linestyle='dotted', linewidth=1)


ax.set_xticks(x + 1.5 * bar_width)  
ax.set_xticklabels(subjects, fontsize=8, rotation=45, ha='right')  
ax.set_xlabel("Subject ID", fontsize=10)

ax.set_ylabel("Accuracy (%)", fontsize=10)
ax.set_ylim(50, 100)  #
ax.set_yticks(np.arange(50, 101, 10))


ax.legend(fontsize=8, loc='upper left', frameon=False, ncol=3)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)


plt.subplots_adjust(bottom=0.25, top=0.95, left=0.08, right=0.98)

plt.savefig("output_S20_manual.png")
print("output_S20_manual.png，。")

print("std：")
for model, std in zip(models, std_accuracy):
    print(f"{model}: {std:.2f}%")  # 
print("\nAccuracy：")
for model, mean in zip(models, mean_accuracy):
    print(f"{model}: {mean:.2f}%")