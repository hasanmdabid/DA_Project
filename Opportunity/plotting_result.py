import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

classes = ['Jittering', 'Rotation', 'Time_warping', 'Slicing', 'Convolve']
df = pd.DataFrame([[57.05, 57.76, 59.39, 57.02, 62.30], [54.38, 60.75, 56.82, 57.70, 58.65], [61.71, 56.77, 64.32, 59.19, 60.56], [62.43, 58.36, 62.25, 60.65, 60.59], [60.97, 58.89, 61.38, 60.49, 58.98]], classes, classes)
print(df)
colormap = sns.color_palette("Blues")
# plotting the heatmap
sns.set(font_scale=.8)
hm = sns.heatmap(data=df, annot=True, linewidth=.5, cmap=colormap, square=True, )
# displaying the plotted heatmap
# plt.title('Linear Evaluation(Opportunity F1-Macro) under individual or composition of Random Transformation Data Augmentation', fontsize = 20) # title with fontsize 20
plt.xlabel('2nd Transformation', fontsize=12) # x-axis label with fontsize 15
plt.ylabel('1st Transformation', fontsize=12) # y-axis label with fontsize 15
plt.savefig('output.png', dpi=500)
#sns.plt.show()
