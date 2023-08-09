
import matplotlib.pyplot as plt

# Sample data
categories = ["None", "jitter", "scaling", "rotation", "crop", "time_warp", "magnitude_warp", "random_guided_warp", "discriminative_guided_warp"]
values = [83.53, 83.68, 83.62, 83.94, 83.42, 83.74, 83.76, 83.56, 83.53]

edge_color = 'black'
edge_width = 2
bar_color = 'skyblue'
# Create a bar plot
plt.bar(categories, values, width=0.5, color = bar_color, edgecolor = edge_color, linewidth=edge_width)

# Add labels and title
plt.xlabel('Augmentation method')
plt.ylabel('Accuracy_mean')
plt.title('Comaparative accuracy for single augmentation factor')
plt.xticks(categories, fontsize=8, rotation = 45, fontstyle='italic', color = 'green') 
plt.tight_layout()
# Save the plot
plt.savefig("/home/abidhasan/Documents/DA_Project/results/bar_plot.png", dpi = 300)