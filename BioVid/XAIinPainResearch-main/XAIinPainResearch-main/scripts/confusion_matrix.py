import numpy as np
import matplotlib.pyplot as plt

def confusion_matrix(actual, predicted):
	"""Function to calculate a confusion matrix. confusion_matrix([1,2,3], [1,1,1]) = [[1, 1, 1], [0, 0, 0], [0, 0, 0]]

	Parameters
	----------
	actual: list. List with the actual labels.
	predicted: list. List with the predicted labels.

	Returns
	-------
	list: Unique labels found given in the acutal and predicted elements
	list of lists: confusion matrix
	"""
	# Assert both list have the same length
	assert len(actual) == len(predicted)
	# Concat actual and predicted
	all_elements = list(actual) + list(predicted)
	# Retrieve unique elements in actual
	unique = set(all_elements)
	# Retrieve number of classes
	num_classes = len(np.unique(all_elements))

	# Create empty matrix
	matrix = [([0]*num_classes) for i in range(num_classes)]

	# Create an  lookup matrix: For each class we save the index in the matrix
	lookup = {}
	for i, value in enumerate(unique):
		lookup[value] = i

	# Create the matrix
	for i in range(len(actual)):
		x = lookup[actual[i]]
		y = lookup[predicted[i]]
		matrix[x][y] += 1

	# Return the elements
	return unique, matrix

def visualize_confusion_matrix(unique, matrix, title = "Confusion matrix", show= True, ax= None):
	"""Function to visualize a confusion matrix. Plt is used to create a figure displaying the confusion matrix.
	Inputs can come from confusion_matrix().

	Parameters
	----------
	unique: list. List with unique labels.
	matrix: list of lists. Describes the condusion matrix.
	title: String. A string describing the title of the figure. Default value is set to "Confusion matrix".
	show: Bool. Whether to call plt.show directly or not.
	ax: axes. Axe to plot on. If none is given, create new. Defaults to None.

	Returns
	-------
	Returns None if either unique or matrix is None.
	"""

	if (unique is None or matrix is None):
		return None
		
	cm = np.array(matrix)
	if ax is None:
		fig, ax = plt.subplots()
	im = ax.imshow(cm, interpolation='nearest', cmap= "Blues")
	ax.figure.colorbar(im, ax=ax)
	thresh = (cm.max() - cm.min()) / 2. + cm.min()

	ax.set(xticks=np.arange(cm.shape[1]),
		   yticks=np.arange(cm.shape[0]),
		   xticklabels=unique, yticklabels=unique,
		   title = title,
		   ylabel='True label',
		   xlabel='Predicted label')

	for i in range(cm.shape[0]):
		for j in range(cm.shape[1]):
			ax.text(j, i, format(cm[i, j]),
					ha="center", va="center",
					color="white" if cm[i, j] > thresh else "black")

	if show:
		plt.show()


if __name__ == "__main__":
	"""Main function.
	"""
	visualize_confusion_matrix(unique= ["No pain", "Pain"], matrix= [[383, 33], [46, 370]])