import numpy as np
x = np.arange(100).reshape(20, 5)
print('Before Overlapping 2D array')
print(x)


def get_strides(a, L, ov):
    out = []
    for i in range(0, a.shape[0] - L + 1, L - ov):
        out.append(a[i:i + L, :])

    return np.array(out)

# I want a 3d array with fixed window size=5 and the formation is processed by "slided window algorithm" with stride of 3

print('After Overlapping')
new_array = get_strides(x, 5, 3)
print(new_array)
print(new_array.shape)
