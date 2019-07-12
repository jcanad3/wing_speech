import numpy as np

for i in range(300):
	a = np.random.randn(5, 160, 40)
	np.save(str(i) + '.npy', a)
