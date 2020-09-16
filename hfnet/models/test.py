import tensorflow as tf
import numpy as np

a = np.random.randint(1, 100, [2, 3, 4])
print(a)
print("----------------")

for i in range(3):
    print(a.max(i))
    print("----------------")
# x = tf.nn.l2_normalize(x, dim=-1)