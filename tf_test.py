import numpy as np
import tensorflow as tf

s = tf.ones((5, 4))

alpha = 0.01
gamma = 0.99

optimizer = tf.keras.optimizers.SGD(learning_rate=alpha)
policy = tf.keras.Sequential([tf.keras.layers.Dense(2, input_shape=(4, ), name="dense_1")])

with tf.GradientTape(watch_accessed_variables=True) as gt:
    gt.watch(policy.trainable_weights)
    pi = policy(s)[:, 0]

    print(pi[:, 0])
    for w in policy.trainable_weights:
        print(w.name, w.shape)

grad_pi = gt.gradient(pi, policy.trainable_weights)

# optimizer.apply_gradients(zip(grad_pi, policy.trainable_variables))
print(grad_pi)
# for grad in grad_pi:
#     print(grad, grad.shape)
