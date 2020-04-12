import tensorflow as tf
import tensorflow_probability as tfp
import matplotlib.pyplot as plt

tf.keras.backend.set_floatx("float64")

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

model = tf.keras.models.Sequential(
    [
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10),
    ]
)
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

model.compile(optimizer="adam", loss=loss_fn, metrics=["accuracy"])
model.fit(x_train, y_train, epochs=5, verbose=0)
model.evaluate(x_test, y_test, verbose=0)

probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])

print(probability_model(x_test[:5]).numpy())

plt.figure()
plt.imshow(x_test[0, :, :])
plt.show()


def policy(state):
    model = tf.keras.models.Sequential(
        [
            tf.keras.layers.Dense(
                24, input_shape=(4,), activation="relu", name="dense_1", trainable=True
            ),
            tf.keras.layers.Dense(
                24, activation="relu", name="dense_2", trainable=True
            ),
            tf.keras.layers.Dense(
                2, activation="softmax", name="dense_3", trainable=True
            ),
        ]
    )
    action_probs = model(state)  # (1, K)

    action = tfp.distributions.Categorical(
        probs=action_probs, name="action_sampling"
    ).sample(1)
    action = tf.reshape(action, ())  # ()

    # action = action_space.sample()  # a_t
    return action.numpy()
