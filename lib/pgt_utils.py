import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp


class ActorReinforce(tf.keras.Model):
    def __init__(self, n_states, n_actions, n_hidden, alpha):
        super(ActorReinforce, self).__init__()
        self.n_states = n_states
        self.n_actions = n_actions

        # Actor
        self.layer_input = tf.keras.layers.Dense(
            n_hidden[0],
            input_shape=(n_states,),
            activation="relu",
            name="actor_layer_input",
            trainable=True,
        )

        self.layers_hidden = []
        for i, n in enumerate(n_hidden[1:]):
            self.layers_hidden.append(
                tf.keras.layers.Dense(
                    n, activation="relu", name=f"actor_layer_hidden_{i}", trainable=True
                )
            )

        self.layer_output = tf.keras.layers.Dense(
            n_actions, activation="softmax", name="actor_layer_output", trainable=True
        )

        self.probs, self.log_probs, = None, None
        self.action, self.action_log_probs = None, None

        # Training
        self.alpha = alpha
        self.optimizer = tf.keras.optimizers.SGD()

        # Initialize variables
        _ = self.act(np.zeros((1, n_states)).astype(np.float32))

        for var in self.trainable_variables:
            print(var.name, var.shape)

    @tf.function(autograph=False)
    def act(self, state):
        with tf.GradientTape() as gt:
            # Policy DNN
            a = self.layer_input(state)  # (1 , n_hidden_l1)
            for layer in self.layers_hidden:
                a = layer(a)  # (1, n_hidden)

            a = self.layer_output(a)  # (1, n_actions)

            # Probabilities
            self.probs = a  # (1, n_actions)
            self.log_probs = tf.math.log(self.probs)  # (1, n_actions)

            # Sample action
            self.action = tfp.distributions.Categorical(probs=self.probs).sample(1)
            self.action = tf.squeeze(self.action)  # ()

            self.action_log_probs = self.log_probs[:, self.action]

        grads = gt.gradient(self.action_log_probs, self.trainable_variables)

        return self.action, grads

    @tf.function(autograph=False)
    def apply_gradients(self, grads, alpha_factor):
        # - for Gradient Ascent
        # grads = [tf.clip_by_norm(grad, 1.0) for grad in grads]

        self.optimizer.learning_rate = -self.alpha * alpha_factor
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

    @tf.function(autograph=False)
    def initialize_gradients(self):
        _, grads = self.act(np.zeros((1, self.n_states)).astype(np.float32))
        return [tf.zeros_like(grad) for grad in grads]


def pgt_gradients(t_grads, gradients, returns, gamma):
    for g in range(len(t_grads)):
        t_grads[g] = t_grads[g] + tf.add_n(
            [
                grads[g] * returns[t] * tf.math.pow(gamma, t)
                for t, grads in enumerate(gradients)
            ]
        )

    return t_grads
