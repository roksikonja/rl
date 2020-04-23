import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp


class ActorReinforce(tf.keras.Model):
    def __init__(self, n_states, n_actions, n_hidden_l1, n_hidden_l2, alpha):
        super(ActorReinforce, self).__init__()
        self.n_states = n_states
        self.n_actions = n_actions

        # Actor
        self.actor_dense_l1 = tf.keras.layers.Dense(
            n_hidden_l1,
            input_shape=(n_states,),
            activation="tanh",
            name="actor_dense_l1",
            trainable=True,
        )
        self.actor_dense_l2 = tf.keras.layers.Dense(
            n_hidden_l2, activation="tanh", name="actor_dense_l2", trainable=True
        )
        self.actor_dense_l3 = tf.keras.layers.Dense(
            n_actions, activation="softmax", name="actor_dense_l3", trainable=True
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
            a1 = self.actor_dense_l1(state)  # (1 , n_hidden_l1)
            a2 = self.actor_dense_l2(a1)  # (1, n_hidden_l2)
            a3 = self.actor_dense_l3(a2)  # (1, n_actions)

            # Probabilities
            self.probs = a3  # (1, n_actions)
            self.log_probs = tf.math.log(a3)  # (1, n_actions)

            # Sample action
            self.action = tfp.distributions.Categorical(probs=self.probs).sample(1)
            self.action = tf.squeeze(self.action)  # ()

            self.action_log_probs = self.log_probs[:, self.action]

        grads = gt.gradient(self.action_log_probs, self.trainable_variables)

        return self.action, grads

    @tf.function(autograph=False)
    def apply_gradients(self, grads, alpha_factor):
        # - for Gradient Ascent
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
