import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp


class ActorReinforce(tf.keras.Model):
    def __init__(self, n_states, n_actions, n_hidden_l1, n_hidden_l2, alpha):
        super(ActorReinforce, self).__init__()
        # Policy
        self.dense_l1 = tf.keras.layers.Dense(
            n_hidden_l1,
            input_shape=(n_states,),
            activation="tanh",
            name="dense_l1",
            trainable=True,
        )
        self.dense_l2 = tf.keras.layers.Dense(
            n_hidden_l2, activation="tanh", name="dense_l2", trainable=True
        )
        self.dense_l3 = tf.keras.layers.Dense(
            n_actions, activation="softmax", name="dense_l3", trainable=True
        )

        self.probs, self.log_probs, self.action, self.action_log_probs = (
            None,
            None,
            None,
            None,
        )

        # Training
        self.alpha, self.optimizer = alpha, tf.keras.optimizers.SGD()

    @tf.function(autograph=False)
    def act(self, state):
        with tf.GradientTape() as gt:
            # Policy DNN
            a1 = self.dense_l1(state)  # (n_batch, n_hidden_l1)
            a2 = self.dense_l2(a1)  # (n_batch, n_hidden_l2)
            a3 = self.dense_l3(a2)  # (n_batch, n_actions)

            # Probabilities
            self.probs = a3  # (n_batch, n_actions)
            self.log_probs = tf.math.log(a3)  # (n_batch, n_actions)

            # Sample action
            self.action = tfp.distributions.Categorical(probs=self.probs).sample(1)
            self.action = tf.squeeze(self.action)  # (n_batch, ) or ()

            self.action_log_probs = self.log_probs[:, self.action]

        grads = gt.gradient(self.action_log_probs, self.trainable_variables)

        return self.action, grads

    @tf.function(autograph=False)
    def apply_gradients(self, grads, alpha_factor):
        # - for Gradient Ascent
        self.optimizer.learning_rate = -self.alpha * alpha_factor
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))


class ActorRandom(tf.keras.Model):
    def __init__(self, n_actions):
        super(ActorRandom, self).__init__()
        self.n_actions = n_actions

        self.action = None

    @tf.function(autograph=False)
    def act(self, state):
        logits = np.ones(shape=(state.shape[0], self.n_actions))
        self.action = tfp.distributions.Categorical(logits=logits).sample(1)
        self.action = tf.squeeze(self.action)
        return self.action, []

    def apply_gradients(self, grads=None, alpha_factor=None):
        pass


class ActorBaseline(tf.keras.Model):
    def __init__(self, n_states, n_actions, n_hidden_l1, n_hidden_l2, alpha_a, alpha_b):
        super(ActorBaseline, self).__init__()
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

        # Baseline
        self.baseline_dense_l1 = tf.keras.layers.Dense(
            n_hidden_l1,
            input_shape=(n_states,),
            activation="tanh",
            name="baseline_dense_l1",
            trainable=True,
        )
        self.baseline_dense_l2 = tf.keras.layers.Dense(
            n_hidden_l2, activation="tanh", name="baseline_dense_l2", trainable=True
        )
        self.baseline_dense_l3 = tf.keras.layers.Dense(
            1, activation=None, name="baseline_dense_l3", trainable=True
        )

        self.probs, self.log_probs, = None, None
        self.action, self.action_log_probs, self.value = None, None, None

        # Training
        self.alpha_a, self.alpha_b, self.optimizer = (
            alpha_a,
            alpha_b,
            tf.keras.optimizers.SGD(),
        )

        # Initialize variables
        _ = self.act(np.zeros((1, n_states)).astype(np.float32))
        _ = self.state_value(np.zeros((1, n_states)).astype(np.float32))

        self.actor_vars = None, None
        for var in self.trainable_variables:
            print(var.name, var.shape)

    @tf.function(autograph=False)
    def act(self, state):
        with tf.GradientTape() as gt:
            # Policy DNN
            a1 = self.actor_dense_l1(state)  # (n_batch, n_hidden_l1)
            a2 = self.actor_dense_l2(a1)  # (n_batch, n_hidden_l2)
            a3 = self.actor_dense_l3(a2)  # (n_batch, n_actions)

            # Probabilities
            self.probs = a3  # (n_batch, n_actions)
            self.log_probs = tf.math.log(a3)  # (n_batch, n_actions)

            # Sample action
            self.action = tfp.distributions.Categorical(probs=self.probs).sample(1)
            self.action = tf.squeeze(self.action)  # (n_batch, ) or ()

            self.action_log_probs = self.log_probs[:, self.action]

        grads = gt.gradient(self.action_log_probs, self.actor_variables())

        return self.action, grads

    @tf.function(autograph=False)
    def state_value(self, state):
        assert state.shape[0] == 1
        with tf.GradientTape() as gt:
            a1 = self.baseline_dense_l1(state)  # (n_batch, n_hidden_l1)
            a2 = self.baseline_dense_l2(a1)  # (n_batch, n_hidden_l2)
            a3 = self.baseline_dense_l3(a2)  # (n_batch, 1)

            # Value function
            self.value = tf.squeeze(a3)  # (n_batch, ) or ()

        grads = gt.gradient(self.value, self.baseline_variables())

        return self.value, grads

    @tf.function(autograph=False)
    def apply_gradients_b(self, grads, alpha_factor):
        # - for Gradient Ascent
        self.optimizer.learning_rate = -self.alpha_b * alpha_factor
        self.optimizer.apply_gradients(zip(grads, self.baseline_variables()))

    @tf.function(autograph=False)
    def apply_gradients_a(self, grads, alpha_factor):
        # - for Gradient Ascent
        self.optimizer.learning_rate = -self.alpha_a * alpha_factor
        self.optimizer.apply_gradients(zip(grads, self.actor_variables()))

    def actor_variables(self):
        return [var for var in self.trainable_variables if "actor" in var.name]

    def baseline_variables(self):
        return [var for var in self.trainable_variables if "baseline" in var.name]


class ActorConstant(tf.keras.Model):
    def __init__(self, n_states, n_actions, n_hidden_l1, n_hidden_l2, alpha_a, alpha_b):
        super(ActorConstant, self).__init__()
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

        # Baseline
        self.baseline = tf.Variable(initial_value=1.0, name="baseline")

        self.probs, self.log_probs, = None, None
        self.action, self.action_log_probs, self.value = None, None, None

        # Training
        self.alpha_a, self.alpha_b, self.optimizer = (
            alpha_a,
            alpha_b,
            tf.keras.optimizers.SGD(),
        )

        # Initialize variables
        _ = self.act(np.zeros((1, n_states)).astype(np.float32))
        _ = self.state_value(np.zeros((1, n_states)).astype(np.float32))

        self.actor_vars = None, None
        for var in self.trainable_variables:
            print(var.name, var.shape)

    @tf.function(autograph=False)
    def act(self, state):
        with tf.GradientTape() as gt:
            # Policy DNN
            a1 = self.actor_dense_l1(state)  # (n_batch, n_hidden_l1)
            a2 = self.actor_dense_l2(a1)  # (n_batch, n_hidden_l2)
            a3 = self.actor_dense_l3(a2)  # (n_batch, n_actions)

            # Probabilities
            self.probs = a3  # (n_batch, n_actions)
            self.log_probs = tf.math.log(a3)  # (n_batch, n_actions)

            # Sample action
            self.action = tfp.distributions.Categorical(probs=self.probs).sample(1)
            self.action = tf.squeeze(self.action)  # (n_batch, ) or ()

            self.action_log_probs = self.log_probs[:, self.action]

        grads = gt.gradient(self.action_log_probs, self.actor_variables())

        return self.action, grads

    @tf.function(autograph=False)
    def state_value(self, state=None):
        assert state.shape[0]
        with tf.GradientTape() as gt:
            # Value function
            self.value = tf.squeeze(self.baseline)  # (n_batch, ) or ()

        grads = gt.gradient(self.value, self.baseline)

        return self.value, grads

    @tf.function(autograph=False)
    def apply_gradients_b(self, grads, alpha_factor):
        # - for Gradient Ascent
        self.optimizer.learning_rate = -self.alpha_b * alpha_factor
        self.optimizer.apply_gradients([(grads, self.baseline)])

    @tf.function(autograph=False)
    def apply_gradients_a(self, grads, alpha_factor):
        # - for Gradient Ascent
        self.optimizer.learning_rate = -self.alpha_a * alpha_factor
        self.optimizer.apply_gradients(zip(grads, self.actor_variables()))

    def actor_variables(self):
        return [var for var in self.trainable_variables if "actor" in var.name]
