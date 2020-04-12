import gpflow
import numpy as np
import tensorflow as tf
from tensorflow_probability import distributions as tfd


def randomize(model):
    mean = 1
    sigma = 0.01

    model.kern.lengthscales.assign(
        mean + sigma * np.random.normal(size=model.kern.lengthscales.shape)
    )
    model.kern.variance.assign(
        mean + sigma * np.random.normal(size=model.kern.variance.shape)
    )
    if model.likelihood.variance.trainable:
        model.likelihood.variance.assign(mean + sigma * np.random.normal())


class MGPR(gpflow.Module):
    def __init__(self, X, Y, name=None):
        super(MGPR, self).__init__(name)

        self.num_outputs = Y.shape[1]
        self.num_dims = X.shape[1]
        self.num_datapoints = X.shape[0]

        self.models = []
        self.optimizers = []

        self.create_models(X, Y)

        print(gpflow.config.default_float())

    def create_models(self, X, Y):
        """
        Construct a separate GP model for every output/target dimensions, i.e. for every Delta_{t, i}.

        :param X: Data points, state-action pairs. (num_steps, state_dim + control_dim)
        :param Y: Data points, state differences. (num_steps, state_dim)
        :return:
        """
        for i in range(self.num_outputs):
            kern = gpflow.kernels.RBF()
            kern.lengthscales.prior = tfd.Gamma(1.0, 10.0)
            kern.variance.prior = tfd.Gamma(1.5, 2.0)

            model = gpflow.models.GPR(
                data=(X, Y[:, i : i + 1]), kernel=kern, mean_function=None
            )
            model.likelihood.variance.assign(2e-6)
            gpflow.set_trainable(model.likelihood, False)

            self.models.append(model)
            self.optimizers.append(gpflow.optimizers.Scipy())

    def set_XY(self, X, Y):
        for i in range(self.num_outputs):
            self.models[i].X = X
            self.models[i].Y = Y[:, i : i + 1]

    def optimize(self, restarts=1):
        for i, (model, optimizer) in enumerate(zip(self.models, self.optimizers)):
            init_loss = model.training_loss()
            init_lml = model.log_marginal_likelihood()
            optimizer.minimize(
                model.training_loss,
                model.trainable_variables,
                options=dict(maxiter=100),
            )
            opt_loss = model.training_loss()
            opt_lml = model.log_marginal_likelihood()
            print(
                "MODEL{:>10}\t\tloss:{:>10.3f}\t\t{:>10.3f}\t\tlml:{:>10.3f}\t\t{:>10.3f}".format(
                    i, init_loss, opt_loss, init_lml, opt_lml
                )
            )

    def predict_on_noisy_inputs(self, m, s):
        iK, beta = self.calculate_factorizations()
        return self.predict_given_factorizations(m, s, iK, beta)

    def calculate_factorizations(self):
        K = self.K(self.X)
        batched_eye = tf.eye(
            tf.shape(self.X)[0], batch_shape=[self.num_outputs], dtype=np.float64
        )
        L = tf.cholesky(K + self.noise[:, None, None] * batched_eye)
        iK = tf.cholesky_solve(L, batched_eye)
        Y_ = tf.transpose(self.Y)[:, :, None]
        # Why do we transpose Y? Maybe we need to change the definition of self.Y() or beta?
        beta = tf.cholesky_solve(L, Y_)[:, :, 0]
        return iK, beta

    def predict_given_factorizations(self, m, s, iK, beta):
        """
        Approximate GP regression at noisy inputs via moment matching
        IN: mean (m) (row vector) and (s) variance of the state
        OUT: mean (M) (row vector), variance (S) of the action
             and inv(s)*input-ouputcovariance
        """

        s = tf.tile(s[None, None, :, :], [self.num_outputs, self.num_outputs, 1, 1])
        inp = tf.tile(self.centralized_input(m)[None, :, :], [self.num_outputs, 1, 1])

        # Calculate M and V: mean and inv(s) times input-output covariance
        iL = tf.matrix_diag(1 / self.lengthscales)
        iN = inp @ iL
        B = iL @ s[0, ...] @ iL + tf.eye(self.num_dims, dtype=np.float64)

        # Redefine iN as in^T and t --> t^T
        # B is symmetric so its the same
        t = tf.linalg.transpose(
            tf.matrix_solve(B, tf.linalg.transpose(iN), adjoint=True),
        )

        lb = tf.exp(-tf.reduce_sum(iN * t, -1) / 2) * beta
        tiL = t @ iL
        c = self.variance / tf.sqrt(tf.linalg.det(B))

        M = (tf.reduce_sum(lb, -1) * c)[:, None]
        V = tf.matmul(tiL, lb[:, :, None], adjoint_a=True)[..., 0] * c[:, None]

        # Calculate S: Predictive Covariance
        R = s @ tf.matrix_diag(
            1 / tf.square(self.lengthscales[None, :, :])
            + 1 / tf.square(self.lengthscales[:, None, :])
        ) + tf.eye(self.num_dims, dtype=np.float64)

        X = inp[None, :, :, :] / tf.square(self.lengthscales[:, None, None, :])
        X2 = -inp[:, None, :, :] / tf.square(self.lengthscales[None, :, None, :])
        Q = tf.matrix_solve(R, s) / 2
        Xs = tf.reduce_sum(X @ Q * X, -1)
        X2s = tf.reduce_sum(X2 @ Q * X2, -1)
        maha = (
            -2 * tf.matmul(X @ Q, X2, adjoint_b=True)
            + Xs[:, :, :, None]
            + X2s[:, :, None, :]
        )
        #
        k = tf.log(self.variance)[:, None] - tf.reduce_sum(tf.square(iN), -1) / 2
        L = tf.exp(k[:, None, :, None] + k[None, :, None, :] + maha)
        S = (
            tf.tile(beta[:, None, None, :], [1, self.num_outputs, 1, 1])
            @ L
            @ tf.tile(beta[None, :, :, None], [self.num_outputs, 1, 1, 1])
        )[:, :, 0, 0]

        diagL = tf.transpose(tf.linalg.diag_part(tf.transpose(L)))
        S = S - tf.diag(tf.reduce_sum(tf.multiply(iK, diagL), [1, 2]))
        S = S / tf.sqrt(tf.linalg.det(R))
        S = S + tf.diag(self.variance)
        S = S - M @ tf.transpose(M)

        return tf.transpose(M), S, tf.transpose(V)

    def centralized_input(self, m):
        return self.X - m

    def K(self, X1, X2=None):
        return tf.stack([model.kern.K(X1, X2) for model in self.models])

    @property
    def Y(self):
        return tf.concat([model.Y.parameter_tensor for model in self.models], axis=1)

    @property
    def X(self):
        return self.models[0].X.parameter_tensor

    @property
    def lengthscales(self):
        return tf.stack(
            [model.kern.lengthscales.constrained_tensor for model in self.models]
        )

    @property
    def variance(self):
        return tf.stack(
            [model.kern.variance.constrained_tensor for model in self.models]
        )

    @property
    def noise(self):
        return tf.stack(
            [model.likelihood.variance.constrained_tensor for model in self.models]
        )
