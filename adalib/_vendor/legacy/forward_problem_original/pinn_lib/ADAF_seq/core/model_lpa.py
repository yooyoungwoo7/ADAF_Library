import numpy as np
import tensorflow as tf
import sympy as sp


def get_Legendre_coefs(order=1, n_panel=10):
    x = sp.symbols('x')
    P = sp.legendre(order, x)
    P_int = sp.integrate(P, x)

    inds = np.linspace(-1, 1, n_panel + 1)
    coefs = np.array([P_int.subs(x, ind) for ind in inds], dtype='float32')
    coefs = coefs[1:] - coefs[:-1]
    coefs *= (2. * order + 1.) / 2.
    return coefs


def Leg_Poly(x, order):
    if order == 1:
        return x
    elif order == 2:
        return 0.5 * (3. * tf.math.square(x) - 1.)
    elif order == 3:
        return 0.5 * (5. * tf.math.pow(x, 3) - 3. * x)
    elif order == 4:
        return (1. / 8.) * (35. * tf.math.pow(x, 4) - 30. * tf.math.square(x) + 3.)
    elif order == 5:
        return (1. / 8.) * (63. * tf.math.pow(x, 5) - 70. * tf.math.pow(x, 3) + 15. * x)
    elif order == 6:
        return (1. / 16.) * (
            231. * tf.math.pow(x, 6)
            - 315. * tf.math.pow(x, 4)
            + 105. * tf.math.pow(x, 2)
            - 5.
        )


class LPA(tf.keras.layers.Layer):
    def __init__(self, order=3, N_p=10, DTYPE='float32', kernel_regularizer=None, name=None):
        super(LPA, self).__init__(name=name)

        self.N_p = N_p
        self.order = order
        self.DTYPE = tf.as_dtype(DTYPE)
        self.kernel_regularizer = tf.keras.regularizers.get(kernel_regularizer)

        coefs_np = np.array(
            [get_Legendre_coefs(i, N_p) for i in range(1, order + 1)],
            dtype=self.DTYPE.as_numpy_dtype
        )
        self.coefs = tf.constant(coefs_np, dtype=self.DTYPE)

    def build(self, input_shape):
        self.W_i = self.add_weight(
            name='W_i',
            shape=(self.N_p,),
            initializer='random_normal',
            regularizer=self.kernel_regularizer,
            trainable=True,
            dtype=self.DTYPE
        )

    def call(self, inputs):
        if inputs.dtype != self.DTYPE:
            inputs = tf.cast(inputs, self.DTYPE)

        Am = tf.tensordot(self.coefs, self.W_i, axes=1)
        out = tf.reduce_mean(self.W_i)

        for i in range(self.order):
            out += Leg_Poly(inputs, i + 1) * Am[i]

        return out