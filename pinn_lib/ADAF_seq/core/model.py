import tensorflow as tf
import numpy as np



class ADAF_Reusable(tf.Module):
    """
    Implements g1/g2/g3 on a fixed normalized grid x in [0, gamma] with Nt points.
    You can update init terms per segment (continuity) and warm-start weights.
    """
    def __init__(self, xgrid_norm, init1=0., init2=0., init3=0.,
                 N_p=100, N_m=100, L=1.0, gamma=0.8, dtype=tf.float32, name=None):
        super().__init__(name=name)
        self.dtype = dtype
        self.L = tf.constant(L, dtype)
        self.gamma = float(gamma)
        self.N_p = int(N_p)
        self.N_m = int(N_m)
        self.PI = tf.constant(np.pi, self.dtype)

        # ---- x_i / Mb,Ma depend only on panels and (L,gamma) ----
        if gamma == 1.0:
            x_i_np = np.linspace(0.0, L, N_p + 1).astype(np.float32)
        else:
            x_i_np = np.concatenate(
                (np.linspace(0, gamma, N_p - 1)[:-1], np.linspace(gamma, 1, 3)),
                axis=0
            ).astype(np.float32)

        self.x_i = tf.constant(x_i_np, dtype=dtype)
        x1 = self.x_i[1:]     # (N_p,)
        x2 = self.x_i[:-1]    # (N_p,)

        # ---- trainable weights ----
        w0 = np.random.uniform(-1.0, 1.0, N_p).astype(np.float32)
        self.W = tf.Variable(w0, dtype=dtype, name="W")

        # ---- init terms: make them assignable per segment ----
        self.init1 = tf.Variable(float(init1), dtype=dtype, trainable=False, name="init1")
        self.init2 = tf.Variable(float(init2), dtype=dtype, trainable=False, name="init2")
        self.init3 = tf.Variable(float(init3), dtype=dtype, trainable=False, name="init3")

        # ---- mode vectors ----
        n = tf.range(1, self.N_m + 1, dtype=dtype)     # (N_m,)
        self.scale  = (2.0 / (n * self.PI))                 # (N_m,)
        self.factor = self.L / (n * self.PI)                # (N_m,)

        k = (n * self.PI) / self.L                          # (N_m,)
        k = k[:, None]                                 # (N_m,1)

        cos_kx1 = tf.cos(k * x1[None, :])              # (N_m,N_p)
        cos_kx2 = tf.cos(k * x2[None, :])
        sin_kx1 = tf.sin(k * x1[None, :])
        sin_kx2 = tf.sin(k * x2[None, :])

        self.Mb = (-cos_kx1 + cos_kx2)                 # (N_m,N_p)
        self.Ma = ( sin_kx1 - sin_kx2)                 # (N_m,N_p)

        # ---- fixed grid cache (xgrid_norm) ----
        x = tf.convert_to_tensor(xgrid_norm, dtype=dtype)   # (Nt,)
        self.x  = x
        self.x2 = tf.square(x)
        self.x3 = tf.pow(x, 3)

        xf = x[None, :] / self.factor[:, None]         # (N_m,Nt)
        self.S = tf.sin(xf)
        self.C = tf.cos(xf)
        self.Cm1 = self.C - 1.0
        self.one_m_C = 1.0 - self.C

        self.f  = self.factor[:, None]                  # (N_m,1)
        self.f2 = tf.square(self.f)                     # (N_m,1)

    def set_inits(self, init1, init2, init3):
        self.init1.assign(tf.cast(init1, self.dtype))
        self.init2.assign(tf.cast(init2, self.dtype))
        self.init3.assign(tf.cast(init3, self.dtype))

    @tf.function(jit_compile=True)
    def coeffs(self):
        bn_num = tf.linalg.matvec(self.Mb, self.W)     # (N_m,)
        an_num = tf.linalg.matvec(self.Ma, self.W)     # (N_m,)
        b_n = self.scale * bn_num
        a_n = self.scale * an_num
        a0  = tf.reduce_mean(self.W)
        return a0, a_n, b_n

    @tf.function(jit_compile=True)
    def g1_from_coeffs(self, a0, a_n, b_n):
        term = 0.5 * (-self.f * b_n[:, None]) * self.Cm1 + 0.5 * (self.f * a_n[:, None]) * self.S
        return 0.5 * a0 * self.x + tf.reduce_sum(term, axis=0) + self.init1

    @tf.function(jit_compile=True)
    def g2_from_coeffs(self, a0, a_n, b_n):
        term1 = 0.5 * (-self.f * b_n[:, None]) * (self.f * self.S - self.x[None, :])
        term2 = 0.5 * (self.f2 * a_n[:, None]) * self.one_m_C
        return (a0 / 4.0) * self.x2 + tf.reduce_sum(term1 + term2, axis=0) + self.init1 * self.x + self.init2

    @tf.function(jit_compile=True)
    def g3_from_coeffs(self, a0, a_n, b_n):
        term1 = 0.5 * (self.f * b_n[:, None]) * (0.5 * self.x2[None, :] + self.f2 * self.Cm1)
        term2 = 0.5 * (self.f2 * a_n[:, None]) * (self.x[None, :] - self.f * self.S)
        return (a0 / 12.0) * self.x3 + tf.reduce_sum(term1 + term2, axis=0) + 0.5 * self.init1 * self.x2 + self.init2 * self.x + self.init3