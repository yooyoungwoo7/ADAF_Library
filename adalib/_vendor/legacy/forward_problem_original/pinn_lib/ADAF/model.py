import tensorflow as tf
import numpy as np


class ADAFModel():
    def __init__(self, ic , N_p = 40, N_m = 20, L = 1.0, gamma=1.0, model_num=0,dtype = 'float32'):
        self.dtype = dtype
        self.L = L        
        self.N_p = N_p
        self.N_m = N_m   
        self.model_num = model_num

        if gamma == 1.0:
            self.x_i = np.linspace(0, self.L, N_p+1).astype(dtype)
        else:
            self.x_i = np.concatenate((np.linspace(0, gamma, N_p-1)[:-1],np.linspace(gamma,1,3)),axis=0).astype(dtype)
        self.W_i = np.random.uniform(-1.0, 1.0, self.N_p).astype(dtype)
        self.W_i = tf.Variable(self.W_i)                           
        self.init1 = tf.constant(0.0, dtype=dtype)
        self.init2 = tf.constant(ic,  dtype=dtype)
    
    def out_bn(self, n, x_1, x_2):                
        sum_1 = -tf.math.cos(n*np.pi/self.L* x_1)
        sum_2 = tf.math.cos(n*np.pi/self.L* x_2)
        
        b_n = self.W_i *(sum_1 + sum_2)
        b_n = tf.reduce_sum(b_n)
        b_n = (2./ (n*np.pi))*b_n
        return b_n
    def out_an(self, n, x_1, x_2):
        if n == 0:
            a_n = tf.reduce_sum(self.W_i)
            a_n = a_n/self.N_p
        else:
            sum_1 = tf.math.sin(n*np.pi/self.L* x_1)
            sum_2 = -tf.math.sin(n*np.pi/self.L* x_2)
            a_n = self.W_i * (sum_1 + sum_2)
            a_n = tf.reduce_sum(a_n)
            a_n = (2./(n*np.pi)) * a_n
        return a_n        
    def out_g_x_1(self, x):
        # --- dtype 통일 (문자열 dtype 쓰는 경우를 그대로 존중) ---
        x = tf.convert_to_tensor(x)
        if x.dtype != tf.as_dtype(self.dtype):
            x = tf.cast(x, tf.as_dtype(self.dtype))

        # --- x_i를 tensor로 변환 (매 호출마다 변환되는 걸 피하려면 __init__에서 tf.constant로 만들어두는 게 더 좋음) ---
        x_i = tf.convert_to_tensor(self.x_i, dtype=tf.as_dtype(self.dtype))
        x_1 = x_i[1:]   # (N_p,)
        x_2 = x_i[:-1]  # (N_p,)

        # --- n 벡터: (N_m,) ---
        n = tf.range(1, self.N_m + 1, dtype=tf.as_dtype(self.dtype))  # float로 둬야 브로드캐스팅 편함
        pi = tf.constant(np.pi, dtype=tf.as_dtype(self.dtype))

        # --- 공통 상수/스케일 ---
        # factor(n) = L/(n*pi)  -> (N_m,)
        factor = tf.constant(self.L, dtype=tf.as_dtype(self.dtype)) / (n * pi)

        # =========================
        # 1) a_n, b_n을 벡터로 한 번에
        # =========================
        # cos/sin 안쪽: (N_m, 1) * (1, N_p) -> (N_m, N_p)
        # k = n*pi/L
        k = (n * pi) / tf.constant(self.L, dtype=tf.as_dtype(self.dtype))   # (N_m,)
        k = k[:, None]  # (N_m,1)

        # (N_m, N_p)
        cos_kx1 = tf.cos(k * x_1[None, :])
        cos_kx2 = tf.cos(k * x_2[None, :])
        sin_kx1 = tf.sin(k * x_1[None, :])
        sin_kx2 = tf.sin(k * x_2[None, :])

        # (N_p,)
        W = tf.cast(self.W_i, tf.as_dtype(self.dtype))[None, :]  # (1, N_p)

        # b_n = (2/(n*pi)) * sum_i W_i * (-cos(k*x1_i) + cos(k*x2_i))
        bn_num = tf.reduce_sum(W * (-cos_kx1 + cos_kx2), axis=1)  # (N_m,)
        b_n = (2.0 / (n * pi)) * bn_num                            # (N_m,)

        # a_n = (2/(n*pi)) * sum_i W_i * ( sin(k*x1_i) - sin(k*x2_i))
        an_num = tf.reduce_sum(W * (sin_kx1 - sin_kx2), axis=1)    # (N_m,)
        a_n = (2.0 / (n * pi)) * an_num                             # (N_m,)

        # a_0 (스칼라)
        a0 = tf.reduce_sum(tf.cast(self.W_i, tf.as_dtype(self.dtype))) / tf.cast(self.N_p, tf.as_dtype(self.dtype))

        # =========================
        # 2) g_x 계산 (n에 대해 reduce_sum)
        # =========================
        # x/factor: (N_t,) / (N_m,) -> (N_m, N_t)
        xf = x[None, :] / factor[:, None]  # (N_m, N_t)

        term1 = 0.5 * (-factor[:, None] * b_n[:, None]) * (tf.cos(xf) - 1.0)  # (N_m, N_t)
        term2 = 0.5 * ( factor[:, None] * a_n[:, None]) * (tf.sin(xf))        # (N_m, N_t)

        g_x = tf.zeros_like(x, dtype=tf.as_dtype(self.dtype))
        g_x += 0.5 * a0 * x
        g_x += tf.reduce_sum(term1 + term2, axis=0)  # sum over n -> (N_t,)

        # init1 더하기
        g_x += tf.cast(self.init1, tf.as_dtype(self.dtype))
        return g_x

    def out_g_x_2(self, x):
        # --- dtype 통일 ---
        x = tf.convert_to_tensor(x)
        if x.dtype != tf.as_dtype(self.dtype):
            x = tf.cast(x, tf.as_dtype(self.dtype))

        # --- x_i 텐서화 (가능하면 __init__에서 tf.constant로 고정 추천) ---
        x_i = tf.convert_to_tensor(self.x_i, dtype=tf.as_dtype(self.dtype))
        x_1 = x_i[1:]   # (N_p,)
        x_2 = x_i[:-1]  # (N_p,)

        # --- n 벡터: (N_m,) ---
        n = tf.range(1, self.N_m + 1, dtype=tf.as_dtype(self.dtype))
        pi = tf.constant(np.pi, dtype=tf.as_dtype(self.dtype))

        # --- 공통 상수/스케일 ---
        # factor(n) = L/(n*pi)  -> (N_m,)
        Lc = tf.constant(self.L, dtype=tf.as_dtype(self.dtype))
        factor = Lc / (n * pi)

        # =========================
        # 1) a_n, b_n을 벡터로 한 번에
        # =========================
        # k = n*pi/L : (N_m,)
        k = (n * pi) / Lc
        k = k[:, None]  # (N_m,1)

        # (N_m, N_p)
        cos_kx1 = tf.cos(k * x_1[None, :])
        cos_kx2 = tf.cos(k * x_2[None, :])
        sin_kx1 = tf.sin(k * x_1[None, :])
        sin_kx2 = tf.sin(k * x_2[None, :])

        # weights: (1, N_p)
        W = tf.cast(self.W_i, tf.as_dtype(self.dtype))[None, :]

        # b_n = (2/(n*pi)) * sum_i W_i * (-cos(k*x1_i) + cos(k*x2_i))  -> (N_m,)
        bn_num = tf.reduce_sum(W * (-cos_kx1 + cos_kx2), axis=1)
        b_n = (2.0 / (n * pi)) * bn_num

        # a_n = (2/(n*pi)) * sum_i W_i * ( sin(k*x1_i) - sin(k*x2_i))  -> (N_m,)
        an_num = tf.reduce_sum(W * (sin_kx1 - sin_kx2), axis=1)
        a_n = (2.0 / (n * pi)) * an_num

        # a_0 (스칼라)
        a0 = tf.reduce_sum(tf.cast(self.W_i, tf.as_dtype(self.dtype))) / tf.cast(self.N_p, tf.as_dtype(self.dtype))

        # =========================
        # 2) g_x_2 계산 (n에 대해 reduce_sum)
        # =========================
        # x/factor: (N_t,) / (N_m,) -> (N_m, N_t)
        xf = x[None, :] / factor[:, None]  # (N_m, N_t)

        # 원래 코드:
        # g_x_2 += 0.5*(-factor*b_n) * ( factor*sin(x/factor) - x )
        # g_x_2 += 0.5*factor^2*a_n * (1 - cos(x/factor))
        term1 = 0.5 * (-factor[:, None] * b_n[:, None]) * (factor[:, None] * tf.sin(xf) - x[None, :])
        term2 = 0.5 * (tf.square(factor)[:, None] * a_n[:, None]) * (1.0 - tf.cos(xf))

        g_x_2 = tf.zeros_like(x, dtype=tf.as_dtype(self.dtype))
        g_x_2 += (a0 / 4.0) * tf.square(x)
        g_x_2 += tf.reduce_sum(term1 + term2, axis=0)  # sum over n -> (N_t,)

        # init1*x + init2
        g_x_2 += tf.cast(self.init1, tf.as_dtype(self.dtype)) * x + tf.cast(self.init2, tf.as_dtype(self.dtype))
        return g_x_2