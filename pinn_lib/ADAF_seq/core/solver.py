import time
import numpy as np
import tensorflow as tf
import scipy.optimize  # <-- L-BFGS-B

from .model import ADAF_Reusable


class Solver:
    """
    ADAF_seq Solver (1st-order ODE only)

    User provides:
      user_residual_fn(var_list, i) -> residual tensor (Nt_seg,)
    where
      var_list[k] = (y_k, y_k_t)
      y_k   = model_k.g2(...)
      y_k_t = (gamma/dt_seg) * d/dx(model_k.g2) = s1 * model_k.g1(...)
    """

    def __init__(
        self,
        user_residual_fn,
        t_step=0.05,      # total horizon T (네 코드가 이렇게 사용하고 있음)
        n_seg=10,
        Nt_total=2000,
        Nt_seg=None,
        gamma=0.8,
        N_p=100,
        N_m=100,
        L=1.0,
        adam_epochs=40,
        adam_inner=50,
        adam_lr=1e-3,
        seed=0,
        DTYPE=tf.float32,
        ode_num=1,
        xla_predict=True,
        xla_step=False,  # Windows/XLA에서 optimizer 업데이트가 터질 수 있어 기본 False 권장
    ):
        # ---------------- config ----------------
        self.user_residual_fn = user_residual_fn
        self.t_step = float(t_step)
        self.n_seg = int(n_seg)
        self.Nt_total = int(Nt_total)
        self.Nt_seg = Nt_seg  # infer later
        self.gamma = float(gamma)
        self.N_p = int(N_p)
        self.N_m = int(N_m)
        self.L = float(L)
        self.adam_epochs = int(adam_epochs)
        self.adam_inner = int(adam_inner)
        self.adam_lr = float(adam_lr)
        self.seed = int(seed)
        self.DTYPE = DTYPE
        self.ode_num = int(ode_num)
        self.xla_predict = bool(xla_predict)
        self.xla_step = bool(xla_step)
        self.results_list = []

        if self.ode_num <= 0:
            raise ValueError("ode_num must be >= 1")

        # ---------------- seeds ----------------
        np.random.seed(self.seed)
        tf.random.set_seed(self.seed)

        # ---------------- infer Nt_seg ----------------
        if self.Nt_seg is None:
            if self.Nt_total % self.n_seg != 0:
                raise ValueError("Nt_total must be divisible by n_seg when Nt_seg is None")
            self.Nt_seg = self.Nt_total // self.n_seg
        else:
            self.Nt_seg = int(self.Nt_seg)

        # ---------------- consistency check (인덱싱 안전) ----------------
        if self.Nt_total != self.n_seg * self.Nt_seg:
            raise ValueError(
                f"Inconsistent sizes: Nt_total({self.Nt_total}) != n_seg({self.n_seg}) * Nt_seg({self.Nt_seg})"
            )

        # ---------------- segment layout ----------------
        self.dt_seg = self.t_step / self.n_seg  # segment physical time length
        self.s1 = tf.constant(self.gamma / self.dt_seg, dtype=self.DTYPE)  # d/dt = s1 * d/dx

        # fixed x-grid for each segment (shape fixed -> no retrace)
        x_norm_np = np.linspace(0.0, self.L * self.gamma, self.Nt_seg).astype(np.float32)
        self.x_norm = tf.constant(x_norm_np, dtype=self.DTYPE)

        # ---------------- models & optimizers ----------------
        self.models = [
            ADAF_Reusable(
                self.x_norm,
                init1=0.0,
                init2=0.0,
                init3=0.0,  # unused in 1st-order mode
                N_p=self.N_p,
                N_m=self.N_m,
                L=self.L,
                gamma=self.gamma,
                dtype=self.DTYPE,
                name=f"ADAF_{i}",
            )
            for i in range(self.ode_num)
        ]

        self.optims = [tf.keras.optimizers.Adam(learning_rate=self.adam_lr) for _ in range(self.ode_num)]

        # optimizer 슬롯 변수(momentum/velocity) eager에서 미리 생성
        for opt, m in zip(self.optims, self.models):
            opt.build([m.W])

        # ---------------- per-ODE loss & train_step ----------------
        # user_residual_fn이 if i==0 스타일이어도 안전하게 i를 python 상수로 고정
        self.loss_fns = []
        self.train_steps = []

        for i in range(self.ode_num):

            @tf.function(jit_compile=self.xla_predict)
            def _loss(i=i):
                var_list = self.predict_vars()
                res = self.user_residual_fn(var_list, i)  # (Nt_seg,)
                return tf.reduce_mean(tf.square(res))

            @tf.function(jit_compile=self.xla_step)
            def _step(i=i):
                with tf.GradientTape() as tape:
                    li = _loss(i=i)
                g = tape.gradient(li, self.models[i].W)
                # safety: None -> zeros (그래프/잔차가 W에 의존 안 하면 None 나옴)
                if g is None:
                    g = tf.zeros_like(self.models[i].W)
                self.optims[i].apply_gradients([(g, self.models[i].W)])
                return li

            self.loss_fns.append(_loss)
            self.train_steps.append(_step)

    # ---------------- compiled predict ----------------
    @tf.function(jit_compile=True)
    def predict_vars(self):
        var_list = []
        for m in self.models:
            a0, a, b = m.coeffs()
            y = m.g2_from_coeffs(a0, a, b)     # (Nt_seg,)
            dy_dx = m.g1_from_coeffs(a0, a, b) # (Nt_seg,)
            y_t = dy_dx * self.s1              # (Nt_seg,)
            var_list.append((y, y_t))
        return var_list

    # ---------------- total loss (for L-BFGS) ----------------
    @tf.function(jit_compile=True)
    def loss_total_fn(self):
        var_list = self.predict_vars()
        losses = []
        for j in range(self.ode_num):
            res = self.user_residual_fn(var_list, j)  # (Nt_seg,)
            losses.append(tf.reduce_mean(tf.square(res)))
        return tf.add_n(losses)

    # ------------------- L-BFGS helpers ---------------------------------
    def _lbfgs_build_slices(self):
        sizes = [int(np.prod(m.W.shape)) for m in self.models]
        offsets = np.cumsum([0] + sizes)
        return sizes, offsets

    def _lbfgs_pack_weights_np(self):
        flats = [m.W.numpy().reshape(-1) for m in self.models]
        return np.concatenate(flats).astype(np.float64)

    def _lbfgs_unpack_weights_np(self, w_flat):
        _, offsets = self._lbfgs_build_slices()
        w_flat_tf = tf.convert_to_tensor(w_flat, dtype=self.DTYPE)
        for j, m in enumerate(self.models):
            s0 = offsets[j]
            s1 = offsets[j + 1]
            m.W.assign(tf.reshape(w_flat_tf[s0:s1], m.W.shape))

    @tf.function(jit_compile=True)
    def _lbfgs_loss_and_grads_tf(self):
        with tf.GradientTape() as tape:
            for m in self.models:
                tape.watch(m.W)
            loss = self.loss_total_fn()
        grads = tape.gradient(loss, [m.W for m in self.models])

        fixed_grads = []
        for g, m in zip(grads, self.models):
            fixed_grads.append(tf.zeros_like(m.W) if g is None else g)
        return loss, fixed_grads

    def _lbfgs_loss_and_grad_np(self, w_flat):
        self._lbfgs_unpack_weights_np(w_flat)
        loss_tf, grads_tf = self._lbfgs_loss_and_grads_tf()
        loss_val = float(loss_tf.numpy())
        grad = np.concatenate([g.numpy().reshape(-1) for g in grads_tf]).astype(np.float64)
        return loss_val, grad



    # -------------------------- ADAM + L-BFGS piecewise training --------------------------
    def train_adam_lbfgs_piecewise(
        self,
        ic,
        use_lbfgs=True,
        lbfgs_method="L-BFGS-B",
        lbfgs_options=None,
        verbose=True,
    ):
        """
        ADAM alternating -> L-BFGS refine, piecewise segments.
        continuity (1st-order):
          init2_next = y_end
          init1_next = dy_dx_end

        Returns:
          y_pred   : (Nt_total, ode_num)
          W_bank   : list length ode_num, each is list of W snapshots per segment
          seg_meta : list of dict per segment (timing + optimizer meta)
        """
        if len(ic) != self.ode_num:
            raise ValueError("ic length must match ode_num")

        if lbfgs_options is None:
            lbfgs_options = {
                "maxiter": 500,
                "maxfun": 50000,
                "maxcor": 50,
                "maxls": 50,
                "ftol": np.finfo(float).eps,
                "gtol": np.finfo(float).eps,
                "iprint": -1,
            }

        # storages
        y_pred = np.zeros((self.Nt_total, self.ode_num), dtype=np.float32)
        W_bank = [[] for _ in range(self.ode_num)]
        seg_meta = []

        # initial conditions
        init1 = [tf.constant(0.0, dtype=self.DTYPE) for _ in range(self.ode_num)]            # dy/dx(0)
        init2 = [tf.constant(float(ic[j]), dtype=self.DTYPE) for j in range(self.ode_num)]  # y(0)

        prev_W = [None for _ in range(self.ode_num)]
        zero3 = tf.constant(0.0, dtype=self.DTYPE)

        t_all0 = time.perf_counter()

        for k in range(self.n_seg):
            t_start = k * self.dt_seg
            t_end   = (k + 1) * self.dt_seg

            i0 = k * self.Nt_seg
            i1 = (k + 1) * self.Nt_seg

            # set inits for this segment
            for j in range(self.ode_num):
                self.models[j].set_inits(init1[j], init2[j], zero3)

            # warm-start weights
            for j in range(self.ode_num):
                if prev_W[j] is not None:
                    self.models[j].W.assign(prev_W[j])

            # ---------------- ADAM alternating ----------------
            t0 = time.perf_counter()
            loss_last = None
            for _ in range(self.adam_epochs):
                for j in range(self.ode_num):
                    for _ in range(self.adam_inner):
                        loss_last = self.train_steps[j]()
            t1 = time.perf_counter()

            if verbose:
                loss_parts = [float(self.loss_fns[j]().numpy()) for j in range(self.ode_num)]
                loss_total = float(np.sum(loss_parts))
                print(
                    f"[seg {k}/{self.n_seg-1}] Adam done in {t1 - t0:.3f}s | "
                    f"loss={loss_total:.3e} | parts={['%.3e'%lp for lp in loss_parts]}"
                )

            # ---------------- L-BFGS refine (segment-wise) ----------------
            lbfgs_msg = "skip"
            lbfgs_success = None
            lbfgs_nit = None
            lbfgs_nfev = None

            if use_lbfgs:
                t2 = time.perf_counter()
                w0 = self._lbfgs_pack_weights_np()
                res = scipy.optimize.minimize(
                    fun=self._lbfgs_loss_and_grad_np,
                    x0=w0,
                    jac=True,
                    method=lbfgs_method,
                    options=lbfgs_options,
                )
                t3 = time.perf_counter()

                lbfgs_msg = str(res.message).split("\n")[0]
                lbfgs_success = bool(res.success)
                lbfgs_nit = int(res.nit) if hasattr(res, "nit") else None
                lbfgs_nfev = int(res.nfev) if hasattr(res, "nfev") else None

                if verbose:
                    loss_parts = [float(self.loss_fns[j]().numpy()) for j in range(self.ode_num)]
                    loss_total = float(np.sum(loss_parts))
                    print(
                        f"[seg {k}/{self.n_seg-1}] L-BFGS done in {t3 - t2:.3f}s | "
                        f"loss={loss_total:.3e} | parts={['%.3e'%lp for lp in loss_parts]} | {lbfgs_msg}"
                    )

            # ---------------- predictions + continuity update ----------------
            next_init1 = []
            next_init2 = []

            for j in range(self.ode_num):
                a0, a, b = self.models[j].coeffs()
                y_seg_tf = self.models[j].g2_from_coeffs(a0, a, b)
                dy_dx_tf = self.models[j].g1_from_coeffs(a0, a, b)

                y_pred[i0:i1, j] = y_seg_tf.numpy().astype(np.float32)

                next_init2.append(tf.cast(y_seg_tf[-1], self.DTYPE))
                next_init1.append(tf.cast(dy_dx_tf[-1], self.DTYPE))

                W_bank[j].append(self.models[j].W.numpy().copy())
                prev_W[j] = tf.identity(self.models[j].W)

            init1 = next_init1
            init2 = next_init2

            seg_meta.append(
                {
                    "k": int(k),
                    "t_start": float(t_start),
                    "t_end": float(t_end),
                    "dt_seg": float(self.dt_seg),
                    "Nt_seg": int(self.Nt_seg),
                    "gamma": float(self.gamma),
                    "ode_num": int(self.ode_num),
                    "adam_time": float(t1 - t0),
                    "adam_last_loss": float(loss_last.numpy()) if loss_last is not None else None,
                    "lbfgs": bool(use_lbfgs),
                    "lbfgs_method": str(lbfgs_method),
                    "lbfgs_msg": str(lbfgs_msg),
                    "lbfgs_success": lbfgs_success,
                    "lbfgs_nit": lbfgs_nit,
                    "lbfgs_nfev": lbfgs_nfev,
                }
            )

        t_all1 = time.perf_counter()
        if verbose:
            print(f"[total] elapsed: {t_all1 - t_all0:.3f} sec")


        # 유저에게 반환될 솔버 출력을 results_list라는 속성에 저장
        for i in range(self.ode_num):
            self.results_list.append(y_pred[:,i])
        