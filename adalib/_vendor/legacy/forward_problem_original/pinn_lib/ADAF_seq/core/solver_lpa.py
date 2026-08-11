import time
import numpy as np
import tensorflow as tf
import scipy.optimize

from .model_lpa import LPA


class Solver:
    """
    LPA solver for full-domain training of 1st-order ODE systems.

    User provides:
      user_residual_fn(var_list, i) -> residual tensor (Nt_total,)
    where
      var_list[k] = (y_k, y_k_t)
    """

    def __init__(
        self,
        user_residual_fn,
        t_step=1.0,
        Nt_total=2000,
        N_p=10,
        order=3,
        lb=0.0,
        ub=1.0,
        adam_epochs=40,
        adam_inner=50,
        adam_lr=1e-3,
        seed=0,
        DTYPE=tf.float32,
        ode_num=1,
        kernel_regularizer=None,
        xla_predict=True,
        xla_step=False,
    ):
        # ---------------- config ----------------
        self.user_residual_fn = user_residual_fn
        self.t_step = float(t_step)
        self.Nt_total = int(Nt_total)
        self.N_p = int(N_p)
        self.order = int(order)
        self.lb = float(lb)
        self.ub = float(ub)

        self.adam_epochs = int(adam_epochs)
        self.adam_inner = int(adam_inner)
        self.adam_lr = float(adam_lr)
        self.seed = int(seed)
        self.DTYPE = tf.as_dtype(DTYPE)
        self.ode_num = int(ode_num)
        self.kernel_regularizer = kernel_regularizer
        self.xla_predict = bool(xla_predict)
        self.xla_step = bool(xla_step)

        self.results_list = []

        if self.ode_num <= 0:
            raise ValueError("ode_num must be >= 1")
        if self.Nt_total <= 1:
            raise ValueError("Nt_total must be >= 2")
        if self.ub <= self.lb:
            raise ValueError("ub must be greater than lb")
        if self.order < 1:
            raise ValueError("order must be >= 1")
        if self.N_p < 1:
            raise ValueError("N_p must be >= 1")

        # ---------------- seeds ----------------
        np.random.seed(self.seed)
        tf.random.set_seed(self.seed)

        # ---------------- full-domain time grid ----------------
        np_dtype = self.DTYPE.as_numpy_dtype

        t_np = np.linspace(self.lb, self.ub, self.Nt_total).astype(np_dtype).reshape(-1, 1)
        self.t = tf.constant(t_np, dtype=self.DTYPE)

        # normalized time for LPA basis: map [lb, ub] -> [-1, 1]
        tau_np = (2.0 * (t_np - self.lb) / (self.ub - self.lb) - 1.0).astype(np_dtype)
        self.tau = tf.constant(tau_np, dtype=self.DTYPE)

        # ---------------- models & optimizers ----------------
        self.models = [
            LPA(
                order=self.order,
                N_p=self.N_p,
                DTYPE=self.DTYPE,
                kernel_regularizer=self.kernel_regularizer,
                name=f"LPA_{i}",
            )
            for i in range(self.ode_num)
        ]

        # build variables once
        for m in self.models:
            _ = m(self.tau)

        self.optims = [
            tf.keras.optimizers.Adam(learning_rate=self.adam_lr)
            for _ in range(self.ode_num)
        ]

        for opt, m in zip(self.optims, self.models):
            opt.build([m.W_i])

        # ---------------- per-ODE loss & train_step ----------------
        self.loss_fns = []
        self.train_steps = []

        for i in range(self.ode_num):

            @tf.function(jit_compile=self.xla_predict)
            def _loss(i=i):
                var_list = self.predict_vars()
                res = self.user_residual_fn(var_list, i)
                return tf.reduce_mean(tf.square(res))

            @tf.function(jit_compile=self.xla_step)
            def _step(i=i):
                with tf.GradientTape() as tape:
                    li = _loss(i=i)
                g = tape.gradient(li, self.models[i].W_i)
                if g is None:
                    g = tf.zeros_like(self.models[i].W_i)
                self.optims[i].apply_gradients([(g, self.models[i].W_i)])
                return li

            self.loss_fns.append(_loss)
            self.train_steps.append(_step)

    # ---------------- compiled predict ----------------
    @tf.function(jit_compile=True)
    def predict_vars(self):
        var_list = []

        scale = tf.cast(2.0 / (self.ub - self.lb), self.DTYPE)

        for m in self.models:
            with tf.GradientTape() as tape:
                tape.watch(self.tau)
                y = m(self.tau)
            dy_dtau = tape.gradient(y, self.tau)
            y_t = dy_dtau * scale

            y = tf.reshape(y, (-1,))
            y_t = tf.reshape(y_t, (-1,))

            var_list.append((y, y_t))

        return var_list

    # ------------------- L-BFGS helpers ---------------------------------
    def _lbfgs_build_slices(self):
        sizes = [int(np.prod(m.W_i.shape)) for m in self.models]
        offsets = np.cumsum([0] + sizes)
        return sizes, offsets

    def _lbfgs_pack_weights_np(self):
        flats = [m.W_i.numpy().reshape(-1) for m in self.models]
        return np.concatenate(flats).astype(np.float64)

    def _lbfgs_unpack_weights_np(self, w_flat):
        _, offsets = self._lbfgs_build_slices()
        w_flat_tf = tf.convert_to_tensor(w_flat, dtype=self.DTYPE)

        for j, m in enumerate(self.models):
            s0 = offsets[j]
            s1 = offsets[j + 1]
            m.W_i.assign(tf.reshape(w_flat_tf[s0:s1], m.W_i.shape))

    # ---------------- initial condition loss ----------------
    @tf.function(jit_compile=True)
    def ic_loss_fn(self, ic):
        var_list = self.predict_vars()
        losses = []

        for j in range(self.ode_num):
            y0 = var_list[j][0][0]
            target = tf.cast(ic[j], self.DTYPE)
            losses.append(tf.square(y0 - target))

        return tf.add_n(losses)

    # ---------------- full-domain training ----------------
    def train(self, ic, use_lbfgs=True, lbfgs_method="L-BFGS-B", lbfgs_options=None, verbose=True, ic_weight=1.0):
        """
        Full-domain ADAM + optional L-BFGS training for LPA basis.
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

        # redefine train steps to include IC penalty
        train_steps = []
        loss_fns = []

        for i in range(self.ode_num):

            @tf.function(jit_compile=self.xla_predict)
            def _loss(i=i):
                var_list = self.predict_vars()
                res = self.user_residual_fn(var_list, i)
                res_loss = tf.reduce_mean(tf.square(res))
                ic_loss = self.ic_loss_fn(ic)
                return res_loss + tf.cast(ic_weight, self.DTYPE) * ic_loss

            @tf.function(jit_compile=self.xla_step)
            def _step(i=i):
                with tf.GradientTape() as tape:
                    li = _loss(i=i)
                g = tape.gradient(li, self.models[i].W_i)
                if g is None:
                    g = tf.zeros_like(self.models[i].W_i)
                self.optims[i].apply_gradients([(g, self.models[i].W_i)])
                return li

            loss_fns.append(_loss)
            train_steps.append(_step)

        self.loss_fns = loss_fns
        self.train_steps = train_steps

        t_all0 = time.perf_counter()

        # ---------------- ADAM alternating ----------------
        loss_last = None

        for ep_idx in range(self.adam_epochs):
            t0 = time.perf_counter()

            for j in range(self.ode_num):
                for _ in range(self.adam_inner):
                    loss_last = self.train_steps[j]()

            t1 = time.perf_counter()

            if verbose:
                loss_parts = [float(self.loss_fns[j]().numpy()) for j in range(self.ode_num)]
                loss_total = float(np.sum(loss_parts))
                print(
                    f"[epoch {ep_idx+1}/{self.adam_epochs}] Adam done in {t1 - t0:.3f}s | "
                    f"loss={loss_total:.3e} | parts={['%.3e' % lp for lp in loss_parts]}"
                )

        # ---------------- L-BFGS refine ----------------
        lbfgs_msg = "skip"
        lbfgs_success = None
        lbfgs_nit = None
        lbfgs_nfev = None

        if use_lbfgs:
            def total_loss_and_grad_np(w_flat):
                self._lbfgs_unpack_weights_np(w_flat)

                with tf.GradientTape() as tape:
                    var_list = self.predict_vars()

                    loss_res = []
                    for j in range(self.ode_num):
                        res = self.user_residual_fn(var_list, j)
                        loss_res.append(tf.reduce_mean(tf.square(res)))

                    loss_ic = self.ic_loss_fn(ic)
                    loss = tf.add_n(loss_res) + tf.cast(ic_weight, self.DTYPE) * loss_ic

                vars_ = [m.W_i for m in self.models]
                grads = tape.gradient(loss, vars_)
                grads = [tf.zeros_like(v) if g is None else g for g, v in zip(grads, vars_)]

                loss_val = float(loss.numpy())
                grad = np.concatenate([g.numpy().reshape(-1) for g in grads]).astype(np.float64)
                return loss_val, grad

            t2 = time.perf_counter()
            w0 = self._lbfgs_pack_weights_np()
            res = scipy.optimize.minimize(
                fun=total_loss_and_grad_np,
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
                    f"[full] L-BFGS done in {t3 - t2:.3f}s | "
                    f"loss={loss_total:.3e} | parts={['%.3e' % lp for lp in loss_parts]} | {lbfgs_msg}"
                )

        # ---------------- final prediction ----------------
        var_list = self.predict_vars()
        y_pred = np.zeros((self.Nt_total, self.ode_num), dtype=self.DTYPE.as_numpy_dtype)

        self.results_list = []
        for j in range(self.ode_num):
            y_pred[:, j] = var_list[j][0].numpy().astype(self.DTYPE.as_numpy_dtype)
            self.results_list.append(y_pred[:, j])

        t_all1 = time.perf_counter()

        if verbose:
            print(f"[total] elapsed: {t_all1 - t_all0:.3f} sec")

        self.train_meta = {
            "adam_last_loss": float(loss_last.numpy()) if loss_last is not None else None,
            "lbfgs": bool(use_lbfgs),
            "lbfgs_method": str(lbfgs_method),
            "lbfgs_msg": str(lbfgs_msg),
            "lbfgs_success": lbfgs_success,
            "lbfgs_nit": lbfgs_nit,
            "lbfgs_nfev": lbfgs_nfev,
            "runtime_sec": float(t_all1 - t_all0),
        }