class BasisOptions:
    def __init__(
        self,
        name='adaf',
        order=3,
        N_p=100,
        N_m=100,
        Nt_total=None,
        n_seg=10,
        Nt_seg=None,
        gamma=0.8,
        L=1.0,
        kernel_regularizer=None,
    ):
        self.name = name
        self.order = order
        self.N_p = N_p
        self.N_m = N_m
        self.Nt_total = Nt_total
        self.n_seg = n_seg
        self.Nt_seg = Nt_seg
        self.gamma = gamma
        self.L = L
        self.kernel_regularizer = kernel_regularizer


class GridOptions:
    def __init__(
        self,
        lb=0.0,
        ub=1.0,
        Nt_total=2000,
        n_seg=10,
        Nt_seg=None,
        gamma=0.8,
        L=1.0,
    ):
        self.lb = lb
        self.ub = ub
        self.Nt_total = Nt_total
        self.n_seg = n_seg
        self.Nt_seg = Nt_seg
        self.gamma = gamma
        self.L = L


class AdamOptions:
    def __init__(
        self,
        epochs=10,
        inner=50,
        lr=1e-3,
        seed=0,
        dtype="float32",
        xla_predict=True,
        xla_step=False,
    ):
        self.epochs = epochs
        self.inner = inner
        self.lr = lr
        self.seed = seed
        self.dtype = dtype
        self.xla_predict = xla_predict
        self.xla_step = xla_step


class LBFGSOptions:
    def __init__(
        self,
        use=True,
        method="L-BFGS-B",
        options=None,
    ):
        self.use = use
        self.method = method

        if options is None:
            self.options = {
                'maxiter': 4000,
                'maxfun': 50000,
                'maxcor': 50,
                'maxls': 50,
                'ftol': 1e-15,
                'gtol': 1e-15,
                'iprint': 50,
            }
        else:
            self.options = options