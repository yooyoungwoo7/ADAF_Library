from .solver import Solver
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

def ode(ode_res, ic, lb=0.0, ub=3.0,ep=10, gamma=1.0, dtype = 'float32',N_p=20,N_m=20):

    # collocation points 생성
    L = 1.0
    t = np.linspace(lb, ub, 2000)
    t_train = (t - np.min(t))
    t_train = (t_train/np.max(t_train))*L*gamma
    t_train = t_train.astype(dtype)
    t_train = tf.constant(t_train)

    # 솔버 생성 : 솔버 내에서 각 ODE별 독립적 모델 생성 
    ode_num = len(ic)  # ODE 수는 초기조건의 수와 동일
    adaf_solver = Solver(ode_res=ode_res,ic=ic,t=t,t_tf=t_train,model_num=ode_num,gamma=gamma,lb=lb,ub=ub,N_p=N_p,N_m=N_m)
    print("\nModel built Completely.")
    adaf_solver.status_print()

    
    # 모델 학습: 아담
    adaf_solver.train(epoch=ep)

    # 결과 플롯
    print( "ODE solved with ADAM Optimizer.")
    adaf_solver.plot()

    
    print( "\nStarting L-BFGS-B Optimization")

    
    # 모델 학습: L-BFGS-B 
    adaf_solver.ScipyOptimizer(method='L-BFGS-B', 
    options={'maxiter': 4000, 
        'maxfun': 50000, 
        'maxcor': 50, 
        'maxls': 50, 
        'ftol': np.finfo(float).eps,
        'gtol': np.finfo(float).eps,            
        'factr':np.finfo(float).eps,
        'iprint':50})

    
    
    
    # 결과 플롯
    adaf_solver.plot()
    

    return adaf_solver
