from .model import PINNModel
from .solver import Solver
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

def ode(ode_res, ic, lb=0.0, ub=3.0,ep=10):

    # collocation points 생성
    t_np = np.linspace(lb,ub,1000).reshape(-1,1).astype(np.float32) 
    t_tf = tf.convert_to_tensor(t_np)

    # 모델 생성 : 각 ODE 별로 독립적으로 생성
    ode_num = len(ic)  # ODE 수는 초기조건의 수와 동일
    model = PINNModel(model_num=ode_num)

    print("Model built Completely.")

    # 솔버 생성 
    pinn_solver = Solver(ode_res,ic,model,t_tf)


    # 솔버에 모델 전달 및 학습
    pinn_solver.train(epoch=ep)

    # 결과 플롯
    print( "ODE solved Completely.")
    pinn_solver.plot()

    return pinn_solver
