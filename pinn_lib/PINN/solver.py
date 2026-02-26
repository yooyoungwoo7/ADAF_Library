import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import uuid

from .model import PINNModel



class Solver: 
    def __init__(self, ode_res, ic, model, t_tf): 
        self.ode_res = ode_res # 유저로부터 입력받는 잔차식
        self.ic = ic
        self.t_tf = t_tf


        # 유저로부터 입력받은 손실함수(잔차), 초기조건
        self.ode_res = ode_res
        self.ic = ic

        # NN 모델
        self.model = model # 인스턴스
        self.models = model.models # 모델 리스트

        # 각 모델별 옵티마이저 생성
        self.optims = [tf.keras.optimizers.Adam(1e-1) for _ in range(self.model.model_num)]


    def get_var(self,i): # 각 모델별 종속 변수들 계산
        with tf.GradientTape() as tape:
            tape.watch(self.t_tf)
            u = self.models[i](self.t_tf)
            u_t = tape.gradient(u, self.t_tf)
            u0 =  self.models[i](self.t_tf[:1]) 
        return (u,u_t,u0)
    
    def get_grad(self,i): # 손실함수 및 기울기 계산
        with tf.GradientTape() as tape:
            var_list = []
            for k in range(self.model.model_num):
                var_list.append(self.get_var(k))                   
            loss = tf.reduce_mean(self.ode_res(var_list,i,self.ic))
        grads = tape.gradient(loss, self.models[i].trainable_variables)
        return loss, grads

    def train(self,epoch=20):
        for i in range(epoch):
            for j in range(self.model.model_num): # 학습 루프
                for k in range(50):
                    loss, grads = self.get_grad(j)
                    self.optims[j].apply_gradients(zip(grads, self.models[j].trainable_variables))
            print(f'epoch: {i}, Loss: {loss.numpy()}')
        return
    
    def plot(self):
        # plot 
        plt.figure()
        for i in range(self.model.model_num):
            plt.plot(self.t_tf, self.models[i](self.t_tf), label=f'Model #{i}')

        plt.title('Solution')
        plt.legend()
        filename = f"solution_{uuid.uuid4().hex}.png" # 랜덤한 이름으로 플롯 저장
        plt.savefig(filename)
        plt.show()
        
        return
    

    def status_print(self):
        print('\n<Summary>')
        print(f'Number of models: {self.model.model_num}')
        print(f'Detail: {self.models}')
        return