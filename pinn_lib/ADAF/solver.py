import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import uuid
import scipy.optimize

from .model import ADAFModel


class Solver: 
    def __init__(self, ode_res, ic, gamma,t,t_tf,model_num,lb,ub,N_p=20,N_m=20): 
        self.ode_res = ode_res # 유저로부터 입력받는 잔차식
        self.lb = lb 
        self.ub = ub
        self.ic = ic
        self.t_tf = t_tf
        self.t = t 
        self.model_num = model_num
        self.gamma = gamma 


        # 유저로부터 입력받은 손실함수(잔차), 초기조건
        self.ode_res = ode_res
        self.ic = ic

        # 각 ODE별 ADA-F 모델 생성해서 리스트에 저장
        self.models = [ADAFModel(ic[_],gamma=gamma,N_p=N_p,N_m=N_m) for _ in range(self.model_num)] 

        # 각 모델별 옵티마이저 생성
        self.optims = [tf.keras.optimizers.Adam(1e-3) for _ in range(self.model_num)]

 
    def get_var(self,i): # 각 모델별 종속 변수들 계산
        with tf.GradientTape() as tape:
            # x 도메인에서 출력
            u = self.models[i].out_g_x_2(self.t_tf)
            u_t = self.models[i].out_g_x_1(self.t_tf) 

            # t 도메인에서 출력
            u_t *= self.gamma/(self.ub - self.lb)
        return (u,u_t)
    

    def get_grad(self,i): # 손실함수 및 기울기 계산
        with tf.GradientTape() as tape:
            tape.watch(self.models[i].W_i)
            var_list = []
            for k in range(self.model_num):
                var_list.append(self.get_var(k))                   
            loss = tf.reduce_mean(tf.square(self.ode_res(var_list,i)))
        grads = tape.gradient(loss, [self.models[i].W_i])
        return loss, grads

    def train(self,epoch=20):
        for i in range(epoch):
            for j in range(self.model_num): # 학습 루프
                for k in range(50):
                    loss, grads = self.get_grad(j)
                    self.optims[j].apply_gradients(zip(grads, [self.models[j].W_i]))
            print(f'epoch: {i}, Loss: {loss.numpy()}')
        return


    def plot(self):
        # plot 
        plt.figure()
        for i in range(self.model_num):
            plt.plot(self.t, self.models[i].out_g_x_2(self.t_tf), label=f'Model #{i}')
        plt.title('Solution')
        plt.legend()
        filename = f"solution_{uuid.uuid4().hex}.png" # 랜덤한 이름으로 플롯 저장
        plt.savefig(r'C:\Users\young\Desktop\python codes\pinn_lib\results\ADA-F\\'+filename)
        plt.show()

    def status_print(self):
        print('\n<Summary>')
        print(f'Number of models: {self.model_num}')
        print(f'Detail: {self.models}\n')
        return
    
#------------------- 여기부터는 L-BFGS 옵티마이저를 위한 메써드들 ---------------------

    def get_total_loss(self): # 각 모델의 Loss를 불러들여 전체 Loss를 구함
        var_list = [self.get_var(k) for k in range(self.model_num)]

        losses = []
        for i in range(self.model_num):
            res_i = self.ode_res(var_list, i)             
            losses.append(tf.reduce_mean(tf.square(res_i))) 

        loss_total = tf.add_n(losses)
        return loss_total

    def get_gradient(self): # 전체 Loss를 전체 파라미터에대해서 자동미분해서 gradient 구함

        W_list = [self.models[i].W_i for i in range(self.model_num)]

        with tf.GradientTape() as tape:
            for W in W_list:
                tape.watch(W)
            loss = self.get_total_loss()
        grads = tape.gradient(loss, W_list)
        return grads
    
    def ScipyOptimizer(self,method='L-BFGS-B', **kwargs):    
        def get_weight_tensor():
            weight_list = []
            shape_list = []
            W_list = [self.models[_].W_i for _ in range(self.model_num)]
            
            for v in W_list:
                shape_list.append(v.shape)
                weight_list.extend(v.numpy().flatten())
            weight_list = tf.convert_to_tensor(weight_list)
            
            return weight_list, shape_list    
        x0, shape_list = get_weight_tensor()
        def set_weight_tensor(weight_list):        
            idx=0
            W_list = [self.models[_].W_i for _ in range(self.model_num)]

            for v in W_list:
                vs = v.shape
                
                if len(vs) == 2:
                    sw = vs[0]*vs[1]
                    new_val = tf.reshape(weight_list[idx:idx+sw], (vs[0],vs[1]))
                    idx += sw
                elif len(vs) == 1:
                    new_val = weight_list[idx:idx+vs[0]]
                    idx+=vs[0]
                elif len(vs) ==0:
                    new_val = weight_list[idx]
                    idx+=1
                elif len(vs) ==3:
                    sw = vs[0]*vs[1]*vs[2]
                    new_val = tf.reshape(weight_list[idx:idx+sw], (vs[0],vs[1],vs[2]))                    
                    idx += sw
                elif len(vs) == 4:
                    sw = vs[0]*vs[1]*vs[2]*vs[3]
                    new_val = tf.reshape(weight_list[idx:idx+sw], (vs[0],vs[1],vs[2],vs[3]))                    
                    idx += sw                    
                v.assign(tf.cast(new_val, 'float32'))   
        
        def get_loss_and_grad(w):
            set_weight_tensor(w)
            
            loss = self.get_total_loss()
            grad = self.get_gradient()
            loss = loss.numpy().astype(np.float64)
            grad_flat=[]

            for g in grad:
                grad_flat.extend(g.numpy().flatten())
            
            grad_flat = np.array(grad_flat, dtype=np.float64)
            return loss, grad_flat
        

        # iteration counter (closure)
        it = {'k': 0}

        def callback(xk):
            it['k'] += 1
            set_weight_tensor(xk)
            loss = self.get_total_loss().numpy()
            print(f"[Iter {it['k']}] loss = {loss:.6e}")
        
        return scipy.optimize.minimize(fun=get_loss_and_grad,
                            x0 = x0,
                            jac = True,
                            callback = callback,
                            method=method,
                            **kwargs)



    