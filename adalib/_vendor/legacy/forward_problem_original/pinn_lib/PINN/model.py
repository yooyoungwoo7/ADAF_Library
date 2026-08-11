import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

class PINNModel:
    def __init__(self, lb=0.0, ub=3.0,model_num=0):
        
        # collocation points
        self.lb = lb
        self.ub = ub  
        self.t_np = np.linspace(self.lb,self.ub,100).reshape(-1,1).astype(np.float32) 
        self.t_tf = tf.convert_to_tensor(self.t_np)

        # ODE별 모델 생성해서 속성으로 저장 
        self.model_num = model_num
        self.models = []

        for i in range(self.model_num): # ODE 갯수만큼 모델 생성
            self.models.append(self.build_model())


    def build_model(self,num_hidden=3, num_neurons=20):
        # Build Model 
        T_in = tf.keras.Input(shape=(1,))
        hiddens = tf.keras.layers.Lambda(lambda t: 2.0*(t - self.lb)/(self.ub - self.lb) - 1.0, output_shape=(1,))(T_in) # Normalize input to [-1,1]
        for _ in range(num_hidden):
            hiddens = tf.keras.layers.Dense(10, 
                                            #activation = tf.keras.activations.get('tanh'))
                                            kernel_initializer = 'glorot_normal')(hiddens)
        
        hiddens = tf.keras.layers.Lambda(lambda x: x + tf.math.square(tf.math.sin(x)))(hiddens)
        prediction = tf.keras.layers.Dense(1)(hiddens)

        model = tf.keras.Model(T_in,prediction) # T_in 부터 prediction 까지 모델 생성

        return model
    
