import numpy as np
        
# Layer Class
class DenseLayer:
    def __init__(self, output_size, init=None, reg_type=None, reg_param=0.0, t = 1, beta_1 = 0.9, beta_2 = 0.999, epsilon = 1e-8):
        self.reg_type = reg_type
        self.reg_param = reg_param
        self.weights = None
        self.biases = np.zeros((1, output_size))
        self.init = init
        self.output_size = output_size
        self.optimizer = None

        self.t = t
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.m_dw, self.v_dw = 0, 0
        self.m_db, self.v_db = 0, 0
    
    def _initialize_weights(self, input_size):
        # Weight Initialization
        if self.init is None or self.init == "random":
            self.weights = np.random.randn(input_size, self.output_size) - 0.5
        elif self.init == "Xavier":
            self.weights = np.random.randn(input_size, self.output_size) * np.sqrt(1. / input_size)
        elif self.init == "He":
            self.weights = np.random.randn(input_size, self.output_size) * np.sqrt(2. / input_size)
        elif self.init == "Zero":
            self.weights = np.zeros((input_size, self.output_size))
        elif self.init == "One":
            self.weights = np.ones((input_size, self.output_size))
        else:
            raise ValueError(f"Unknown initialization method: {self.init}")
        
    def calculate_update(self, dw, db):
        # weight momentum
        self.m_dw = self.beta_1*self.m_dw + (1-self.beta_1)*dw
        # biases momentum
        self.m_db = self.beta_1*self.m_db + (1-self.beta_1)*db

        # weight rms
        self.v_dw = self.beta_2*self.v_dw + (1-self.beta_2)*(dw**2)
        # biases rms
        self.v_db = self.beta_2*self.v_db + (1-self.beta_2)*(db**2)
    
        ## bias correction
        m_dw_corr = self.m_dw/(1-self.beta_1**self.t)
        m_db_corr = self.m_db/(1-self.beta_1**self.t)
        v_dw_corr = self.v_dw/(1-self.beta_2**self.t)
        v_db_corr = self.v_db/(1-self.beta_2**self.t)

        update_w = self.learning_rate*(m_dw_corr/(np.sqrt(v_dw_corr)+self.epsilon))
        update_b = self.learning_rate*(m_db_corr/(np.sqrt(v_db_corr)+self.epsilon))

        return update_w, update_b

    def forward(self, input):
        if len(input.shape) > 2:
            input = np.reshape(input, (input.shape[0], input.shape[1]))
        if self.weights is None:
            self.m, self.input_size = input.shape
            self._initialize_weights(self.input_size)
        self.input = input
        return np.dot(self.input, self.weights) + self.biases
    
    def backward(self, output_gradient, learning_rate):
        self.learning_rate = learning_rate
        input_gradient = np.dot(output_gradient, self.weights.T)
        weight_gradient = np.dot(self.input.T, output_gradient) / self.m
        biases_gradient = np.sum(output_gradient, axis=0, keepdims=True) / self.m
        
        # Apply regularization
        if self.reg_type == 'l1':
            weight_gradient += self.reg_param * np.sign(self.weights)
        elif self.reg_type == 'l2':
            weight_gradient += self.reg_param * self.weights

        if self.optimizer == "adam":
            update_weights, update_biases = self.calculate_update(weight_gradient, biases_gradient)
        elif self.optimizer == "gradient_descent":
            update_weights, update_biases = learning_rate * weight_gradient, learning_rate * biases_gradient
        
        self.weights -= update_weights
        self.biases -=  update_biases
        
        return input_gradient