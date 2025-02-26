# Regressão Linear

# p = parametros(pesos)
# m = numero de amostras
# x = entradas/features
# y = saidas/targets
# xi = i-nésima feature
# x1, y1 = primeira amostra/exemplo
# n = numero de features

# Função
# h(x) = p0 + p1 * x1 + p2 * x2 + ... + pn * xn

# Função de perda (MSE)
# 1/m * sum(yi - yhati) ** 2

# Função de Otimização (Gradient Descent)
# pi = pi - alfa * (dpMSE / dppi)
# onde:
# dpMSE / dppi = - 2 / m * (sum(yi - yhati) * xi)
import numpy as np

class LinearRegression():
    
    def __init__(self, learningRate = 0.01):
        
        self.a = learningRate
        
    def fit(self, X, y, iterations = 5):
        self.X = X
        self.y = y
        self.m = X.shape[0]
        try: 
            self.n = X.shape[1]
        except:
            self.n = 1
        
        # print(self.m, self.n)
        print(self.X, self.y)
        
        self.p = [0.001 for _ in range(self.n + 1)]
        self.p[0] = 0
        # print('p' , self.p)
        
        for _ in range(iterations):
            
            yhat = self._yhat()
            # print('yhay: ', yhat)
            
            for i in range(self.n + 1):
                derivada = self._gradientDescent(self.p[i], yhat, i)
                # print('i: ', i)
                # print('d: ', derivada)
                self.p[i] = self.p[i] - self.a * derivada
        
            # print('parametros ' ,self.p)
                
                
    def _yhat(self):
        predictions = np.array([self.p[0] + sum(pi * xi for pi, xi in zip(self.p[1:], self.X[i])) for i in range(self.m)])
        # print('predictions : ', predictions)
        return predictions
    
    def _predict(self, X):
        print('x: ', X)
        predictions = np.array([self.p[0] + sum(pi * xi for pi, xi in zip(self.p[1:], X[i])) for i in range(self.m)])
        # print('predictions : ', predictions)
        return predictions
        
        
        
        
    def _gradientDescent(self, x, yhat, i):
        if i == 0: x = 1
        # print('x:' , x)
        # print('sum: ', sum((self.y - yhat)))
        partialDerivative = - 2 / self.m * sum((self.y - yhat) * x)
        # print('pd: ' , partialDerivative)
        
        return partialDerivative
    
    def predict(self, X):
        print('X: ', X)
        predictions = [self._predict(x) for x in X]
        return predictions
    
    

    
        
        
    