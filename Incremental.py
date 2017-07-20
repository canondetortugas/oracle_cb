import numpy as np
import scipy.linalg as la

class IncrementalModel():

    def __init__(self):
        pass
        
    def fit(self, X, y):
        '''
        Fit the model to a dataset (X is n x d dimensional, where n is number of datapoints and d is data dimension).
        '''
        pass
    def predict(self, X):
        '''
        Return model predictions for each row of the matrix X
        '''
        pass
        
    def fit_incremental(self, x, keep=False):
        '''
        Update the model given a single context x.
        
        If keep is False, just return MSE. Otherwise return a copy of the new model .
        '''
        
        # TODO: Should give MSE both on the original dataset and the new dataset
        
        pass
        
    def get_dataset_size(self):
        '''
        Return size of the dataset the model was fit on (if it was fit)
        '''
        pass
        

        
class IncrementalRegressionModel(IncrementalModel):

    def get_mse(self):
        '''
        If the model has been fit, return MSE. Otherwise None
        '''
        
        pass
    
    def get_range(self):
        '''
        Return max and min values this regressor class can produce.
        '''    
            
class IncrementalLinearRegression(IncrementalRegressionModel):

    def __init__(self, reg = 0):
        
        self.reg = reg
        
        self.is_fit = False
        
    def get_dataset_size(self):
        
        assert self.is_fit
        
        return self.n
        
    def fit(self, X, y, weights = None):
        
        assert len(X.shape) == 2
        
        self.n = X.shape[0]
        self.d = X.shape[1]
       
        assert self.n == len(y)
       
        if weights is not None:
            assert self.n == len(weights)
            self.weights = weights
        else:
            self.weights = np.ones(self.n)
       
        self.X = X
        self.y = y
        
        W = np.diag(self.weights)
        
        self.A = np.eye(self.d)*self.reg + self.X.transpose().dot(W.dot(self.X))
        self.A = la.pinv(self.A)
        
        self.b = self.X.transpose().dot(W.dot(self.y))
        
        # (parameter vector)
        self.theta = self.A.dot(self.b)
        
        self.mse = self.weights.dot((self.X.dot(self.theta)- self.y)**2)
        
        self.is_fit = True
        
        return
        
    def get_mse(self):
    
        assert self.is_fit
        
        return self.mse
        
    def fit_incremental(self, x, y , weight = 1., keep = False):
        
        assert self.is_fit
        
        if keep is False:
            A = self.A.copy()
            b = self.b.copy()
        else:
            A = self.A
            b = self.b
        
        # update using sherman morrison (only works if A is invertible -- can be incorrect if regularization is turned off).
        z = A.dot(x)
        
        A = A - weight*np.outer(z, z)/(1 + weight*x.dot(z))
        
        b = b + weight*x*y
        
        theta = A.dot(b)
        
        if keep is True:
            self.theta = theta
        
        past_pred = self.X.dot(theta)
        
        past_mse = self.weights.dot((past_pred - self.y)**2)
        full_mse = past_mse + weight*(x.dot(theta) - y)**2
        
        return (past_mse, full_mse)
        
    def predict(self, X):
        
        assert self.is_fit
        
        assert X.shape[1] == self.data
        
        return X.dot(self.theta)
        
      
class IncrementalRegressionTree(IncrementalRegressionModel):

    def __init__(self):
        pass
        
class IncrementalRegressionTreeEnsemble(IncrementalRegressionModel):
    
    def __init__(self):
        pass