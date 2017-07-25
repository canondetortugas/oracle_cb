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
        self.Ainv = la.pinv(self.A)
        
        self.b = self.X.transpose().dot(W.dot(self.y))
        
        # (parameter vector)
        self.theta = self.Ainv.dot(self.b)
        
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
            Ainv = self.Ainv.copy()
            b = self.b.copy()
        else:
            A = self.A
            Ainv = self.Ainv
            b = self.b
        
        # update using sherman morrison (only works if A is invertible -- can be incorrect if regularization is turned off).
        z = Ainv.dot(x)
        
        Ainv = Ainv - weight*np.outer(z, z)/(1 + weight*x.dot(z))

        b = b + weight*x*y
        
        theta = Ainv.dot(b)
        
        if keep is True:
            self.theta = theta
        # past_pred = self.X.dot(theta)

        pred = theta.dot(x)

        # past_mse = self.weights.dot((past_pred - self.y)**2)
        past_mse = self.get_mse() + theta.dot(A.dot(theta))
        
        full_mse = past_mse + weight*(x.dot(theta) - y)**2
        
        if keep is True:
            A = A + weight*np.outer(x, x)
            self.mse = full_mse

        return (pred, past_mse, full_mse)

    def fit_incremental_slow(self, x, y , weight = 1., keep = False):
        '''
        Test version of fit_incremental that refits the whole dataset
        '''
        
        assert self.is_fit
        
        # if keep is False:
        #     A = self.A.copy()
        #     b = self.b.copy()
        # else:
        #     A = self.A
        #     b = self.b
        
        # update using sherman morrison (only works if A is invertible -- can be incorrect if regularization is turned off).

        Xp = np.vstack((self.X, x))
        yp = np.concatenate((self.y, [y]))
        wp = np.concatenate((self.weights, [weight]))
        
        # z = A.dot(x)
        
        # A = A - weight*np.outer(z, z)/(1 + weight*x.dot(z))
        
        # b = b + weight*x*y
        
        # theta = A.dot(b)

        W = np.diag(wp)
                
        A = np.eye(self.d)*self.reg + Xp.transpose().dot(W.dot(Xp))
        A = la.pinv(A)
        
        b = Xp.transpose().dot(W.dot(yp))
        
        # (parameter vector)
        theta = A.dot(b)
        
        if keep is True:
            self.theta = theta
            self.A = A
            self.b = b
            self.weights = wp
            self.X = Xp
            self.y = yp
        
        past_pred = self.X.dot(theta)
        
        pred = theta.dot(x)

        past_mse = self.weights.dot((past_pred - self.y)**2)
        full_mse = past_mse + weight*(x.dot(theta) - y)**2
        
        return (pred, past_mse, full_mse)
    
    def predict(self, X):
        
        assert self.is_fit

        if len(X.shape)==1:
            assert X.shape[0] == self.d
        elif len(X.shape) ==2:
            assert X.shape[1] == self.d
        else:
            assert False, "not implemented"
        
        return X.dot(self.theta)

    def pred_range(self, x, delta):

        assert self.is_fit

        mid = self.theta.dot(x)
        
        half_width = np.sqrt(x.dot(self.Ainv.dot(x)))*np.sqrt(delta)

        return (mid - half_width, mid + half_width)

    def pred_range_coarse(self, x, delta):
        assert self.is_fit

        mid = self.theta.dot(x)
        
        half_width = np.linalg.norm(x)*np.sqrt(delta)/self.reg

        return (mid - half_width, mid + half_width)

      
class IncrementalRegressionTree(IncrementalRegressionModel):

    def __init__(self):
        pass
        
class IncrementalRegressionTreeEnsemble(IncrementalRegressionModel):
    
    def __init__(self):
        pass
