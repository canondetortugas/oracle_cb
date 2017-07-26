import numpy as np
import Simulators
from Util import *
import scipy.linalg
import sklearn.linear_model
from Policy import *
import Argmax
import pickle

import Incremental

"""
Module of Semibandit algorithms
"""

class Semibandit(object):
    """
    Default semibandit learning algorithm.  Provides interface for
    semibandit learning but also implements the random policy that
    picks random slates on each iteration and does no updating.
    """
    def __init__(self, B):
        """
        Args: 
        B -- A SemibanditSim instance.
        """
        self.B = B
        
    def init(self, T, params={}):
        """
        Initialize the semibandit algorithm for a run.
        Args:
        T -- maximum number of rounds.
        """
        self.reward = []
        self.opt_reward = []
        self.dist = [1.0/self.B.N for i in range(self.B.N)]

    def play(self, T, params={}, verbose=True, validate=None):
        """
        Execute this algorithm on the semibandit simulator for T rounds.

        Returns: 
        cumulative reward (np.array),
        cumulative optimal reward (np.array)
        cumulative regret (np.array)
        """
        self.verbose=verbose
        self.init(T, params=params)
        self.val_scores = []
        for t in range(T):
            if t != 0 and t % 10 == 0 and verbose:
                print("t = %d, r = %0.3f, ave_regret = %0.3f" % (t, np.cumsum(self.reward)[len(self.reward)-1],
                                                                 (np.cumsum(self.opt_reward) - np.cumsum(self.reward))[len(self.reward)-1]/(t+1)), flush=True)

                # if self.leaders is not None:
                #     print([leader.model.weights for leader in self.leaders])
                # print(self.cov)

                # if "num_unif" in dir(self):
                #     print("num_unif = %d" % (self.num_unif))
                
                # print(self.subset_sizes)
                # if self.upper_range is not None:
                #     print(np.array(self.upper_range) - np.array(self.lower_range))
                # print(self.lower_range)
                
            if validate != None and t != 0 and t % 500 == 0:
                val = validate.offline_evaluate(self, train=False)
                if verbose:
                    print("t=%d: val = %0.3f" % (t, val))
                self.val_scores.append(val)
            x = self.B.get_new_context()
            if x == None:
                break
            p = self.get_action(x)
            r = self.B.get_slate_reward(p) ## cumulative reward of action subset selected
            o = self.B.get_best_reward() # Best instantaneous reward
            # if verbose:
            #     print('context: ', x.get_name())
            #     print('action: ', " ".join([str(x) for x in p]))
            #     print('reward: ', " ".join([str(x) for x in self.B.get_base_rewards(p)]))
            self.reward.append(r)
            self.opt_reward.append(o)
            self.update(x, p, self.B.get_base_rewards(p), self.B.get_slate_reward(p))
        l1 = np.cumsum(self.reward)
        l2 = np.cumsum(self.opt_reward)
        return (l1[9::10], (l2-l1)[9::10], self.val_scores)

    def update(self, x, a, y_vec, r):
        """
        Update the state of the semibandit algorithm with the most recent
        context, composite action, and reward.

        The default semibandit doesn't do any updating.

        Args:
        x -- context (should be hashable)
        a -- composite action (np.array of length L)
        r -- reward vector (np.array of length K)
        """
        pass

    def get_action(self, x):
        """
        Pick a composite action to play for this context. 

        Args:
        x -- context

        Returns:
        Composite action (np.array of length L)
        """
        act = np.random.choice(x.get_K(), size=x.get_L(), replace=False)
        self.action = act
        return self.action
        # dist = [1.0/self.B.K for i in range(self.B.K)]
        # p = np.random.multinomial(1, dist) ## Distribution over ACTIONS
        # p = int(np.nonzero(p)[0])
        # return p

class RegressorUCB(Semibandit):
    """
    """

    STRATEGIES= ['greedy', 'pull_if_uncertain']

    def __init__(self, B, learning_alg):
        """
        Initialize object
        Args:
        B -- Semibandit Simulator
        learning_alg -- scikit learn regression algorithm.
                        Should support fit() and predict() methods.
        """
        self.B = B
        if self.B.L is not 1:
            raise NotImplementedError('Semibandit functionality not implemented')    
        

    def init(self, T, params={}):
        """
        Initialize the semibandit algorithm for a run.
        Args:
        T -- maximum number of rounds.

        """

        # Algorithm parameters ###################

        self.stack_actions = False
        self.cheat = True
        
        self.pull_strategy = 'greedy'
        # self.pull_strategy = 'pull_if_uncertain'
        
        self.T = T ## max # rounds
        self.burn_in = 200 ## how many rounds before we start using algorithm


        # confidence parameters (mostly not used currently)
        self.beta = 0.1 ## safety parameter
        self.failure_prob = 0.05 ## failure probability
        self.nu = np.log(2*T**2*self.B.K/self.failure_prob) ## ignoring class G size
        self.kappa = 80
        self.eps = self.T**(self.beta)*self.nu
        self.delta = 1 ## arbitrary, updated at each timestep

        ##########################################

        self.t = 1 ## current round

        # points in time at which we do a full batch optimization
        self.training_points = []
        self.training_indices = [] #indices into history where we did full batch optimization

        self.leaders = None

        self.uncertain_t = None ## last time we were uncertain about the best action for a given context.

        self.subset_sizes = None
        self.upper_range = None
        self.lower_range = None
        
        t = 1
        while t + self.burn_in <= T:
            self.training_points.append(self.burn_in + t)
            t *=2
        # while t + self.burn_in <= T:
        #     self.training_points.append(self.burn_in + t)
        #     t += 500
        
        self.reward = []
        self.opt_reward = []
#        self.dist = [1.0/self.B.N for i in range(self.B.N)]
        
        # Datapoints we've added to the dataset so far. In general size will be < self.t because we do not add at every round
        self.history = []
        

    def update_confidence(self, i):
        self.eta = 1./np.sqrt(i)
        self.delta = self.kappa*self.eps/(float(i)-1)
        self.eps = (self.T/float(i))**(self.beta)*self.nu
        return
       
    def update(self, x, act, y_vec, r):
        """
        Update the state of the semibandit algorithm with the most recent
        context, composite action, and reward.

        Note: Currently assumes that x is same as x given as argument to get_action.

        Args:
        x -- context (should be hashable)
        a -- composite action (np.array of length L)
        y_vec -- reward of current action (non-slate setting)
        r -- reward vector (np.array of length K)
        """

        if self.uncertain_t is not None and self.uncertain_t == self.t:
            full_rvec = np.zeros(self.B.K)
            full_rvec[act] = y_vec ##/self.imp_weights[act]
            self.history.append((x, act, full_rvec, np.ones(self.B.K))) # add current context to dataset. Last element of tuple is treated as weights by argmax2
            
        if self.t in self.training_points: #and self.t > self.burn_in:

            print("training. (t={})".format(self.t))
            
            m = len(self.history)
            
            # (Xs, Ys) -- sub-datasets split accoridng to which action was taken.
            
            if self.stack_actions is True:

                Xs = [np.zeros((m, self.B.d)) for idx in range(self.B.K)]
                Ys = [np.zeros(m) for idx in range(self.B.K)]
                subset_sizes = [0 for idx in range(self.B.K)]
            
                for item in self.history:
                    context = item[0]
                    # print(item)
                    # print(item[1])
                
                    act = item[1][0] ### Assumes K = 1
                    reward = item[2]
                    weight = item[3]
                
                    ss = subset_sizes[act]
                    Xs[act][ss] = context.get_ld_features()[act,:]
                    Ys[act][ss] = reward[act]
                
                    subset_sizes[act] += 1

                self.subset_sizes = subset_sizes

                self.leaders = []

                for idx in range(self.B.K):
                    Xs[idx] = Xs[idx][:subset_sizes[idx]]
                    Ys[idx] = Ys[idx][:subset_sizes[idx]]

                    # print("Action {} data:".format(idx))
                    # print(Xs[idx])
                    # print(Ys[idx])

                    pred = learning_alg()
                    pred.fit(Xs[idx], Ys[idx])

                    self.leaders.append(RegressionPolicy(pred))
                    # print(pred.theta)
            else: # self.stack_actions is False
                X = np.zeros((m, self.B.d))
                Y = np.zeros(m)

                for (idx, item) in enumerate(self.history):
                    context = item[0]
                    act = item[1][0] ### Assumes K = 1
                    reward = item[2]
                    weight = item[3]
                    X[idx] = context.get_ld_features()[act,:]
                    Y[idx] = reward[act]

                pred = learning_alg()
                pred.fit(X, Y)
                self.leader = RegressionPolicy(pred)

            # self.leader, (X, Y, W) = Argmax.argmax2(self.B, self.history, policy_type = RegressionPolicy, learning_alg = self.learning_alg) #leader, dataset used to train leader
            
            # Y_pred = self.leader.model.predict(X)
            
            # self.leader_square_loss = metrics.mean_squared_error(Y, Y_pred)
            
        self.training_indices.append(len(self.history)-1) # mark index into history where we updated.
        
        self.t += 1
        
        self.update_confidence(self.t) ## update confidence parameters


    def _min_reward(self):
        pass
    
    def _max_reward(self):
        pass
        
    def get_action(self, x):
        """
        Pick a action to play for this context. 

        Args:
        x -- context

        Returns:
        Action (np.array of length L)
        """
        
        if self.t <= self.training_points[0]:
            act = np.random.choice(self.B.K, size=1, replace=False)
            self.imp_weights = np.ones(self.B.K)/float(self.B.K)
            self.uncertain_t = self.t
            self.action = act
            return act

        
        # Upper and lower reward ranges predicted by regressors
        upper_range = np.zeros(self.B.K)
        lower_range = np.zeros(self.B.K)
        
        ######
        # compute confidence ranges
        
        for idx in range(self.B.K):
            
            # RegressionPolicy wrapping an IncrementalRegressionModel class.

        
            # context features for current action
            xa=x.get_ld_features()[idx,:]

            if self.stack_actions is True:
                leader = self.leaders[idx]
            else:
                leader = self.leader
                
            model = leader.model
            m=model.get_dataset_size()

            prec = 0.01 

            # Compute confidence range using binary search
            if self.cheat == False:

                # Binary search

                # radius = self.delta
                radius = prec
                # Get worst-case range params from the model
                (rmin, rmax) = model.pred_range_coarse(xa, radius)
                lmin = prec
                lmax = m+prec
                # lmin = 1
                # lmax = m/prec

                # lmax = 10
                # print(m)

                leader_mse = model.get_mse()

                r_upper = self._binary_search(model, xa, radius, rmax, prec, lmin, lmax, leader_mse)
                r_lower = self._binary_search(model, xa, radius, rmin, prec, lmin, lmax, leader_mse)

            # Get confidence range for built-in model computation
            else:
                (r_lower, r_upper) = model.pred_range(xa, prec)


            # (r_lower, r_upper) = model.pred_range(xa, radius)

            # print("Action {} confidence range:".format(idx))
            # print((r_upper, r_lower))

            # assert r_upper >= r_lower, (r_upper, r_lower, model.theta, leader_mse, self.t)
            if r_upper < r_lower:
                # import ipdb
                from IPython import embed
                embed()
            
            upper_range[idx] = r_upper
            lower_range[idx] = r_lower

        self.upper_range = upper_range
        self.lower_range = lower_range
            
        ######
        
        # TODO: Add option: Greedy arm pull vs. pull when confused.

        if self.pull_strategy == 'pull_if_uncertain':
        
            max_lower = np.max(lower_range)
        
            possible_winners = []
        
            for idx in range(self.B.K):
            
                if upper_range[idx] >= max_lower:
                    possible_winners.append(idx)
        
            assert len(possible_winners) > 0

            uncertain = (len(possible_winners) > 1)

                # TODO: Only declare uncertain if gap exceeds gamma
        
            if uncertain:
                self.uncertain_t = self.t # Set flag if we played randomly
                # print(possible_winners)
                act = np.array([possible_winners[np.random.choice(len(possible_winners))]])

                #self.imp_weights = ??? # uniform over confused set
            else:
                act = np.array([possible_winners[0]])

        elif self.pull_strategy == 'greedy':
            act = np.array([np.argmax(upper_range)])

            # Add context to the dataset even though we pulled greedily
            self.uncertain_t = self.t
            
        self.action = act
        return self.action
        
    def _binary_search(self, model, x, radius, r, prec, lmin, lmax, min_mse):
        '''
        r -- Value (eg 1 for max cost, 0 for min cost) value we would like the regressor class to try to match
        model --- is assumed to be an IncrementalRegressionModel
        '''
        
        # print("Binary search, r={}".format(r))

        ll = lmin
        lh = lmax
        
        lt = None

        it = 0
        
        while lh - ll > prec:


                
            lt = (ll + lh)/2.

            # (pred, past_mse, full_mse) = model.fit_incremental_slow(x, r, weight=1./lt, keep=False)
            (pred, past_mse, full_mse) = model.fit_incremental(x, r, weight=1./lt, keep=False)
            # print(1./lt)
            # print(min_mse)
            # print(ll, lh)
            # print(1./lt)
            # print("binary search iteration {}, gap {}, prec {}".format(it, lh - ll, prec))
            # print(1./lh, 1./ll)
            # print(1./lt)            
            # print(pred, model.get_mse(), past_mse, full_mse)
            # print(lt)

            # If model has regularization or isn't optimized well, we can have past_mse < min_mse.
            if max(past_mse- min_mse, 0) > radius:
                ll = lt
            else:
                lh = lt

            it += 1
                
        # No optimization was performed because precision was too low. (should never happen if prec < 1)
        if lt is None:
            val = model.predict(x)
            return val
        else:
            val = pred
            # # Value of MSE optimization problem
            # val = lt*(full_mse - min_mse - radius)
#            print("val {}".format(val))
            # # convert to cost
            # return 1. - np.sqrt(val)
            return val
        

class EpsGreedy(Semibandit):
    """
    Epsilon Greedy algorithm for semibandit learning.
    Can use scikit_learn LearningAlg as oracle

    Current implementation uses a constant value for epsilon. 
    """
    def __init__(self, B, learning_alg="enumerate", classification=False):
        """
        Initialize the epsilon greedy algorithm.
        
        Args:
        B -- Semibandit Simulator
        learning_alg -- scikit learn regression algorithm.
                        Should support fit() and predict() methods.
        """
        self.B = B
        self.link = "linear"
        self.learning_alg = learning_alg
        if learning_alg == "enumerate":
            assert 'Pi' in dir(B), "No learning algorithm but simulator has no policies!"
            self.policy_type = EnumerationPolicy
            self.learning_alg = "enumerate"
        elif classification:
            assert B.L == 1, "Cannot use classification reduction for semibandits"
            self.policy_type = ClassificationPolicy
        else:
            self.policy_type = RegressionPolicy

    def init(self, T, params={}):
        """
        Initialize the current run. 
        The EpsGreedy algorithm maintains a lot more state.

        Args:
        T -- Number of rounds of interaction.
        """
        if "eps" in params.keys():
            self.eps = params['eps']
        else:
            self.eps = 0.1
        if "reward" in params.keys() and params['reward'] == True:
            self.use_reward_features = False
        else:
            self.use_reward_features = True
            if 'weight' in params.keys():
                self.weights = params['weight']
            else:
                self.weights = self.B.weight

        if 'train_all' in params.keys() and params['train_all']:
            self.train_all = True
        else:
            self.train_all = False

        if 'learning_alg' in params.keys():
            self.learning_alg = params['learning_alg']
        if self.learning_alg == "enumerate":
            assert 'Pi' in dir(self.B), "No learning algorithm but simulator has no policies!"
            self.policy_type = EnumerationPolicy
        elif 'classification' in params.keys():
            assert B.L == 1, "Cannot use classification reduction for semibandits"
            self.policy_type = ClassificationPolicy
        else:
            self.policy_type = RegressionPolicy

        if "link" in params.keys():
            self.link = params['link']

        self.training_points = []
        i = 4
        while True:
            self.training_points.append(int(np.sqrt(2)**i))
            i+=1
            if np.sqrt(2)**i > T:
                break
        print(self.training_points)

        self.reward = []
        self.opt_reward = []
        self.T = T
        self.t = 1
        self.action = None
        self.leader = None
        self.history = []
        self.ber = 0
        self.num_unif = 0

    def get_action(self, x):
        """
        Select a composite action to play for this context.
        Also updates the importance weights.

        Args:
        x -- context.
        """
        self.imp_weights = (x.get_L()/x.get_K())*np.ones(x.get_K())
        if self.leader is not None and self.train_all:
            self.imp_weights = (self._get_eps())*self.imp_weights
            self.imp_weights[self.leader.get_action(x)] += (1-self._get_eps())
        self.ber = np.random.binomial(1,np.min([1, self._get_eps()]))
        if self.leader is None or self.ber:
            A = np.random.choice(x.get_K(), size=x.get_L(), replace=False)
            self.action = A
            self.num_unif += 1
        elif self.use_reward_features:
            A = self.leader.get_weighted_action(x,self.weights)
        else:
            A = self.leader.get_action(x)
        self.action = A
        return A
            
    def update(self, x, act, y_vec, r):
        """
        Update the state for the algorithm.
        We currently call the AMO whenever t is a perfect square.

        Args:
        x -- context.
        act -- composite action (np.array of length L)
        r_vec -- reward vector (np.array of length K)
        """
        if self.use_reward_features:
            full_rvec = np.zeros(self.B.K)
            full_rvec[act] = y_vec ##/self.imp_weights[act]
            if self.train_all or self.ber:
                self.history.append((x, act, full_rvec, 1.0/self.imp_weights))
        elif self.ber:
            self.history.append((x,act,r))
        ##if self.t >= 10 and np.log2(self.t) == int(np.log2(self.t)):
        if self.t >= 10 and self.t in self.training_points:
            if self.verbose:
                print("----Training----", flush=True)
            if self.use_reward_features:
                ## self.leader = Argmax.weighted_argmax(self.B, self.history, self.weights, link=self.link, policy_type = self.policy_type, learning_alg = self.learning_alg)
                self.leader = Argmax.argmax2(self.B, self.history, policy_type = self.policy_type, learning_alg = self.learning_alg)
            else:
                ## self.leader = Argmax.argmax(self.B, self.history, policy_type = self.policy_type, learning_alg = self.learning_alg)
                self.leader = Argmax.reward_argmax(self.B, self.history, policy_type = self.policy_type, learning_alg = self.learning_alg)
        self.t += 1

    def _get_eps(self):
        """
        Return the current value of epsilon.
        """
        return np.max([1.0/np.sqrt(self.t), self.eps])
        


class LinUCB(Semibandit):
    """
    Implementation of Semibandit Linucb.
    This algorithm of course only works if features are available 
    in the SemibanditSim object. 

    It must also have a fixed dimension featurization and expose
    B.d as an instance variable.
    """
    def __init__(self, B):
        """
        Initialize a linUCB object.
        """
        self.B = B

    def init(self, T, params={}):
        """
        Initialize the regression target and the 
        feature covariance. 
        """
        self.T = T
        self.d = self.B.d
        self.b_vec = np.matrix(np.zeros((self.d,1)))
        self.cov = np.matrix(np.eye(self.d))
        self.Cinv = scipy.linalg.inv(self.cov)
        self.weights = self.Cinv*self.b_vec
        self.t = 1

        if "delta" in params.keys():
            self.delta = params['delta']
        else:
            self.delta = 0.05

        self.reward = []
        self.opt_reward = []

    def update(self, x, A, y_vec, r):
        """
        Update the regression target and feature cov. 
        """
        features = np.matrix(x.get_ld_features())

        # for idx in range(features.shape[0]):
        #     if np.linalg.norm(features[idx,:]) != 0:
        #         features[idx,:] = features[idx,:]/np.linalg.norm(features[idx,:])
        
        for i in range(len(A)):
            self.cov += features[A[i],:].T*features[A[i],:]
            self.b_vec += y_vec[i]*features[A[i],:].T

        self.t += 1
        if self.t % 100 == 0:
            self.Cinv = scipy.linalg.inv(self.cov)
            self.weights = self.Cinv*self.b_vec

    def get_action(self, x):
        """
        Find the UCBs for the predicted reward for each base action
        and play the composite action that maximizes the UCBs
        subject to whatever constraints. 
        """
        features = np.matrix(x.get_ld_features())
        K = x.get_K()
        # Cinv = scipy.linalg.inv(self.cov)
        # self.weights = Cinv*self.b_vec

        alpha = np.sqrt(self.d)*self.delta ## *np.log((1+self.t*K)/self.delta)) + 1
        ucbs = [features[k,:]*self.weights + alpha*np.sqrt(features[k,:]*self.Cinv*features[k,:].T) for k in range(K)]

        ucbs = [a[0,0] for a in ucbs]
        ranks = np.argsort(ucbs)
        return ranks[K-self.B.L:K]



class MiniMonster(Semibandit):
    """
    Implementation of MiniMonster with a scikit_learn learning algorithm as the AMO.
    """
    def __init__(self, B, learning_alg = None, classification=True):
        self.B = B
        self.learning_alg = learning_alg
        if learning_alg == None:
            assert 'Pi' in dir(B), "No learning algorithm but simulator has no policies!"
            self.policy_type = EnumerationPolicy
            self.learning_alg = "enumerate"
        elif classification:
            assert B.L == 1, "Cannot use classification reduction for semibandits"
            self.policy_type = ClassificationPolicy
        else:
            self.policy_type = RegressionPolicy


    def init(self, T, params={}):
        self.training_points = []
        i = 4
        while True:
            ## self.training_points.append(int(np.sqrt(2)**i))
            self.training_points.append(int(np.sqrt(2)**i))
            i+=1
            if np.sqrt(2)**i > T:
                break
        print(self.training_points)

        self.reward = []
        self.opt_reward = []
        self.weights = []
        self.T = T
        self.action = None
        self.imp_weights = None
        self.t = 1
        self.leader = None
        self.history = []
        self.num_amo_calls = 0
        if 'mu' in params.keys():
            self.mu = params['mu']
        else:
            self.mu = 1.0
        self.num_unif = 0
        self.num_leader = 0

    def update(self, x, A, y_vec, r):
        full_rvec = np.zeros(x.get_K())
        full_rvec[A] = y_vec
        self.history.append((x, A, full_rvec, 1.0/self.imp_weights))
        ##if np.log2(self.t) == int(np.log2(self.t)):
        if self.t >= 10 and self.t in self.training_points:
            if self.verbose:
                print("---- Training ----", flush=True)
            pi = Argmax.argmax2(self.B, self.history, policy_type=self.policy_type, learning_alg = self.learning_alg)                
            self.num_amo_calls += 1
            self.leader = pi
            self.weights = self._solve_op()
            if self.verbose:
                print("t: ", self.t, " Support: ", len(self.weights), flush=True)
            # print("----Evaluating policy distribution on training set----")
            # print("leader weight: %0.2f, score = %0.2f" % (1 - np.sum([z[1] for z in self.weights]), self.B.offline_evaluate(self.leader, train=True)))
            # for item in self.weights:
            #     pi = item[0]
            #     w = item[1]
            #     print("weight %0.2f, score = %0.2f" % (w, self.B.offline_evaluate(pi, train=True)))
            
        self.action = None
        self.imp_weights = None
        self.t += 1

    def get_action(self, x):
        """
        Choose a composite action for context x.
        
        Implements projection, smoothing, mixing etc. 
        Computes importance weights and stores in self.imp_weights.
        """
        ## Compute slate and base action distributions
        p = {}
        p2 = np.zeros(x.get_K())
        for item in self.weights:
            pi = item[0]
            w = item[1]
            (p, p2) = self._update_dists(p, p2, pi.get_action(x), w)

        ## Mix in leader
        if self.leader != None:
            (p, p2) = self._update_dists(p, p2, self.leader.get_action(x), 1 - np.sum([z[1] for z in self.weights]))
        ## Compute importance weights by mixing in uniform
        p2 = (1-self._get_mu())*p2 + (self._get_mu())*x.get_L()/x.get_K()
        self.imp_weights = p2

        ## Decide what action to play
        unif = np.random.binomial(1, self._get_mu())
        ## print("Exploration probability %0.3f" % (self._get_mu()))
        if unif or self.leader is None:
            ## Pick a slate uniformly at random
            ## print("Random action!")
            act = np.random.choice(x.get_K(), size=x.get_L(), replace=False)
            self.num_unif += 1
        else:
            ## Pick a slate from the policy distribution. 
            p = [(k,v) for (k,v) in p.items()]
            draw = np.random.multinomial(1, [x[1] for x in p])
            ## print("Sampled action: ", p[np.where(draw)[0][0]][0], " %0.2f " % (p[np.where(draw)[0][0]][1]))
            ## print("Leader action: ", self.leader.get_action(x))
            act = self._unhash_list(p[np.where(draw)[0][0]][0])
            if p[np.where(draw)[0][0]][1] > 0.5:
                self.num_leader += 1
        self.action = act
        return act

    def _update_dists(self, slate_dist, base_dist, slate, weight):
        """
        This is a subroutine for the projection step. 
        Update the slate_dist and base_dist distributions by 
        incorporating slate played with weight.
        """
        key = self._hash_list(slate)
        if key in slate_dist.keys():
            slate_dist[key] = slate_dist[key] + weight
        else:
            slate_dist[key] = weight
        for a in slate:
            base_dist[a] += weight
        return (slate_dist, base_dist)
        
    def _get_mu(self):
        """
        Return the current value of mu_t
        """
        ## a = 1.0/(2*self.B.K)
        ## b = np.sqrt(np.log(16.0*(self.t**2)*self.B.N/self.delta)/float(self.B.K*self.B.L*self.t))
        a = self.mu
        b = self.mu*np.sqrt(self.B.K)/np.sqrt(self.B.L*self.t)
        c = np.min([a,b])
        return np.min([1,c])

    def _hash_list(self, lst):
        """
        There is a hashing trick we are using to make a dictionary of
        composite actions. _hash_list and _unhash_list implement
        the hash and inverse hash functions. 

        The hash is to write the composite action as a number in base K.
        """
        return np.sum([lst[i]*self.B.K**i for i in range(len(lst))])

    def _unhash_list(self, num):
        lst = []
        if num == 0:
            return np.array([0 for i in range(self.B.L)])
        for i in range(self.B.L):
            rem = num % self.B.K
            lst.append(rem)
            num = (num-rem)/self.B.K
        return lst

    def _solve_op(self):
        """
        Main optimization logic for MiniMonster.
        """
        H = self.history
        mu = self._get_mu()
        Q = [] ## self.weights ## Warm-starting
        psi = 1

        ## MEMOIZATION
        ## 1. Unnormalized historical reward for each policy in supp(Q)
        ## 2. Feature matrix on historical contexts
        ## 3. Recommendations for each policy in supp(Q) for each context in history.
        predictions = {}
        ## vstack the features once here. This is a caching optimization
        if self.policy_type == ClassificationPolicy:
            features = np.zeros((1, H[0][0].get_dim()))
            for x in H:
                features = np.vstack((features, x[0].get_features()))
            features = features[1:,:]
        elif self.policy_type == RegressionPolicy:
            features = np.zeros((1, H[0][0].get_ld_dim()))
            for x in H:
                features = np.vstack((features, x[0].get_ld_features()))
            features = features[1:,:]
        else:
            features = None

        ## Invariant is that leader is non-Null
        (leader_reward, predictions) = self._get_reward(H, self.leader, predictions, features=features)
        leader_reward = leader_reward

        q_rewards = {}
        for item in Q:
            pi = item[0]
            (tmp,predictions) = self._get_reward(H, pi, predictions, features=features)
            q_rewards[pi] = tmp

        updated = True
        iterations = 0
        while updated and iterations < 20:
            print("OP Iteration")
            iterations += 1
            updated = False
            ## First IF statement
            score = np.sum([x[1]*(2*self.B.K*self.B.L/self.B.L + self.B.K*(leader_reward - q_rewards[x[0]])/(psi*self.t*self.B.L*mu)) for x in Q])
            if score > 2*self.B.K*self.B.L/self.B.L:
                # if self.verbose:
                #     print("Shrinking", flush=True)
                c = (2*self.B.K*self.B.L/self.B.L)/score
                Q = [(x[0], c*x[1]) for x in Q]
                updated = True

            ## argmax call and coordinate descent update. 
            ## Prepare dataset
            Rpi_dataset = []
            Vpi_dataset = []
            Spi_dataset = []
            for i in range(self.t):
                context = H[i][0]
                q = self._marginalize(Q, context, predictions)
                act = np.arange(context.get_K())
                r1 = 1.0/(self.t*q)
                r2 = self.B.K/(self.t*psi*mu*self.B.L)*H[i][2]/H[i][3]
                r3 = 1.0/(self.t*q**2)
                weight = np.ones(context.get_K())

                Vpi_dataset.append((context, act, r1, weight))
                Rpi_dataset.append((context, act, r2, weight))
                Spi_dataset.append((context, act, r3, weight))
            dataset = Rpi_dataset
            dataset.extend(Vpi_dataset)

            ## AMO call
            pi = Argmax.argmax2(self.B, dataset, policy_type=self.policy_type, learning_alg = self.learning_alg)
            self.num_amo_calls += 1

            ## This is mostly to make sure we have the predictions cached for this new policy
            if pi not in q_rewards.keys():
                (tmp,predictions) = self._get_reward(H, pi, predictions, features=features)
                q_rewards[pi] = tmp
                if q_rewards[pi] > leader_reward:
                    # if self.verbose:
                    #     print("Changing leader", flush=True)
                    self.leader = pi
                    leader_reward = q_rewards[pi]

            assert pi in predictions.keys(), "Uncached predictions for new policy pi"
            ## Test if we need to update
            (Dpi,predictions) = self._get_reward(dataset, pi, predictions)
            target = 2*self.B.K*self.B.L/self.B.L + self.B.K*leader_reward/(psi*self.t*mu*self.B.L)
            if Dpi > target:
                ## Update
                updated = True
                Dpi = Dpi - (2*self.B.K*self.B.L/self.B.L + self.B.K*leader_reward/(psi*self.t*mu*self.B.L))
                (Vpi,ptwo) = self._get_reward(Vpi_dataset, pi, predictions)
                (Spi,ptwo) = self._get_reward(Spi_dataset, pi, predictions)
                toadd = (Vpi + Dpi)/(2*(1-mu)*Spi)
                Q.append((pi, toadd))
        return Q

    def _get_reward(self, dataset, pi, predictions, features=None):
        """
        For a policy pi whose predictions are cached in predictions dict,
        compute the cumulative reward on dataset.
        """
        ## assert pi in predictions.keys() or features is not None, "Something went wrong with caching"
        if pi not in predictions.keys():
            ## This is going to go horribly wrong if dataset is not the right size
            assert len(dataset) == self.t, "If predictions not yet cached, dataset should have len = self.t"
            predictions[pi] = dict(zip([y[0].get_name() for y in dataset], pi.get_all_actions([y[0] for y in dataset], features=features)))
        score = 0.0
        for item in dataset:
            x = item[0].get_name()
            r = item[2]
            w = item[3]
            score += np.sum(r[predictions[pi][x]]*w[predictions[pi][x]])

        return (score, predictions)
            
    def _marginalize(self, Q, x, predictions):
        """
        Marginalize a set of weights Q for context x
        using the predictions cache. 
        """
        p = np.zeros(self.B.K, dtype=np.longfloat)
        for item in Q:
            pi = item[0]
            w = item[1]
            p[predictions[pi][x.get_name()]] += w
        p = (1.0-self._get_mu())*p + (self._get_mu())*float(self.B.L)/float(self.B.K)
        return p



if __name__=='__main__':
    import sklearn.ensemble
    import sklearn.tree
    import sys, os
    import argparse
    import settings
    import time

    parser = argparse.ArgumentParser()
    parser.add_argument('--T', action='store',
                        default=1000,
                        help='number of rounds', type=int)
    parser.add_argument('--dataset', action='store', choices=['synth','mq2007','mq2008', 'yahoo', 'mslr', 'mslrsmall', 'mslr30k', 'xor', 'static_linear'])
    parser.add_argument('--L', action='store', default=5, type=int)
    parser.add_argument('--I', action='store', default=0, type=int)
    parser.add_argument('--noise', action='store', default=None)
    parser.add_argument('--alg', action='store' ,default='all', choices=['mini', 'eps', 'lin', 'rucb'])
    parser.add_argument('--learning_alg', action='store', default=None, choices=[None, 'gb2', 'gb5', 'tree', 'lin'])
    parser.add_argument('--param', action='store', default=None)
    
    Args = parser.parse_args(sys.argv[1:])
    print(Args, flush=True)
    if Args.noise is not None:
        Args.noise = float(Args.noise)
        outdir = './results/%s_T=%d_L=%d_e=%0.1f/' % (Args.dataset, Args.T, Args.L, Args.noise)
        if not os.path.isdir(outdir):
            os.mkdir(outdir)
    else:
        outdir = './results/%s_T=%d_L=%d_e=0.0/' % (Args.dataset, Args.T, Args.L)
        if not os.path.isdir(outdir):
            os.mkdir(outdir)

    if Args.param is not None:
        Args.param = float(Args.param)

    loop = True
    if Args.dataset=='mslr' or Args.dataset=='mslrsmall' or Args.dataset=='mslr30k' or Args.dataset=='yahoo':
        loop = False

    B = Simulators.DatasetBandit(dataset=Args.dataset, L=Args.L, loop=loop, metric=None, noise=Args.noise)
    Bval = None
    if Args.dataset == 'static_linear':
        Bval = Simulators.DatasetBandit(dataset=Args.dataset, L=Args.L, loop=False, metric=None, noise=Args.noise)
    elif Args.dataset == 'mslr30k':
        Bval = None
    else:
        Bval = Simulators.DatasetBandit(dataset=Args.dataset, L=Args.L, loop=False, metric=None, noise=Args.noise)
    
    # if Args.dataset != 'yahoo':
    #     Bval = Simulators.DatasetBandit(dataset=Args.dataset, L=Args.L, loop=False, metric=None, noise=Args.noise)

    # if Args.dataset == 'mslr30k' and Args.I < 20:
        # order = np.load(settings.DATA_DIR+"mslr/mslr30k_train_%d.npz" % (Args.I))
        # print("Setting order for Iteration %d" % (Args.I), flush=True)
        # B.contexts.order = order['order']
        # B.contexts.curr_idx = 0
    # if Args.dataset == 'yahoo' and Args.I < 20:
        # order = np.load(settings.DATA_DIR+"yahoo/yahoo_train_%d.npz" % (Args.I))
        # print("Setting order for Iteration %d" % (Args.I), flush=True)
        # B.contexts.order = order['order']
        # B.contexts.curr_idx = 0

    # print("Setting seed for Iteration %d" % (Args.I), flush=True)
    # B.set_seed(Args.I)

    learning_alg = None
    if Args.learning_alg == "gb2":
        learning_alg = lambda: sklearn.ensemble.GradientBoostingRegressor(n_estimators=100, max_depth=2)
    elif Args.learning_alg == "gb5":
        learning_alg = lambda: sklearn.ensemble.GradientBoostingRegressor(n_estimators=100, max_depth=5)
    elif Args.learning_alg == "tree":
        learning_alg = lambda: sklearn.tree.DecisionTreeRegressor(max_depth=2)
    elif Args.learning_alg == "lin":
        learning_alg = lambda: sklearn.linear_model.LinearRegression()
    
    if Args.alg == "rucb":
        if Args.learning_alg == 'lin':
            learning_alg = lambda: Incremental.IncrementalLinearRegression(reg=1.)
        elif Args.learning_alg == 'tree':
            learning_alg = lambda: Incremental.IncrementalRegressionTree(max_depth=4)
        elif Args.learning_alg == 'gb5':
            learning_alg = lambda: Incremental.IncrementalRegressionTreeEnsemble(max_depth=5, n_estimators=100)
        else:
            assert False, "not implemented"
        
        print('Using RegressorUCB with model {}'.format(Args.learning_alg))
        
        R = RegressorUCB(B, learning_alg=learning_alg)
        if Args.param is not None:
            if os.path.isfile(outdir+"rucb_%s_%0.5f_rewards_%d.out" % (Args.learning_alg, Args.param,Args.I)):
                print('---- ALREADY DONE ----')
                sys.exit(0)
            start = time.time()
            # TODO: Change param string writing
            (r,reg,val_tmp) = R.play(Args.T, verbose=True, validate=None)
            stop = time.time()
            # TODO: Don't write param string
            np.savetxt(outdir+"rucb_%s_%0.5f_rewards_%d.out" % (Args.learning_alg, Args.param,Args.I), r)
            np.savetxt(outdir+"rucb_%s_%0.5f_validation_%d.out" % (Args.learning_alg, Args.param,Args.I), val_tmp)
            np.savetxt(outdir+"rucb_%s_%0.5f_time_%d.out" % (Args.learning_alg, Args.param, Args.I), np.array([stop-start]))
        else:
            if os.path.isfile(outdir+"rucb_%s_default_rewards_%d.out" % (Args.learning_alg, Args.I)):
                print('---- ALREADY DONE ----')
                sys.exit(0)
            start = time.time()
            (r,reg,val_tmp) = R.play(Args.T, verbose=True, validate=Bval)
            stop = time.time()
            np.savetxt(outdir+"rucb_%s_default_rewards_%d.out" % (Args.learning_alg, Args.I), r)
            np.savetxt(outdir+"rucb_%s_default_validation_%d.out" % (Args.learning_alg, Args.I), val_tmp)
            np.savetxt(outdir+"rucb_%s_default_time_%d.out" % (Args.learning_alg, Args.I), np.array([stop-start]))

    if Args.alg == "lin":
        L = LinUCB(B)
        if Args.param is not None:
            if os.path.isfile(outdir+"lin_%0.5f_rewards_%d.out" % (Args.param,Args.I)):
                print('---- ALREADY DONE ----')
                sys.exit(0)
            start = time.time()
            (r,reg,val_tmp) = L.play(Args.T, verbose=True, validate=Bval, params={'delta': Args.param})
            stop = time.time()
            np.savetxt(outdir+"lin_%0.5f_rewards_%d.out" % (Args.param,Args.I), r)
            np.savetxt(outdir+"lin_%0.5f_validation_%d.out" % (Args.param,Args.I), val_tmp)
            np.savetxt(outdir+"lin_%0.5f_time_%d.out" % (Args.param, Args.I), np.array([stop-start]))
        else:
            if os.path.isfile(outdir+"lin_default_rewards_%d.out" % (Args.I)):
                print('---- ALREADY DONE ----')
                sys.exit(0)
            start = time.time()
            (r,reg,val_tmp) = L.play(Args.T, verbose=True, validate=Bval)
            stop = time.time()
            np.savetxt(outdir+"lin_default_rewards_%d.out" % (Args.I), r)
            np.savetxt(outdir+"lin_default_validation_%d.out" % (Args.I), val_tmp)
            np.savetxt(outdir+"lin_default_time_%d.out" % (Args.I), np.array([stop-start]))
    if Args.alg == "mini":
        if learning_alg is None:
            print("Cannot run MiniMonster without learning algorithm")
            sys.exit(1)
        M = MiniMonster(B, learning_alg = learning_alg, classification=False)
        if Args.param is not None:
            if os.path.isfile(outdir+"mini_%s_%0.3f_rewards_%d.out" % (Args.learning_alg,Args.param,Args.I)):
                print('---- ALREADY DONE ----')
                sys.exit(0)
            start = time.time()
            (r,reg,val_tmp) = M.play(Args.T, verbose=True, validate=Bval, params={'weight': np.ones(Args.L), 'mu': Args.param})
            stop = time.time()
            np.savetxt(outdir+"mini_%s_%0.3f_rewards_%d.out" % (Args.learning_alg, Args.param,Args.I), r)
            np.savetxt(outdir+"mini_%s_%0.3f_validation_%d.out" % (Args.learning_alg, Args.param,Args.I), val_tmp)
            np.savetxt(outdir+"mini_%s_%0.3f_time_%d.out" % (Args.learning_alg, Args.param, Args.I), np.array([stop-start]))
        else:
            if os.path.isfile(outdir+"mini_%s_default_rewards_%d.out" % (Args.learning_alg,Args.I)):
                print('---- ALREADY DONE ----')
                sys.exit(0)
            start = time.time()
            (r,reg,val_tmp) = M.play(Args.T, verbose=True, validate=Bval, params={'weight': np.ones(Args.L)})
            stop = time.time()
            np.savetxt(outdir+"mini_%s_default_rewards_%d.out" % (Args.learning_alg, Args.I), r)
            np.savetxt(outdir+"mini_%s_default_validation_%d.out" % (Args.learning_alg,Args.I), val_tmp)
            np.savetxt(outdir+"mini_%s_default_time_%d.out" % (Args.learning_alg,Args.I), np.array([stop-start]))
        
    if Args.alg == "eps":
        if learning_alg is None:
            print("Cannot run EpsGreedy without learning algorithm")
            sys.exit(1)
        E = EpsGreedy(B, learning_alg = learning_alg, classification=False)
        if Args.param is not None:
            if os.path.isfile(outdir+"epsall_%s_%0.3f_rewards_%d.out" % (Args.learning_alg,Args.param,Args.I)):
                print('---- ALREADY DONE ----')
                sys.exit(0)
            start = time.time()
            (r,reg,val_tmp) = E.play(Args.T, verbose=True, validate=Bval, params={'weight': np.ones(Args.L), 'eps': Args.param, 'train_all': True})
            stop = time.time()
            np.savetxt(outdir+"epsall_%s_%0.3f_rewards_%d.out" % (Args.learning_alg, Args.param,Args.I), r)
            np.savetxt(outdir+"epsall_%s_%0.3f_validation_%d.out" % (Args.learning_alg, Args.param,Args.I), val_tmp)
            np.savetxt(outdir+"epsall_%s_%0.3f_time_%d.out" % (Args.learning_alg, Args.param,Args.I), np.array([stop-start]))
        else:
            if os.path.isfile(outdir+"epsall_%s_default_rewards_%d.out" % (Args.learning_alg,Args.I)):
                print('---- ALREADY DONE ----')
                sys.exit(0)
            start = time.time()
            (r,reg,val_tmp) = E.play(Args.T, verbose=True, validate=Bval, params={'weight': np.ones(Args.L), 'train_all': True})
            stop = time.time()
            np.savetxt(outdir+"epsall_%s_default_rewards_%d.out" % (Args.learning_alg, Args.I), r)
            np.savetxt(outdir+"epsall_%s_default_validation_%d.out" % (Args.learning_alg,Args.I), val_tmp)
            np.savetxt(outdir+"epsall_%s_default_time_%d.out" % (Args.learning_alg,Args.I), np.array([stop-start]))
    print("---- DONE ----")
