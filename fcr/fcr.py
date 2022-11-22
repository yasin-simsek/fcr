import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.optimize import minimize, LinearConstraint
from scipy.stats import norm
from concurrent import futures

class FCR():
    """
    This class creats Fuzzy Clustering Regression Model objects. It has several attributes.
    Several methods like estimation,prediction etc are also provided.

    For details see Lewis et al. (2022)
    
    """

    # class wide constants
    eps = 1e-6

    def __init__(self, m, G, y=None, timed=None, X=None, Z=None, params=None):
        self.m = m
        self.G = G
        self.params = params
        self.y = y
        self.timed = timed
        self.X = X
        self.Z = Z
    

    ## PROPERTIES

    @property
    def m(self):
        return self._m

    @m.setter
    def m(self,value):
        if value < 1 + FCR.eps:
            raise ValueError('m is too small.')

        self._m = value

    @property
    def G(self):
        return self._G

    @G.setter
    def G(self,value):
        if (value < 1) or (int(value) != value):
            raise ValueError('G must be positive integer.')

        self._G = value

    ## METHODS

    def estimate(self, y, timed, X=None, Z=None, grouped_time_FE=False, grouped_level_FE=False, time_FE=False, parallel=False, n_startingVal=20):
        """
        This method estimates parameters of FCR model based on user inputs. 
        Then, it updates attributes of FCR object.
        
        Inputs:
            self             : FCR object
            y                : (N*T)x1 vector, dependent variable
            timed            : (N*T)x1 vector, time vector
            X                : (N*T)xK matrix, heterogenous covariates, None by default
            Z                : (N*T)xL matrix, homogenous covariates, None by default
            grouped_time_FE  : T/F, grouped time fixed effect, False by default
            grouped_level_FE : T/F, grouped level fixed effect, False by default
            time_FE          : T/F, time fixed effect, False by default
            
        Outputs:
            self             : FCR object, updated with estimation results

        """
        np.random.seed(seed=1453)
        
        # input checking
        FCR.input_checking(y,timed, X, Z, grouped_time_FE, grouped_level_FE, time_FE)

        # construction of data matrices
        X, Z = FCR.construct_data(timed, X, Z, grouped_time_FE, grouped_level_FE, time_FE)

        # hyperparameters
        m = self.m
        G = self.G

        # dimensions
        T = len(np.unique(timed))
        N = int(len(y)/T)
        K = X.shape[1]
        L = Z.shape[1] if Z is not None else 0

        # fit homogenous model
        Xnew = np.column_stack((X,Z)) if Z is not None else X
        homo_reg = sm.OLS(y, Xnew).fit()
        homo_params = homo_reg.params # (K+L)x1 vector

        # starting value generation 
        startingVals = [None]*n_startingVal
        for i in range(n_startingVal):
            '''
            temp = np.tile(homo_params[:K], G)
            temp += np.random.normal(0,0.01,G*K)

            for g in range(G-1):
                if temp[g*K] > temp[g*K+K]:
                    foo = temp[g*K]
                    temp[g*K] = np.copy(temp[g*K+K])
                    temp[g*K+K] = foo
            
            startingVals[i] = np.append(temp,homo_params[K:]) 
            '''

            temp = np.zeros((G,K))
            for j in range(G):
                temp[j,:] = homo_params[:K] + np.random.normal(0,0.1,K)    

            temp = temp[temp[:, 0].argsort()]
            startingVals[i] = np.append(temp.flatten(),homo_params[K:]) 

        # optimization of objective function
        # TODO: data driven bounds?
        #bounds = [(0,10)]*(G*T*K)
        bounds = [(0,10)]*(L+G*K)

        if parallel is True:
                
            with futures.ProcessPoolExecutor() as executor:
                
                results = executor.map(FCR.optimize_FCR,
                                        [y for i in range(len(startingVals))],
                                        [timed for i in range(len(startingVals))],
                                        [X for i in range(len(startingVals))],
                                        [Z for i in range(len(startingVals))],
                                        [m for i in range(len(startingVals))],
                                        [G for i in range(len(startingVals))],
                                        startingVals)

                results_mat = np.asarray(list(results))
                idx = np.argmin(results_mat[:,1])

                estimates = results_mat[idx,0]
                obj_val = results_mat[idx,1]

        else:
            obj_val = np.inf
            for startingVal in startingVals:
                
                # estimate the parameters given startingVal
                estimates_new, obj_val_new = FCR.optimize_FCR(y, timed, X, Z, m, G, startingVal)
                
                # update the estimates if necessary
                if obj_val_new < obj_val:
                    estimates = np.copy(estimates_new)
                    obj_val = np.copy(obj_val_new)
        
        # update the model object(self)
        self.params = estimates
        self.y = y
        self.timed = timed
        self.X = X
        self.Z = Z

        return self


    def predict(self):
        """
        This method gives in-sample model predictions for dependent variable.

        Inputs:
            self : FCR object, estimated FCR model

        Outputs:
            yhat : (N*Tx1) vector, dependent variable predictions 
        
        """

        # data matrices
        y = self.y
        timed = self.timed
        X = self.X
        Z = self.Z
        
        # dimensions
        T = len(np.unique(timed))
        N = int(len(y)/T)
        K = X.shape[1]
        L = Z.shape[1] if Z is not None else 0

        # number of groups
        G = self.G

        # parameters
        params = self.params
        theta = params[0:G*K] # G*Kx1 vector
        beta = params[G*K:] # Lx1 vector

        # estimated clusters
        clusters = self.get_clusters()

        # calculate predictions
        yhat = np.zeros(N*T)
        for i in range(N):
            
            # find cluster for unit i
            g = clusters[i]
            
            for t in range(T):
                
                yhat[i*T+t] = np.sum(X[i*T+t,:] @ theta[g*K:(g+1)*K].T - Z[i*T+t,:] @ beta.T)

        return yhat

    def residuals(self):
        """
        This method gives in-sample residuals of estimated model.

        Inputs:
            self : FCR object, estimated FCR model

        Outputs:
            nuhat : (N*Tx1) vector, residual estimates 
        
        """ 

        y = self.y # true values
        yhat = self.predict() # predictions

        nuhat = y - yhat # residual

        return nuhat

    def momentfun(self, params):
        """
        This method returns the moment functions fo FCR model in the spirit of GMM.
        See Equation 12 of Lewis et al. 2022

        Inputs:
            self   : FCR object, estimated FCR model
            params : (G*K+L)x1 vector, parameter values, estimated parameters by default

        Outputs:
            eta    : Nx(G*K+L) matrix, moment function for each unit and parameter

        Note: Extra params input is added to exploit in derivative(gradient) calculation

        """

        # data matrices
        y = self.y
        timed = self.timed
        X = self.X
        Z = self.Z
        
        # dimensions
        T = len(np.unique(timed))
        N = int(len(y)/T)
        K = X.shape[1]
        L = Z.shape[1] if Z is not None else 0

        # number of groups
        G = self.G

        # regularization parameter
        m = self.m

        # parameters
        theta = params[0:G*K] # G*Kx1 vector
        beta = params[G*K:] # Lx1 vector

        # cluster probabilities
        mu = self.get_cluster_probs() # NxG matrix 
        
        # compute eta function
        eta = np.zeros((N,len(params)))
        for i in range(N):
            for g in range(G):

                eta[i,g*K:g*K+K] = (mu[i,g]**m) * ( X[i*T:i*T+T,:].T) @ \
                            (y[i*T:i*T+T] - X[i*T:i*T+T,:] @ theta[g*K:g*K+K].T - Z[i*T:i*T+T,:] @ beta.T)

                eta[i,G*K:] += (mu[i,g]**m) * ( Z[i*T:i*T+T,:].T) @ \
                            (y[i*T:i*T+T] - X[i*T:i*T+T,:] @ theta[g*K:g*K+K].T - Z[i*T:i*T+T,:] @ beta.T)
        
        return eta


    def vcov(self):
        """
        This method returns the variance covariance matrix of the estimated parameters.
        See Proposition 3 of Lewis et al. 2022

        variance-covariance matrix = Hinv * V * Hinv / N

        Inputs:
            self     : FCR object, estimated FCR model

        Outputs:
            vcov_mat : (G*K+L)x(G*K+L) matrix, variance-covariance matrix

        """
        
        # parameter estimates
        params = self.params

        # moment function
        eta = self.momentfun(params=params)

        # number of units 
        N = eta.shape[0]

        # outer product of moment function, see Assumption 2 Condition 7 in Lewis et al.(2022)
        V = eta.T @ eta / N

        # gradient of moment function, see Assumption 2 Condition 5 in Lewis et al.(2022)
        H = np.zeros((len(params),len(params)))
        params_abs = np.absolute(params)
        step = np.minimum(1e-5*params_abs,params_abs)
        step = np.diag(step)
        
        for jj in range(len(params)):
            
            eta_ = self.momentfun(params=params+step[:,jj]) # Nx(G*K+L) matrix
            
            temp = (eta_ - eta)/step[jj,jj]
            temp = np.sum(temp, axis=0)
            
            H[:,jj] = temp / N

        try:
            Hinv = np.linalg.inv(H)
        except:
            Hinv = np.linalg.pinv(H)

        vcov_mat =  Hinv @ V @ Hinv / N

        return vcov_mat

    def stderror(self):
        """
        This method return standard errors for parameter estimates.

        Inputs:
            self : FCR object, estimated FCR model

        Outputs:
            se   : ((G*K+L)x1) vector, standard errors

        """
        
        vcov_mat = self.vcov() # variance-covariance matrix
        se = np.sqrt(np.diag(vcov_mat)) # standard errors

        return se

    def tstat(self):
        """
        This method return standard errors for parameter estimates.

        Inputs:
            self    : FCR object, estimated FCR model

        Outputs:
            t_stats : ((G*K+L)x1) vector, t statistics for each parameter

        """
        
        params = self.params # estimates
        se = self.stderror() # standard errors

        t_stats = params/se
        
        return t_stats

    def confint(self,alpha=0.05):
        """
        This method return standard errors for parameter estimates.

        Inputs:
            self  : FCR object, estimated FCR model
            alpha : scalar in [0,1], significance level, 0.05 by default

        Outputs:
            ci    : ((G*K+L)x2) matrix, confidence intervals

        """

        params = self.params # parameter estimates
        z = norm.ppf(q=1-alpha) # critical values
        se = self.stderror() # standard errors

        ci = np.zeros((len(params),2)) # confidence intervals
        ci[:,0] = params - z*se
        ci[:,1] = params + z*se

        return ci

    def Rsquared(self):
        """
        This method computes R^2 for fuzzy clustering regression.

        Inputs:
            self : FCR object, estimated FCR model

        Outputs:
            R2   : R^2 of FCR model
            
        """

        y = self.y # true values
        yhat = self.predict() # model predictions

        SSR = np.sum((y-yhat)**2) # sum of squared residuals
        SST = np.sum((y-np.mean(y))**2) # total sum of squares
        R2 = 1-SSR/SST

        return R2 

    def AIC(self):
        """
        This method computes AIC for fuzzy clustering regression.

        AIC = 2*K + (N*T)*log(SSR)

        Inputs:
            self : FCR object, estimated FCR model

        Outputs:
            aic   : AIC statistic of FCR model
            
        """

        y = self.y # true values
        yhat = self.predict() # model predictions

        K = len(self.params) # number of parameters
        NT = len(y) # number of data points

        SSR = np.sum((y-yhat)**2) # sum of squared residuals

        aic = 2*K + NT*np.log(SSR) # AIC

        return aic     

    def BIC(self):
        """
        This method computes BIC for fuzzy clustering regression.

        BIC = K*log(N*T) + (N*T)*log(SSR/(N*T))

        Inputs:
            self : FCR object, estimated FCR model

        Outputs:
            aic   : AIC statistic of FCR model
            
        """
        
        y = self.y # true values
        yhat = self.predict() # model predictions

        K = len(self.params) # number of parameters
        NT = len(y) # number of data points

        SSR = np.sum((y-yhat)**2) # sum of squared residuals

        bic = K*np.log(NT) + NT*np.log(SSR/NT) # BIC

        return bic

    def grouped_time_FE(self):

        T = len(np.unique(self.timed)) # time dimension
        G = self.G # number of groups

        params = self.params # parameter estimates

        # consider first G*T heterogenous covariates
        grouped_time_FE = np.zeros((G,T))
        for g in range(G):
            grouped_time_FE[g,:] = params[g*T:g*T+T]

        return grouped_time_FE

    def get_cluster_probs(self):
        """
        This method returns the implied cluster probabilities for each unit. 
        
        See Equation 8 in Lewis et al. (2022)

        Inputs:
            self          : FCR object, estimated FCR model

        Outputs:
            cluster_probs : (NxG) matrix, implied cluster probabilities for each cluster(group) and unit

        """
        
        # data
        y = self.y
        timed = self.timed
        X = self.X
        Z = self.Z
        m = self.m
        G = self.G

        y = y.flatten()
        
        # dimensions
        T = len(np.unique(timed))
        N = int(len(y)/T)
        K = X.shape[1]
        L = Z.shape[1] if Z is not None else 0
        
        # parameters
        params = self.params
        theta = params[0:G*K] # G*Kx1 vector
        beta = params[G*K:] # Lx1 vector

        # residual as a tensor
        nu = np.zeros((N,T,G))

        for g in range(G):
            
            if Z is not None:
                temp = y - X @ theta[g*K:(g+1)*K].T - Z @ beta.T # N*Tx1 vector
            else:
                temp = y - X @ theta[g*K:(g+1)*K].T # N*Tx1 vector

            nu[:,:,g] = np.reshape(temp, (N,T))
        
        # cluster probabilities(Equation 8 in Lewis et al. (2022))
        nu2 = nu**2
        nu2 = np.sum(nu2, axis=1) # NxG matrix

        cluster_probs = np.zeros((N,G))
        for g in range(G):

            temp1 = nu2[:,[g]] # Nx1 vector
            temp2 = temp1 / nu2 # Nxg matrix
            temp2 = temp2 ** (1/(m-1))

            cluster_probs[:,g] = np.sum(temp2, axis=1) # Nx1 vector

        cluster_probs = cluster_probs ** (-1)

        return cluster_probs

    def get_clusters(self):
        """
        This method returns the implied cluster assignments for each unit. 
        
        Inputs:
            self     : FCR object, estimated FCR model

        Outputs:
            clusters : (Nx1) matrix, implied cluster assignments for each cluster(group) and unit

        """

        # get cluster probabilities
        cluster_probs = self.get_cluster_probs()

        # pick the cluster with the greatest probability
        clusters = np.argmax(cluster_probs, axis=1)

        return clusters

    def param_names(self):
        """
        This method returns names for estimated parameters. 
        
        Inputs:
            self  : FCR object, estimated FCR model

        Outputs:
            names : ((G*K+L)x1) vector, parameter names

        """
        
        # data
        y = self.y
        timed = self.timed
        X = self.X
        Z = self.Z

        # hyper parameters
        m = self.m
        G = self.G
        
        # dimensions
        T = len(np.unique(timed))
        N = int(len(y)/T)
        K = X.shape[1]
        L = Z.shape[1] if Z is not None else 0

        params = self.params
        names = [None]*len(params)
        for g in range(G):
            for k in range(K):
                names[g*K+k] = 'theta_{0:.0f}{1:.0f}'.format(g,k)

        for l in range(L):
            names[G*K+l] = 'beta_{0:.0f}'.format(l)

        return names

    def summarize(self,alpha=0.05):
        """
        This method return standard errors for parameter estimates.

        Inputs:
            self    : FCR object, estimated FCR model
            alpha   : scalar in [0,1], significance level, 0.05 by default

        Outputs:
            results : ((G*K+L)x6) table, summary of estimation results

        """

        # general estimation inference results
        names = self.param_names()
        params = self.params
        se = self.stderror()
        t_stats = params/se
        p_values = 1-norm.cdf(np.absolute(t_stats))
        ci_mat = self.confint(alpha=alpha)

        # transform ci to list of tuples
        ci = [np.nan]*len(params)
        for i in range(len(params)):
            ci[i] = (ci_mat[i,0],ci_mat[i,1])

        # results table
        results = pd.DataFrame(columns=['Parameter','Estimate','Std Error','t-stat','Pr(>|t|)','CI'])

        results['Parameter'] = names
        results['Estimate'] = params
        results['Std Error'] = se
        results['t-stat'] = t_stats
        results['Pr(>|t|)'] = p_values
        results['CI'] = ci

        return results

    ## STATIC METHODS

    @staticmethod
    def objective_fun(params,y,timed,X,Z,m,G):
        '''
        This is the objectice function of FCR.

        y_it = x_it'*theta_g + z_it*beta + eps_it
        
        Inputs:
            params   : (L+G*K)x1 vector, model parameters
            y        : (N*T)x1 vector, dependent variable
            timed    : (N*T)x1 vector, time vector
            X        : (N*T)xK matrix, heterogenous covariates
            Z        : (N*T)xL matrix, homogenous covariates
            m        : scalar, regularization parameter
            G        : scalar, number of groups
            
            
        Outputs:
            val      : scalar, value of the objective function
            
        '''
        
        y = y.flatten()

        # dimensions
        T = len(np.unique(timed))
        N = int(len(y)/T)
        K = X.shape[1]
        L = Z.shape[1] if Z is not None else 0

        # parameters
        theta = params[0:G*K] # G*Kx1 vector
        beta = params[G*K:] # Lx1 vector

        # residual as a tensor
        nu = np.zeros((N,T,G))

        for g in range(G):
            
            if Z is not None:
                temp = y - X @ theta[g*K:(g+1)*K].T - Z @ beta.T # N*Tx1 vector
            else:
                temp = y - X @ theta[g*K:(g+1)*K].T # N*Tx1 vector

            nu[:,:,g] = np.reshape(temp, (N,T))

        # value of objective function(Equation 9 in Lewis et al. (2022))
        
        nu2 = nu**2
        val = np.sum(nu2, axis=1) # NxG matrix
        val = val**(1/(1-m))
        val = np.sum(val, axis=1) # Nx1 matrix
        val = val**(1-m)
        val = np.sum(val)
        val = val/N

        return val

    # estimation function given startignVals for optimization
    @staticmethod
    def optimize_FCR(y, timed, X, Z, m, G, startingVal):
        """
        This function optimize the FCR objective function given starting values for the optimization routine.

        Inputs:
            y           : (N*T)x1 vector, dependent variable
            timed       : (N*T)x1 vector, time vector
            X           : (N*T)xK matrix, heterogenous covariates, None by default
            Z           : (N*T)xL matrix, homogenous covariates, None by default
            m           : scalar, regularization parameter
            G           : scalar, number of groups
            startingVal : ((K*G+L)x1) vector, starting values for optimization
            
        Outputs:
            results.x   : ((K*G+L)x1) vector, estimated parameters
            results.fun : scalar, value of objective function at estimated parameters

        """
        # dimensions
        T = len(np.unique(timed))
        N = int(len(y)/T)
        K = X.shape[1]
        L = Z.shape[1] if Z is not None else 0

        # FE ordering
        A = np.zeros((G-1,len(startingVal)))
        for j in range(G-1):
            A[j, j*K] = 1 # first element theta_g for g=0,..,G-2
            A[j, (j+1)*K] = -1 # first element theta_g for g=1,..,G-1

        lb = [-np.inf]*(G-1)
        ub = [0]*(G-1)
        lin_cons = LinearConstraint(A, lb, ub)

        # bounds
        bounds = [(-10,10)]*(L+G*K)

        if np.all(A @ startingVal < ub) == False:
            print('startingVal does not satisfy FE ordering')

        # optimization
        results = minimize(FCR.objective_fun, startingVal, method='SLSQP', constraints=lin_cons,
                                    options={'ftol':1e-10,'maxiter':1e5},
                                    bounds=bounds, args=(y,timed,X,Z,m,G))

        return results.x, results.fun

    # construction function of data matrices
    @staticmethod       
    def construct_data(timed, X, Z, grouped_time_FE, grouped_level_FE, time_FE):
        """
        This method returns data matrices after making them suitable for FCR estimation.

        Inputs:
            timed            : (N*T)x1 vector, time vector
            X                : (N*T)xK matrix, heterogenous covariates
            Z                : (N*T)xL matrix, homogenous covariates
            grouped_time_FE  : T/F, grouped time fixed effect
            grouped_level_FE : T/F, grouped level fixed effect
            time_FE          : T/F, time fixed effect
        
        Outputs:
            X,Z              : data matrices, dimensions may vary
        
        
        """
        X_ = pd.get_dummies(timed).values # create time dummies
        
        if grouped_time_FE == True and grouped_level_FE==True:
            # add constant to X matrix for gorpued level FE
            X = np.column_stack((np.ones(X.shape[0]),X)) if X is not None else np.ones((X.shape[0],1))

            # discard first column T=1
            X__ = np.delete(X_,0,axis=1)

            # add time dummies to heterogenous covaraites X
            X = np.column_stack((X__,X))           

        elif grouped_time_FE == True and grouped_level_FE==False:
            # add time dummies to heterogenous covaraites X
            X = np.column_stack((X_,X)) if X is not None else np.copy(X_)

        elif grouped_time_FE == False and grouped_level_FE==True:
            if time_FE == True:
                # add constant to X matrix for gorpued level FE
                X = np.column_stack((np.ones(X.shape[0]),X)) if X is not None else np.ones((X.shape[0],1))

                # discard first column T=1
                X__ = np.delete(X_,0,axis=1)

                # add time dummies to homogenous covaraites Z
                Z = np.column_stack((X__,Z)) if Z is not None else np.copy(X__)

            else:
                # add constant to X matrix for gorpued level FE
                X = np.column_stack((np.ones(X.shape[0]),X)) if X is not None else np.ones((X.shape[0],1))

        elif grouped_time_FE == False and grouped_level_FE==False:
            if time_FE == True:
                # add time dummies to homogenous covaraites Z
                Z = np.column_stack((X_,Z)) if Z is not None else np.copy(X_)
            else:
                pass

        else:
            pass

        return X, Z

    # input checking function
    @staticmethod
    def input_checking(y,timed, X, Z, grouped_time_FE, grouped_level_FE, time_FE):
        """
        This function checks user inputs and raises exceptions. It returns nothing.

        Inputs:
            y                : (N*T)x1 vector, dependent variable
            timed            : (N*T)x1 vector, time vector
            X                : (N*T)xK matrix, heterogenous covariates, None by default
            Z                : (N*T)xL matrix, homogenous covariates, None by default
            grouped_time_FE  : T/F, grouped time fixed effect, False by default
            grouped_level_FE : T/F, grouped level fixed effect, False by default
            time_FE          : T/F, time fixed effect, False by default
        
        Outputs:

        """

        # check heterogenous coefficents
        if X is None and grouped_level_FE==False and grouped_time_FE==False:
            raise ValueError('The model has to contain at least one heterogenous coefficient.')

        # check matrix dimensions
        if y.shape[0]!=timed.shape[0]:
            raise ValueError('The dimensions of input y and timed must match.')
        else:
            if X is not None:
                if (y.shape[0]!=X.shape[0]) or (timed.shape[0]!=X.shape[0]):
                    raise ValueError('The dimensions of input X must match with y and timed.')
        
                if (Z is not None) and  (Z.shape[0]!=X.shape[0]):
                    raise ValueError('The dimensions of input X and Z must match.')
            else:
                if Z is not None:
                    if (y.shape[0]!=Z.shape[0]) or (timed.shape[0]!=Z.shape[0]):
                        raise ValueError('The dimensions of input Z must match with y and timed.')

        # check input types
        if not isinstance(grouped_level_FE, bool):
            raise TypeError('The input option grouped_level_FE can only take True or False.')
        
        if not isinstance(grouped_time_FE, bool):
            raise TypeError('The input option grouped_time_FE can only take True or False.')

        if not isinstance(time_FE, bool):
            raise TypeError('The input option time_FE can only take True or False.')

        # check nan values
        if np.any(np.isnan(y)):
            raise ValueError('y contains nan values')

        if np.any(np.isnan(timed)):
            raise ValueError('timed contains nan values')

        if X is not None:
            if np.any(np.isnan(X)):
                raise ValueError('X contains nan values')

        if Z is not None:
            if np.any(np.isnan(Z)):
                raise ValueError('Z contains nan values')

        # check not numeric values


