import numpy as np
import pandas as pd
import statsmodels.api as sm
import os
from scipy.optimize import minimize, LinearConstraint
from scipy.stats import norm
from concurrent import futures

class FCR():
    """
    This class creats Fuzzy Clustering Regression Model objects. It has several attributes.
    Several methods like estimation,prediction etc are also provided.

    For details see Lewis et al. (2022).
    
    """

    # class wide constants
    eps = 1e-6

    def __init__(self, m, G):
        self.m = m
        self.G = G
        self._coef = None
        self._y = None
        self._timed = None
        self._X = None
        self._Z = None
        self._grouped_time_FE = False
        self._grouped_level_FE = False
        self._time_FE = False
        self._vcov = None
        self._df = None
    
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

    def estimate(self, y, timed, X=None, Z=None, grouped_time_FE=False, grouped_level_FE=False, time_FE=False,
                    algorithm='slsqp', tolerance=1e-6, bounds=None, parallel=False,
                    n_cores=os.cpu_count(), n_startingVal=20, **kwargs):
        """
        This method estimates coefficients of FCR model based on user inputs. 
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
            
            algorithm        : 'slsqp' or 'trust-constr', optimization algorithm, 'slsqp' by default
            tolerance        : scalar, optimization tolerance, 1e-6 by default
            bounds           : (G*K+L)x1 list of tuples, bounds for coefficients, None by default
            parallel         : T/F, parallelize the estimation procedure, False by default
            n_cores          : scalar, number of cores for parallel estimation, max avaliable core in the machine by default
            n_startingVal    : scalar, number of starting values for optimization, 20 by default
            
            **kwargs         : dict, keyword arguments (for data frame option)

        Outputs:
            self             : FCR object, updated with estimation results

        """
        np.random.seed(seed=1453)

        # check data frame presence
        if 'df' in kwargs.keys():

            df = kwargs['df']

            # keep column names
            self._X_names = X
            self._Z_names = Z

            # assign data matrices
            y = np.asarray(df[y]).flatten()
            timed = np.asarray(df[timed]).flatten()
            X = np.asarray(df[X]) if X is not None else None
            Z = np.asarray(df[Z]) if Z is not None else None

            # update model object
            self._df = df
        
        else: # keep column names
            self._X_names = np.arange(X.shape[1]) if X is not None else None
            self._Z_names = np.arange(Z.shape[1]) if Z is not None else None

        # input checking
        FCR.input_checking(y,timed, X, Z, grouped_time_FE, grouped_level_FE, time_FE)

        # keep column names

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
        homo_coef = homo_reg.params # (K+L)x1 vector

        # starting value generation 
        startingVals = [None]*n_startingVal
        for i in range(n_startingVal):
             
            temp = np.zeros((G,K))
            for j in range(G):
                temp[j,:] = homo_coef[:K] + np.random.normal(0,0.1,K)    

            temp = temp[temp[:, 0].argsort()]
            startingVals[i] = np.append(temp.flatten(),homo_coef[K:]) 

        # optimization of objective function

        if parallel is True:
                
            with futures.ProcessPoolExecutor(max_workers=n_cores) as executor:
                
                results = executor.map(FCR.optimize_FCR,
                                        [y for i in range(len(startingVals))],
                                        [timed for i in range(len(startingVals))],
                                        [X for i in range(len(startingVals))],
                                        [Z for i in range(len(startingVals))],
                                        [m for i in range(len(startingVals))],
                                        [G for i in range(len(startingVals))],
                                        [algorithm for i in range(len(startingVals))],
                                        [tolerance for i in range(len(startingVals))],
                                        [bounds for i in range(len(startingVals))],
                                        startingVals)

                results_mat = np.asarray(list(results))
                idx = np.argmin(results_mat[:,1])

                estimates = results_mat[idx,0]
                obj_val = results_mat[idx,1]

        else:
            obj_val = np.inf
            for startingVal in startingVals:
                
                # estimate coefficients given startingVal
                estimates_new, obj_val_new = FCR.optimize_FCR(y, timed, X, Z, m, G, startingVal)
                
                # update the estimates if necessary
                if obj_val_new < obj_val:
                    estimates = np.copy(estimates_new)
                    obj_val = np.copy(obj_val_new)
        
        # update the model object(self)
        self._coef = estimates
        self._y = y
        self._timed = timed
        self._X = X
        self._Z = Z
        self._grouped_time_FE = grouped_time_FE
        self._grouped_level_FE = grouped_level_FE
        self._time_FE = time_FE

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
        y = self._y
        timed = self._timed
        X = self._X
        Z = self._Z
        
        # dimensions
        T = len(np.unique(timed))
        N = int(len(y)/T)
        K = X.shape[1]
        L = Z.shape[1] if Z is not None else 0

        # number of groups
        G = self.G

        # coefficients
        coef = self._coef
        theta = coef[0:G*K] # G*Kx1 vector
        beta = coef[G*K:] # Lx1 vector

        # estimated clusters
        clusters = self.clusters()

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

        y = self._y # true values
        yhat = self.predict() # predictions

        nuhat = y - yhat # residual

        return nuhat

    def momentfun(self, coef):
        """
        This method returns the moment functions fo FCR model in the spirit of GMM.
        See Equation 12 of Lewis et al. 2022

        Inputs:
            self   : FCR object, estimated FCR model
            coef   : (G*K+L)x1 vector, coefficients values, estimates by default

        Outputs:
            eta    : Nx(G*K+L) matrix, moment function for each unit and coefficient

        Note: Extra coef input is added to exploit in derivative(gradient) calculation

        """

        # data matrices
        y = self._y
        timed = self._timed
        X = self._X
        Z = self._Z
        
        # dimensions
        T = len(np.unique(timed))
        N = int(len(y)/T)
        K = X.shape[1]
        L = Z.shape[1] if Z is not None else 0

        # number of groups
        G = self.G

        # regularization parameter
        m = self.m

        # coefficients
        theta = coef[0:G*K] # G*Kx1 vector
        beta = coef[G*K:] # Lx1 vector

        # cluster probabilities
        mu = self.cluster_probs() # NxG matrix 
        
        # compute eta function
        eta = np.zeros((N,len(coef)))
        for i in range(N):
            for g in range(G):

                eta[i,g*K:g*K+K] = (mu[i,g]**m) * ( X[i*T:i*T+T,:].T) @ \
                            (y[i*T:i*T+T] - X[i*T:i*T+T,:] @ theta[g*K:g*K+K].T - Z[i*T:i*T+T,:] @ beta.T)

                eta[i,G*K:] += (mu[i,g]**m) * ( Z[i*T:i*T+T,:].T) @ \
                            (y[i*T:i*T+T] - X[i*T:i*T+T,:] @ theta[g*K:g*K+K].T - Z[i*T:i*T+T,:] @ beta.T)
        
        return eta

    def vcov(self):
        """
        This method returns the variance covariance matrix of the estimated coefficients.
        See Proposition 3 of Lewis et al. 2022

        variance-covariance matrix = Hinv * V * Hinv / N

        Inputs:
            self     : FCR object, estimated FCR model

        Outputs:
            vcov_mat : (G*K+L)x(G*K+L) matrix, variance-covariance matrix

        """
        if self._vcov is not None: # if it is estimated before don't estimate again

            return self._vcov 

        else:
        
            # estimates
            coef = self._coef

            # moment function
            eta = self.momentfun(coef=coef)

            # number of units 
            N = eta.shape[0]

            # outer product of moment function, see Assumption 2 Condition 7 in Lewis et al.(2022)
            V = eta.T @ eta / N

            # gradient of moment function, see Assumption 2 Condition 5 in Lewis et al.(2022)
            H = np.zeros((len(coef),len(coef)))
            coef_abs = np.absolute(coef)
            step = np.minimum(1e-5*coef_abs,coef_abs)
            step = np.diag(step)
            
            for jj in range(len(coef)):
                
                eta_ = self.momentfun(coef=coef+step[:,jj]) # Nx(G*K+L) matrix
                
                temp = (eta_ - eta)/step[jj,jj]
                temp = np.sum(temp, axis=0)
                
                H[:,jj] = temp / N

            try:
                Hinv = np.linalg.inv(H)
            except:
                Hinv = np.linalg.pinv(H)

            vcov_mat =  Hinv @ V @ Hinv / N

            # update self
            self._vcov = vcov_mat

            return vcov_mat

    def stderror(self):
        """
        This method return standard errors for coefficient estimates.

        Inputs:
            self : FCR object, estimated FCR model

        Outputs:
            se   : ((G*K+L)x1) vector, standard errors

        """

        # variance-covariance matrix
        if self._vcov is None:
            vcov_mat = self.vcov()
        else:
            vcov_mat = self._vcov

        # standard errors
        se = np.sqrt(np.diag(vcov_mat)) 

        return se

    def tstat(self):
        """
        This method return standard errors for coefficient estimates.

        Inputs:
            self    : FCR object, estimated FCR model

        Outputs:
            t_stats : ((G*K+L)x1) vector, t statistics for each coefficient

        """
        
        coef = self._coef # estimates
        se = self.stderror() # standard errors

        t_stats = coef/se
        
        return t_stats

    def confint(self,alpha=0.05):
        """
        This method return standard errors for coefficient estimates.

        Inputs:
            self  : FCR object, estimated FCR model
            alpha : scalar in [0,1], significance level, 0.05 by default

        Outputs:
            ci    : ((G*K+L)x2) matrix, confidence intervals

        """

        coef = self._coef # coefficient estimates
        z = norm.ppf(q=1-alpha) # critical values
        se = self.stderror() # standard errors

        ci = np.zeros((len(coef),2)) # confidence intervals
        ci[:,0] = coef - z*se
        ci[:,1] = coef + z*se

        return ci

    def bic(self):
        """
        This method computes BIC for fuzzy clustering regression.

        Inputs:
            self : FCR object, estimated FCR model

        Outputs:
            bic   : BIC statistic of FCR model
            
        """
        
         # data matrices
        y = self._y
        timed = self._timed
        X = self._X
        Z = self._Z
        
        # dimensions
        T = len(np.unique(timed))
        N = int(len(y)/T)
        K = X.shape[1]
        L = Z.shape[1] if Z is not None else 0

        G = self.G # number of groups

        yhat = self.predict() # model predictions

        SSR = np.sum((y-yhat)**2) # sum of squared residuals

        sigma2hat = SSR/(N*T-G*K-L) # variance of nu estimate

        bic =  SSR/(N*T) + sigma2hat*(G*K+L)*np.log(N*T)/(N*T) # BIC

        return bic

    def coef(self):
        """
        This method returns coefficient estimates.

        Inputs:
            self         : FCR object, estimated FCR model

        Outputs:
            self._coef   : coefficients

        """
        return self._coef

    
    def grouped_time_FE(self):
        """
        This method returns grouped time fixed efffect estimates, if any.

        Inputs:
            self              : FCR object, estimated FCR model

        Outputs:
            grouped_time_FE   : (GxT) matrix, grouped time fixed effect estimations

        """

        if self._grouped_time_FE != True:
            raise NameError('This model does not have grouped time fixed effect!')
        else:
            T = len(np.unique(self._timed)) # time dimension
            K = (self._X).shape[1] # number of heterogenous coefs
            G = self.G # number of groups

            coef = self._coef # coefficient estimates

            # consider first G*T heterogenous covariates
            grouped_time_FE = np.zeros((G,T))
            for g in range(G):
                grouped_time_FE[g,:] = coef[g*K:g*K+T]

        return grouped_time_FE

    def grouped_level_FE(self):
        """
        This method returns grouped level fixed efffect estimates, if any.

        Inputs:
            self              : FCR object, estimated FCR model

        Outputs:
            grouped_level_FE  : (Gx1) vector, grouped level fixed effect estimations

        """

        if self._grouped_level_FE != True:
            raise NameError('This model does not have grouped level fixed effect!')
        
        else:
            if self._grouped_time_FE == False:
                K = (self._X).shape[1] # number of heterogenous coefs
                G = self.G # number of groups

                coef = self._coef # coefficient estimates

                grouped_level_FE = np.zeros(G,)
                for g in range(G):
                    grouped_level_FE[g] = coef[g*K]

            else:
                T = len(np.unique(self._timed)) # time dimension
                K = (self._X).shape[1] # number of heterogenous coefs
                G = self.G # number of groups

                coef = self._coef # coefficient estimates

                grouped_level_FE = np.zeros((G,T))
                for g in range(G):
                    grouped_level_FE[g] = coef[g*K+T-1]

        return grouped_level_FE


    def cluster_probs(self):
        """
        This method returns the implied cluster probabilities for each unit. 
        
        See Equation 8 in Lewis et al. (2022)

        Inputs:
            self          : FCR object, estimated FCR model

        Outputs:
            cluster_probs : (NxG) matrix, implied cluster probabilities for each cluster(group) and unit

        """
        
        # data
        y = self._y
        timed = self._timed
        X = self._X
        Z = self._Z
        m = self.m
        G = self.G

        y = y.flatten()
        
        # dimensions
        T = len(np.unique(timed))
        N = int(len(y)/T)
        K = X.shape[1]
        L = Z.shape[1] if Z is not None else 0
        
        # coefficient estimates
        coef = self._coef
        theta = coef[0:G*K] # G*Kx1 vector
        beta = coef[G*K:] # Lx1 vector

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

    def clusters(self):
        """
        This method returns the implied cluster assignments for each unit. 
        
        Inputs:
            self     : FCR object, estimated FCR model

        Outputs:
            clusters : (Nx1) matrix, implied cluster assignments for each cluster(group) and unit

        """

        # get cluster probabilities
        cluster_probs = self.cluster_probs()

        # pick the cluster with the greatest probability
        clusters = np.argmax(cluster_probs, axis=1)

        return clusters

    def distribution(self, idx):
        """
        This method returns the distriution of weighted coeffcients.

        Inputs:
            self  : FCR object, estimated FCR model
            idx   : scalar or string, index or name of the hetergenous covariate 
        
        Outputs:
            dist  : (Nx1) vector, weighted coefficients

        """
        
        # dimensions
        T = len(np.unique(self._timed))
        K = (self._X).shape[1]
        G = self.G

        # find the index
        if type(idx) != int and type(idx) != float:
            df = self._df
            idx = (df.columns).index(idx)

        if self._grouped_time_FE == True:
            idx = idx + T 

        elif self._grouped_level_FE == True:
            idx = idx + 1

        # gather related coeffcients
        coef = self._coef
        coef_ = np.zeros(G)
        for g in range(G):
            coef_[g] = np.copy(coef[idx+g*K])

        # group weights
        weights = self.cluster_probs()

        # distribution
        dist = weights * coef_
        dist = np.mean(dist, axis=1)

        return dist

    def coef_names(self):
        """
        This method returns names for estimated coefficients. 
        
        Inputs:
            self  : FCR object, estimated FCR model

        Outputs:
            names : ((G*K+L)x1) vector, coefficient names

        """
        
        # data
        y = self._y
        timed = self._timed
        X = self._X
        Z = self._Z

        # hyper parameters
        m = self.m
        G = self.G
        
        # dimensions
        T = len(np.unique(timed))
        N = int(len(y)/T)
        K = X.shape[1]
        L = Z.shape[1] if Z is not None else 0

        # column names for X and Z
        X_names = self._X_names
        X_names = [] if X_names is None else X_names
        Z_names = self._Z_names
        Z_names = [] if Z_names is None else Z_names
        
        if self._grouped_time_FE:
            if self._grouped_level_FE:
                X_names = ['time'+str(i) for i in range(1,T)]+['level']+ X_names
            else:
                X_names = ['time'+str(i) for i in range(0,T)]+ X_names
        else:
            if self._grouped_level_FE:
                X_names = ['level']+ X_names
            else:
                pass


        coef = self._coef
        names = [None]*len(coef)
        for g in range(G):
            for k in range(K):
                    names[g*K+k] = 'theta_{0:.0f}_{1:s}'.format(g,str(X_names[k]))

        for l in range(L):
                names[G*K+l] = 'beta_{0:s}'.format(str(Z_names[l]))

        return names

    def summarize(self,alpha=0.05):
        """
        This method return standard errors for coefficient estimates.

        Inputs:
            self    : FCR object, estimated FCR model
            alpha   : scalar in [0,1], significance level, 0.05 by default

        Outputs:
            results : ((G*K+L)x6) table, summary of estimation results

        """

        # general estimation inference results
        names = self.coef_names()
        coef = self._coef
        se = self.stderror()
        t_stats = coef/se
        p_values = 1-norm.cdf(np.absolute(t_stats))
        ci_mat = self.confint(alpha=alpha)

        # transform ci to list of tuples
        ci_mat = np.around(ci_mat,decimals=3)
        ci = [np.nan]*len(coef)
        for i in range(len(coef)):
            ci[i] = (ci_mat[i,0],ci_mat[i,1])

        # results table
        results = pd.DataFrame(columns=['Coefficient','Estimate','Std Error','t-stat','Pr(>|t|)','CI'])

        results['Coefficient'] = names
        results['Estimate'] = coef.round(3)
        results['Std Error'] = se.round(3)
        results['t-stat'] = t_stats.round(3)
        results['Pr(>|t|)'] = p_values.round(3)
        results['CI'] = ci

        return results

    ## STATIC METHODS

    @staticmethod
    def objective_fun(coef,y,timed,X,Z,m,G):
        '''
        This is the objectice function of FCR.

        y_it = x_it'*theta_g + z_it*beta + eps_it
        
        Inputs:
            coef     : (L+G*K)x1 vector, model coefficients
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

        # coefficients
        theta = coef[0:G*K] # G*Kx1 vector
        beta = coef[G*K:] # Lx1 vector

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
    def optimize_FCR(y, timed, X, Z, m, G, algorithm, tolerance, bounds, startingVal):
        """
        This function optimize the FCR objective function given starting values for the optimization routine.

        Inputs:
            y           : (N*T)x1 vector, dependent variable
            timed       : (N*T)x1 vector, time vector
            X           : (N*T)xK matrix, heterogenous covariates, None by default
            Z           : (N*T)xL matrix, homogenous covariates, None by default
            m           : scalar, regularization parameter
            G           : scalar, number of groups
            algorithm   : string, name of the optimization algorithm
            tolerance   : scalar, tolerance level for optimization
            bounds      : ((K*G+L)x1) list of tuples, upper and lower bounds
            startingVal : ((K*G+L)x1) vector, starting values for optimization
            
        Outputs:
            results.x   : ((K*G+L)x1) vector, estimated coefficients
            results.fun : scalar, value of objective function at estimated coefficients

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
        results = minimize(FCR.objective_fun, startingVal, method=algorithm, 
                                    constraints=lin_cons,
                                    tol=tolerance,
                                    options={'maxiter':1e5},
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

        







