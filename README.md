# Fuzzy Clustering Regression

This package implements the estimator proposed by [Lewis et al.(2022)](https://drive.google.com/file/d/1U_MJHtJcB7H1Edv3xceilU_HJoxhLssP/view). This paper introduces Fuzzy Clustering Regression to estimate heterogenous coefficients when there is an unobserved heterogeneity with grouped patterns.

It extends "Fuzzy C-Means" algorithms to regression setup and approximates "Hard K-Means" regression, see [Bonhomme and Manresa (2015)](https://onlinelibrary.wiley.com/doi/abs/10.3982/ECTA11319).

The general specification of Fuzzy Clustering Regression is as follows:

$$
\Large
y_i  = \sum_{g=1}^G \gamma_{i,g} x_i \theta_g +  z_i \beta + \nu_i
$$

where $i=1,2,...,N$, $y_i \in \mathbb{R}^T$, $x_i \in \mathbb{R}^T \times \mathbb{R}^K$, $z_i \in \mathbb{R}^T \times \mathbb{R}^L$ and $\gamma_{i,g} = \mathbb{1}[g_i=g]$. Note that $\theta_g$ is a $K \times 1$ vector representing the group specific (heterogenous) coeffcieints and $\beta$ is $Lx1$ vector stands for homogenous coeffcients. Finally, we assume $\mathbb{E}[\nu_i|g_i=g]=0$.

In case of $G=1$, the model boils down to standard panel data regression model.

This specification encompasses many different variations.

* $x_i$ may contain vector of constants, that corresponds **grouped level FE**
* $x_i$ may contain time dummies, then $\theta_g$ includes **gropued time FE**
* $z_i$ may have time dummies in it, it refers to **time FE**
 
The goal is to estimate the group memberships ($\gamma_{i,g}$), heterogenous coefficients ($\theta_g$) and homogenous coefficients($\beta$). Estimation of discrete group memberships requires computationally expensive step-wise procedures. Indeed, FCR objective function introduces the group membership function which assigns probability to each group/cluster for each observation. Specifically, the objective function is

$$
\Large
L_m^{F C R}(\theta, \mu)=\mathbb{E}\left[\sum_{g=1}^G \mu_g^m\left\|y-\theta_g x\right\|^2\right] \\
$$

$$
\large
\mu_g(y, x ; \theta, m)=\left(\sum_{h=1}^G \frac{\left\|y-\theta_g x\right\|^{2 /(m-1)}}{\left\|y-\theta_h x\right\|^{2 /(m-1)}}\right)^{-1}; g=1, \ldots, G
$$

where $m>1$ is the regularization parameter and $\mu_g$ represent group probabilities/weights. After some algebra the objective function becomes 

$$
\Large
L_m^{F C R}(\theta, \mu)=\mathbb{E}\left[\left(\sum_{g=1}^G\left\|y-\theta_g x\right\|^{-2 /(m-1)}\right)^{1-m}\right].
$$

This approximation enables us to describe FCR as a GMM and also estimation becomes a nonlinear one step optimization problem.

**Why is it useful?**
1. *Compuation time*: Comparing to "Hard K-Means" regression approach in [Bonhomme and Manresa(2015)](https://onlinelibrary.wiley.com/doi/abs/10.3982/ECTA11319), estimation takes considerably less time. Also, fuzzy parroach can be easily parallelized. 
2. *Inference*: Since it can be described as GMM problem, the inference is straightforward.

**Note**: For a detailed treatment of FCR, check [Lewis et al.(2022)](https://drive.google.com/file/d/1U_MJHtJcB7H1Edv3xceilU_HJoxhLssP/view).

## Package Installation

To install the package type and run the following. It installs the package with dependencies which are avaliable in `requirements.txt` file. 


```python
pip install fcr
```

## Estimation of a FCR Model

Before estimation, fist we need to define a FCR model object which is defined by FCR class. 


```python
from fcr import FCR

fcr_model = FCR(m,G)
```

A typical FCR object has 2 attributes: regularization parameter $m$ and number of groups in the model $G$. m can only take values greater than 1 and G has to be integer greater than 1. Otherwise an error message shows up.

To estimate the model, the package has `estimation()` method that can be applied to an FCR object.


```python
fcr_model.estimate(y, timed, X, Z, grouped_time_FE, grouped_level_FE, time_FE, parallel, n_startingVal)
```

**Arguments:**
* `y`                : (N*T)x1 vector, dependent variable
* `timed`            : (N*T)x1 vector, time vector to track the time dimension in panel data

    - We recommend to create timed vector in the form of $[0,1,...,T-1,0,...,T-1,......]$
    
* `X`                : (N*T)xK matrix, heterogenous covariates, None by default
* `Z`                : (N*T)xL matrix, homogenous covariates, None by default
* `grouped_time_FE`  : T/F, grouped time fixed effect, False by default
* `grouped_level_FE` : T/F, grouped level fixed effect, False by default
* `time_FE`          : T/F, time fixed effect, False by default
    
    - Last 3 boolean options are provided to facilitiate user access to package features. User can put time dummies or constant vector in `X` matrix instead of assigning `True` for `grouped_time_FE` or `grouped_level_FE` options. Similiarly, if `Z` matrix has time dummies, no longer need to assign `True` to `time_FE` option.
    
    - If `grouped_time_FE=True` and/or `grouped_level_FE=True`, the code creates time dummies and a constant vector and stack them with `X` column-wise: $[\text{time dummies}, \text{constant}, X]$ becomes new heterogenous covariates.
    
    - If `time_FE=True`, the code creates time dummies and stack them with `Z` column-wise: $[\text{time dummies}, Z]$ becomes new homogenous covariates.

* `n_startingVal`    : scalar, number of starting values for optimization, 20 by default
    
    - This method creates lots of starting values for nonlinear optimization step to avoid local optima.
    
* `parallel`         : T/F, parallelize the estimation procedure, False by default

    - If `parallel=True`, then the package runs the estimation in parallel cores over the set of starting values. This substantially reduces the computation time.

## Methods for an Estimated Model
This package provides lots of useful methods which can be applied to an estimated model(except `estimate()` method). Here is the list of them. The user can access more information by typing `help(FCR.method_name)`.
* `estimate()`: returns an estimated FCR object.
* `predict()`: returns fitted values for `y` dependent variable.
* `cluster_probs()`: returns cluster probabilities, see Equation 8 in [Lewis et al.(2022)](https://drive.google.com/file/d/1U_MJHtJcB7H1Edv3xceilU_HJoxhLssP/view).
* `cluster()`: returns cluster assignments for each unit.
* `coef()`: returns coefficient estimates as a vector

    The ordering of coefficient is as follows: it starts with the heterogenous coeffcients of group 1 and it continues until group G. Lastly, homogenous coefficients are added.
    
$$    
\begin{equation}
[\theta_0,...,\theta_{G-1},\beta]
\end{equation}
$$

* `grouped_time_FE()`: returns grouped time fixed effects if any.
* `vcov()`: return variance-covariance matrix of coefficient estimates
* `stderror()`: return standard erors of coefficient estimates
* `confint()`: return confidence intervals coefficient estimates
* `tstat()`: return t-statistics of coefficient estimates
* `bic()`: return Bayesian Information Criteria of estimated model
* `summarize()`: returns a table which is a summary of estimation results.

## An Example

We replicate some results of [Lewis et al.(2022)](https://drive.google.com/file/d/1U_MJHtJcB7H1Edv3xceilU_HJoxhLssP/view) in this notebook. Users can consider that notebook as a tutorial on how to us `fcr` package with real data.
