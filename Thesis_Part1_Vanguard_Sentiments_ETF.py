#!/usr/bin/env python
# coding: utf-8

# In[2]:


from numpy import mat, cov, mean, hstack, multiply,sqrt,diag, \
    squeeze, ones, array, vstack, kron, zeros, eye, savez_compressed
from numpy.linalg import inv
from scipy.stats import chi2
from pandas import read_csv
import statsmodels.api as sm
import os


# In[3]:


os.getcwd()


# In[4]:


os.chdir("/Users/mandarphatak/Downloads/")


# In[5]:


data = read_csv('fama_french.csv')

data


# In[6]:


# Split using both named colums and ix for larger blocks
dates = data['date'].values
factors = data[['mkt_excess', 'smb', 'hml','cci','cci_mkt']].values
riskfree = data['rf'].values
portfolios = data.iloc[:, 7:].values

# Use mat for easier linear algebra
factors = mat(factors)
riskfree = mat(riskfree)
portfolios = mat(portfolios)

# Shape information
T,K = factors.shape
T,N = portfolios.shape
# Reshape rf and compute excess returns
riskfree.shape = T,1
excessReturns = portfolios - riskfree


# In[8]:


# Time series regressions
X = sm.add_constant(factors)
ts_res = sm.OLS(excessReturns, X).fit()
alpha = ts_res.params[0]
beta = ts_res.params[1:]
avgExcessReturns = mean(excessReturns, 0)
# Cross-section regression
cs_res = sm.OLS(avgExcessReturns.T, beta.T).fit()
riskPremia = cs_res.params


# In[9]:


# Moment conditions
X = sm.add_constant(factors)
p = vstack((alpha, beta))
epsilon = excessReturns - X @ p
moments1 = kron(epsilon, ones((1, K + 1)))
moments1 = multiply(moments1, kron(ones((1, N)), X))
u = excessReturns - riskPremia[None,:] @ beta
moments2 = u * beta.T
# Score covariance
S = mat(cov(hstack((moments1, moments2)).T))
# Jacobian
G = mat(zeros((N * K + N + K, N * K + N + K)))
SigmaX = (X.T @ X) / T
G[:N * K + N, :N * K + N] = kron(eye(N), SigmaX)
G[N * K + N:, N * K + N:] = -beta @ beta.T
for i in range(N):
    temp = zeros((K, K + 1))
    values = mean(u[:, i]) - multiply(beta[:, i], riskPremia)
    temp[:, 1:] = diag(values)
    G[N * K + N:, i * (K + 1):(i + 1) * (K + 1)] = temp

vcv = inv(G.T) * S * inv(G) / T


# In[10]:


vcvAlpha = vcv[0:N * K + N:6, 0:N * K + N:6]
J = alpha @ inv(vcvAlpha) @ alpha.T
J = J[0, 0]
Jpval = 1 - chi2(11).cdf(J)


# In[11]:


Jpval


# In[17]:


vcvRiskPremia = vcv[N * K + N:, N * K + N:]
annualizedRP = 12 * riskPremia
arp = list(squeeze(annualizedRP))
arpSE = list(sqrt(12 * diag(vcvRiskPremia)))
print('        Annualized Risk Premia')
print('           Market       SMB         HML     CCI  CCI_MKT')
print('---------------------------------------------------------------')
print('Premia     {0:0.4f}    {1:0.4f}    {2:0.4f}   {3:0.4f}   {4:0.4f}'.format(arp[0], arp[1], arp[2], arp[3], arp[4]))
print('Std. Err.  {0:0.4f}    {1:0.4f}    {2:0.4f}   {3:0.4f}   {4:0.4f}'.format(arpSE[0], arpSE[1], arpSE[2], arpSE[3], arpSE[4]))
print('\n\n')

print('J-test:   {:0.4f}'.format(J))
print('P-value:   {:0.4f}'.format(Jpval))

i = 0
betaSE = []
for j in range(11):
    for k in range(1):
        a = alpha[i]
        b = beta[:, i]
        variances = diag(vcv[(K + 1) * i:(K + 1) * (i + 1), (K + 1) * i:(K + 1) * (i + 1)])
        betaSE.append(sqrt(variances))
        s = sqrt(variances)
        c = hstack((a, b))
        t = c / s
        print('Size: {:}, Value:{:}   Alpha   Beta(VWM)   Beta(SMB)   Beta(HML) Beta(CCI) Beta(CCI_MKT)'.format(j + 1, k + 1))
        print('Coefficients: {:>10,.4f}  {:>10,.4f}  {:>10,.4f}  {:>10,.4f} {:>10,.4f}  {:>10,.4f}'.format(a, b[0], b[1], b[2], b[3],b[4]))
        print('Std Err.      {:>10,.4f}  {:>10,.4f}  {:>10,.4f}  {:>10,.4f}  {:>10,.4f} {:>10,.4f}'.format(s[0], s[1], s[2], s[3],s[4],s[5]))
        print('T-stat        {:>10,.4f}  {:>10,.4f}  {:>10,.4f}  {:>10,.4f}   {:>10,.4f} {:>10,.4f}'.format(t[0], t[1], t[2], t[3],t[4],t[5]))
        print('')
        i += 1


# In[33]:


### if we use the linear GMM library "IYW", "IYR" , "IYH" , "IYF" , "IYE" , "IYK",  "IYJ" , "IYC" , "IDU" , "IYM" , "IYZ"

from linearmodels.asset_pricing import LinearFactorModelGMM
data = read_csv('fama_french.csv')
data = data.dropna()
data


# In[31]:


data.iloc[:, 7:] = data.iloc[:, 7:].values - data[["rf"]].values


# In[32]:


portfolios= data[["IYW", "IYR" , "IYH" , "IYF" , "IYE" , "IYK",  "IYJ" , "IYC" , "IDU" , "IYM" , "IYZ"]]
portfolios


# In[34]:


factors = data[["mkt_excess", "smb", "hml", "cci","cci_mkt"]]


# In[35]:


mod = LinearFactorModelGMM(portfolios, factors)
res = mod.fit()
print(res)


# In[36]:


res = mod.fit(cov_type="kernel", kernel="bartlett", disp=0)
print(res)


# In[37]:


res.full_summary


# In[ ]:




