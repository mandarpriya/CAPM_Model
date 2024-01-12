#!/usr/bin/env python
# coding: utf-8

# In[1]:


from numpy import mat, cov, mean, hstack, multiply,sqrt,diag, \
    squeeze, ones, array, vstack, kron, zeros, eye, savez_compressed
from numpy.linalg import inv
from scipy.stats import chi2
from pandas import read_csv
import statsmodels.api as sm
import os


# In[2]:


os.chdir("/Users/mandarphatak/Downloads/")


# ## Case 1:- CFNAI index

# In[7]:


data=read_csv("ff3_sentiments.csv")
data


# In[9]:


# Split using both named colums and ix for larger blocks
dates = data['date'].values
factors = data[['mkt_excess', 'smb', 'hml','cfnai','cfnai_mkt']].values
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


# In[11]:


# Time series regressions
X = sm.add_constant(factors)
ts_res = sm.OLS(excessReturns, X).fit()
alpha = ts_res.params[0]
beta = ts_res.params[1:]
avgExcessReturns = mean(excessReturns, 0)
# Cross-section regression
cs_res = sm.OLS(avgExcessReturns.T, beta.T).fit()
riskPremia = cs_res.params


# In[12]:


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


# In[13]:


# We increase by sentiments and augmented factor. so 5  as we are using three factors
vcvAlpha = vcv[0:N * K + N:6, 0:N * K + N:6]
J = alpha @ inv(vcvAlpha) @ alpha.T
J = J[0, 0]
Jpval = 1 - chi2(11).cdf(J)
print(J)
print(Jpval)

### We dont rejec the Jtest, the null hypthesis that alpha is equal to zero is satisfied in our case


# In[23]:


vcvRiskPremia = vcv[N * K + N:, N * K + N:]
annualizedRP = 12 * riskPremia
arp = list(squeeze(annualizedRP))
arpSE = list(sqrt(12 * diag(vcvRiskPremia)))
print('        Annualized Risk Premia')
print('           Market       SMB      HML   CFNAI   CFNAI_MKT')
print('---------------------------------------------------------------')
print('Premia     {0:0.4f}    {1:0.4f}   {2:0.4f}  {3:0.4f}  {4:0.4f}'.format(arp[0], arp[1], arp[2], arp[3], arp[4]))
print('Std. Err.  {0:0.4f}    {1:0.4f}      {2:0.4f}  {3:0.4f}   {4:0.4f} '.format(arpSE[0], arpSE[1], arpSE[2],arpSE[3],arpSE[4]))
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
        print('Size: {:}, Value:{:}   Alpha   Beta(MKT)   Beta(SMB)   Beta(HML) Beta(CFNAI) Beta(CFNAI_MKT)'.format(j + 1, k + 1))
        print('Coefficients: {:>10,.4f}  {:>10,.4f}  {:>10,.4f}  {:>10,.4f} {:>10,.4f}  {:>10,.4f}'.format(a, b[0], b[1], b[2], b[3],b[4]))
        print('Std Err.      {:>10,.4f}  {:>10,.4f}  {:>10,.4f}  {:>10,.4f} {:>10,.4f}  {:>10,.4f}'.format(s[0], s[1], s[2], s[3],s[4],s[5]))
        print('T-stat        {:>10,.4f}  {:>10,.4f}  {:>10,.4f}  {:>10,.4f} {:>10,.4f}  {:>10,.4f}'.format(t[0], t[1], t[2], t[3], t[4],t[5]))
        print('')
        i += 1


# In[ ]:


## We can clearly see that Beta(mkt) is dominant and significat for all the ETFs, 


# ## Case 2:- UMCSENT Index:-
# 

# In[25]:


data = read_csv("ff3_sentiments1.csv")
data


# In[26]:


# Split using both named colums and ix for larger blocks
dates = data['date'].values
factors = data[['mkt_excess', 'smb', 'hml','umcsent','umcsent_mkt']].values
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


# In[27]:


# Time series regressions
X = sm.add_constant(factors)
ts_res = sm.OLS(excessReturns, X).fit()
alpha = ts_res.params[0]
beta = ts_res.params[1:]
avgExcessReturns = mean(excessReturns, 0)
# Cross-section regression
cs_res = sm.OLS(avgExcessReturns.T, beta.T).fit()
riskPremia = cs_res.params


# In[28]:


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


# In[29]:


# We increase by 6 as we are using three factors , sentiment and augmented factor
vcvAlpha = vcv[0:N * K + N:6, 0:N * K + N:6]
J = alpha @ inv(vcvAlpha) @ alpha.T
J = J[0, 0]
Jpval = 1 - chi2(11).cdf(J)
print(J)
print(Jpval)


# In[ ]:


### So we have the similar outcome as before we are accepting the J test


# In[30]:


vcvRiskPremia = vcv[N * K + N:, N * K + N:]
annualizedRP = 12 * riskPremia
arp = list(squeeze(annualizedRP))
arpSE = list(sqrt(12 * diag(vcvRiskPremia)))
print('        Annualized Risk Premia')
print('           Market       SMB      HML   UMCSENT   UMCSENT_MKT')
print('---------------------------------------------------------------')
print('Premia     {0:0.4f}    {1:0.4f}   {2:0.4f}  {3:0.4f}  {4:0.4f}'.format(arp[0], arp[1], arp[2], arp[3], arp[4]))
print('Std. Err.  {0:0.4f}    {1:0.4f}      {2:0.4f}  {3:0.4f}   {4:0.4f} '.format(arpSE[0], arpSE[1], arpSE[2],arpSE[3],arpSE[4]))
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
        print('Size: {:}, Value:{:}   Alpha   Beta(MKT)   Beta(SMB)   Beta(HML) Beta(UMCSENT) Beta(UMCSENT_MKT)'.format(j + 1, k + 1))
        print('Coefficients: {:>10,.4f}  {:>10,.4f}  {:>10,.4f}  {:>10,.4f} {:>10,.4f}  {:>10,.4f}'.format(a, b[0], b[1], b[2], b[3],b[4]))
        print('Std Err.      {:>10,.4f}  {:>10,.4f}  {:>10,.4f}  {:>10,.4f} {:>10,.4f}  {:>10,.4f}'.format(s[0], s[1], s[2], s[3],s[4],s[5]))
        print('T-stat        {:>10,.4f}  {:>10,.4f}  {:>10,.4f}  {:>10,.4f} {:>10,.4f}  {:>10,.4f}'.format(t[0], t[1], t[2], t[3], t[4],t[5]))
        print('')
        i += 1


# In[ ]:


### We can clearly see that Beta(mkt) is highly significant 


# ## Case 3: PMI Index
# 

# In[32]:


data=read_csv('ff3_sentiment2.csv')
data


# In[33]:


# Split using both named colums and ix for larger blocks
dates = data['date'].values
factors = data[['mkt_excess', 'smb', 'hml','pmi','pmi_mkt']].values
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


# In[34]:


# Time series regressions
X = sm.add_constant(factors)
ts_res = sm.OLS(excessReturns, X).fit()
alpha = ts_res.params[0]
beta = ts_res.params[1:]
avgExcessReturns = mean(excessReturns, 0)
# Cross-section regression
cs_res = sm.OLS(avgExcessReturns.T, beta.T).fit()
riskPremia = cs_res.params


# In[35]:


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


# In[36]:


# We increase by 6 as we are using three factors , sentiment and augmented factor
vcvAlpha = vcv[0:N * K + N:6, 0:N * K + N:6]
J = alpha @ inv(vcvAlpha) @ alpha.T
J = J[0, 0]
Jpval = 1 - chi2(11).cdf(J)
print(J)
print(Jpval)


# In[ ]:


### We donot reject the J test and accept the null H0: Alpha is equal to zero


# In[37]:


vcvRiskPremia = vcv[N * K + N:, N * K + N:]
annualizedRP = 12 * riskPremia
arp = list(squeeze(annualizedRP))
arpSE = list(sqrt(12 * diag(vcvRiskPremia)))
print('        Annualized Risk Premia')
print('           Market       SMB      HML   PMI   PMI_MKT')
print('---------------------------------------------------------------')
print('Premia     {0:0.4f}    {1:0.4f}   {2:0.4f}  {3:0.4f}  {4:0.4f}'.format(arp[0], arp[1], arp[2], arp[3], arp[4]))
print('Std. Err.  {0:0.4f}    {1:0.4f}      {2:0.4f}  {3:0.4f}   {4:0.4f} '.format(arpSE[0], arpSE[1], arpSE[2],arpSE[3],arpSE[4]))
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
        print('Size: {:}, Value:{:}   Alpha   Beta(MKT)   Beta(SMB)   Beta(HML) Beta(PMI) Beta(PMI)'.format(j + 1, k + 1))
        print('Coefficients: {:>10,.4f}  {:>10,.4f}  {:>10,.4f}  {:>10,.4f} {:>10,.4f}  {:>10,.4f}'.format(a, b[0], b[1], b[2], b[3],b[4]))
        print('Std Err.      {:>10,.4f}  {:>10,.4f}  {:>10,.4f}  {:>10,.4f} {:>10,.4f}  {:>10,.4f}'.format(s[0], s[1], s[2], s[3],s[4],s[5]))
        print('T-stat        {:>10,.4f}  {:>10,.4f}  {:>10,.4f}  {:>10,.4f} {:>10,.4f}  {:>10,.4f}'.format(t[0], t[1], t[2], t[3], t[4],t[5]))
        print('')
        i += 1


# In[ ]:


### In the case we can clearly see that  Beta(hml) is significat for most of the sectors, Beta(smb) is for a few. 
#surprisingly Beta(mkt) is not significat


# ##  Case4: Consumer Confidence Index 

# In[39]:


data=read_csv('ff3_sentiments3.csv')
data


# In[40]:


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


# In[41]:


# Time series regressions
X = sm.add_constant(factors)
ts_res = sm.OLS(excessReturns, X).fit()
alpha = ts_res.params[0]
beta = ts_res.params[1:]
avgExcessReturns = mean(excessReturns, 0)
# Cross-section regression
cs_res = sm.OLS(avgExcessReturns.T, beta.T).fit()
riskPremia = cs_res.params


# In[42]:


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


# In[43]:


# We increase by 6 as we are using three factors , sentiment and augmented factor
vcvAlpha = vcv[0:N * K + N:6, 0:N * K + N:6]
J = alpha @ inv(vcvAlpha) @ alpha.T
J = J[0, 0]
Jpval = 1 - chi2(11).cdf(J)
print(J)
print(Jpval)


# In[ ]:


### We are not rejecting the J test and thus accepting the null :- Alpha is zero


# In[44]:


vcvRiskPremia = vcv[N * K + N:, N * K + N:]
annualizedRP = 12 * riskPremia
arp = list(squeeze(annualizedRP))
arpSE = list(sqrt(12 * diag(vcvRiskPremia)))
print('        Annualized Risk Premia')
print('           Market       SMB      HML   CCI   CCI_MKT')
print('---------------------------------------------------------------')
print('Premia     {0:0.4f}    {1:0.4f}   {2:0.4f}  {3:0.4f}  {4:0.4f}'.format(arp[0], arp[1], arp[2], arp[3], arp[4]))
print('Std. Err.  {0:0.4f}    {1:0.4f}      {2:0.4f}  {3:0.4f}   {4:0.4f} '.format(arpSE[0], arpSE[1], arpSE[2],arpSE[3],arpSE[4]))
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
        print('Size: {:}, Value:{:}   Alpha   Beta(MKT)   Beta(SMB)   Beta(HML) Beta(CCI) Beta(CCI_MKT)'.format(j + 1, k + 1))
        print('Coefficients: {:>10,.4f}  {:>10,.4f}  {:>10,.4f}  {:>10,.4f} {:>10,.4f}  {:>10,.4f}'.format(a, b[0], b[1], b[2], b[3],b[4]))
        print('Std Err.      {:>10,.4f}  {:>10,.4f}  {:>10,.4f}  {:>10,.4f} {:>10,.4f}  {:>10,.4f}'.format(s[0], s[1], s[2], s[3],s[4],s[5]))
        print('T-stat        {:>10,.4f}  {:>10,.4f}  {:>10,.4f}  {:>10,.4f} {:>10,.4f}  {:>10,.4f}'.format(t[0], t[1], t[2], t[3], t[4],t[5]))
        print('')
        i += 1


# In[ ]:


## Beta(MKT) is significant for all the ETFs, foe a few Beta(SMB) is also significant.

