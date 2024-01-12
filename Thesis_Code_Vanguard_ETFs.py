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


os.getcwd()


# # Importing the data for vanguard and the famafrench factors
#     

# In[11]:


data=read_csv("vanguard_famafrench.csv")


# In[14]:


data = data.dropna()

data


# # we can start with the analysis as per the methodology in the JH Cochrane (Asset Pricing)
# 

# In[15]:


# Split using both named colums and ix for larger blocks
dates = data['date'].values
factors = data[['mkt_excess', 'smb', 'hml']].values
riskfree = data['rf'].values
portfolios = data.iloc[:, 5:].values

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


# In[16]:


# Time series regressions
X = sm.add_constant(factors)
ts_res = sm.OLS(excessReturns, X).fit()
alpha = ts_res.params[0]
beta = ts_res.params[1:]
avgExcessReturns = mean(excessReturns, 0)
# Cross-section regression
cs_res = sm.OLS(avgExcessReturns.T, beta.T).fit()
riskPremia = cs_res.params


# In[17]:


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


# In[18]:


# We increase by 4 as we are using three factors
vcvAlpha = vcv[0:N * K + N:4, 0:N * K + N:4]
J = alpha @ inv(vcvAlpha) @ alpha.T
J = J[0, 0]
Jpval = 1 - chi2(11).cdf(J)


# In[19]:


print(J)
print(Jpval)


# In[ ]:


## We end up accepting the null for Vanguard Sectoral ETFs , alpha is equal to zero . 
#We proceed to calculate the annual risk premia and for individual sectoral funds


# In[20]:


vcvRiskPremia = vcv[N * K + N:, N * K + N:]
annualizedRP = 12 * riskPremia
arp = list(squeeze(annualizedRP))
arpSE = list(sqrt(12 * diag(vcvRiskPremia)))
print('        Annualized Risk Premia')
print('           Market       SMB         HML ')
print('---------------------------------------------------------------')
print('Premia     {0:0.4f}    {1:0.4f}    {2:0.4f}   '.format(arp[0], arp[1], arp[2]))
print('Std. Err.  {0:0.4f}    {1:0.4f}     {2:0.4f}   '.format(arpSE[0], arpSE[1], arpSE[2]))
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
        print('Size: {:}, Value:{:}   Alpha   Beta(MKT)   Beta(SMB)   Beta(HML) '.format(j + 1, k + 1))
        print('Coefficients: {:>10,.4f}  {:>10,.4f}  {:>10,.4f}  {:>10,.4f}'.format(a, b[0], b[1], b[2]))
        print('Std Err.      {:>10,.4f}  {:>10,.4f}  {:>10,.4f}  {:>10,.4f}'.format(s[0], s[1], s[2], s[3]))
        print('T-stat        {:>10,.4f}  {:>10,.4f}  {:>10,.4f}  {:>10,.4f}'.format(t[0], t[1], t[2], t[3]))
        print('')
        i += 1


# In[ ]:


## We can clearly observe that for most Beta(MKT) dominates and is significant , for few Beta(SMB) is significant


# In[ ]:


# We can also use the linearGMM package 


# In[21]:


from linearmodels.asset_pricing import LinearFactorModelGMM
data = read_csv('vanguard_famafrench.csv')
data = data.dropna()
data


# In[22]:


data.iloc[:, 5:] = data.iloc[:, 5:].values - data[["rf"]].values



# In[23]:


portfolios=data[["Communications Services","Consumer Discretionary", "Consumer Staples","Energy","Financials","Health Care","Industrials","Information Technology","Materials","Real Estate","Utilities"]]


# In[24]:


portfolios


# In[26]:


factors = data[["mkt_excess", "smb", "hml"]]
factors


# In[27]:


mod = LinearFactorModelGMM(portfolios, factors)
res = mod.fit()
print(res)


# In[28]:


# Now we may be having heteroscedasticity, so we go for  robustness
res = mod.fit(cov_type="kernel", kernel="bartlett", disp=0)
print(res)


# In[29]:


from scipy import linalg
from scipy.stats import f


# In[30]:


res.full_summary


# In[31]:


def grs(res_output, N, K, factors):
    T = res_output.nobs
    N = N  # number of portfolios
    K = K  # number of factors
    # dividing the GRS equation
    a = (T - N - K) / N
    # omega hat should be a K x K matrix (verified and True)
    E_f = factors.mean()
    omega_hat = (1 / T) * (factors - E_f).T.dot(factors - E_f)
    # b should be a scalar (verified and True)
    omega_hat_inv = linalg.pinv(omega_hat)  # pseudo-inverse
    b = 1 + ((E_f.T).dot(omega_hat_inv).dot(E_f))
    b_inv = b**(-1)
    # Part c
    # sigma hat should be a N x N matrix (verified and True)
    sigma_hat = res_output.std_errors
    sigma_hat = (sigma_hat).dot(sigma_hat.T)
    sigma_hat_inv = linalg.pinv(sigma_hat)  # pseudo-inverse
    alpha_hat = res_output.alphas
    c = alpha_hat.dot(sigma_hat_inv).dot(alpha_hat.T)
    # Putting the 3 GRS parts together
    grs = a * b_inv * c
    print(grs)
    dfn = N
    dfd = T - N - K
    p_value = 1 - f.cdf(grs, dfn, dfd)
    print('p-value', p_value)


# In[32]:


grs(res, 11, 3, factors)


# # the GLS test (Gibbons Ross and Shanken Test):-Gibbons, Ross & Shanken (1989)  which tests whether the factors fully explain the expected returns of various portfolios, the test suggests that the three -factor model improves the explanatory power of the returns of stocks relative to the three-factor model in our case. 

# In[33]:


## We proceed further to check for five facotr model

data=read_csv('vanguard_five_factors.csv')
data


# In[34]:


# Split using both named colums and ix for larger blocks
dates = data['date'].values
factors = data[['mkt_excess', 'smb', 'hml','rmw','cma']].values
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


# In[48]:


# Time series regressions
X = sm.add_constant(factors)
ts_res = sm.OLS(excessReturns, X).fit()
alpha = ts_res.params[0]
beta = ts_res.params[1:]
avgExcessReturns = mean(excessReturns, 0)
# Cross-section regression
cs_res = sm.OLS(avgExcessReturns.T, beta.T).fit()
riskPremia = cs_res.params


# In[36]:


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


# In[37]:


##The 𝐽-test examines whether the average pricing errors, 𝛼̂ , are zero. The 𝐽statistic has an asymptotic 𝜒2𝑁 distribution
## Our model does not rejec the J-Test, alpha is equal to zero.
vcvAlpha = vcv[0:N * K + N:6, 0:N * K + N:6]
J = alpha @ inv(vcvAlpha) @ alpha.T
J = J[0, 0]
Jpval = 1 - chi2(11).cdf(J)
J
Jpval


# In[39]:


print(J)
print(Jpval)


# In[40]:


### We use the GRS test our results

factors = data[["mkt_excess", "smb", "hml","rmw","cma"]]
factors

portfolios = data[["VOX","VCR","VDC","VDE","VFH" ,"VHT" ,"VIS" ,"VGT" ,"VAW" ,"VNQ" ,"VPU" ]]

mod = LinearFactorModelGMM(portfolios, factors)
res = mod.fit(cov_type="kernel", kernel="bartlett", disp=0)


factors


# In[41]:


def grs(res_output, N, K, factors):
    T = res_output.nobs
    N = N  # number of portfolios
    K = K  # number of factors
    # dividing the GRS equation
    a = (T - N - K) / N
    # omega hat should be a K x K matrix (verified and True)
    E_f = factors.mean()
    omega_hat = (1 / T) * (factors - E_f).T.dot(factors - E_f)
    # b should be a scalar (verified and True)
    omega_hat_inv = linalg.pinv(omega_hat)  # pseudo-inverse
    b = 1 + ((E_f.T).dot(omega_hat_inv).dot(E_f))
    b_inv = b**(-1)
    # Part c
    # sigma hat should be a N x N matrix (verified and True)
    sigma_hat = res_output.std_errors
    sigma_hat = (sigma_hat).dot(sigma_hat.T)
    sigma_hat_inv = linalg.pinv(sigma_hat)  # pseudo-inverse
    alpha_hat = res_output.alphas
    c = alpha_hat.dot(sigma_hat_inv).dot(alpha_hat.T)
    # Putting the 3 GRS parts together
    grs = a * b_inv * c
    print(grs)
    dfn = N
    dfd = T - N - K
    p_value = 1 - f.cdf(grs, dfn, dfd)
    print('p-value', p_value)


# In[42]:


grs(res,11,5,factors)


# In[ ]:


### We performed the Gibbons , Ross and Shanken test, 
##where  Gibbons, Ross and Shanken (1989, GRS hereafter) developedand analyzed a test for the ex ante mean-variance efficiency of portfolios.


# In[ ]:


## So now we can move forward for 6 factors ie using momentum


# In[45]:


data = read_csv('vanguard_ff6_data.csv')


# In[46]:


data


# In[47]:


# Split using both named colums and ix for larger blocks
dates = data['date'].values
factors = data[['mkt_excess', 'smb', 'hml','rmw','cma','mom']].values
riskfree = data['rf'].values
portfolios = data.iloc[:, 8:].values

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


# In[49]:


# Time series regressions
X = sm.add_constant(factors)
ts_res = sm.OLS(excessReturns, X).fit()
alpha = ts_res.params[0]
beta = ts_res.params[1:]
avgExcessReturns = mean(excessReturns, 0)
# Cross-section regression
cs_res = sm.OLS(avgExcessReturns.T, beta.T).fit()
riskPremia = cs_res.params


# In[50]:


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


# In[51]:


##The 𝐽-test examines whether the average pricing errors, 𝛼̂ , are zero. The 𝐽statistic has an asymptotic 𝜒2𝑁 distribution
## Our model does not rejec the J-Test, alpha is equal to zero.
vcvAlpha = vcv[0:N * K + N:7, 0:N * K + N:7]
J = alpha @ inv(vcvAlpha) @ alpha.T
J = J[0, 0]
Jpval = 1 - chi2(11).cdf(J)




# In[52]:


print(J)
print(Jpval)


# In[ ]:


## We again have the similar outcome that our model doesnt reject the J statistics, alpha is = zero ie no pricing erorr


# In[54]:


data=read_csv('vanguard_ff6_data.csv')
data


# In[55]:


data.iloc[:, 8:] = data.iloc[:, 8:].values - data[["rf"]].values


# In[59]:


## We can test the GRS test
### We use the GRS test our results

factors = data[["mkt_excess", "smb", "hml","rmw","cma",'mom']]


portfolios = data[["Communications Services","Consumer Discretionary", "Consumer Staples","Energy","Financials","Health Care","Industrials","Information Technology","Materials","Real Estate","Utilities" ]]

mod = LinearFactorModelGMM(portfolios, factors)
res = mod.fit(cov_type="kernel", kernel="bartlett", disp=0)



print(res)


# In[60]:


res.full_summary


# In[61]:


def grs(res_output, N, K, factors):
    T = res_output.nobs
    N = N  # number of portfolios
    K = K  # number of factors
    # dividing the GRS equation
    a = (T - N - K) / N
    # omega hat should be a K x K matrix (verified and True)
    E_f = factors.mean()
    omega_hat = (1 / T) * (factors - E_f).T.dot(factors - E_f)
    # b should be a scalar (verified and True)
    omega_hat_inv = linalg.pinv(omega_hat)  # pseudo-inverse
    b = 1 + ((E_f.T).dot(omega_hat_inv).dot(E_f))
    b_inv = b**(-1)
    # Part c
    # sigma hat should be a N x N matrix (verified and True)
    sigma_hat = res_output.std_errors
    sigma_hat = (sigma_hat).dot(sigma_hat.T)
    sigma_hat_inv = linalg.pinv(sigma_hat)  # pseudo-inverse
    alpha_hat = res_output.alphas
    c = alpha_hat.dot(sigma_hat_inv).dot(alpha_hat.T)
    # Putting the 3 GRS parts together
    grs = a * b_inv * c
    print(grs)
    dfn = N
    dfd = T - N - K
    p_value = 1 - f.cdf(grs, dfn, dfd)
    print('p-value', p_value)


# In[62]:


grs(res_output=res,N=11,K=6,factors=factors)


# In[ ]:


###Our GRS test is also valid.

