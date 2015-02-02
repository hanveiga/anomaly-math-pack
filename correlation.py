import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.nonparametric.api as nparam
import math
import commonfct as cm

"""Implementation of Local Correlation from the following paper:
   url(http://www.cs.cmu.edu/~spapadim/pdf/loco_icdm06.pdf)
"""

def local_autocovariance(ts,window,n_windows,t, exponential = False):
  """ Computes the sum of autocovariance matrices """
  ranges = range(t-window+1,t)
  cov_matrix = np.zeros((window-1,window-1))

  if exponential:
  	print 'Exponential weight not implemented yet. :('

  for index in ranges:
    cov_matrix += np.outer(ts[index:index+window-1],ts[index:index+window-1])

  return cov_matrix

def local_corr_score(ts1,ts2, t, k=0.7, lag=5):
  """ Computes the local correlation score between two auto covariance matrices """
  autocov1 = local_autocovariance(ts1,lag,1,t)
  autocov2 = local_autocovariance(ts2,lag,1,t) 

  v1, w1 = np.linalg.eigh(autocov1)
  v2, w2 = np.linalg.eigh(autocov2)
  v1, w1 = cm.order_eigenvectors(w1,v1, dim='c')
  v2, w2 = cm.order_eigenvectors(w2,v2, dim='c')

  k0 = 0 # calculate number of vectors needed
  U_k_1 = []
  U_k_2 = []
  for i in range(w1.shape[1]):
    if k0 > k:
      break;
    U_k_1.append(w1[:,i])
    U_k_2.append(w2[:,i])
    k0 += max(v1[i],v2[i])

  loco_score = 0.5 * ( np.linalg.norm(np.dot(U_k_1,w2[:,np.argmax(v2)]))  
                      + np.linalg.norm(np.dot(U_k_2,w1[:,np.argmax(v1)])) ) 
  return loco_score

def sign(ts1,ts2,window,t):
  """ Returns the sign of the product of the difference between two points in the two ts """
  delta1 = ts1[t+window-1] - ts1[t-window+1] 
  delta2 = ts2[t+window-1] - ts2[t-window+1]

  return np.sign(delta1*delta2)

"""
Implementation of the paper:
url()
"""

"""
Estimators:
"""
def mean_est(X):
  """ unbiased mean estimator """
  return 1/float(len(X))*sum(X)

def var_est(X):
  """ unbiased var estimator """
  mean_x = mean_est(X)
  summation = sum(map(lambda x: ((x - mean_x)**2), X))
  return 1/float(len(X)-1)*summation

def loc_pearson(X,Y):
  """ local pearson correlation estimator """
  mean_x = mean_est(X)
  mean_y = mean_est(Y)
  var_x = var_est(X)
  var_y = var_est(Y)
  summation = sum(map(lambda x,y: ((x - mean_x)**2)*((y-mean_y)**2), X,Y))
  return 1/float(len(X)-1)*summation*1/float(math.sqrt(var_x)*math.sqrt(var_y))

def conditional_est(X,Y):
  """
  CDF estimate using Nadaraya watson. Independent - X, dependent - Y
  """
  return nparam.KernelReg(endog=Y,
                      exog=X, reg_type='lc',
                      var_type='c', bw='cv_ls',
                      defaults=nparam.EstimatorSettings(efficient=True))

def local_correlation(ts1,ts2,t,mu_x_model=None,mu_y_model=None):
  mu_x = mean_est(ts1)
  mu_y = mean_est(ts2)
  r_xy = loc_pearson(ts1,ts2)
  var_x = var_est(ts1)
  var_y = var_est(ts2)

  if not mu_x_model and not mu_y_model:
    mu_x_model = conditional_est(ts2,ts1)
    mu_y_model = conditional_est(ts1,ts2)

  a = mu_x_model.fit([ts2[t]])[0]
  b = mu_y_model.fit([ts1[t]])[0]
  loc_corr = (r_xy + ((mu_x - a)/math.sqrt(var_x))*((mu_y - b)/math.sqrt(var_y)))/(math.sqrt(1+((mu_x-a)/math.sqrt(var_x))**2)*math.sqrt(1+((mu_y-b)/math.sqrt(var_y))**2))
  return loc_corr[0]

def local_correlation_curve(ts1,ts2,window):
  if len(ts1) != len(ts2):
    raise Exception("Error: Time series have to be the same length.")
  ts1 = np.array(ts1).astype('float')
  ts2 = np.array(ts2).astype('float')
  mu_x_model = conditional_est(ts2,ts1)
  mu_y_model = conditional_est(ts1,ts2)
  corr_values = [np.nan]*window
  for t in range(window,len(ts1)-1):
    corr_values.append(local_correlation(ts1,ts2,t,mu_x_model=mu_x_model,mu_y_model=mu_y_model))
  return corr_values

def loco_curve(ts1,ts2,lag,k=0.7,exponential=False):
  """ Computes the local correlation curve over two timeseries of the same length """
  if len(ts1) != len(ts2):
    raise Exception("Error: Time series have to be the same length.")
  local_cov_list = [np.nan]*lag
  for t in range(lag,len(ts1)-lag+1):
    local_cov_list.append(local_corr_score(ts1,ts2,t,k=k, lag=lag))
  return local_cov_list

def correlation_matrix(data_slice, t, correlation_function, **kwargs):
  tsteps , ndims = data_slice.shape
  correlation_matrix = np.empty((ndims,ndims))
  correlation_matrix.fill(np.nan)
  for i in range(ndims):
    for j in range(i,ndims):
      correlation_matrix[i,j] = correlation_function(data_slice.iloc[:,i],data_slice.iloc[:,j], t, **kwargs)
      correlation_matrix[j,i] = correlation_matrix[i,j]
  return correlation_matrix

if __name__ == '__main__':
  """ Usage example """
  indices = np.array(range(0,1000,1))/float(1000)
  ts1 = pd.Series(data=np.sin(2*indices))
  ts2 = pd.Series(data=np.cos(5*indices))

  plt.plot(ts1, alpha=0.5, label='TS1')
  plt.plot(ts2, alpha=0.5, label='TS2')
  plt.plot(local_correlation_curve(ts1,ts2,10),label='LocalC')
  plt.plot(loco_curve(ts1,ts2,10),linestyle='None',marker='.',label='LoCo')
  plt.plot(pd.rolling_corr(ts1,ts2,10), alpha=0.6, label='PeCo') #pearson correlation
  plt.legend()
  plt.show()