import numpy as np
import pandas as pd
import math

"""
--------------
Common
--------------
"""

def order_eigenvectors(eigenvectors, eigenvalues, dim='r'):
  idx = eigenvalues.argsort()[::-1]
  ordered_eigenvalues = eigenvalues[idx]
  if dim == 'r':
    ordered_eigenvectors = eigenvectors[idx,:]
  elif dim == 'c':
    ordered_eigenvectors = eigenvectors[:,idx]
  return np.array(ordered_eigenvalues)/float(sum(ordered_eigenvalues)), np.array(ordered_eigenvectors)

def compute_weights(eigenvalues):
  """
  Receive a list of eigenvalues and normalise the eigenvalues
  """
  sum_eigenvalues = sum(eigenvalues)
  normalised_weights = [ eigenvalue/float(sum_eigenvalues) for eigenvalue in eigenvalues ]
  return normalised_weights
