import numpy as np
from bisect import bisect_left
import math


class EqualWidthDiscretizer3Sigma(object):

  def __init__(self, data, nbins):

    self.n_data = len(data)
    self.n_dim = len(data[0])
    self.nbins = nbins
    self.bin_cuts = [[] for i in range(self.n_dim)]
    self.bin_counts = [[] for i in range(self.n_dim)]
    self.num_bins = [0 for i in range(self.n_dim)]
    for i in range(self.n_dim):
      b_cuts, b_counts = self.equal_width_histograms(data[:,i], nbins)
      self.bin_cuts[i] = b_cuts
      self.bin_counts[i] = b_counts
      self.num_bins[i] = len(b_counts)
      # self.num_bins[i] = len(b_cuts)

  def get_bin_cuts_counts(self):
    return self.bin_cuts, self.bin_counts

  def get_num_bins(self):
    return self.num_bins

  def adjust_extreme_bins(self, b_val):
    if (b_val < 0):
      b_val = 0
    if (b_val >= self.nbins):
      b_val = self.nbins-1
    #if (b_val > self.nbins):
    #  b_val = self.nbins  
    return b_val          

  def get_bin_id(self, x):
    x_bin_ids = [-1 for i in range(self.n_dim)]
    for i in range(self.n_dim):
      x_i = self.nbins * (x[i] - self.bin_cuts[i][0]) / (self.bin_cuts[i][-1] - self.bin_cuts[i][0])
      x_bin_ids[i] = self.adjust_extreme_bins(int(np.ceil(x_i)) - 1)
      #x_bin_ids[i] = self.adjust_extreme_bins(int(np.ceil(x_i)))
    return np.array(x_bin_ids)

  def equal_width_histograms(self, xd, nbins):
    min_val = min(xd)
    max_val = max(xd)
    mean_val = np.mean(xd)
    std_val = np.std(xd)
    r_min = max(min_val, mean_val-(3*std_val))
    r_max = min(max_val, mean_val+(3*std_val))
    [bCnts, bEdgs] = np.histogram(xd, bins=nbins, range=(r_min, r_max), normed=False, weights=None, density=False) 
    # find values smaller and larger than r_min and r_max
    low_vals = sum(xd < bEdgs[0])
    high_vals = sum(xd > bEdgs[len(bEdgs)-1])
    bCnts[0] += low_vals
    bCnts[len(bCnts)-1] += high_vals
    # now return
    return np.array(bEdgs), np.array(bCnts)
