import numpy as np
from equal_width_discretization import EqualWidthDiscretizer
from equal_width_discretization_3sigma import EqualWidthDiscretizer3Sigma

class SPAD(object):

  def __init__(self, data, nbins, disc_type='EW'):
    # get the required statistics
    self.ndata = len(data)
    self.ndim = len(data[0])
    self.nbins = nbins
    self.disc_type = disc_type
    self.dimVec = np.array([i for i in range(self.ndim)])
    self.discretiser = None
    self.bin_mass = None
    if (self.disc_type == 'EW'):
      print('\t ... Equal Width discretisation (b=%s) ...' % (self.nbins))
      self.discretiser = EqualWidthDiscretizer(data, self.nbins)
    elif (self.disc_type == 'EW3S'):  
      print('\t ... Equal Frequency discretisation with 3 Sigma from mean (b=%s) ...' % (self.nbins))
      self.discretiser = EqualWidthDiscretizer3Sigma(data, self.nbins)    
    else:
      print('\t ... Equal Width discretisation (b=%s) ...' % (self.nbins))
      self.discretiser = EqualWidthDiscretizer(data, self.nbins)   
    # compute dissimilarity between eacb bins in each dimension
    self.bin_cuts, self.bin_counts = self.discretiser.get_bin_cuts_counts()
    self.num_bins =  self.discretiser.get_num_bins()
    self.bin_mass_log = self.get_bin_mass_log()
    #print('Lenght of bin_mass: %s, Lenght of bin_mass[0]: %s' %(len(self.bin_mass), len(self.bin_mass[0])))
    
  def get_bin_mass_log(self):
    bin_mass = [[] for i in range(self.ndim)]
    max_num_bins = max(self.num_bins)
    min_num_bins = min(self.num_bins)
    print('\t Max. num. of bins: %s, Min. num. of bins: %s' % (max_num_bins, min_num_bins))    
    for i in range(self.ndim):
      b_mass = [0.0 for j in range(max_num_bins)]
      for j in range(self.num_bins[i]):
        b_mass[j] = np.log((self.bin_counts[i][j] + 1) / (self.ndata + self.num_bins[i]))
      bin_mass[i] = b_mass    
    return np.array(bin_mass)    
    
  def log_prob_mass(self, x_bin_ids):   
    log_pmass = self.bin_mass_log[self.dimVec,x_bin_ids]
    return np.sum(log_pmass)