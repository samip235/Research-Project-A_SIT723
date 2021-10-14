#from tesrt_1 import getsplit
import numpy as np
import pandas as pd



class iTree(object):

  def __init__(self):
    self.split_att = -1
    self.approx_pathlength = 0
    self.split_point = -1.0
    self.left_child = None
    self.right_child = None

  def getsplit(self, s_x, s_y):
   
    self.maximum_value = []
    self.maximum_index = []
    self.maximum_IG = []
    self.maximum_IG = []
    u_s_x = np.unique(s_x,s_y)
    l = len(u_s_x)
    #print(l)
    IG = []
    S = []
    
    for i in range(l-1):
        s = (u_s_x[i] + u_s_x[i+1])/2
        S.append(s)
        
        l_y = s_y[s_x < S[i]]
        r_y = s_y[s_x >= S[i]]
         
        p_l_0 = (len(l_y[l_y==0])+1)/(len(l_y)+2)
        p_l_1 = (len(l_y[l_y==1])+1)/(len(l_y)+2)
        
        p_r_0 = (len(r_y[r_y==0])+1)/(len(r_y)+2)
        p_r_1 = (len(r_y[r_y==0])+1)/(len(r_y)+2)
        
        ig= 1 -(p_l_0 * np.log2(p_l_0) + p_l_1 * np.log2(p_l_1)) + (p_r_0 * np.log2(p_r_0) + p_r_1 * np.log2(p_r_1))
        IG.append(ig)
        
    max_value = max(S)
    max_index = S.index(max_value)

    self.maximum_value.append(S[max_index])
    self.maximum_index.append(max_index)
    self.maximum_IG.append(IG[max_index])
    split_point = self.maximum_value[0]
    return split_point()


  def build_iTree(self, data, sample_ids, height, h_max):
    num_samples = len(sample_ids)
    print(type(data))
    if (num_samples <= 1) or (height == h_max):
        self.approx_pathlength = self.c(num_samples)
        return
    else:
      samples = data[sample_ids,:]
      sample_size =len(samples)
      att_range = np.max(samples, axis=0) - np.min(samples, axis=0)
      att_list = np.where(att_range >= 0)[0]
      if (len(att_list) == 0): 
        self.approx_pathlength = self.c(num_samples)       
        return 
      else:
        np.random.shuffle(att_list)
        self.split_att = att_list[0]
        sample_vals = samples[:,self.split_att]
        max_val = max(sample_vals)
        min_val = min(sample_vals)

        # synthetic - negative class
        synthetic_data = np.array(np.random.uniform(min_val,max_val,data.shape[0]))
        #synthetic_data = list(np.random.uniform(min_val,max_val,data.shape[0]))

        #print(synthetic_data)
        #print(type(synthetic_data))

        
        y = [0 for i in range(sample_size)] # temporary positive class label # temporary class variable
        y.append([1 for i in range(sample_size)]) # temporary negative class label
       
        x = samples # temporary  positive sampels
        synthetic_data = np.append(synthetic_data, x)
        #x = list(x)
        #np.append(x,synthetic_data)
        #x.append(synthetic_data) # temporary negative samples
        
        #self.split_point = min_val + (max_val - min_val) * np.random.random()
        self.split_point = self.getsplit(data['ID0'],data['Class'])
        
        left_sample_ids = sample_ids[np.where(sample_vals < self.split_point)[0]]
        right_sample_ids = np.setdiff1d(sample_ids, left_sample_ids)
        # build a tree
        self.left_child = iTree()
        leaf_id = self.left_child.build_iTree(data, left_sample_ids, height+1, h_max)
        self.right_child = iTree()
        leaf_id = self.right_child.build_iTree(data, right_sample_ids, height+1, h_max)
    return
        
  def get_path_length(self, x):
    if (self.split_att == -1): # if leaf, return
      return self.approx_pathlength
    else:
      if (x[self.split_att] < self.split_point):
        return (1.0 + self.left_child.get_path_length(x))
      else:
        return (1.0 + self.right_child.get_path_length(x))

  def c(self, n):
    if (n <= 1):
      return 0
    else:
      return 2 * (np.log(n-1) + 0.5772156649) - (2 * (n-1) / n )

class IForest(object):

  def __init__(self, data, sample_size, num_trees, rseed):
    self.sample_size = sample_size
    self.num_trees = num_trees
    self.h_max = int(np.log2(self.sample_size))
    data_size = len(data)
    np.random.seed(rseed)
    print('\t iForest (samp. size: %s, num. trees: %s)' % (self.sample_size, self.num_trees))
    self.iforest = [object for i in range(self.num_trees)]
    d_ids = np.array([i for i in range(data_size)])   
    for i in range(self.num_trees):     
      np.random.shuffle(d_ids)
      sample_ids = d_ids[0:self.sample_size]
      self.iforest[i] = iTree()
      self.iforest[i].build_iTree(data, sample_ids, 0, self.h_max)



  def get_anomaly_score(self, x):
    avg_path_length = 0.0
    for i in range(self.num_trees):
      avg_path_length += self.iforest[i].get_path_length(x)
    return pow(2, -avg_path_length/self.num_trees)
