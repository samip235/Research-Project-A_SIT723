from argument_parser import ArgumentParser
from spad import SPAD
from iforest import IForest
import sklearn.decomposition as decom
import sklearn.manifold as mf
from itertools import combinations
from sklearn.svm import OneClassSVM
from local_outlier_factor import LocalOutlierFactor
from sp import Sp
import random
import numpy as np
import pandas as pd
import timeit
import random
import sys, os

class RunAnomalyDecttion(object):

  def __init__(self, argv):
    # read the command-line arguments
    self.argument_parser = ArgumentParser(argv)
    # read the data file 
    data_file = pd.read_csv(self.argument_parser.get_data_file(), header=0)
    # data stats
    self.nrows = len(data_file.index)
    self.ncols = len(data_file.columns)
    # print data statistics
    print("Number of data records: " + str(self.nrows))
    print("Number of attributes: " + str(self.ncols))
    # column index for class label
    self.class_att_index = self.get_class_att_index(self.ncols)
    # print the class label attribute index
    print("Class attribute index: " + str(self.class_att_index))
    # slice the class lable
    data_targets = data_file.iloc[:,self.class_att_index-1]
    data_feature_values = data_file.drop(data_file.columns[self.class_att_index-1], axis=1)
    # convert target labels into integers if they are not
    unique_targets = data_targets.unique()
    map_to_int = {name: n for n, name in enumerate(unique_targets)}
    data_targets_int = data_targets.replace(map_to_int)
    # convert data into feature values and label as arrays
    self.Y = np.array(data_targets_int)
    self.X = np.array(data_feature_values)
    # number of classes
    self.nclasses = len(unique_targets)
    # print the number of classes
    print("Number of class labels: " + str(self.nclasses))
    # find the anomaly label
    ul, ulc = np.unique(self.Y, return_counts=True)
    self.anomaly_label = 0
    if (ulc[0] > ulc[1]):
      self.anomaly_label = 1
    self.num_anomalies = ulc[self.anomaly_label]
    print("Number of anomalies: " + str(self.num_anomalies))
    # get the random seed
    self.rseed = self.argument_parser.get_random_seed_methods()
    # print the random seed 
    print("Random seed (used in random methods - sp and iforest): " + str(self.rseed))
    # get train and test fold
    np.random.seed(0)
    anomaly_indices = np.array(np.where(self.Y == self.anomaly_label)[0])
    normal_indices = np.array(np.where(self.Y != self.anomaly_label)[0])
    np.random.shuffle(normal_indices)
    self.train_indices = normal_indices[:int(len(normal_indices)/2)]
    self.test_indices = np.concatenate((anomaly_indices, normal_indices[int(len(normal_indices)/2):]))
    # get train and test segment
    self.data_X = self.X[self.train_indices]
    self.data_Y = self.Y[self.train_indices]
    self.query_X = self.X[self.test_indices]
    self.query_Y = self.Y[self.test_indices]
    self.data_size = len(self.data_X)
    self.query_size= len(self.query_X)
    # perform min-max normalisation using training data range just in case
    self.min_max_normalisation()
    # check if anomaly is there in the training set
    if (len(np.where(self.data_Y == self.anomaly_label)[0]) > 0):
      print("Something is wrong, anomalies in the training data!")
      exit()  
    # Ranking measure to use
    self.ranking = self.argument_parser.get_ranking_measure()
    print("Ranking measure: " + str(self.ranking))
    # Check if any projection is required
    projection = self.argument_parser.get_data_projection_method()      
    if (projection != 'pca') and (projection != 'kpca'):
      projection = 'None'
    self.projection = projection
    print("Projection: " + str(self.projection))
    # Check if attributes are to be added
    self.add_attributes = self.argument_parser.get_add_attributes_flag()
    print("Attributes are added?: " + str(self.add_attributes))    
    # Number of bins
    self.nbins = self.argument_parser.get_number_of_bins()
    if (self.nbins <= 0):
      self.nbins = int(np.log2(self.data_size) + 1)
    print("Number of bins (b): " + str(self.nbins))
    # Histrogram type
    self.hist_type = self.argument_parser.get_histogram_type()
    print("Partition type: " + str(self.hist_type))   
    # Number of trees for tree based methods
    self.num_trees = self.argument_parser.get_number_of_trees()
    print("Number of trees (t): " + str(self.num_trees) + ' (For tree-based methods.)')
    # Neighborhood parameter k
    self.k = self.argument_parser.get_param_k()
    '''
    if (self.k <= 0):  
      self.k = int(np.sqrt(self.data_size)) + 1
    '''  
    print("Parameter k for lof: " + str(self.k))


  def  min_max_normalisation(self):
    print('\t ... Performing min-max normalisation (using training data range) on training and query data ...')
    # perform min-max normalisation using training data
    d_min = np.amin(self.data_X, axis=0)
    d_max = np.amax(self.data_X, axis=0)
    ndim = len(self.data_X[0])
    d_range = np.array([1.0 for i in range(ndim)])
    for i in range(ndim):
      if (d_min[i] < d_max[i]):
        d_range[i] = d_max[i]-d_min[i]
    for i in range(self.data_size):
      self.data_X[i,:] = (self.data_X[i,:]-d_min)/d_range
    for i in range(self.query_size):
      self.query_X[i,:] = (self.query_X[i,:]-d_min)/d_range     

  def get_class_att_index(self, ndims):
    class_att_index = ndims; # default set the first column as class
    # get the class label
    class_label_index = self.argument_parser.get_class_index()
    # check the value passed
    if (class_label_index == 'first'):
        class_att_index = 1
    elif (class_label_index == 'last'):
        class_att_index = ndims
    else:
        class_att_index = int(class_label_index)
    # check if the class attribute index is valid
    if (class_att_index < 1) or (class_att_index > ndims):
        print("ERROR: Invalid class label attribute index!")
        exit()
    # return the class attribute value index
    return class_att_index

  def run_experiment(self):
    print('Running Experiment... Please wait...')  

    # do transformation if required
    ptime1 = timeit.default_timer()
    if self.projection != 'None':
      # pca
      if self.projection == 'pca':
        print("\t ... Running PCA ...")
        pca = decom.PCA()
        # pca.fit(self.data_X[s_indices,:])
        data_pca = pca.fit_transform(self.data_X)
        query_pca = pca.transform(self.query_X)
        if (self.add_attributes):
          print('\t\t ... Adding original dimensions and PCs ...')      
          self.data_X = np.concatenate((self.data_X, data_pca),axis=1)
          self.query_X = np.concatenate((self.query_X, query_pca),axis=1)
        else:  
          print('\t\t ... Using PCs only ...')      
          self.data_X = data_pca
          self.query_X = query_pca

    ptime2 = timeit.default_timer()           
    print('\t Projection time: %.4f seconds.' % (ptime2 - ptime1))

    # display data size and parameter k  
    print('\t Data size: %s; Query size: %s' %(self.data_size, self.query_size))
    print('\t Dimensionality of the space: %s' % (len(self.data_X[0]))) 

    # Number of trees for tree based methods
    if (self.num_trees == 0):
      self.num_trees = max(len(self.data_X[0]), 100)

    # use the measure and perform task
    scores = None;
    if (self.ranking == 'iforest'):
      scores = self.iforest()         
    elif (self.ranking == 'lof'):
      scores = self.lof()                  
    elif (self.ranking == 'spad'):   
      scores = self.spad()                                      
    elif (self.ranking == 'sp25'):   
      scores = self.sp25()
    elif (self.ranking == 'svm'):   
      scores = self.one_class_svm()                                                  
    else:     
      scores = self.spad()

    # process result
    auc, pr = self.process_ad_result(scores)  
    print('AUC: %.4f' % (auc))
    print('P@%d: %.4f' % (self.num_anomalies, pr))
    print('Done!')

  def lof(self):
    # set k value
    if (self.k <= 0):  
      self.k = int(np.sqrt(self.data_size)) + 1
    print("\t ... using LOF (k=" + str(self.k) + ") ...")
    res = [0.0 for j in range(self.query_size)]
    time1 = timeit.default_timer()
    lof = LocalOutlierFactor(n_neighbors=self.k)
    lof.fit(self.data_X)
    time2 = timeit.default_timer()
    for i in range(self.query_size):
      res[i] = lof._decision_function([self.query_X[i,:]])[0]
    time3 = timeit.default_timer()
    print('\t Training time: %.4f seconds.' % (time2 - time1))
    print('\t Testing time: %.4f seconds.' % (time3 - time2))
    return res 

  def spad(self):
    print("\t ... using SPAD (b=" + str(self.nbins) + ") ...")  
    res = [0.0 for j in range(self.query_size)]
    time1 = timeit.default_timer()    
    d_ranker = SPAD(self.data_X, self.nbins, self.hist_type)
    time2 = timeit.default_timer()
    for i in range(self.query_size): 
      q_bin_ids = d_ranker.discretiser.get_bin_id(self.query_X[i,:])
      res[i] = d_ranker.log_prob_mass(q_bin_ids)
    time3 = timeit.default_timer()
    print('\t Training time: %.4f seconds.' % (time2 - time1))
    print('\t Testing time: %.4f seconds.' % (time3 - time2))
    return res

  def iforest(self):
    s_size = min(256, self.data_size)
    print("\t ... using iforest (s=" + str(s_size) + ", t=" + str(self.num_trees) + ") ...")
    res = [0.0 for j in range(self.query_size)]
    time1 = timeit.default_timer()    
    d_ranker = IForest(self.data_X, s_size, self.num_trees, self.rseed)
    time2 = timeit.default_timer()
    for i in range(self.query_size):
      res[i] = -d_ranker.get_anomaly_score(self.query_X[i,:])
    time3 = timeit.default_timer()
    print('\t Training time: %.4f seconds.' % (time2 - time1))
    print('\t Testing time: %.4f seconds.' % (time3 - time2))  
    return res    

  def sp25(self):
    s_size = 25
    print("\t ... Sp (s=" + str(s_size) + ") ...")
    res = [0.0 for j in range(self.query_size)] 
    time1 = timeit.default_timer()      
    d_ranker = Sp(self.data_X, s_size, self.rseed)
    time2 = timeit.default_timer()
    for i in range(self.query_size):
      res[i] = -d_ranker.get_anomaly_score(self.query_X[i,:])
    time3 = timeit.default_timer()
    print('\t Training time: %.4f seconds.' % (time2 - time1))
    print('\t Testing time: %.4f seconds.' % (time3 - time2))  
    return res  

  def one_class_svm(self):
    print('\t ... using one-class svm ...')
    res = [0.0 for j in range(self.query_size)]
    time1 = timeit.default_timer()
    svm = OneClassSVM()
    svm.fit(self.data_X)
    time2 = timeit.default_timer()
    for i in range(self.query_size):
      res[i] = svm.decision_function([self.query_X[i,:]])[0,0]
    time3 = timeit.default_timer()
    print('\t Training time: %.4f seconds.' % (time2 - time1))
    print('\t Testing time: %.4f seconds.' % (time3 - time2))  
    return res

  def process_ad_result(self, scores):
    # sort scores
    sorted_idices = np.argsort(np.array(scores))
    # local variables
    true_positive = 0
    false_positive = 0
    total = 0
    # Precision@num_anomalies
    pr = 0.0
    # compute AUC
    for i in range(len(sorted_idices)):
      # check the label
      if (self.query_Y[sorted_idices[i]] == self.anomaly_label):
        true_positive += 1
      else:
        false_positive += 1
        total += true_positive
      # calculate precision@num_anomalies
      if (i == self.num_anomalies-1):
        pr = true_positive / self.num_anomalies  
    # calculate auc
    auc = total / (true_positive * false_positive)
    # return auc
    return auc, pr    


start_time = timeit.default_timer()

ad = RunAnomalyDecttion(sys.argv)
ad.run_experiment()

stop_time = timeit.default_timer()

print('Total runtime: %.4f seconds.' % (stop_time - start_time))
