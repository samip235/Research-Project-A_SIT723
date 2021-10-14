import argparse

# Class responsible for parsing command-line arguments
class ArgumentParser(object):

  def __init__(self, argv):
    arg_parser = argparse.ArgumentParser(description="Data and algorithm parameters")
    arg_parser.add_argument('-D', '--datafile', dest='data_file', help='A datafile in csv format.', nargs='?', default="C:/Users/folder/spambase.csv")
    arg_parser.add_argument('-C', '--class-index', dest='class_index', help='The index of class label column in the data file.', default='last', type=str)
    arg_parser.add_argument('-r', '--rseed-methods', dest='random_seed_methods', help='The random seed to use in random sampling for random methods.', default=0, type=int)    
    arg_parser.add_argument('-P', '--projection', dest='data_projection_method', help='The projection method (PCA or None) if required.', default='None', type=str)
    arg_parser.add_argument('-S', '--ranking-measure', dest='ranking_measure', help='Ranking measure to use.', default='prob', type=str)
    arg_parser.add_argument('-a', '--add-attributes', dest='add_attributes', action='store_true', help='Flag indicating if additional attributes (using PCs) to be added. Applicable only when projection is used.', default=False)
    arg_parser.add_argument('-b', '--nbins', dest='number_of_bins', help='The number of bins for discretisation.', default=0, type=int)
    arg_parser.add_argument('-t', '--histogram', dest='histogram_type', help='The type of histogram (EW or EW3S).', default='EW3S', type=str)  
    arg_parser.add_argument('-n', '--trees', dest='number_of_trees', help='The number of trees for iforest.', default=100, type=int)
    arg_parser.add_argument('-w', '--sample-size', dest='sample_size', help='The number of subsamples for subsample based methods.', default=256, type=int)    
    arg_parser.add_argument('-k', '--paramK', dest='param_k', help='Parameter k for LOF.', default=0, type=int)    
    self.parsed_arguments = arg_parser.parse_args(argv[1:])

  # get the data file
  def get_data_file(self):
    return self.parsed_arguments.data_file

  # get the class index
  def get_class_index(self):
    return self.parsed_arguments.class_index

  # get the number of bins for discretisation
  def get_number_of_bins(self):
    return self.parsed_arguments.number_of_bins

   # get the histogram type
  def get_histogram_type(self):
    return self.parsed_arguments.histogram_type  

  # get the random seed1
  def get_random_seed_projections(self):
    return self.parsed_arguments.random_seed_projections

  # get the random seed2
  def get_random_seed_methods(self):
    return self.parsed_arguments.random_seed_methods

  # get the random seed
  def get_data_projection_method(self):
    return self.parsed_arguments.data_projection_method

  # get the random seed
  def get_ranking_measure(self):
    return self.parsed_arguments.ranking_measure

  # get the number of trees
  def get_number_of_trees(self):
    return self.parsed_arguments.number_of_trees

  # get the subsample size
  def get_sample_size(self):
    return self.parsed_arguments.sample_size     

  # get neighborhood parameter k
  def get_param_k(self):
    return self.parsed_arguments.param_k  

  # get the flag if attributes are to be added
  def get_add_attributes_flag(self):
    return self.parsed_arguments.add_attributes 

  # get monotonic transformation
  def monotonic_transformation(self):
    return self.parsed_arguments.monotonic_transformation       
