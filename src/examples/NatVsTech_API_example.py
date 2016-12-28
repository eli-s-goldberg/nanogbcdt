# coding: utf-8

# ## Imports
# Note: python3. Please install requirements using requirments.txt in main directory.

# In[1]:

import sys
import glob
import fnmatch
import os.path
import pandas as pd
import matplotlib.pyplot as plt

# import API one directory above
from DataUtil import DataUtil

sys.path.append(os.path.abspath(os.path.join(os.path.dirname("__file__"), os.path.pardir)))
from NatVsTech import NatVsTech
from sklearn.model_selection import GridSearchCV

# ## Directory structure
# Note: we generically define directory so it will work on any OS: mac/pc/linux.
# Note: drop the "" around "__file__" when in a regular python file.

# In[2]:
PARENT_PATH = os.path.abspath(os.path.join(os.path.dirname("__file__"), os.path.pardir))
DATABASES_BASEPATH = os.path.abspath(os.path.join(os.path.dirname("__file__"), 'databases'))
IMPORT_TRAINING_DATABASE_PATH = os.path.abspath(
	os.path.join(DATABASES_BASEPATH, 'training_data'))
IMPORT_TESTING_DATABASE_PATH = os.path.abspath(
	os.path.join(DATABASES_BASEPATH, 'test_data'))
OUTPUT_DATA_SUMMARY_PATH = os.path.abspath(
	os.path.join(os.path.dirname("__file__"), 'output'))

# print the paths, just to make sure things make sense
print(PARENT_PATH)
print(DATABASES_BASEPATH)
print(IMPORT_TRAINING_DATABASE_PATH)
print(OUTPUT_DATA_SUMMARY_PATH)

# ## Training files
# Import training files, combine, and concatenate into dataframes.
# Note: if you re-run the notebook without resetting the kernal, you'll get an error. Restart the notebook kernal and
#  it will work.

# In[3]:
# set the natural and technical database training file names
NATURAL_TRAINING_DATABASE_NAME_ = 'natural_training_data.csv'
TECHNICAL_TRAINING_DATABASE_NAME_ = 'technical_training_data.csv'

# change the directory to the import training data path
os.chdir(IMPORT_TRAINING_DATABASE_PATH)

# find all csv's in the directory
training_files = glob.glob('*.csv')

# iterate through files and assign classification id
for file in training_files:
	if fnmatch.fnmatchcase(file, TECHNICAL_TRAINING_DATABASE_NAME_):
		technical_training_database = pd.DataFrame.from_csv(
			os.path.join(file), header=0, index_col=None)

		# assign classification id
		technical_training_database['classification'] = 0

	elif fnmatch.fnmatchcase(file, NATURAL_TRAINING_DATABASE_NAME_):
		natural_training_database = pd.DataFrame.from_csv(
			os.path.join(file), header=0, index_col=None)

		# assign classification id
		natural_training_database['classification'] = 1

print(training_files)
# concatenate all the data into a single file
training_data = pd.concat([natural_training_database,
						   technical_training_database])

# remoove all the na values (other filtering done later)
training_data = DataUtil.filter_na(training_data)

# ## Using the API
# Before you can use the API, you have to initialize the class. We'll then work through how the data is easily
# filtered, stored, and used for training and prediction.

# In[4]:
# initialize class
nat_v_tech = NatVsTech()

print(nat_v_tech)

# In[5]:
# filter the data of negative values
neg_filt_training_data = DataUtil.filter_negative(data=training_data)

# threshold the data with a single isotope trigger
thresh_neg_filt_training_data = DataUtil.apply_detection_threshold(data=neg_filt_training_data, threshold_value=5)

# print to maake sure we're on target
print(thresh_neg_filt_training_data.head())

# In[6]:
# right now training data contains the classification data. Split it.
(training_df, target_df) = DataUtil.split_target_from_training_data(df=thresh_neg_filt_training_data)

# print training data to check structure
print(training_df.head())

# print target data to check structure
print(target_df.head())

# In[8]:
# conform the test data for ML and store it as X and y.
# (X, y) = nat_v_tech.conform_data_for_ML(training_df=training_df, target_df=target_df)

# initialize gbc parameters to determine max estimators with least overfitting
GBC_INIT_PARAMS = {'loss': 'deviance', 'learning_rate': 0.1,
				   'min_samples_leaf': 100, 'n_estimators': 1000,
				   'max_depth': 5, 'random_state': None, 'max_features': 'sqrt'}

# print to verify parameter init structure
print(GBC_INIT_PARAMS)

# outline grid search parameters
# set optimum boosting stages. Note: n_estimators automatically set
GBC_GRID_SEARCH_PARAMS = {'loss': ['exponential', 'deviance'],
						  'learning_rate': [0.01, 0.1],
						  'min_samples_leaf': [50, 100],
						  'random_state': [None],
						  'max_features': ['sqrt', 'log2'],
						  'max_depth': [5],
						  'n_estimators': [50]}

print(GBC_GRID_SEARCH_PARAMS)

# determining optimum feature selection with rfecv
result = nat_v_tech.rfecv_feature_identify(training_df=training_df, target_df=target_df,
								  gbc_grid_params=GBC_GRID_SEARCH_PARAMS,
								  gbc_init_params=GBC_INIT_PARAMS,
								  n_splits=3)

print(result.name_list_)
print(result.grid_scores_)
print(result.holdout_predictions_)
print(result.class_scores_f1_)
print(result.class_scores_r2_)
print(result.class_scores_mae_)
print(result.feature_importances_)
result.grid_scores_.plot(kind="box", ylim=[0, 1])
plt.show()


# In[8]:
# find optimum boosting stages
optimum_boosting_stages = nat_v_tech.find_min_boosting_stages(gbc_base_params=GBC_INIT_PARAMS,
															  training_df=training_df,
															  target_df=target_df)[1]

# print optimum boosting stages
print(optimum_boosting_stages)

# In[9]:
# create grid search parameters in which to find the optimum set,
# set optimum boosting stages. Note: n_estimators automatically set
GBC_GRID_SEARCH_PARAMS = {'loss': ['exponential', 'deviance'],
						  'learning_rate': [0.01, 0.1],
						  'min_samples_leaf': [50, 100],
						  'random_state': [None],
						  'max_features': ['sqrt', 'log2'],
						  'max_depth': [5],
						  'n_estimators': [optimum_boosting_stages]}

# print search parameter grid to verify init structure
print(GBC_GRID_SEARCH_PARAMS)

# In[10]:
# find the optimum gbc parameters
gbc_fitted = nat_v_tech.find_optimum_gbc_parameters(crossfolds=5,
													training_df=training_df,
													target_df=target_df,
													gbc_search_params=GBC_GRID_SEARCH_PARAMS)

# print the optimum gbc structure
print(gbc_fitted)

# In[12]:
# use the X and y data to train the model. Then test the trained model against the test data and output results.
nat_v_tech.apply_trained_classification(test_data_path=IMPORT_TESTING_DATABASE_PATH,
										output_summary_data_path=OUTPUT_DATA_SUMMARY_PATH,
										output_summary_base_name='summary.csv',
										track_class_probabilities=[0.1, 0.1],
										isotope_trigger='140Ce',
										gbc_fitted=gbc_fitted,
										training_df=training_df,
										target_df=target_df)

