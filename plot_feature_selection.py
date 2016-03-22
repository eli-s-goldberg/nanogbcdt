import matplotlib.pyplot as plt
import pandas as pd
import os


## Input file summary

RFECV_RESULTS_BASEPATH = os.path.join(os.path.dirname(__file__),
                                      'output', 'rfecv_results')

fileName1 = os.path.join(RFECV_RESULTS_BASEPATH,'f1_score_all.csv')
fileName2 = os.path.join(RFECV_RESULTS_BASEPATH,'class_IFIRS.csv')
fileName3 = os.path.join(RFECV_RESULTS_BASEPATH,'class_optimum_length.csv')
fileName4 = os.path.join(RFECV_RESULTS_BASEPATH,'class_sel_feature_importances.csv')
fileName5 = os.path.join(RFECV_RESULTS_BASEPATH,'class_rfecv_grid_scores.csv')
fileName6 = os.path.join(RFECV_RESULTS_BASEPATH,'class_Rsq_score_all.csv')


# --- initialize dataframes for speed ---
nameListAll = pd.DataFrame()
optimumLengthAll = pd.DataFrame()
f1ScoreAll = pd.DataFrame()
featureImportancesAll = pd.DataFrame()
rfecvGridScoresAll = pd.DataFrame()

os.chdir(RFECV_RESULTS_BASEPATH)
path = os.getcwd()
# --- import data
fileName1 = 'f1_score_all.csv'
fileName2 = 'class_IFIRS.csv'
fileName3 = 'class_optimum_length.csv'
fileName4 = 'class_sel_feature_importances.csv'
fileName5 = 'class_rfecv_grid_scores.csv'
fileName6 = 'class_Rsq_score_all.csv'
# fileName7 = 'coreIsotopes.csv'
# fileName8 = 'allIsotopes.csv'

# grab data locations
f1_score_all = os.path.join(path,fileName1)
class_IFIRS = os.path.join(path,fileName2)
class_optimum_length =os.path.join(path,fileName3)
class_sel_feature_importances =os.path.join(path,fileName4)
class_rfecv_grid_scores =os.path.join(path,fileName5)
class_Rsq_score_all =os.path.join(path,fileName6)
# coreIsotopes = os.path.join(path,fileName7)
# allIsotopes = os.path.join(path,fileName8)

# pull in CSVs
F1_score_agg = pd.DataFrame.from_csv(f1_score_all, header=0, sep=',', index_col=0, parse_dates=True, encoding=None,
                                       tupleize_cols=False, infer_datetime_format=False)
IFIRS_agg = pd.DataFrame.from_csv(class_IFIRS, header=0, sep=',', index_col=0, parse_dates=True, encoding=None,
                                       tupleize_cols=False, infer_datetime_format=False)
optimum_length = pd.DataFrame.from_csv(class_optimum_length, header=0, sep=',', index_col=0, parse_dates=True, encoding=None,
                                       tupleize_cols=False, infer_datetime_format=False)
sel_feature_importances = pd.DataFrame.from_csv(class_sel_feature_importances, header=0, sep=',', index_col=0, parse_dates=True, encoding=None,
                                       tupleize_cols=False, infer_datetime_format=False)
rfecv_grid_scores = pd.DataFrame.from_csv(class_rfecv_grid_scores, header=0, sep=',', index_col=0, parse_dates=True, encoding=None,
                                       tupleize_cols=False, infer_datetime_format=False)
r2_score = pd.DataFrame.from_csv(class_Rsq_score_all, header=0, sep=',', index_col=0, parse_dates=True, encoding=None,
                                       tupleize_cols=False, infer_datetime_format=False)
# coreIsotopes = pd.DataFrame.from_csv(coreIsotopes, header=0, sep=',', index_col=0, parse_dates=True, encoding=None,
#                                        tupleize_cols=False, infer_datetime_format=False)
# allIsotopes = pd.DataFrame.from_csv(allIsotopes, header=0, sep=',', index_col=0, parse_dates=True, encoding=None,
#                                        tupleize_cols=False, infer_datetime_format=False)
#
# print list(coreIsotopes)
# print list(allIsotopes)

print F1_score_agg.head()
# number of iterations is defined by the number of rfecv runs
iterator1 = len(rfecv_grid_scores)

# create a copy of the RFECV scores and plot
gradientBoostRFECVScores = rfecv_grid_scores.copy(deep=True)
gradientBoostRFECVScores.plot(kind='box', ylim=[0.9, 1])
plt.savefig('rfecvForced.eps')


# create a new dataframe for the holdout boxplot
holdoutBox = pd.DataFrame(F1_score_agg)


# add in a optimum length as a new column
holdoutBox['opLen'] = optimum_length
print holdoutBox.head()

# plot the scores as a function of the op length.
holdoutBox.boxplot(by='opLen')
plt.savefig('performance_by_optimum_length.eps')

print holdoutBox.groupby(by='opLen').size()
holdout_prints =  holdoutBox.groupby(by=['opLen','0']).size()

holdout_prints.to_csv('score_by_len.csv')

## Frequency distribution generation
IFIRS_agg_count = list(IFIRS_agg.values)
ISOTOPE_LIST_ = ['107Ag', '109Ag', '139La', '140Ce', '141Pr', '143Nd',
                 '146Nd', '147Sm', '149Sm', '153Eu', '157Gd', '159Tb',
                 '182W', '206Pb', '208Pb', '232Th', '238U', '25Mg',
                 '55Mn', '59Co', '60Ni', '65Cu', '66Zn', '88Sr',
                 '90Zr', '93Nb', '95Mo']
nameFeatures = ISOTOPE_LIST_

df1 = pd.DataFrame()
df2 = pd.DataFrame()

# determine frequency of features
for feature_ids in nameFeatures:
    df11 = IFIRS_agg_count.count(feature_ids)
    df1 = df1.append([df11],ignore_index=True)

    df22 = [feature_ids]
    df2 = df2.append(df22,ignore_index=True)

# turn into a percent
df1 = df1/pd.Series(iterator1)*pd.Series(100)

# sort and plot
df3 = pd.concat((df2,df1),axis=1)
df3.index = nameFeatures
df3.columns = ['feature','observation %']
df3.sort_values(by=['observation %','feature'],ascending = [True,True], inplace=True)
df3.plot(kind='barh',title = 'Feature Frequency (%) Inclusion')
plt.savefig('ffi.eps')
plt.show()
