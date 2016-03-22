import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.grid_search import GridSearchCV
import numpy as np

def techVnatCount (target_feature):

    target_feature= list(target_feature)
    technicalCount = target_feature.count(0)
    naturalCount = target_feature.count(1)
    return (technicalCount,naturalCount)

def heldout_score(clf, X_test, y_test,max_n_estimators):
    """compute deviance scores on ``X_test`` and ``y_test``. """
    clf.fit(X_test, y_test)
    score = np.zeros((max_n_estimators,), dtype=np.float64)
    for i, y_pred in enumerate(clf.staged_decision_function(X_test)):
        score[i] = clf.loss_(y_test, y_pred)
    return score

def import_agg_rfecv_outputs(fileName1,fileName2,fileName3,fileName4,fileName5,fileName6):
    import pandas as pd
    import os
    from os import getcwd

    # This is the script working directory. It is where the file is located and where things start.
    scriptDir = getcwd()
    print(scriptDir)

    # Move up one directory and check working directory.
    # os.chdir("..")
    # gitMasterDir = getcwd()
    # print(gitMasterDir)

    # os.chdir('Data')
    transportDatabaseDir = getcwd()
    print(transportDatabaseDir)

    path1 = os.path.join(transportDatabaseDir,fileName1)
    path2 = os.path.join(transportDatabaseDir,fileName2)
    path3 = os.path.join(transportDatabaseDir,fileName3)
    path4 = os.path.join(transportDatabaseDir,fileName4)
    path5 = os.path.join(transportDatabaseDir,fileName5)
    path6 = os.path.join(transportDatabaseDir,fileName6)

    reader_f1 = pd.DataFrame.from_csv(path1, header=0, sep=',', index_col=0, parse_dates=True, encoding=None,
                                   tupleize_cols=False, infer_datetime_format=False)

    reader_f2 = pd.DataFrame.from_csv(path2, header=0, sep=',', index_col=0, parse_dates=True, encoding=None,
                                   tupleize_cols=False, infer_datetime_format=False)

    reader_f3 = pd.DataFrame.from_csv(path3, header=0, sep=',', index_col=0, parse_dates=True, encoding=None,
                                   tupleize_cols=False, infer_datetime_format=False)

    reader_f4 = pd.DataFrame.from_csv(path4, header=0, sep=',', index_col=0, parse_dates=True, encoding=None,
                                   tupleize_cols=False, infer_datetime_format=False)

    reader_f5 = pd.DataFrame.from_csv(path5, header=0, sep=',', index_col=0, parse_dates=True, encoding=None,
                                   tupleize_cols=False, infer_datetime_format=False)

    reader_f6 = pd.DataFrame.from_csv(path6, header=0, sep=',', index_col=0, parse_dates=True, encoding=None,
                                   tupleize_cols=False, infer_datetime_format=False)

    return (reader_f1,reader_f2,reader_f3,reader_f4,reader_f5,reader_f6)