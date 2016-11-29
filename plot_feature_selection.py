import os
import fnmatch
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

RFECV_RESULTS_BASEPATH = os.path.join(os.path.dirname(__file__),
                                      'output', 'rfecv_results')



ISOTOPE_LIST_ = ['107Ag', '109Ag', '139La', '140Ce', '141Pr', '143Nd',
                 '146Nd', '147Sm', '149Sm', '153Eu', '157Gd', '159Tb',
                 '182W', '206Pb', '208Pb', '232Th', '238U', '25Mg',
                 '55Mn', '59Co', '60Ni', '65Cu', '66Zn', '88Sr',
                 '90Zr', '93Nb', '95Mo']

CRITICAL_ISOTOPE_LIST_ = ['140Ce', '139La', '206Pb', '208Pb',
                          '88Sr', '90Zr', '66Zn', '107Ag']

LIST_OF_RFECV_RESULTS_ = ['1_ifirs.csv', '2_f1.csv', '3_r2.csv', '4_mae.csv',
                          '5_optimum_length.csv', '6_feature_importance.csv', '7_rfecv_grid.csv']




def main(path='.', feature_selection_filepath=RFECV_RESULTS_BASEPATH, plot_rfecv_grid = False,
         rfecv_y_limits = [0.9,1], holdout_y_limits= [0.9,1],
         holdout_statistics_filename = 'summary_holdout.csv',plot_holdout=True,
         feature_frequency_plot = True):
    os.chdir(feature_selection_filepath)


    ifirs = pd.DataFrame.from_csv(os.path.join(feature_selection_filepath,LIST_OF_RFECV_RESULTS_[0]),
                                         header=0,
                                         index_col=0)

    f1 = pd.DataFrame.from_csv(os.path.join(feature_selection_filepath, LIST_OF_RFECV_RESULTS_[1]),
                                  header=0,
                                  index_col=0)

    r2 = pd.DataFrame.from_csv(os.path.join(feature_selection_filepath, LIST_OF_RFECV_RESULTS_[2]),
                               header=0,
                               index_col=0)
    mae = pd.DataFrame.from_csv(os.path.join(feature_selection_filepath, LIST_OF_RFECV_RESULTS_[3]),
                               header=0,
                               index_col=0)

    optimum_length = pd.DataFrame.from_csv(os.path.join(feature_selection_filepath, LIST_OF_RFECV_RESULTS_[4]),
                               header=0,
                               index_col=0)

    feature_importance = pd.DataFrame.from_csv(os.path.join(feature_selection_filepath, LIST_OF_RFECV_RESULTS_[5]),
                               header=0,
                               index_col=0)

    rfecv_grid = pd.DataFrame.from_csv(os.path.join(feature_selection_filepath, LIST_OF_RFECV_RESULTS_[6]),
                               header=0,
                               index_col=0)

    # set number of iterations by inspectino of grid score length
    iterator1 = len(rfecv_grid)

    #
    if plot_rfecv_grid:
        rfecv_grid.plot(kind='box', ylim=[rfecv_y_limits])
        plt.savefig(os.path.join(feature_selection_filepath,'rfecv_plot.eps'))
        plt.show()


    # concatenate f1 and optimum length to plot holdout
    holdout = pd.DataFrame(np.array(f1),columns=['f1'])
    holdout['optimum_length'] = np.array(optimum_length)

    if plot_holdout:
        fig, ax = plt.subplots()
        box = holdout.boxplot(ax = ax, by='optimum_length')
        ax.set_ylim(holdout_y_limits)
        plt.savefig(os.path.join(feature_selection_filepath, 'holdout_plot.eps'))
        plt.show()



    # calculate Q1, median, Q3, and variance
    holdout_groups = holdout.groupby(by=['optimum_length'])
    holdout_group_names = list(holdout_groups.groups)
    holdout_group_score_track = pd.DataFrame()
    for group_ in holdout_group_names:

        holdout_group_score = pd.DataFrame()
        holdout_group_score_ = np.array(holdout_groups.get_group(group_)['f1'])
        holdout_group_score['group'] = [group_]
        holdout_group_score['quartile_1'] = [np.percentile(holdout_group_score_,25)]
        holdout_group_score['median'] = [np.percentile(holdout_group_score_, 50)]
        holdout_group_score['quartile_3'] = [np.percentile(holdout_group_score_, 75)]
        holdout_group_score['inner_quartile_range'] = [float(holdout_group_score['quartile_3']-\
                                                      holdout_group_score['quartile_1'])]
        holdout_group_score['number_observed_in_group'] = [len(holdout_groups.groups.get(group_))]
        holdout_group_score['optimality_score'] = [float((1/holdout_group_score['group'])*(1/holdout_group_score['median'])*(1/holdout_group_score['inner_quartile_range'])*(holdout_group_score['number_observed_in_group']))]

        holdout_group_score_track = holdout_group_score_track.append(holdout_group_score)

    holdout_group_score_track.to_csv(os.path.join(feature_selection_filepath,holdout_statistics_filename),index=0)

    if feature_frequency_plot:
        ifirs_count = list(ifirs.values)
        nameFeatures = ISOTOPE_LIST_

        df1 = pd.DataFrame()
        df2 = pd.DataFrame()

        # determine frequency of features
        for feature_ids in nameFeatures:
            df11 = ifirs_count.count(feature_ids)
            df1 = df1.append([df11], ignore_index=True)

            df22 = [feature_ids]
            df2 = df2.append(df22, ignore_index=True)

        # turn into a percent
        df1 = df1 / pd.Series(iterator1) * pd.Series(100)

        # sort and plot
        df3 = pd.concat((df2, df1), axis=1)
        df3.index = nameFeatures
        df3.columns = ['feature', 'observation %']
        df3.sort_values(by=['observation %', 'feature'], ascending=[True, True], inplace=True)
        df3.plot(kind='barh', title='Feature Frequency (%) Inclusion')
        plt.savefig(os.path.join(feature_selection_filepath, 'ffi.eps'))
        plt.show()


if __name__ == '__main__':
    main()
