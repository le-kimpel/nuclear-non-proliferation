import numpy as np
import pandas as pd
from scipy.stats import permutation_test, mannwhitneyu, chisquare

def calc_latency_rank(df):
    '''
    Ranks the nuclear latency of a particular country based on sums of boolean data
    '''

    # first, get all of the data from a particular country

    # then sum this data; higher sums = higher latency.

    # append the latency ranking calculation to the dataset
    return 

if __name__=="__main__":
    df = pd.read_excel("Data/nl_dataset_v.1.2.xlsx")
    print(df)


    # compute the U-sample statistics

    # perform permutation tests over these statistics

    # perform chi-square tests over these statistics 
