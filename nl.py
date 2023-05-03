import numpy as np
import pandas as pd
from scipy.stats import permutation_test, mannwhitneyu, chisquare
from operator import itemgetter

def calc_U():
    '''
    Returns the U-stest statistic for a given piece of data
    '''
    return

def partition_ranks(df):
    '''
    Split rankings up by severity.
    '''
    ranks = df['rank']
    country = df['country_name']

    total = []
    i = 0
    for rank in ranks:
        total.append((country[i], rank))
        i+=1

    total = list(set(total))
    total.sort(key=itemgetter(1), reverse=True)
    
    print("Highest-ranked country: " + str(total[0]))
    print("Lowest-ranked country: " + str(total[-1:]))

    # split the countries by bracket...take the top 16 versus the lower 16.
    high_latency = []
    low_latency = []
    for i in range(0, 17):
        high_latency.append(total[i][1])
    for i in range(17, len(total)):
        low_latency.append(total[i][1])

    return high_latency, low_latency

def calc_latency_rank_by_country(df):
    '''
    Ranks the nuclear latency of a particular country based on sums of ranking data.
    '''

    # first, get all of the data from a particular country
    countries = df['country_name']
    rankings_total = []
    for country in countries:
        cdata = df.loc[df['country_name'] == country]
        cdata = cdata.replace(-99, 0)
        cdata = cdata.replace(-77, 0)
        
        amb = sum(cdata['facility_ambiguity'])
        enr = sum(cdata['enr_type'])
        size = sum(cdata['size'])
        covert=sum(cdata['covert'])
        iaea = sum(cdata['iaea'])
        regional = sum(cdata['regional'])
        military = sum(cdata['military'])
        mil_amb = sum(cdata['military_ambiguity'])
        multinational = sum(cdata['multinational'])
        foreign_ast = sum(cdata['foreign_assistance'])
        foreign_ast_amb = sum(cdata['foreign_assistance_ambiguity'])

        
        # then sum this data; higher sums = higher latency.
        ranking = sum([amb, enr, size, covert, iaea, regional, military, mil_amb, multinational, foreign_ast, foreign_ast_amb])

        rankings_total.append(ranking)
        
    
    df = df.assign(rank=rankings_total)
    # append the latency ranking calculation to the dataset
    return df

if __name__=="__main__":
    df = pd.read_excel("Data/nl_dataset_v.1.2.xlsx")
    print(df)

    df = calc_latency_rank_by_country(df)
    # compute the U-sample statistics

    print(df)

    high_latency, low_latency = partition_ranks(df)
    # perform permutation tests over these statistics
    
    
    # perform chi-square tests over these statistics 
