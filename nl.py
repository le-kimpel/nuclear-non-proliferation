import numpy as np
import pandas as pd
from scipy.stats import permutation_test, mannwhitneyu, chisquare

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

    df = calc_latency_rank(df)
    # compute the U-sample statistics

    print(df)
    # perform permutation tests over these statistics

    
    # perform chi-square tests over these statistics 
