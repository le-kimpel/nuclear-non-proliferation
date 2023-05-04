import numpy as np
import pandas as pd
from scipy.stats import permutation_test, mannwhitneyu, chisquare
from operator import itemgetter

def statistic(hdata, ldata):
    '''
    Returns the U-stest statistic for a given piece of data
    '''
    return mannwhitneyu(hdata, ldata)[0]

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
    for i in range(0, 16):
        high_latency.append(total[i])
    for i in range(16, len(total)):
        low_latency.append(total[i])

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
    df = calc_latency_rank_by_country(df)

    high_latency, low_latency = partition_ranks(df)
    print("High Latency: " + str(high_latency))
    print("Low Latency: " + str(low_latency))

    hdata = []
    ldata = []

    # compute 2-sample U statistics over this data.
    H = [high[0] for high in high_latency]
    L = [low[0] for low in low_latency]

    df = df.replace(np.nan, -1)
    for i in range(0, df.shape[0]):
        row = df.iloc[i]
        if row[0] in H:
            hdata.append(np.array(row[3:], dtype=int))
        elif row[0] in L:
            ldata.append(np.array(row[3:], dtype=int))

    # now perform a permutation test over some data and the U statistic
    # Let's compare the top 5 highest-ranked countries and the lowest-ranked 5 countries.
    highest = H[:5]
    lowest = L[-5:]

    u = np.array(mannwhitneyu(hdata, ldata))
    lowu = len(hdata) * len(ldata) - u
    print("U-test statistic (low latency): " + str(lowu)) 
    print("U-test statistic: " + str(u[0]))

    print("p-values: " + str(u[1]))
    
    PERM_RESULTS = []
    i = 0
    j = 0
    for i in range(0,len(highest)):
        A = hdata[i]
        for j in range(0,len(lowest)):
            B = ldata[j]
            D = np.array([A,B])
            perm_statistic = permutation_test(D, statistic)
        PERM_RESULTS.append(perm_statistic)

    print(PERM_RESULTS)
    for stat in PERM_RESULTS:
        print(stat)
 
    
