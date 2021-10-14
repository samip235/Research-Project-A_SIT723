import pandas as pd
import numpy as np

df = pd.read_csv('C:/Users/folder/spambase.csv', header=0)
print(df.head())

def getBestSplit(s_x,s_y):
    maximum_value = list()
    maximum_index = []
    maximum_IG = []   
    u_s_x = np.unique(s_x)
    l = len(u_s_x)
    IG = []
    S = []
    
    for i in range(l-3):
        s = (u_s_x[i] + u_s_x[i+1])/2
        S.append(s)
        
        l_y = s_y[s_x < S[i]]
        r_y = s_y[s_x >= S[i]]
         
        p_l_0 = (len(l_y[l_y==0])+1)/(len(l_y)+2)
        p_l_1 = (len(l_y[l_y==1])+1)/(len(l_y)+2)
        
        p_r_0 = (len(r_y[r_y==0])+1)/(len(r_y)+2)
        p_r_1 = (len(r_y[r_y==0])+1)/(len(r_y)+2)
        
        ig= 1 -(p_l_0 * np.log2(p_l_0) + p_l_1 * np.log2(p_l_1)) + (p_r_0 * np.log2(p_r_0) + p_r_1 * np.log2(p_r_1))
        IG.append(ig)
        
    max_value = max(S)
    max_index = S.index(max_value)

    maximum_value.append(S[max_index])
    maximum_index.append(max_index)
    maximum_IG.append(IG[max_index])

    
