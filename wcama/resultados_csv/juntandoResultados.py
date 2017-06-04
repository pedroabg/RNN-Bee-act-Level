# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 17:18:38 2017

@author: PedroAlberto
"""

import pandas as pd

dirPath = "lstm_square/"
data_file = 'predicted_abelha-prev-act.csv_'



frame = pd.DataFrame()
list_ = pd.read_csv(dirPath+data_file+str(0)+".csv",index_col=None, header=0)

for i in range(1,10):
    df = pd.read_csv(dirPath+data_file+str(i)+".csv",index_col=0,  header=0)
    
    list_ = pd.concat([list_,df],axis=1)
    
pandas.DataFrame(list_).to_csv(dirPath+"all.csv")

