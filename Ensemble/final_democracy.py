'''
First, loading all data frames, including the base one 

Df is the base model whose predictions will be voted against; 
each model will be given one vote only (not including base model)
- when deciding whether it should overturn the base model prediction, 
majority vote must have at least 'maj_threshold' num of votes from
all models 

Besides, an additional criterion ensures all predictions of 4 
from base model will not be overturned. 
'''
import pandas as pd
import numpy as np

df = pd.read_csv('best_ensemble.csv')
df0 = pd.read_csv('best_single.csv')
df1 = pd.read_csv('caffe.csv')
df2 = pd.read_csv('inception.csv')
df3 = pd.read_csv('lenet.csv')
df4 = pd.read_csv('resnet18_prev_best.csv')
df5 = pd.read_csv('resnet18_x3.csv')
df6 = pd.read_csv('resnet34_p75.csv')
df7 = pd.read_csv('resnet34_x6.csv')
df8 = pd.read_csv('resnet50_full_epo9.csv')
df9 = pd.read_csv('resnet50_p75.csv')
df10 = pd.read_csv('resnet101.csv')
df11 = pd.read_csv('vgg_16.csv')
df12 = pd.read_csv('xgb.csv')
df13 = pd.read_csv('googlenet_pred_prob_Val83.97.csv')
df14 = pd.read_csv('alexnet_pred_prob_Val83.45.csv')
df15 = pd.read_csv('resnet34_scratch.csv')

df.sort_values('Id', inplace=True)
df0.sort_values('Id', inplace=True)
df1.sort_values('Id', inplace=True)
df2.sort_values('Id', inplace=True)
df3.sort_values('Id', inplace=True)
df4.sort_values('Id', inplace=True)
df5.sort_values('Id', inplace=True)
df6.sort_values('Id', inplace=True)
df7.sort_values('Id', inplace=True)
df8.sort_values('Id', inplace=True)
df9.sort_values('Id', inplace=True)
df10.sort_values('Id', inplace=True)
df11.sort_values('Id', inplace=True)
df12.sort_values('Id', inplace=True)
df13.sort_values('Id', inplace=True)
df14.sort_values('Id', inplace=True)
df15.sort_values('Id', inplace=True)

df['best_single'] = df0['label'].values
df['caffe'] = df1['label'].values
df['inception'] = df2['label'].values
df['lenet'] = df3['label'].values
df['resnet18_prev_best'] = df4['label'].values
df['resnet18_x3'] = df5['label'].values
df['resnet34_p75'] = df6['label'].values
df['resnet34_x6'] = df7['label'].values
df['resnet50_full_epo9'] = df8['label'].values
df['resnet50_p75'] = df9['label'].values
df['resnet101'] = df10['label'].values
df['vgg_16'] = df11['label'].values
df['xgb'] = df12['label'].values
df['google'] = df13['label'].values
df['alex'] = df14['label'].values
df['resnet34_scratch'] = df15['label'].values

maj_threshold = 8

def majority_element(seq, thre = maj_threshold):
    seq = list(seq)
    for c in [1,2,3,4]:
        if seq.count(c) >= thre:
            return c
    return -1
    
       
df['maj_vote'] = df.iloc[:, 1:].apply(lambda x: x[0] if x[0] == 4 or majority_element(x[1:]) == -1 else majority_element(x[1:]), axis = 1)
#df['maj_vote'] = df.iloc[:, 1:].apply(lambda x: x[0] if majority_element(x[1:]) == -1 else majority_element(x[1:]), axis = 1)

df['maj_vote'] = df['maj_vote'].apply(np.int)

print("Num of overturned: ", np.sum(df['maj_vote'] != df['label']))

df_out = df.loc[:, ['Id', 'maj_vote']]

df_out.columns = ['Id', 'label']

df_out.to_csv('final_democracy_123_threshold8_16models_151Overturns_FINAL.csv', index = False)












