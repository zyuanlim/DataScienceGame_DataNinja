import pandas as pd
import numpy as np

'''
First, load data frames of those predictions you wish to ensemble. 
below shows example of 11 predictions

Df is the base prediction whose prediction will sit in the first column
Other models - sequence are important, it will go from left to right to 
look for majority vote
Process: The overall majority vote selection sequence is from left to right; 
However, higher priority is given to both first and second col - unless majority 
vote is counted at least 'threshold' more than their vote's count, their vote will 
be selected. Otherwise, selection continue until the majority vote is found. 
'''

df = pd.read_csv('resnet18_Epoch36_Val85.39.csv')
resnet34scrach = pd.read_csv('resnet34_scratch_Epoch32_Val85.03_withProb.csv')
resnet18 = pd.read_csv('resnet18_epoch4_LR0.0075_sub.csv')
resnet101 = pd.read_csv('resnet101_nEpochs2_Val84.213.csv')
resnet50 = pd.read_csv('resnet_layer50_Epochs9.csv')
resnet34 = pd.read_csv('resnet_layer34_nEpochs3_x6_allTraining_Jul5.csv')
resnet34_2 = pd.read_csv('resnet_layer34_nEpochs4.csv')
caffenet = pd.read_csv('caffenet_fix.csv')
alex = pd.read_csv('alexnet_pred_prob_Val83.45.csv')
google = pd.read_csv('googlenet_pred_prob_Val83.97.csv')
inception = pd.read_csv('inception_bn_epoch_9_sub.csv')

df.sort_values('Id', inplace=True)
resnet34scrach.sort_values('Id', inplace=True)
resnet18.sort_values('Id', inplace=True)
inception.sort_values('Id', inplace=True)
resnet101.sort_values('Id', inplace=True)
resnet50.sort_values('Id', inplace=True)
resnet34.sort_values('Id', inplace=True)
resnet34_2.sort_values('Id', inplace=True)
caffenet.sort_values('Id', inplace=True)
google.sort_values('Id', inplace=True)
alex.sort_values('Id', inplace=True)

## ORDER Matters!! adding in all predictions according to the importance sequence 
df['resnet34scrach'] = resnet34scrach['label'].values     # second col 
df['resnet18'] = resnet18['label'].values                 # third col 
df['resnet101'] = resnet101['label'].values               # ...
df['resnet50'] = resnet50['label'].values
df['resnet34'] = resnet34['label'].values
df['resnet34_2'] = resnet34_2['label'].values            
df['caffenet'] = caffenet['label'].values
df['google'] = google['label'].values
df['alex'] = alex['label'].values
df['inception'] = inception['label'].values   

def majority_element(seq, threhold=2):

    seq = list(seq)
    maj_vote_count = 0
    for c in range(len(seq)):                               # find the maj_vote_count first (there could be multiple majority vote that counts as this)
        if seq.count(seq[c]) > maj_vote_count:
            maj_vote_count = seq.count(seq[c])
    
    for j in range(len(seq)):                               # loop from left cols to right
        if j <= 1:                                              # for first two cols 
            if seq.count(seq[j]) > (maj_vote_count - threhold):        # if their vote is greater than maj_vote_count - threhold
                return seq[j]                                       # return it!
        else:                                                   # for cols from third onwards 
            if seq.count(seq[j]) == maj_vote_count:                    # return whichever found first that is equal to maj_vote_count
                return seq[j]
                
            
df['maj_vote'] = df.iloc[:, 1:].apply(lambda x: majority_element(x), axis = 1)

df['maj_vote'] = df['maj_vote'].apply(np.int)

print("Num of overturned: ", np.sum(df['maj_vote'] != df['label']))

df_out = df.loc[:, ['Id', 'maj_vote']]

df_out.columns = ['Id', 'label']

df_out.to_csv('new_maj_vote_resScratch_res18_101_50_34_34_caffenet_google_alex_incept_1095Overturns.csv', index = False)














