import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.cross_validation import StratifiedKFold as KFold
from sklearn.metrics import roc_auc_score
import operator
import matplotlib.pyplot as plt

def evalerror(preds, dtrain):
    labels = dtrain.get_label() 
    result = np.mean(preds == labels)
    return 'Accuracy', result

df = pd.read_csv('./all_images_df.csv')
df.index = df.image

train_ids = pd.read_csv('./id_train.csv')
test_ids = df.image.values.tolist()
test_ids = [i for i in test_ids if i not in train_ids.Id.values.tolist()]
test = df.ix[test_ids]
test_image = test['image'].values
del test['image']
test_X = test.values

train = df.ix[train_ids['Id']]
train_y = train_ids['label']
train_y = train_y - 1
del train['image']
train_X = train.values

print("Train and test dimension: ", train_X.shape, test_X.shape)

param = {
        'booster': 'gbtree',
        'objective': 'multi:softprob',
        'eta': 0.01,
        'gamma': 2,
        #'min_child_weight': 2,
        'max_depth': 8,
        'subsample': 0.7,
        'colsample_bytree': 0.6,        
        'num_round': 900,
        #'nthread': xgb_nthread,
        'silent': 1,
        #'seed': 2016,   
        'num_class': 4,
        #"eval_metric" : "merror",
        'verbose' : 0,
        'lambda':0.7,
        'alpha':0.7,
        'scale_positive_weight': 0.4, 
        }

NUM_MODELS = 1 ## number of models to blend / ensemble; set to 1 for single run of xgboost 

dtrain_fold = xgb.DMatrix(train_X, label=train_y, missing=np.NAN)
dvalid_fold = xgb.DMatrix(test_X, missing=np.NAN)
    
final_pred = 0
for m in range(NUM_MODELS):
    print("\n\nThis is round: ", m)
    
    param['seed'] = m ## change the seed, important!
    
    bst = xgb.train(param, dtrain_fold, param['num_round'])#, watchlist, feval=evalerror)
    
    temp_pred = bst.predict(dvalid_fold)
    
    print("Predictions for this round: ", temp_pred)
    if m == 0:
        final_pred = temp_pred
    else:
        final_pred += temp_pred
    
    temp_df = final_pred.copy() / float(m + 1)
    temp_df = pd.DataFrame(temp_df)
    temp_df['image'] = test_image
    temp_save_name = 'temp_pred_' + str(m+1) + '.csv'
    temp_df.to_csv(temp_save_name, index = False)

final_pred = final_pred / float(m + 1)
final_df = pd.DataFrame(final_pred)
final_df['image'] = test_image
final_save_name = 'xgb_pred_' + str(m+1) + '_rounds' + '.csv'
final_df.to_csv(final_save_name, index = False)
    
print("all done.")




















