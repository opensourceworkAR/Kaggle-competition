# -*- coding: utf-8 -*-
"""
Created on Fri Dec  2 19:04:37 2016

@author: Rodolfo Fernandez R. <rfern@pdx.edu>
"""


def param_search_embedded(n_folds,features, target, kf, unique_sequences, unique_sequences_fold, train, 
                          PCAkey=False):

    num_boost_round = 1000
    early_stopping_rounds = 50
    
    train_index_group=[]
    test_index_group=[]
    
    num_fold1=0
    
    print(train)
    print(train.shape)
    print(train.columns.values)
    

    param_test1={}
#    param_test1['learning_rate']= [i/10.0 for i in range(1,5)]
    param_test1['max_depth']= [1,2,3,4,5,6,7]
    param_test1['min_child_weight']=[1,2,3]
#   param_test1['gamma']=[i/10.0 for i in range(0,7)]
#    param_test1['subsample']=[i/10.0 for i in range(6,10)]
#    param_test1['colsample_bytree']=[i/10.0 for i in range(6,10)]
#    param_test1['scale_pos_weight']=[1,2,3,4,5]
#    param_test1['max_delta_step']=[0,1,2,3,4,5]

    for train_seq_index1, test_seq_index1 in kf:
        num_fold1 += 1
        print('this is creation of Kfold iterator')
        print('Start fold {} from {}'.format(num_fold1, n_folds))
    
        train_seq1 = unique_sequences[train_seq_index1]
        valid_seq1 = unique_sequences[test_seq_index1]
        
#        print(train_seq1)
#        print(valid_seq1)

        train_index = train[unique_sequences_fold.isin(train_seq1)].index.values
        test_index = train[unique_sequences_fold.isin(valid_seq1)].index.values

#        print(train_index, type(train_index))
#        print(test_index, type(test_index))
       
        train_index_group.append(train_index)
        test_index_group.append(test_index)
        
    
#    print('train index group',train_index_group)

    custom_cv = [(train_index_group[i], test_index_group[i]) for i in range(0,n_folds) ]
    
#    custom_cv=GroupShuffleSplit(n_splits=nfolds, test_size=0.5, random_state=0)

#    custom_cv = list(zip(train_index_group, test_index_group))

    print('custom cv', np.array(custom_cv).shape)
                   

#    Scaling and PCA

    if PCAkey:

        scaler1 = MinMaxScaler() 
    
        train_features=train[features]
        train_target=train[target]
            
        train_scaled=pd.DataFrame(scaler1.fit_transform(train_features), columns=train_features.columns, index=train_features.index)


        pcatest=PCA(20)
        train_features_f=pd.DataFrame(pcatest.fit_transform(train_scaled), index=train_scaled.index)

        dmfeatures=train_features_f
        dmtarget=train_target        

    else:
    
        dmfeatures=train[features]
        dmtarget=train[target]
    
#   GridSearch

    print('start grid search')
    
    classifier1=XGBClassifier( learning_rate =0.2, n_estimators=1000, max_depth=4,
        min_child_weight=2, gamma=0.1, subsample=0.9, colsample_bytree=0.9,
        objective= 'binary:logistic', nthread=4, scale_pos_weight=2, seed=27, max_delta_step=0)

    gsearch1 = GridSearchCV(estimator = classifier1, param_grid = param_test1, scoring='roc_auc',n_jobs=-1,iid=False, cv=custom_cv)
  
    
#    gsearch1.fit(dmfeatures,dmtarget,groups=train['sequence_id'])
    gsearch1.fit(dmfeatures,dmtarget)

    print('best parameters, scores')    
    print(gsearch1.grid_scores_, gsearch1.best_params_, gsearch1.best_score_)
    
#    classifier2 = xgb.train(params, dtrain, num_boost_round, evals=watchlist, early_stopping_rounds=early_stopping_rounds, verbose_eval=500)

    return gsearch1.best_params_