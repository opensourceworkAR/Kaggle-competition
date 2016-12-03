# -*- coding: utf-8 -*-
"""
Created on Fri Dec  2 20:12:17 2016

@author: rodolforfr
"""

def run_train_predict(train, test, features, target,params_model, num_features, channels, csp_n=4,
                      seq_number=1,csp_init=0,csp_end=0.1, 
                      nfolds=3, random_state=2016,
                      mode=0, PCAkey=False, PCAgraph=False,
                      PCAkeyGS=False, SEQoriginal=False,
                     Oversampling=False, GridSearch=False, pred_per_patient=False, CSPkey=False, CSPkey1=False):
    
    
#    print(train)
#    print(train.shape)
#    print(train.columns.values)
#    print(type(train['result']))
    
    function_params = OrderedDict()
    function_params["nfolds"]=nfolds
    function_params["random_state"]= random_state
    function_params["PCAkey"] = PCAkey
    function_params["PCAgraph"]= PCAgraph
    function_params["PCAkeyGS"]= PCAkeyGS
    function_params["SEQoriginal"]= SEQoriginal
    function_params["Oversampling"]= Oversampling
    function_params["GridSearch"]= GridSearch
    
    np.set_printoptions(suppress=True)
    
    #print('train type', type(train),'train', train, 'train index', train.index.values)
    #print('test type',type(test),'test', test, 'test index', test.index.values)

    #train=train.iloc[0:120]
    #test=test.iloc[0:100]
    
    #print('test',test)
    

    
    #train_seq=train['Id']
    
    #if seq_number==6:
    #    pandas_number=0
    #else:
    #    pandas_number=seq_number
    
    #print('pandas number', pandas_number)
    
    #train=train[(train_seq % 1000) % 6 == pandas_number]
    
    #train_seq_rev=train['Id'].values.tolist()#[0:120]
    #result_seq_rev=np.int64(train[target].values.tolist())#[0:120])
    
    #for item111 in train_seq_rev:
        
    #    if (item111 % 1000) % 6 != pandas_number:
            
    #        print('error pandas numerb',(item111 % 1000) % 6, pandas_number)
        
    #    else:
            
    #        print('all good!', (item111 % 1000) % 6, pandas_number)
    
    #file_name_train_sequence=[]
    #seq_number_list=[]
   
    #for i,f_id in enumerate(train_seq_rev):
                           
    #   real_f_id=f_id % 100000 
    #   name_file1="./train_"+str(mode)+'/'+str(mode)+'_'+str(real_f_id)+'_'+str(result_seq_rev[i])+'.mat'
    #   file_name_train_sequence.append(name_file1)
    #   print(name_file1)
    #  
    #   try:
    #       sequence_from_mat_seq = mat_to_pandas_seq(name_file1)
    #   except:
    #       print('Some error here {}...'.format(name_file1))
    #       seq_number_list.append(0)
    #       continue
       
    #   seq_number_list.append(int(sequence_from_mat_seq[0][0][0][0]))
    #   print('another done', i)

#    pandas_array=np.array(list(np.loadtxt('pandas_sequences_train_3.txt', delimiter=',')))
    
#    print('pandas_array', len(pandas_array), pandas_array)

#    total_sequences_lists=[result_seq_rev, seq_number_list]
        
#   pandas_sequences=pd.Dataframe(total_sequences_lists, columns=['Id_number', 'sequences'])
    
    
    #pandas_sequences=pd.DataFrame(np.array(seq_number_list),columns=['sequences'], index=train.index)
    
    #print('pandas_sequences',pandas_sequences.shape, pandas_sequences)
    print('train shape',train.shape)
    train=train[train['sequence_num']==seq_number]
    
#    for i in pandas_sequences['Id_number'].values.tolist():
        
#        for j in train.index.values.tolist():
            
#            if 
#            train_list.append()
    
#    train=train[train['Id']==Id_number &
                
#                pandas_sequences[pandas_sequences.isin(train['Id'])]

    
#    np.savetxt('pandas_sequences.txt', pandas_sequences, delimiter=',')

      
    
    print(#'train_seq',len(train_seq_rev),
          'train shape',train.shape, 'train', train.shape)
    
    #print('train type', type(train),'train', train, 'train index', train.index.values)
    #print('test type',type(test),'test', test, 'test index', test.index.values)
    
    unique_seq = train.drop_duplicates(subset=['sequence_id'])
    unique_seq_y = unique_seq['result'].values
    
    print('unique seq y', len(unique_seq_y) )
    
    n_samples=len(unique_seq_y)
    print('length',n_samples)
    unique_seq_X = np.zeros(n_samples)
    
    print('unique seq X', len(unique_seq_X)  )
    
    
    print('train pre', train.shape) 

    yfull_train = dict()
    yfull_test = copy.deepcopy(test[['Id']].astype(object))

    unique_sequences = np.array(train['sequence_id'].unique())
    print('unique sequences pre', unique_sequences.shape)

    groups1=np.fix(unique_sequences/1000)
    
    groups2=groups1.astype(int)
    #    print('groups', groups2)
        
        
        
    gkf = GroupKFold(n_splits=3)
    test1=gkf.split(unique_sequences, groups=groups2)
    test2=gkf.split(unique_sequences, groups=groups2)
    
    #random_state=random_state
    print('unique sequences', unique_sequences.shape)
    #    splitKF = KFold(len(unique_sequences), n_folds=nfolds, shuffle=True, random_state=random_state)
    #    kf = NewKF(n_splits=nfolds, shuffle=True, random_state=random_state)
    kf = StratKF(n_splits=nfolds, shuffle=False, random_state=random_state)
    
    num_fold = 0
    num_fold1=0
    

    
    if SEQoriginal:
        sequences_full=np.mod(train['sequence_id'].values,1000)
        print('sequences full', sequences_full.shape)
        unique_sequences2=np.mod(unique_sequences,1000)
        unique_sequences_fold=pd.Series(sequences_full, index=train['sequence_id'].index)
#        print('unique_sequences_fold', unique_sequences_fold)
    
        unique_sequences = np.unique(unique_sequences2)
        print('unique sequences pre', unique_sequences.shape)

    else:
        unique_sequences_fold=pd.Series(train['sequence_id'], index=train['sequence_id'].index)

    
    
    
    num_boost_round = 1000
    early_stopping_rounds = 50
    
    
    eta = 0.1
    max_depth = 4
    subsample = 0.9
    colsample_bytree = 0.9
    start_time = time.time()

    print('XGBoost params. ETA: {}, MAX_DEPTH: {}, SUBSAMPLE: {}, COLSAMPLE_BY_TREE: {}'.format(eta, max_depth, subsample, colsample_bytree))
    
    params = OrderedDict()
    params["objective"]= "binary:logistic"
    params["booster"]= "gbtree"
    params["eval_metric"]= "auc"
    params["eta"] = eta
    params["tree_method"]='exact'
    params["max_depth"]= max_depth
    params["subsample"] =subsample
    params["colsample_bytree"]= colsample_bytree
    params["silent"] =1
    params["seed"] =random_state
    params["gamma"] =0.1
    params["min_child_weight"] =2
    params["scale_pos_weight"]=2
    params["seed"]=27

    xgboost_to_xgb={
    
    "learning_rate" : "eta",
    "reg_alpha" : "alpha",
   
    "reg_lambda" : "lambda" }

#    Parameters from previous run, if any

    if type(params_model) is OrderedDict:
        
        for item in params_model:
        
            params[item]=params_model[item]
            
        print(params)
    
    
    
    
    
#   Using best parameters to train model 


        
        
    if CSPkey1 and mode!=0:
        
        seq_chosen=seq_number
        csp_components=csp_n
        
        print('test length ',test.shape[0])
        
        for item in range(test.shape[0]):
            
            if item==0:
                
                test_3d_pre=test[features].iloc[[item]].drop(['file_size','patient_id'],1).values
                test_3d=np.reshape(test_3d_pre,newshape=(num_features, channels), order='F')
                continue
                
            test_3d_pre = test[features].iloc[[item]].drop(['file_size','patient_id'],1).values
            test_red=np.reshape(test_3d_pre,newshape=(num_features, channels), order='F')
            
            test_3d=np.dstack((test_3d,test_red))
          
        
        
        csp_test=test_3d.transpose((2,1,0))
        print('csp_test',csp_test.shape) 
                        
    if GridSearch and CSPkey:
        
        pass
    
    if GridSearch and CSPkey==False and CSPkey1==False:
        
        pass
            
            
    if GridSearch and CSPkey1 and mode!=0:
        
        for item in range(train.shape[0]):
            
            if item==0:
                
                train_3d_pre=train[features].iloc[[item]].drop(['file_size','patient_id'],1).values
                train_3d=np.reshape(train_3d_pre,newshape=(num_features, channels), order='F')
                continue
                
            train_3d_pre = train[features].iloc[[item]].drop(['file_size','patient_id'],1).values
            train_red=np.reshape(train_3d_pre,newshape=(num_features, channels), order='F')
            
            train_3d=np.dstack((train_3d,test_red))
          
        
        
        csp_train_gs=train_3d.transpose((2,1,0))
        print('csp_train_gs',csp_train_gs.shape, csp_train_gs )
        
        CSPtest=CSP(n_components=csp_components, transform_into='csp_space')
        
        target_gs=train[target].values.astype(np.int64)
        
        print('target gs',target_gs.shape, target_gs)
        
        CSPtest.fit(csp_train_gs,target_gs)
        
        csp_train_gs_final=CSPtest.transform(csp_train_gs)

        #csp_train_gs_final=CSPtest.fit_transform(train[fea.values,target_gs)
        
        print('csp_train_gs_final',csp_train_gs_final.shape, csp_train_gs_final)
        
        
        train_Id=train['Id'].values.astype(np.int64)
        train_seq_Id=train['sequence_id'].values.astype(np.int64)
        train_patient_Id=train['patient_id'].values.astype(np.int64)
        train_result=train[target].values.astype(np.int64)
        
        
        csp_train_gs_final_index=np.column_stack((train_Id, train_seq_Id, train_patient_Id,
                                                  csp_train_gs_final, train_result,
                                                  train.index.values.astype(np.int64)))
        
        print('csp_train_gs_final_index',csp_train_gs_final_index)
        
        csp_train_gs_f_index=csp_train_gs_final_index[~np.any(np.isinf(csp_train_gs_final_index), axis=1)]
        
        #print('csp_train_gs_f_index',csp_train_gs_f_index)
        index_csp_train_gs=csp_train_gs_f_index[:,csp_train_gs_f_index.shape[1]-1].astype(np.int64)
        
        #print('index_csp_train_gs',index_csp_train_gs)
        
        csp_train_gs_f=np.delete(csp_train_gs_f_index,csp_train_gs_f_index.shape[1]-1, 1)
        
        #print('csp_train_gs_f',csp_train_gs_f)
        
        features_names=['feature'+str(i) for i in range(channels)]
        
        train_gs_columns=['Id','sequence_id', 'patient_id']+features_names+['result']
        
        
        train_gs_f=pd.DataFrame(csp_train_gs_f, index=index_csp_train_gs, columns =train_gs_columns)
        
        #print('train gs f',train_gs_f)
    
        splitKF=kf.split(unique_seq_X, unique_seq_y)
        
        #print('splitKF',splitKF)
    
        best_param=param_search_embedded(nfolds, train_gs_columns, target, splitKF, unique_sequences, 
                                         unique_sequences_fold,train_gs_f, PCAkeyGS)
    
        print('after best_param', best_param)
        
        
        for key in best_param:
            if key in xgboost_to_xgb:   
                best_param[xgboost_to_xgb[key]]=best_param[key]
                del best_param[key]
                
        #print ('substitution', best_param)
        params={key : best_param.get(key, value) for key, value in params.items()}
        
        print (params)
        
        
        
        
        
    
    if CSPkey and mode!=0:
        
        seq_chosen=seq_number
        csp_components=csp_n

#        train_seq_Id=train['sequence_id'].values.astype(np.int64)
        
#        files_id_test=test['Id'].tolist()
#        
#        print('files_number',len(files_id_test))
#        
#        file_name_test=[]
#        
#        for i,f_id in enumerate(files_id_test):
#            
##            if train_seq_Id[i]==seq_number:
#                
#            real_f_id=f_id % 100000   
#            file_name_test.append("./test_"+str(mode)+'_new/new_'+str(mode)+'_'+str(real_f_id)+'.mat')        
#
#
#        print('files_id_test',len(files_id_test),'file_name_test',len(file_name_test))
#        
#        test_div=20
#        
#        parts_test=int(len(file_name_test)/test_div)+1
#        
#        print('parts_test', parts_test)
#        
#        csp_test=[]
#        
#        for k in range(parts_test):
#        
#            track1=0
#            
#            ki=k
#            if k==(parts_test-1):
#                kfin=len(file_name_test)
#                print('kfin',kfin)
#            else:
#                kfin=(k+1)*test_div
#                print('kfin',kfin)
#
#            for i, fl1 in enumerate(file_name_test[ki*test_div:kfin]):
#            #    print(i)
#
#                if i==track1:
#        
#                    print('checking')
#     
#
#                    tables, sequence_from_mat, samp_freq = mat_to_pandas(fl1)
#                
#                    csp_left=int(csp_init*10*60*samp_freq)
#                    csp_right=int(csp_end*10*60*samp_freq)
#                
#                    print('csp left right', csp_left, csp_right)
#        
#        
#
#                    data_csp_test=np.transpose(tables.values[csp_left:csp_right,:])
#                
#                
#          
#                    print('done!')
#        
#                    continue
#              
#   
#                try:
#                    tables1, sequence_from_mat1, samp_freq1 = mat_to_pandas(fl1)
#                except:
#                    print('Some error here {}...'.format(fl1))
#                    continue
#                
#                csp_left=int(csp_init*10*60*samp_freq1)
#                csp_right=int(csp_end*10*60*samp_freq1)
#            
#                print('csp left right', csp_left, csp_right)
#            
#        
#                temp_matrix=np.transpose(tables1.values[csp_left:csp_right,:])
#                data_csp_test=np.dstack((data_csp_test,temp_matrix))
#                print('data_csp_test',data_csp_test.shape)
#
##            csp_test.append(data_csp_test.transpose((2,0,1)))
#
#            outfile = "csp_test_"+str(mode) +"_"+ "part_"+str(k)+ ".txt"
#            data_csp_test.transpose((2,0,1)).tofile(outfile)
#
#            del data_csp_test
            
#            print('csp_test',len(csp_test))     
    
        
        
        
        if GridSearch:
            
            seq_chosen=seq_number
            csp_components=csp_n
        

        
            files_id_train=train['Id'].tolist()
            results_id_train=train[target].tolist()
        
            print('files_number',len(files_id_train))
            
       
            file_name_train=[]
        
            for i,f_id in enumerate(files_id_train):
                
                if (f_id % 1000) % 6 == 0:
                     train_seq_Id=6
                else:
                     train_seq_Id=(f_id % 1000) % 6
                
                if train_seq_Id==seq_chosen:
                
                    real_f_id=f_id % 100000   
                    file_name_train.append("./train_"+str(mode)+'/'+str(mode)+'_'
                                      +str(real_f_id)+'_'+str(results_id_train[i])+'.mat')        


            print('files_id_train',len(files_id_train),'file_name_train',len(file_name_train))
        
       
        
            track1=0

            for i, fl1 in enumerate(file_name_train):
                #    print(i)

                if i==track1:
        
                    print('checking')
     

                    tables, sequence_from_mat, samp_freq = mat_to_pandas(fl1)
                
                    csp_left=int(csp_init*10*60*samp_freq)
                    csp_right=int(csp_end*10*60*samp_freq)
                
                    print('csp left right', csp_left, csp_right)
        
        
                    if int(sequence_from_mat[0][0][0][0])==seq_chosen:
                
                        data_csp_train=tables.values[csp_left:csp_right,:]
                        print('done!')
                        continue
                    
                    else:
                        track1+=1
                        continue
                
   
                try:
                    tables1, sequence_from_mat1, samp_freq1 = mat_to_pandas(fl1)
                except:
                    print('Some error here {}...'.format(fl1))
                    continue
                
                csp_left=int(csp_init*10*60*samp_freq1)
                csp_right=int(csp_end*10*60*samp_freq1)
            
                print('csp left right', csp_left, csp_right)
                
                if int(sequence_from_mat1[0][0][0][0])==seq_chosen:
        
                    temp_matrix=tables1.values[csp_left:csp_right,:]
                    data_csp_train=np.dstack((data_csp_train,temp_matrix))
                    print('data_csp_train',data_csp_train.shape)

            csp_train_gs=data_csp_train.transpose((2,1,0))

            print('csp_test',csp_train.shape, csp_train_gs)
            
            
            
        
            CSPtest=CSP(n_components=csp_components)
        
            target_gs=train[target].values.astype(np.int64)
        
            print('target gs',target_gs.shape, target_gs)
        
            CSPtest.fit(csp_train_gs,target_gs)
        
            csp_train_gs_final=CSPtest.transform(csp_train_gs)

            #csp_train_gs_final=CSPtest.fit_transform(train[fea.values,target_gs)
        
            print('csp_train_gs_final',csp_train_gs_final.shape, csp_train_gs_final)
        
        
            train_Id=train['Id'].values.astype(np.int64)
            train_seq_Id=train['sequence_id'].values.astype(np.int64)
            train_patient_Id=train['patient_id'].values.astype(np.int64)
            train_result=train[target].values.astype(np.int64)
        
        
            csp_train_gs_final_index=np.column_stack((train_Id, train_seq_Id, train_patient_Id,
                                                  csp_train_gs_final, train_result,
                                                  train.index.values.astype(np.int64)))
        
            print('csp_train_gs_final_index',csp_train_gs_final_index)
            
            csp_train_gs_f_index=csp_train_gs_final_index[~np.any(np.isinf(csp_train_gs_final_index), axis=1)]
        
            #print('csp_train_gs_f_index',csp_train_gs_f_index)
            index_csp_train_gs=csp_train_gs_f_index[:,csp_train_gs_f_index.shape[1]-1].astype(np.int64)
        
            #print('index_csp_train_gs',index_csp_train_gs)
        
            csp_train_gs_f=np.delete(csp_train_gs_f_index,csp_train_gs_f_index.shape[1]-1, 1)
        
            #print('csp_train_gs_f',csp_train_gs_f)
        
            features_names=['feature'+str(i) for i in range(channels)]
        
            train_gs_columns=['Id','sequence_id', 'patient_id']+features_names+['result']
        
        
            train_gs_f=pd.DataFrame(csp_train_gs_f, index=index_csp_train_gs, columns =train_gs_columns)
            
            #print('train gs f',train_gs_f)
    
            splitKF=kf.split(unique_seq_X, unique_seq_y)
        
            #print('splitKF',splitKF)
    
            best_param=param_search_embedded(nfolds, train_gs_columns, target, splitKF, unique_sequences, 
                                         unique_sequences_fold,train_gs_f, PCAkeyGS)
    
            print('after best_param', best_param)
        
        
            for key in best_param:
                if key in xgboost_to_xgb:   
                    best_param[xgboost_to_xgb[key]]=best_param[key]
                    del best_param[key]
                
            #print ('substitution', best_param)
            params={key : best_param.get(key, value) for key, value in params.items()}
        
            print (params)
        
        
        
        
        
        
        
        
        
        
        
        

    
    for train_seq_index, test_seq_index in kf.split(unique_seq_X, unique_seq_y):
        num_fold += 1
        print('Start fold {} from {}'.format(num_fold, nfolds))
        train_seq = unique_sequences[train_seq_index]
        valid_seq = unique_sequences[test_seq_index]
        print('Length of train people: {}'.format(len(train_seq)))
        print('Length of valid people: {}'.format(len(valid_seq)))
        
#        print('train_seq',train_seq)
#        print('valid_seq',valid_seq)

        X_train, X_valid = train[unique_sequences_fold.isin(train_seq)][features], train[unique_sequences_fold.isin(valid_seq)][features]
        y_train, y_valid = train[unique_sequences_fold.isin(train_seq)][target], train[unique_sequences_fold.isin(valid_seq)][target]
        X_test = test[features]
        
        X_train_seq, X_valid_seq =train[unique_sequences_fold.isin(train_seq)]['sequence_id'],\
                                    train[unique_sequences_fold.isin(valid_seq)]['sequence_id']
        
        print('X_train index',X_train.index)
        print('X_valid index',y_train.index)
        print('X_test index', X_test.index.shape)

        print('Length train:', len(X_train))
        print('Length valid:', len(X_valid))
        
        print('X_train_seq', X_train_seq.shape)
        print('X_valid_seq', X_valid_seq.shape)
        
#       Scaling for PCA


        scaler = MinMaxScaler()   
        
        Xtrain_scaled=pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
        Xvalid_scaled=pd.DataFrame(scaler.fit_transform(X_valid), columns=X_valid.columns, index=X_valid.index )

        Xtest_scaled=pd.DataFrame(scaler.fit_transform(X_test), columns=X_test.columns, index=X_test.index)


        if PCAgraph:
            
            pcatest=KernelPCA(n_components=20)
            pcatest.fit(Xtrain_scaled)
            var1=np.cumsum(np.round(pcatest.explained_variance_ratio_, decimals=4)*100)
            f1 = plt.figure()
            print(var1)
            plt.plot(var1)
            plt.show()

    
        if PCAkey:      
        
#       PCA transformation 
            pcatest=PCA(n_components=20)
            X_train_f=pd.DataFrame(pcatest.fit_transform(Xtrain_scaled), index=Xtrain_scaled.index)
            X_valid_f=pd.DataFrame(pcatest.fit_transform(Xvalid_scaled), index=Xvalid_scaled.index)

            X_test_f=pd.DataFrame(pcatest.fit_transform(Xtest_scaled), index=Xtest_scaled.index)


        
        if CSPkey1 and mode!=0:
            
            print('train length ', X_train.shape[0])
            
            for item in range(X_train.shape[0]):
            
                if item==0:
                
                    X_train_3d_pre=X_train[features].iloc[[item]].drop(['file_size','patient_id'],1).values
                    X_train_3d=np.reshape(X_train_3d_pre,newshape=(num_features, channels), order='F')
                    continue
                
                X_train_3d_pre = X_train[features].iloc[[item]].drop(['file_size','patient_id'],1).values
                X_train_red=np.reshape(X_train_3d_pre,newshape=(num_features, channels), order='F')
                X_train_3d=np.dstack((X_train_3d,X_train_red))
          
        
        
            csp_train=X_train_3d.transpose((2,1,0))
            
            
            print('valid length ',X_valid.shape[0])
            
            for item in range(X_valid.shape[0]):
            
                if item==0:
                
                    X_valid_3d_pre=X_valid[features].iloc[[item]].drop(['file_size','patient_id'],1).values
                    X_valid_3d=np.reshape(X_valid_3d_pre,newshape=(num_features, channels), order='F')
                    continue
                
                X_valid_3d_pre = X_valid[features].iloc[[item]].drop(['file_size','patient_id'],1).values
                X_valid_red=np.reshape(X_valid_3d_pre,newshape=(num_features, channels), order='F')
                X_valid_3d=np.dstack((X_valid_3d,X_valid_red))
          
        
        
            csp_valid=X_valid_3d.transpose((2,1,0))
            
            y_csp_train=y_train.values
            y_csp_valid=y_valid.values
            
            
            print('csp_train',csp_train.shape)
            print('csp_valid',csp_valid.shape)
            print('csp_test',csp_test.shape) 
            
            CSPtest=CSP(n_components=csp_components)
        
            csp_train_final=CSPtest.fit_transform(csp_train, y_csp_train)
            csp_valid_final=CSPtest.transform(csp_valid)
            csp_test_final=CSPtest.transform(csp_test)
            
            print('csp train final' ,csp_train_final.shape, 'csp valid final', csp_valid_final.shape)
            print('csp test final' ,csp_test_final.shape)
            
            csp_train_final_index=np.column_stack((csp_train_final, X_train.index.values))
            csp_valid_final_index=np.column_stack((csp_valid_final, X_valid.index.values))
            csp_test_final_index=np.column_stack((csp_test_final, X_test.index.values))
            
            csp_test_final_index[csp_test_final_index == inf] = 100000
            csp_test_final_index[csp_test_final_index == -inf] = 100000
            
            print('csp train final_index' ,csp_train_final_index.shape, 'csp valid final_index', csp_valid_final_index.shape)
            print('csp test final index' ,csp_test_final_index.shape)

            
            csp_train_f_index=csp_train_final_index[~np.any(np.isinf(csp_train_final_index), axis=1)] 
            csp_valid_f_index=csp_valid_final_index[~np.any(np.isinf(csp_valid_final_index), axis=1)]
            csp_test_f_index=csp_test_final_index[~np.any(np.isinf(csp_test_final_index), axis=1)]
            
            print('csp train f index' ,csp_train_f_index.shape, 'csp valid f index', csp_valid_f_index.shape)
            print('csp test f index' ,csp_test_f_index.shape)
            
            index_csp_train=csp_train_f_index[:,csp_train_f_index.shape[1]-1].astype(np.int64)
            index_csp_valid=csp_valid_f_index[:,csp_valid_f_index.shape[1]-1].astype(np.int64)
            index_csp_test=csp_test_f_index[:,csp_test_f_index.shape[1]-1].astype(np.int64)
            
            print('index csp train' ,index_csp_train.shape, 'index csp valid', index_csp_valid.shape)
            print('index_csp_test' ,index_csp_test.shape)
            
            
            csp_train_f=np.delete(csp_train_f_index,csp_train_f_index.shape[1]-1, 1)
            csp_valid_f=np.delete(csp_valid_f_index,csp_valid_f_index.shape[1]-1, 1)
            csp_test_f=np.delete(csp_test_f_index,csp_test_f_index.shape[1]-1, 1)
            
            print('csp train f' ,csp_train_f.shape, 'csp valid f', csp_valid_f.shape)
            print('csp test f' ,csp_test_f.shape)
            
            X_train_f=pd.DataFrame(csp_train_f, index=index_csp_train)
            X_valid_f=pd.DataFrame(csp_valid_f, index=index_csp_valid)

            X_test_f=pd.DataFrame(csp_test_f, index=index_csp_test)
            
            
            #print('X_train_f',X_train_f.shape,'X_valid_f', X_valid_f.shape, 'X_test_f', X_test_f.shape )
            #print('y_train',y_train.shape, 'y_valid', y_valid.shape)
            
            print('X_train_f',X_train_f.shape,'X_valid_f', X_valid_f.shape, 'X_test_f', X_test_f.shape )
            print('y_train',y_train.shape, 'y_valid', y_valid.shape)
            
        else:
        
            X_train_f=X_train
            X_valid_f=X_valid
            X_test_f=X_test
            
            
        
        if CSPkey and mode!=0:
            
            seq_chosen=seq_number
            csp_components=csp_n            
            
            #X_train, X_valid = train[(unique_sequences_fold.isin(train_seq)) & ((X_train_seq % 1000) % 6 == seq_chosen)][features],\
                                #train[(unique_sequences_fold.isin(valid_seq))&((X_valid_seq % 1000) % 6 == seq_chosen)][features]
            #y_train, y_valid = train[(unique_sequences_fold.isin(train_seq)) & ((X_train_seq % 1000) % 6 == seq_chosen)][target],\
                                #train[(unique_sequences_fold.isin(valid_seq)) &((X_valid_seq % 1000) % 6 == seq_chosen)][target]
            
            print('X_train csp',X_train.shape)
            print('X_valid csp',X_valid.shape)
            print('y_train csp', y_train.shape)
            print('y_valid csp', y_valid.shape)
            
            
            #taking the 'Id' files from fold
            
            #files_id_train=train[unique_sequences_fold.isin(train_seq)&((X_train_seq % 1000) % 6 == seq_chosen)]['Id'].tolist()
            files_id_train=train[unique_sequences_fold.isin(train_seq)]['Id'].tolist()
            results_id_train=y_train.tolist()
            
            
            #files_id_valid=train[unique_sequences_fold.isin(valid_seq)&((X_valid_seq % 1000) % 6 == seq_chosen)]['Id'].tolist()
            files_id_valid=train[unique_sequences_fold.isin(valid_seq)]['Id'].tolist()
            results_id_valid=y_valid.tolist()
            
            
            print('files_id_train',len(files_id_train),'results_id_train',len(results_id_train))
            print('files_id_valid',len(files_id_valid),'results_id_valid',len(results_id_valid))
            
            file_name_train=[]
            file_name_valid=[]
            
            for i,f_id in enumerate(files_id_train):
                
#                sequence_validator=(X_train_seq[i] % 1000) % 6
                
#                print('sequence validator', sequence_validator)
                
                
#                if sequence_validator==seq_chosen:
                
                real_f_id=f_id % 100000 
                file_name_train.append("./train_"+str(mode)+'/'+str(mode)+'_'+str(real_f_id)+'_'
                                           +str(results_id_train[i])+'.mat')        


            print('files_id_train',len(files_id_train),'results_id_train',len(results_id_train),'file_name_train',len(file_name_train))

            
            
            for i,f_id in enumerate(files_id_valid):
                
#                sequence_validator=(X_valid_seq[i] % 1000) % 6
                
#                if sequence_validator==seq_chosen:
                
                real_f_id=f_id % 100000 
                file_name_valid.append("./train_"+str(mode)+'/'+str(mode)+'_'+str(real_f_id)+'_'
                                           +str(results_id_valid[i])+'.mat')        


            print('files_id_valid',len(files_id_valid),'results_id_valid',len(results_id_valid),'file_name',len(file_name_valid))

            
    
#            result_list_train=[]
#            result_list_valid=[]
            track1=0
        
            for i, fl1 in enumerate(file_name_train):
                #    print(i)
                
#                result = results_id_train[i]
                if i==track1:
                    print('checking')
                    
                    tables, sequence_from_mat, samp_freq = mat_to_pandas(fl1)
                    csp_left=int(csp_init*10*60*samp_freq)
                    csp_right=int(csp_end*10*60*samp_freq)

                    data_csp_train=np.transpose(tables.values[csp_left:csp_right,:])
                    
                    if int(sequence_from_mat[0][0][0][0])!=seq_chosen:
                        print('error train seq here!',int(sequence_from_mat[0][0][0][0]), seq_chosen, fl1)
        
#                    result_list_train.append(result)
        
                    print('done!')
        
                    continue
              
   
                try:
                    tables1, sequence_from_mat1, samp_freq1 = mat_to_pandas(fl1)
                    csp_left=int(csp_init*10*60*samp_freq)
                    csp_right=int(csp_end*10*60*samp_freq)
                    
                except:
                    print('Some error here {}...'.format(fl))
                    continue
    
#                if sequence_from_mat1==seq_chosen:
                if int(sequence_from_mat1[0][0][0][0])!=seq_chosen:
                        print('error train seq here!',int(sequence_from_mat1[0][0][0][0]), seq_chosen, fl1)
        
                temp_matrix=np.transpose(tables1.values[csp_left:csp_right,:])
                data_csp_train=np.dstack((data_csp_train,temp_matrix))
#                result_list_train.append(result)
                
#                print(data_csp_train.shape)
                    
                    
                    
            for i, fl1 in enumerate(file_name_valid):
                #    print(i)
                
#                result = results_id_valid[i]
                if i==track1:
        
                    print('checking')
                  
        
                    tables, sequence_from_mat, samp_freq = mat_to_pandas(fl1)
                    csp_left=int(csp_init*10*60*samp_freq)
                    csp_right=int(csp_end*10*60*samp_freq)

                    data_csp_valid=np.transpose(tables.values[csp_left:csp_right,:])
                
                    if int(sequence_from_mat[0][0][0][0])!=seq_chosen:
                        print('error valid seq here!',int(sequence_from_mat[0][0][0][0]), seq_chosen, fl1)
        
#                    result_list_valid.append(result)
        
                    print('done!')
        
                    continue
              
   
                try:
                    tables1, sequence_from_mat1, samp_freq1 = mat_to_pandas(fl1)
                    csp_left=int(csp_init*10*60*samp_freq)
                    csp_right=int(csp_end*10*60*samp_freq)
                except:
                    print('Some error here {}...'.format(fl1))
                    continue
    
#                if sequence_from_mat1==seq_chosen:
                if int(sequence_from_mat1[0][0][0][0])!=seq_chosen:
                    print('error valid seq here!',int(sequence_from_mat1[0][0][0][0]), seq_chosen, fl1)

                temp_matrix=np.transpose(tables1.values[csp_left:csp_right,:])
                data_csp_valid=np.dstack((data_csp_valid,temp_matrix))
#                result_list_valid.append(result)
                
#                print(data_csp_valid.shape)

#            y_csp_train=np.array(result_list_train)
#            y_csp_valid=np.array(result_list_valid)

            y_csp_train=y_train.values
            y_csp_valid=y_valid.values
                   
            
            print('y_csp_train',y_csp_train.shape,'y_csp_valid', y_csp_valid.shape)
            print('y_csp_train',y_csp_train,'y_csp_valid', y_csp_valid)


            #print(data_csp.shape)    

            csp_train=data_csp_train.transpose((2,0,1))
            csp_valid=data_csp_valid.transpose((2,0,1))
            
#            print('csp_train',csp_train.shape,'csp_valid', csp_valid.shape)
#            print('csp_train',type(csp_train),'csp_valid', type(csp_valid))

            #print(csp_data.shape)     
    
            CSPtest=CSP(n_components=csp_components)
        
            csp_train_final=CSPtest.fit_transform(csp_train, y_csp_train)
            csp_valid_final=CSPtest.transform(csp_valid)
            
            
            track1=0
            
            print('starting reading test files bits')
            
            
            
            
            files_id_test=test['Id'].tolist()
        
            print('files_number',len(files_id_test))
        
            file_name_test=[]
        
            for i,f_id in enumerate(files_id_test):
            
#            if train_seq_Id[i]==seq_number:
                
                real_f_id=f_id % 100000   
                file_name_test.append("./test_"+str(mode)+'_new/new_'+str(mode)+'_'+str(real_f_id)+'.mat')        


            print('files_id_test',len(files_id_test),'file_name_test',len(file_name_test))
            
            test_div=20
            
            parts_test=int(len(file_name_test)/test_div)+1
            
            print('parts_test', parts_test)
            
            csp_test=[]
            
            track=0
            
            for k in range(parts_test):
            
                track1=0
                
                
                ki=k
                if k==(parts_test-1):
                    kfin=len(file_name_test)
                    print('kfin',kfin)
                else:
                    kfin=(k+1)*test_div
                    print('kfin',kfin)
    
                for i, fl1 in enumerate(file_name_test[ki*test_div:kfin]):
                #    print(i)
    
                    if i==track1:
            
                        print('checking')
         
    
                        tables, sequence_from_mat, samp_freq = mat_to_pandas(fl1)
                    
                        csp_left=int(csp_init*10*60*samp_freq)
                        csp_right=int(csp_end*10*60*samp_freq)
                    
 #                       print('csp left right', csp_left, csp_right)
            
            
    
                        data_csp_test=np.transpose(tables.values[csp_left:csp_right,:])
                    
                    
              
                        print('done!')
            
                        continue
                  
       
                    try:
                        tables1, sequence_from_mat1, samp_freq1 = mat_to_pandas(fl1)
                    except:
                        print('Some error here {}...'.format(fl1))
                        continue
                    
                    csp_left=int(csp_init*10*60*samp_freq1)
                    csp_right=int(csp_end*10*60*samp_freq1)
                
#                    print('csp left right', csp_left, csp_right)
                
            
                    temp_matrix=np.transpose(tables1.values[csp_left:csp_right,:])
                    data_csp_test=np.dstack((data_csp_test,temp_matrix))
#                    print('data_csp_test',data_csp_test.shape)
                    
                    del temp_matrix
                
                temp_test_csp=CSPtest.transform(data_csp_test.transpose((2,0,1)))
                
#                print('test shape bit',temp_test_csp.shape)
                
                if k==track:
                    csp_test_final=temp_test_csp
                    continue
                    
                csp_test_final=np.concatenate((csp_test_final,temp_test_csp))
    
    #            csp_test.append(data_csp_test.transpose((2,0,1)))
    
#                outfile = "csp_test_"+str(mode) +"_"+ "part_"+str(k)+ ".txt"
#                data_csp_test.transpose((2,0,1)).tofile(outfile)
    
                del data_csp_test
                del temp_test_csp
            
#            print('csp_test',len(csp_test))             
        
            
            
                                 
#            for k in range(parts_test):
#                
#                if track1==k:
#                    
#                    outfile = "csp_test_"+str(mode) +"_"+ "part_"+str(k)+ ".txt"
#                    temp_test_csp=np.fromfile(outfile)
#                    
#                    csp_test_final=CSPtest.transform(temp_test_csp)
#                    continue
#                    
#                outfile = "csp_test_"+str(mode) +"_"+ str(patient_id) + "part_"+str(k)+ ".csv"
#                temp_test_csp=np.fromfile(outfile)
#                temp_test_csp1=CSPtest.transform(temp_test_csp)
#                csp_test_final=np.concatenate((csp_test_final,temp_test_csp1))
                print('done! ',k)
            
            print('csp train final' ,csp_train_final.shape,
                  'csp valid final', csp_valid_final.shape, 'csp_test_final', csp_test_final.shape)
            
            csp_train_final_index=np.column_stack((csp_train_final, y_train.values, X_train.index.values))
            csp_valid_final_index=np.column_stack((csp_valid_final, y_valid.values, X_valid.index.values))
            csp_test_final_index=np.column_stack((csp_test_final, X_test.index.values))
            
#            print('csp train final_index' ,csp_train_final_index, 'csp valid final_index', csp_valid_final_index)
            
            
            csp_train_f_index=csp_train_final_index[~np.any(np.isinf(csp_train_final_index), axis=1)] 
            csp_valid_f_index=csp_valid_final_index[~np.any(np.isinf(csp_valid_final_index), axis=1)]
 #           csp_test_f_index=csp_test_final_index[~np.any(np.isinf(csp_test_final_index), axis=1)]
            csp_test_final_index[np.isinf(csp_test_final_index)]=0
            csp_test_f_index=csp_test_final_index

#           replace_test_csp=np.isinf()
#           print('post_test_csp',replace_test_csp)

            
            
#            print('csp train f index' ,csp_train_f_index, 'csp valid f index', csp_valid_f_index)
            
            index_csp_train=csp_train_f_index[:,csp_train_f_index.shape[1]-1]
            index_csp_valid=csp_valid_f_index[:,csp_valid_f_index.shape[1]-1]
            index_csp_test=csp_test_f_index[:,csp_test_f_index.shape[1]-1]
            
#            print('index csp train' ,index_csp_train, 'index csp valid', index_csp_valid)

            y_train=csp_train_f_index[:,csp_train_f_index.shape[1]-2]
            y_valid=csp_valid_f_index[:,csp_valid_f_index.shape[1]-2]
          
#            print('index csp train' ,index_csp_train, 'index csp valid', index_csp_valid)
            
            
            csp_train_f=np.delete(csp_train_f_index,np.s_[csp_train_f_index.shape[1]-2,csp_train_f_index.shape[1]-1], 1)
            csp_valid_f=np.delete(csp_valid_f_index,np.s_[csp_train_f_index.shape[1]-2,csp_train_f_index.shape[1]-1], 1)
            csp_test_f=np.delete(csp_test_f_index,csp_test_f_index.shape[1]-1, 1)
            
            print('csp train f' ,csp_train_f.shape, 'csp valid f', csp_valid_f.shape, 'csp_test_f', csp_test_f.shape)
            
            X_train_f=pd.DataFrame(csp_train_f, index=np.int64(index_csp_train))
            X_valid_f=pd.DataFrame(csp_valid_f, index=np.int64(index_csp_valid))

            X_test_f=pd.DataFrame(csp_test_f, index=np.int64(index_csp_test))
            
            
            print('X_train_f',X_train_f.shape,'X_valid_f', X_valid_f.shape, 'X_test_f', X_test_f.shape )
            print('y_train',y_train.shape, 'y_valid', y_valid.shape)
            
            #print('X_train_f',X_train_f,'X_valid_f', X_valid_f, 'X_test_f', X_test_f )
            #print('y_train',y_train, 'y_valid', y_valid)
            

#       SMOTE oversampling
        
#        print('Original dataset shape {}'.format(Counter(y_train)))
#        print('Original dataset shape {}'.format(Counter(X_train_f)))
#        print(X_train_f)
#        print(y_train)

        if Oversampling:
        
            sm1 = SMOTETomek(random_state=42)
            X_res,y_res = sm1.fit_sample(X_train_f,y_train)
            X_train_f=pd.DataFrame(X_res, columns=X_train_f.columns)
            y_train=pd.Series(y_res)# Does it need the index of y_train?
        
#        print('Resampled dataset shape {}'.format(Counter(y_train)))
#        print(X_train_f)
#        print(y_train)
    
    
                        
        
        
        
#       Preparation for XGB training

        dtrain = xgb.DMatrix(X_train_f, y_train)
        dvalid = xgb.DMatrix(X_valid_f, y_valid)

        watchlist = [(dtrain, 'train'), (dvalid, 'eval')]       
        
        gbm = xgb.train(params, dtrain, num_boost_round, evals=watchlist, early_stopping_rounds=early_stopping_rounds, verbose_eval=500)

        yhat = gbm.predict(xgb.DMatrix(X_valid_f), ntree_limit=gbm.best_iteration+1)

#       Each time store portion of precicted data in train predicted values

        for i in range(len(X_valid_f.index)):
            yfull_train[X_valid_f.index[i]] = yhat[i]
            
        print("Validating...")
        check = gbm.predict(xgb.DMatrix(X_valid_f), ntree_limit=gbm.best_iteration+1)
        score = roc_auc_score(y_valid.tolist(), check)
        print('Check error value: {:.6f}'.format(score))

        print("Predict test set...")
        test_prediction1 = gbm.predict(xgb.DMatrix(X_test_f), ntree_limit=gbm.best_iteration+1)
        print('test prediction', test_prediction1.shape, 'yfull_test', yfull_test.shape)
        
#        for item in replace_test_csp.tolist():
#            np.insert(test_prediction1, item, 0)
        print('test_prediction1 shape',test_prediction1.shape)
        yfull_test['kfold_' + str(num_fold)] = test_prediction1
        
              

    print('iteration finished')
    # Copy dict to list
    train_res = []
    
#    print('train.index',train.index, train.index[0], 'train shape', train.shape)
    print('yfull_train', len(yfull_train), list(yfull_train.keys())[0], list(yfull_train.keys())
         ,type(list(yfull_train.keys())[0]))
    
#    print('train_indexes',train.index.values.tolist())
    
    iterator_train=train.index.values.tolist()
    for i in iterator_train:
        if i in yfull_train:
            train_res.append(yfull_train[i])
            
        else:
            print('this index is missing! ', i)
            miss_index=i
            
            print('miss_index', miss_index)
            row_miss=train.loc[[i]]
#            print('row_miss', row_miss)
        
            missing_id=row_miss.index.values[0]

            
            print('missing_id', missing_id)
#            print('missing_result', missing_result)
            train=train[(train.index != missing_id)]
            
            print('train shape', train.shape)
            print('train_res shape', len(train_res))
            
#    print('test indexes', test.index.values.tolist())
    
    
#    print('test.index',test.index, test.index[0], 'test shape', test.shape)
#    print('X_test_f index', X_test_f.index.values, 'X test shape', X_test_f.shape)
    
    iterator_test=test.index.values.tolist()
    
    for j in iterator_test :
    
        
        if j in X_test_f.index.values.tolist():
            pass
        else:
            missing_test_index=j
            
            row_test_miss=test.loc[[j]]
            
            missing_test_id=row_miss['Id'].values[0]
            
            
            test=test[test['Id'] != missing_test_id]
            
            print('test shape', test.shape)
            

     
    
    print('train shape', train.shape)

    score = roc_auc_score(train[target], np.array(train_res))
    print('Check error value: {:.6f}'.format(score))

    # Find mean for KFolds on test
    merge = []
    for i in range(1, nfolds+1):
        merge.append('kfold_' + str(i))
    yfull_test['mean'] = yfull_test[merge].mean(axis=1)
    


    print('Training time: {} minutes'.format(round((time.time() - start_time)/60, 2)))


    #Pred_per_patient currently in development. Is deactivated and should not be used.
    if pred_per_patient:
    
        total_results=yfull_test['mean'].values
        hist, bins = np.histogram(total_results, bins=50)
        width = 0.7 * (bins[1] - bins[0])
        center = (bins[:-1] + bins[1:]) / 2
        plt.bar(center, hist, align='center', width=width)
        plt.show()
    
    
        total_results_train=np.array(train_res)
        hist, bins = np.histogram(total_results_train, bins=50)
        width = 0.7 * (bins[1] - bins[0])
        center = (bins[:-1] + bins[1:]) / 2
        plt.bar(center, hist, align='center', width=width)
        plt.show()
        
    #Saving module and xgboost parameters to JSON file
    
    xgboost_params=params
    
    parameters=[]
    
    parameters.append(xgboost_params)
    parameters.append(function_params)
    
    now = datetime.datetime.now()
    
    parameter_file_name=str('parameter-file-'+'mode-'+str(mode)+'-'+now.strftime("%Y-%m-%d-%H-%M"))
    
    json.dump(parameters, open(parameter_file_name+".txt",'w'), indent=4)
#    read_params = json.load(open(parameter_file_name+".txt"), object_pairs_hook=OrderedDict)
    
#    print (read_params)
    
#    print('yfull_test shape', yfull_test.shape)


    
    return yfull_test['mean'].values, score, yfull_train, train_res, test