# -*- coding: utf-8 -*-
"""
Created on Fri Dec  2 19:04:37 2016

@author: Rodolfo Fernandez R. <rfern@pdx.edu>
"""



import pandas as pd
import numpy as np
from scipy.io import loadmat

'''  Modules used for data preprocessing. '''

def mat_to_pandas(path):
    '''Reading a file with *.mat structure.'''
    
    mat = loadmat(path)
    names = mat['dataStruct'].dtype.names
    ndata = {n: mat['dataStruct'][n][0, 0] for n in names}
    samp_freq = ndata['iEEGsamplingRate'][0, 0]
    sequence = -1
    if 'sequence' in names:
        sequence = mat['dataStruct']['sequence']
    return pd.DataFrame(ndata['data'], columns=ndata['channelIndices'][0]), sequence, samp_freq

    # Different EEG file extensions can be found at  
    # http://www.fieldtriptoolbox.org/dataformat

def mat_to_numpy(path):
    
    pass

def avr_to_pandas(path):
    
    pass

def fif_to_pandas(path):
    
    pass

def create_simple_csv_train(patient_id, feature_model, num_features, fini, fend, fovr,
                            short_dataset=False, new_test=False):
    ''' Creating CSV file with features from the Feature_Engineering module.
    Rows represent samples and columns represent features.
    '''
    
    #Constructing main part of feature file name
    feature_file='_'+str(feature_model)+'_short_'+str(short_dataset)+'_new_test_'+ str(new_test)\
            +'_fini_'+str(fini)+'_fend_'+str(fend)+'_fovr_'+str(fovr)
        
    # Short or long dataset
    if short_dataset:
        
        source_dir="./data/train_"
    else:
        source_dir="./train_"
    
    new_label=''
    
    # Using old or new test files
    if new_test:
        
        new_label='_new'

    out = open("simple_train_" + str(patient_id) + feature_file + ".csv", "w")
    out.write("Id,sequence_id,sequence_num,patient_id,")
  
    # Generating column names

    columns=''
    for i in range(16):
        for j in range(num_features):
            columns+= 'ch_'+str(i)+'_'+"band_"+str(j)+","        

    out.write(columns+"file_size,result\n")

    ### Generating features
    
    out_str = ''
    
    # Reading files
    files = sorted(glob.glob(source_dir + str(patient_id) + "/*.mat"), key=natural_key)
    print ('train files'+ str(patient_id), len(files))    
    
    # Indicators for debugging
    pos1=0
    neg1=0
    sequence_id_pre = int(patient_id)*1000
    sequence_id_inter = int(patient_id)*1000
    total_pre = 0
    total_inter=0
    seq1=0
    
    # Selecting 'safe' files from old test
    
    new_train = pd.read_csv('train_and_test_data_labels_safe'+'.csv')
    new_data = new_train['image']
    
    selection = new_train[new_train['safe'] == 1].drop('safe', axis=1)
    
    # Iterating through file
    
    for fl in files:
        
                       
        if os.path.basename(fl) not in selection['image'].values:
            continue
        
        id_str = os.path.basename(fl)[:-4]
        arr = id_str.split("_")
        patient = int(arr[0])
        id = int(arr[1])
        result = int(arr[2])
        
        if result == 1:
            
            total_pre += 1
            sequence_id=int(patient_id)*1000+int((total_pre-1) // 6) + int((total_inter-1) // 6) + 1

            
        elif result == 0:
            
            total_inter += 1            
            sequence_id=int(patient_id)*1000+int((total_pre) // 6) + int((total_inter-1) // 6)

        
        new_id = int(patient*100000 + id)
        try:
            tables, sequence_from_mat, samp_freq = mat_to_pandas(fl)
            seq1=int(sequence_from_mat[0][0][0][0])
        except:
            print('Some error here {}...'.format(fl))
            continue
        
        
        if (new_id % 1000) % 6 == 0:
            sequence_validator=6
        else:
            sequence_validator=(new_id % 1000) % 6
        
        
        if seq1!=sequence_validator:
            print('sequence mismatch!',seq1, sequence_validator)
        else:
            print('sequence match ',seq1, sequence_validator) 
        
        
        print(sequence_id)
        out_str += str(new_id) + "," + str(sequence_id) + "," + str(seq1) + ","+str(patient)

        sizesignal=int(tables.shape[0])       
        
        for f in sorted(list(tables.columns.values)):
            
            out_str=feature_eng(tables[f], out_str,feature_model, sizesignal, samp_freq,  fini, fend, fovr,)
            
            
        out_str += "," + str(os.path.getsize(fl)) + "," + str(result) + "\n"
        #print(sequence_from_mat)
        #print(type(sequence_from_mat))
       
        print('total preictal: ', total_pre,' total interictal: ', total_inter,' sequence local: ', seq1)
        if (total_pre % 6 == 0) and result == 1:
                pos1 += 1
                print('Positive ocurrence sequence finished', pos1)
                if (seq1==6):
                    sequence_id_pre += 1
                    print ('sequence preictal next',sequence_id_pre)
        
        if (total_inter % 6 == 0) and result == 0:                
                neg1 += 1
                print('Negative ocurrence sequence finished', neg1)
                if (seq1==6):
                    sequence_id_inter += 1
                    print ('sequence interictal next',sequence_id_inter)

    out.write(out_str)
    
    out.close()
    print('Train CSV for patient {} has been completed...'.format(patient_id))




def create_simple_csv_test(patient_id, feature_model, num_features, fini, fend, fovr,
                           short_dataset=False, new_test=False):
    ''' Creating CSV from test files with features from the Feature_Engineering module.  
    Rows represent samples and columns represent features.
    '''
    
        
    feature_file='_'+str(feature_model)+'_short_'+str(short_dataset)+'_new_test_'+str(new_test)\
            +'_fini_'+str(fini)+'_fend_'+str(fend)+'_fovr_'+str(fovr)
    
    if short_dataset:
        
        source_dir="./data/test_"
    else:
        source_dir="./test_"
    
    new_label=''
    
    if new_test:
        
        new_label="_new"

    # TEST
    out_str = ''
    files = sorted(glob.glob(source_dir + str(patient_id) + new_label + "/*.mat"), key=natural_key)
    print ('test files'+ str(patient_id), len(files))    
    out = open("simple_test_" + str(patient_id) + feature_file + ".csv", "w")
    out.write("Id,patient_id,")
    
    columns=''
    for i in range(16):
        for j in range(num_features):
            columns+= 'ch_'+str(i)+'_'+"band_"+str(j)+","        
    
    out.write(columns+"file_size\n")
    
        
    for fl in files:
        # print('Go for ' + fl)
        id_str = os.path.basename(fl)[4:-4]
        arr = id_str.split("_")
        patient = int(arr[0])
        id = int(arr[1])
        new_id = int(patient*100000 + id)
        try:
            tables, sequence_from_mat, samp_freq = mat_to_pandas(fl)

        except:
            print('Some error here {}...'.format(fl))
            continue
        out_str += str(new_id) + "," + str(patient)

        sizesignal=int(tables.shape[0])
              
               
        for f in sorted(list(tables.columns.values)):
            
            out_str=feature_eng(tables[f], out_str,feature_model, sizesignal, samp_freq, fini, fend, fovr,)
                        
        out_str += "," + str(os.path.getsize(fl)) + "\n"
        # break

    out.write(out_str)
    out.close()
    print('Test CSV for patient {} has been completed...'.format(patient_id))



def feature_eng(data_sensor, out_str, eng_number, sizesignal, fs, fini5,fend5,fovr5):
    ''' Create features from EEG time series data.'''

                
    yf1 = fft(data_sensor)
    fftpeak=2/sizesignal * np.abs(yf1[0:sizesignal/2])
 
    numberofbands=4

    sizeband=20/numberofbands
    
    if eng_number==5:
        
    
        

        ##Frequency parameters##
        #Start frequency#
        fini = fini5
        #End frequency#
        fend = fend5
        #Frequency band range#
        frng = 4
        #Frequency overlap#
        fovr = fovr5
    
        #Frequency band generator#
        fbands = [[f, f + frng] for f in range(fini, fend - fovr, frng - fovr)]
    
        #Filter order#
        order = 5
        #Filter bandstop attenuation (dB)#
        attenuation = 20.0
        #Nyquist frequency#
        fnyq = fs / 2.0
        

        for fb in fbands:
        
            #Create butterworth bandpass filter#
            #b, a = butter(order, fb  / fnyq, btype='band')
            b, a = cheby2(order, attenuation, fb  / fnyq, btype='band')
            
            #Apply filter#
            data_filter = lfilter(b, a, data_sensor)
            
            #Band pass 'power'#
            band_pwr = np.square(data_filter)
            
            avg_band_pwr = band_pwr.mean()
            
            out_str += "," + str(avg_band_pwr)
        
    
    
    
    
    elif eng_number==4:
        
    
        

        ##Frequency parameters##
        #Start frequency#
        fini = 4
        #End frequency#
        fend = 40
        #Frequency band range#
        frng = 4
        #Frequency overlap#
        fovr = 0
    
        #Frequency band generator#
        fbands = [[f, f + frng] for f in range(fini, fend - fovr, frng - fovr)]
    
        #Filter order#
        order = 5
        #Filter bandstop attenuation (dB)#
        attenuation = 20.0
        #Nyquist frequency#
        fnyq = fs / 2.0
        

        for fb in fbands:
        
            #Create butterworth bandpass filter#
            #b, a = butter(order, fb  / fnyq, btype='band')
            b, a = cheby2(order, attenuation, fb  / fnyq, btype='band')
            
            #Apply filter#
            data_filter = lfilter(b, a, data_sensor)
            
            #Band pass 'power'#
            band_pwr = np.square(data_filter)
            
            avg_band_pwr = band_pwr.mean()
            
            out_str += "," + str(avg_band_pwr)
      
    
    
    
    
    elif eng_number==3:
        
    
        

        ##Frequency parameters##
        #Start frequency#
        fini = 7
        #End frequency#
        fend = 30
        #Frequency band range#
        frng = 4
        #Frequency overlap#
        fovr = 0
    
        #Frequency band generator#
        fbands = [[f, f + frng] for f in range(fini, fend - fovr, frng - fovr)]
    
        #Filter order#
        order = 5
        #Filter bandstop attenuation (dB)#
        attenuation = 20.0
        #Nyquist frequency#
        fnyq = fs / 2.0
        

        for fb in fbands:
        
            #Create butterworth bandpass filter#
            #b, a = butter(order, fb  / fnyq, btype='band')
            b, a = cheby2(order, attenuation, fb  / fnyq, btype='band')
            
            #Apply filter#
            data_filter = lfilter(b, a, data_sensor)
            
            #Band pass 'power'#
            band_pwr = np.square(data_filter)
            
            avg_band_pwr = band_pwr.mean()
            
            out_str += "," + str(avg_band_pwr)
            

    elif eng_number==2:
        
        mean = data_sensor.mean()
        
        peak1=fftpeak[0:3].mean()            
        peak2=fftpeak[3:6].mean()          
        peak3=fftpeak[6:9].mean()
        peak4=fftpeak[9:12].mean()
        peak5=fftpeak[12:15].mean()            
        peak6=fftpeak[15:18].mean()          
        peak7=fftpeak[18:21].mean()
        peak8=fftpeak[21:24].mean()
        peak9=fftpeak[24:27].mean()            
        peak10=fftpeak[27:30].mean()          
        peak11=fftpeak[30:33].mean()
        peak12=fftpeak[33:36].mean()
            
        out_str += "," + str(mean)+ "," + str(peak1) + "," + str(peak2) + "," + str(peak3) +"," + str(peak4) \
                    +"," + str(peak5) + "," + str(peak6) + "," + str(peak7) +"," + str(peak8)+ "," + str(peak9) \
                    +"," + str(peak10) + "," + str(peak11) +"," + str(peak12)
    
    elif eng_number==1:
            
        mean = data_sensor.mean()   
        
        peak1=fftpeak[0:5].mean()            
        peak2=fftpeak[5:10].mean()          
        peak3=fftpeak[10:15].mean()
        peak4=fftpeak[15:20].mean()
        
        out_str += "," + str(mean)+ "," + str(peak1) + "," + str(peak2) + "," + str(peak3) +"," + str(peak4)
    
    elif eng_number==0:
            
        mean = data_sensor.mean()
    
        out_str += "," + str(mean)
    
    return out_str
