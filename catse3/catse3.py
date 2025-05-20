import numpy as np
import pandas as pd
from scipy.signal import find_peaks, butter, filtfilt, resample
from sklearn.preprocessing import OneHotEncoder
import statsmodels.api as sm
from tcn import TCN
from tensorflow.keras.models import load_model
import polars as pl

hot_encode = OneHotEncoder()
hot_encode.fit(np.array(['walking', 'running', 'cycling']).reshape(-1, 1))

def reshape_acc(df, seq_len=400):
    '''Reshape the accelerometer time series data to (n_seq, 400, 3)'''

    # assert df is a pandas df or polars df
    assert isinstance(df, (pd.DataFrame, pl.DataFrame)), 'Input must be a pandas or polars dataframe.'

    # if acc is a pandas dataframe, convert to numpy array
    if isinstance(df, pd.DataFrame):
        acc = df[['x', 'y', 'z']].to_numpy()
    elif isinstance(df, pl.DataFrame):
        acc = df['x', 'y', 'z'].to_numpy()
    n_seq = int(np.floor(acc.shape[0] / seq_len))
    acc = acc[:n_seq * seq_len]
    acc = acc.reshape(n_seq, seq_len, 3)

    return acc

def classify_act(df, model='Lendt_2024', str_label=True, filt_cycling=True):
    '''
    This function classifies the activity type based on the accelerometer data.
    
    Input:
    data: accelerometer data (3D array, shape: [n, num_bins, 3])
    model: trained activity classifier model
    
    Output:
    predictions: activity type predictions
    
    '''
    
    if model == 'Lendt_2024':
        model = load_model('models/CNN_BiLSTM_f1.keras', compile=False)

    acc = reshape_acc(df, seq_len=400)
    predictions = model.predict(acc / 8, verbose=0)
    predictions = np.argmax(predictions, axis=1)

    if filt_cycling:
        # create a new vector 'cycling' with the same length as predictions
        cycling = np.zeros(len(predictions))
        cycling = np.where(predictions == 4, 1, 0)

        # perform a moving average with a window size of 5 (equals 20 seconds with 4s per prediction)
        cycling_filt = np.convolve(cycling, np.ones(10)/10, mode='same')
        cycling_filt = np.where(cycling_filt >= 0.5, 1, 0)
        cycling_filt = np.where(cycling_filt < 0.5, 0, 1)

        # set predictions to 4 if cycling_filt == 1
        predictions = np.where(cycling_filt == 1, 4, predictions)

    if str_label:

        predictions_str = []

        for pred in predictions:
                
                if pred == 0:
                    predictions_str.append('walking')
                elif pred == 1:
                    predictions_str.append('running')
                elif pred == 2:
                    predictions_str.append('standing')
                elif pred == 3:
                    predictions_str.append('sitting')
                elif pred == 4:
                    predictions_str.append('cycling')

        # convert to np array
        predictions_str = np.array(predictions_str)
        predictions = predictions_str

    # create series of predictions
    predictions = np.repeat(predictions, 400, axis=0)
    if len(predictions) < len(df):
        predictions = np.concatenate((predictions, np.repeat('unknown', len(df) - len(predictions))))

    # create a new column in the dataframe with the predictions
    if isinstance(df, pd.DataFrame):
        df['activity'] = predictions
    elif isinstance(df, pl.DataFrame):
        df = df.with_columns(pl.Series('activity', predictions))
    
    return df

def create_sequences(acc, act):
    '''
    Create sequences of the same uninterrupted activity.
    Each activity in the act array corresponds to a row in the acc array.

    The output should be a list of lists, where each list contains the accelerometer data of a single activity block.

    '''
    
    acc = reshape_acc(acc, seq_len=400)


    acc_seq = []
    act_seq = []

    for i in range(len(act)):

        if i == 0:
            acc_seq.append(acc[i])
            act_seq.append(act[i])

        elif act[i] == act[i-1]:
            acc_seq[-1] = np.concatenate((acc_seq[-1], acc[i]), axis=0)

        else:
            acc_seq.append(acc[i])
            act_seq.append(act[i])

    return acc_seq, act_seq

def stride_segmentation(acc_seq, activity_seq, num_bins=30):
    '''
    This function segments the accelerometer data into strides based on the activity type.
    
    Input:
    acc: accelerometer data (3D array, shape: [n, 3])
    activity: activity type (either walking, running, or cycling)
    num_bins: number of bins to resample the segmented data (default: 30)

    Output:
    resampled_splits: segmented accelerometer data (3D array, shape: [m, num_bins, 3])
    
    '''

    splits_f = []
    activity_f = []
    peaks_f = []

    # check if acc_seq and activity_seq are both lists
    if not isinstance(acc_seq, list) or not isinstance(activity_seq, list):
        acc_seq = [acc_seq]
        activity_seq = [activity_seq]

    for i in range(len(activity_seq)):

        acc = acc_seq[i]
        activity = activity_seq[i]
        
        if i == 0: 
            idx_start = 0 # 0 for the first sequence
        else:
            # equal to all sequence lengths before the current sequence
            idx_start = sum([len(seq) for seq in acc_seq[:i]])
            
        if activity == 'walking':

            a, b = butter(4, 0.1, 'low')
            acc_filt = filtfilt(a, b, -acc[:,2], axis=0)
            peaks, _ = find_peaks(acc_filt, height=0.3, distance=70)

        elif activity == 'running':

            a, b = butter(4, 0.1, 'low')
            acc_filt = filtfilt(a, b, -acc[:,0], axis=0)
            peaks, _ = find_peaks(acc_filt, height=1, distance=50)

        elif activity == 'cycling':

            a, b = butter(4, 0.1, 'low')
            acc_filt = filtfilt(a, b, acc[:,0], axis=0)
            peaks, _ = find_peaks(acc_filt, height=-0.5, distance=50)

        split_acc = np.split(acc, peaks)
        split_acc = split_acc[1:-1]

        #split_acc = [data for data in split_acc if len(data) < 200]
        split_acc = np.array(split_acc, dtype=object)

        resampled_splits = []

        for split in split_acc:
                
                resampled_split = np.zeros((num_bins, 3))
        
                for i in range(3):
                    resampled_split[:, i] = resample(split[:, i], num_bins)
        
                resampled_splits.append(resampled_split)

        resampled_splits = np.array(resampled_splits, dtype='float32')

        # create an activity vector with the length of the resampled splits
        activity_vec = np.full((len(resampled_splits), 1), activity).flatten()

        splits_f.append(resampled_splits)
        activity_f.append(activity_vec)
        peaks_f.append(peaks[:-1]+idx_start) # no stride after the last peak

    splits_f = np.concatenate(splits_f)
    activity_f = np.concatenate(activity_f)
    peaks_f = np.concatenate(peaks_f)

    return splits_f, activity_f, peaks_f

def stride_EE_estimation(strides, act, EE_model):
    '''
    Estimate the energy expenditure (EE) based on the accelerometer data.
    
    Input:
    strides: segmented accelerometer data (3D array, shape: [n, num_bins, 3])
    act: activity type (either walking, running, or cycling) )(1D array, shape: [n])
    EE_model: trained energy expenditure model (CondLSTM)
    
    Output:
    EE: estimated energy expenditure
    
    '''

    # check if EE_estimator is a regression model
    if (type(EE_model)) == sm.regression.linear_model.RegressionResultsWrapper:

        strides = np.reshape(strides, (strides.shape[0], strides.shape[1]*strides.shape[2]))
        strides = pd.DataFrame(strides)
        strides.columns = ['x_' + str(i) for i in range(1, 31)] + ['y_' + str(i) for i in range(1, 31)] + ['z_' + str(i) for i in range(1, 31)]
        strides.insert(0, 'activity_group', act)
        EE_strides = EE_model.predict(strides)
        EE_strides = EE_strides.to_numpy()

    elif (str.__contains__(EE_model._name, 'TCN')): # if model is TCN model (only acceleration)

        act_hot = hot_encode.transform(act.reshape(-1, 1)).toarray()
        EE_strides = EE_model.predict([strides, act_hot], verbose=0)
        #EE_strides = EE_model.predict(strides, verbose=0)
        EE_strides = EE_strides.flatten()

    else: # if model is CondLSTM model (acceleration and activity type)

        act_hot = hot_encode.transform(act.reshape(-1, 1)).toarray()
        EE_strides = EE_model.predict([strides, act_hot], verbose=0)
        EE_strides = EE_strides.flatten()
    
    return EE_strides

def reg_EE_estimation(acc, act, EE_model):
    '''
    Estimate the energy expenditure (EE in kcal/min/kg) based on the accelerometer data.
    
    Input:
    acc: accelerometer data (2D array, shape: [n, 3])
    act: activity type (either walking, running, or cycling) )(1D array, shape: [n])
    EE_model: trained energy expenditure model (CondLSTM)
    
    Output:
    EE: estimated energy expenditure
    
    '''

    if act == 'sitting':
        act = 'standing'

    # split acc into 400-sample windows
    acc_split = np.array_split(acc, len(acc)//400)
    acc_split = np.array(acc_split)

    # create activity vector with length of acc_split
    act_split = np.array([act] * len(acc_split))

    # calculate ENMO for each split using the calcENMO function
    ENMO = np.array([calcENMO(split) for split in acc_split])

    # combine ENMO and act_split into a single dataframe
    df_ENMO = pd.DataFrame({'ENMO': ENMO, 'activity_group': act_split})

    # predict the energy expenditure using the ENMO model
    EE_pred = EE_model.predict(df_ENMO)

    # create the idx for each prediction (sequence of 400 samples)
    idx_pred = np.arange(0, len(acc), 400)
    
    return EE_pred, idx_pred

def predict_EE(df, model_strides='TCN', model_reg='ENMO_meLM', **kwargs):
    
    if model_strides == 'TCN':
        from keras.models import load_model
        model_strides = load_model('models/TCN_f1.keras', compile=False, custom_objects={'TCN': TCN})
    else:
        # error if the model is not TCN
        raise ValueError('Currently, only TCN model is supported for stride EE estimation.')

    if model_reg == 'ENMO_meLM':
        import statsmodels.api as sm
        model_reg = sm.load('models/ENMO_meLM.pkl')
    else:
        # error if the model is not ENMO_meLM
        raise ValueError('Currently, only ENMO_meLM model is supported for regression EE estimation.')

    df = df.with_columns(pl.when(pl.col('activity') != pl.col('activity').shift(1)).then(1).otherwise(0).alias('act_seq'))
    df = df.with_columns(pl.col('act_seq').cast(pl.Int32))
    df = df.with_columns(pl.col('act_seq').cum_sum().alias('act_seq'))

    EE_ls = []
    idx_ls = []
    idx_start = 0

    for i in range(df['act_seq'].max()): # Iterate over each sequence
            
            acc = df.filter(pl.col('act_seq') == i).select(['x', 'y', 'z']).to_numpy()
            act = df.filter(pl.col('act_seq') == i).select('activity').to_numpy()[0][0]
            
            if act == 'sitting' or act == 'standing':

                EE_est, idx = reg_EE_estimation(acc, act, model_reg)

            elif act == 'walking' or act == 'running' or act == 'cycling':

                strides, act_strides, idx = stride_segmentation(acc, act, num_bins=30)

                if len(strides) == 0: # no strides where found
                    EE_est, idx = reg_EE_estimation(acc, act, model_reg)
                else:
                    EE_est = stride_EE_estimation(strides, act_strides, model_strides)
                    
            else:
                EE_est = None
                idx = None
            
            EE_ls.append(EE_est)
            idx_ls.append(idx + idx_start)
            idx_start += len(acc)

    EE = np.concatenate(EE_ls)
    idx = np.concatenate(idx_ls)

    ee_df = pl.DataFrame({'EE_kcal_min_kg': EE, 'idx': idx})
    # create integer index for the dataframe
    df = df.with_columns(pl.arange(0, df.shape[0]).alias('idx'))
    df = df.join(ee_df, on='idx', how='left')
    df = df.drop('act_seq', 'idx')
        
    return df

def calcENMO(acc):

    VM = np.sqrt(acc[:, 0] ** 2 + acc[:, 1] ** 2 + acc[:, 2] ** 2)
    ENMO = np.where(VM - 1 < 0, 0, VM - 1)

    return np.mean(ENMO)

def calcMAD(acc):

    VM = np.sqrt(acc[:, 0] ** 2 + acc[:, 1] ** 2 + acc[:, 2] ** 2)
    MAD = np.mean(np.abs(VM - np.mean(VM)))

    return np.mean(MAD)
