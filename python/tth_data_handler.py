from tthAnalysis.bdtTraining import data_loader as dl
import json
import os
import numpy as np
import xgboost as xgb
import pandas
from sklearn.model_selection import train_test_split
from tthAnalysis.bdtHyperparameterOptimization import universal
np.random.seed(1)
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)


def read_trainVars(path):
    parameters = []
    with open(path, 'r') as f:
        for line in f:
            parameters.append(line.strip('\n'))
    return parameters


def to_oneDict(list_of_dicts):
    main_dict = {}
    for elem in list_of_dicts:
        key = list(elem.keys())[0]
        main_dict[key] = elem[key]
    return main_dict


def getParameters(parameters_path):
    paramerList = dl.read_parameters(parameters_path)
    parameter_dict = to_oneDict(paramerList)
    return parameter_dict


def tth_analysis_main(
    channel, bdtType, nthread,
    outputDir, trainvar,
    cf, all_years=True, verbose=False
):
    cmssw_base_path = os.path.expandvars('$CMSSW_BASE')
    datacard_dir = os.path.join(
        cmssw_base_path,
        'src',
        'tthAnalysis',
        'bdtTraining',
        'data'
    )
    parameterFile = 'par_' + channel + '.json'
    parameters_path = os.path.join(
        datacard_dir, channel, parameterFile)
    # some data_cards have read_from, others not.
    # same with trainVar choice
    trainVars = cf.choose_trainVar(
        datacard_dir, channel, trainvar, bdtType)
    parameters = getParameters(parameters_path)
    output = cf.read_from()
    print('::::::: Loading data :::::::')
    if all_years:
        total_data = pandas.DataFrame({})
        yearly_paths = ['inputPath16', 'inputPath17', 'inputPath18']
        for path in yearly_paths:
            data = dl.load_data(
                parameters[path],
                parameters['channelInTree'],
                trainVars, # kus on kasutatud tavaline 'variables'?
                bdtType,
                channel,
                output['keys'],
                os.path.join(datacard_dir, 'ttH')
            )
            total_data = total_data.append(data, ignore_index=True)
    else:
        total_data = dl.load_data(
            parameters['inputPath18'],
            parameters['channelInTree'],
            trainVars, # kus on kasutatud tavaline 'variables'?
            bdtType,
            channel,
            output['keys'],
            os.path.join(datacard_dir, 'ttH')
        )
    if verbose:
        print_info(data)
    return total_data, trainVars


def print_info(data):
    print('Sum of weights: ' + str(
        data.loc[data['target']==0]['totalWeight'].sum()))
    data.loc[
        data['target']==0, ['totalWeight']
    ] *= 100000./data.loc[data['target']==0]['totalWeight'].sum()
    data.loc[
        data['target']==1, ['totalWeight']
    ] *= 100000./data.loc[data['target']==1]['totalWeight'].sum()
    print('Norm:')
    print(
        '\tBackground: '
        + str(data.loc[data['target']==0]['totalWeight'].sum())
    )
    print(
        '\tSignal: '
        + str(data.loc[data['target']==1]['totalWeight'].sum()))
    data.dropna(subset=['totalWeight'],inplace = True)
    data.fillna(0)
    nS = len(data.loc[data.target.values == 1])
    nB = len(data.loc[data.target.values == 0])
    print('Without NaN:')
    print('\tSignal: ' + str(nS))
    print('\tBackground: ' + str(nB))


def create_xgb_data_dict(data, trainVars, global_settings):
    output_dir = os.path.expandvars(global_settings['output_dir'])
    info_dir = os.path.join(output_dir, 'previous_files', 'data_dict')
    if not os.path.exists(info_dir):
        os.makedirs(info_dir)
    data = convert_data_to_correct_format(data)
    print('::::::: Create datasets ::::::::')
    additions = ['target', 'totalWeight', 'process']
    variables = trainVars
    for addition in additions:
        if not addition in variables:
            variables = variables + [addition]
    train, test = train_test_split(
        data[variables],
        test_size=0.2, random_state=1
    )
    training_labels = train['target'].astype(int)
    testing_labels = test['target'].astype(int)
    training_processes = train['process']
    testing_processes = test['process']
    traindataset = np.array(train[trainVars].values)
    testdataset = np.array(test[trainVars].values)
    dtrain = xgb.DMatrix(
        traindataset,
        label=training_labels,
        nthread=global_settings['nthread'],
        feature_names=trainVars
    )
    dtest = xgb.DMatrix(
        testdataset,
        label=testing_labels,
        nthread=global_settings['nthread'],
        feature_names=trainVars
    )
    data_dict = {
        'dtrain': dtrain,
        'dtest': dtest,
        'training_labels': training_labels,
        'testing_labels': testing_labels,
        'training_processes': training_processes,
        'testing_processes': testing_processes
    }
    universal.write_data_dict_info(info_dir, data_dict)
    return data_dict


def convert_data_to_correct_format(data):
    ''' Converts the necessary trainingvariables into the correct numeric type
    '''
    for column in data.columns:
        if 'eta' in column:
            data[column] = abs(data[column])
        if 'Jet' in column:
            data[column] = data[column].astype(int)
    return data


def create_nn_data_dict(data, trainvars, global_settings):
    '''Creates the data_dict to be used by the Neural Network

    Parameters:
    ----------
    data : pandas dataframe
        Dataframe containing the data
    trainvars : list
        List of names of the training variables

    Returns:
    -------
    data_dict : dict
        Dictionary containing training and testing labels
    '''
    output_dir = os.path.expandvars(global_settings['output_dir'])
    info_dir = os.path.join(output_dir, 'previous_files', 'data_dict')
    if not os.path.exists(info_dir):
        os.makedirs(info_dir)
    data = convert_data_to_correct_format(data)
    print('::::::: Create datasets ::::::::')
    additions = ['target', 'totalWeight', 'process']
    variables = trainvars
    for addition in additions:
        if not addition in variables:
            variables = variables + [addition]
    train, test = train_test_split(
        data[variables],
        test_size=0.2, random_state=1
    )
    training_labels = np.array(train['target'].astype(int))
    testing_labels = np.array(test['target'].astype(int))
    training_processes = train['process']
    testing_processes = test['process']
    traindataset = np.array(train[trainvars].values)
    testdataset = np.array(test[trainvars].values)
    data_dict = {
        'train': traindataset,
        'test': testdataset,
        'training_labels': training_labels,
        'testing_labels': testing_labels,
        'training_processes': training_processes,
        'testing_processes': testing_processes,
        'trainvars': trainvars
    }
    universal.write_data_dict_info(info_dir, data_dict)
    return data_dict