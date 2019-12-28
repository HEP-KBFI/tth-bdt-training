'''
Global functions for finding the optimal trainingvariables
'''
import json
import ROOT
from tthAnalysis.bdtHyperparameterOptimization import universal
from tthAnalysis.bdtTraining import data_loader as dl
from tthAnalysis.bdtTraining import xgb_tth as ttHxt
import glob
import os


def write_new_trainvar_list(trainvars, out_dir):
    '''Writes new trainvars to be tested into a file

    Parameters:
    ----------
    trainvars : list
        List of training variables to be outputted into a file
    out_file : str
        Path to the file of the trainvars

    Returns:
    -------
    Nothing
    '''
    out_file = os.path.join(out_dir, 'optimization_trainvars.txt')
    with open(out_file, 'w') as file:
        for trainvar in trainvars[:-1]:
            file.write(str(trainvar) + '\n')
        file.write(str(trainvars[-1]))


def choose_trainVar(datacard_dir, channel, trainvar, bdt_type):
    '''Reads the training variables from the data folder from file 
    'optimization_trainvars.txt'. Is used for the xgb_tth cf function.

    Parametrs:
    ---------
    datacard_dir : dummy argument
        Needed for compability with the other trainvars loading
    channel : dummy argument
        Needed for compability with the other trainvars loading
    trainvar : dummy argument
        Needed for compability with the other trainvars loading
    bdt_type : dummy argument
        Needed for compability with the other trainvars loading

    Returns:
    -------
    trainvars : list
        list of trainvars that are to be used in the optimization.
    '''
    global_settings = universal.read_settings('global')
    out_dir = os.path.expandvars(global_settings['output_dir'])
    trainvars_path = os.path.join(
        out_dir,
        'optimization_trainvars.txt'
    )
    try:
        trainvars = xt.read_trainVars(trainvars_path)
    except:
        print('Could not find trainvars')
        trainvars = ''
    return trainvars


def initialize_trainvars(channel):
    '''Reads in all the possible trainvars for initial run

    Parameters:
    ----------
    None

    Returns:
    trainvars : list
        list of all possible trainvars that are to be used in the optimization
    '''
    cmssw_base_path = os.path.expandvars('$CMSSW_BASE')
    data_folder = os.path.join(
        cmssw_base_path,
        'src',
        'tthAnalysis',
        'bdtTraining',
        'data'
    )
    tth_folder = os.path.join(
        data_folder,
        'ttH'
    )
    trainvars_path = os.path.join(
        data_folder,
        str(channel),
        'par_' + str(channel) + '.json'
    )
    parameter_list = universal.read_parameters(trainvars_path)
    info_dict = universal.to_one_dict(parameter_list)
    path_to_files = info_dict['inputPath16']
    random_sample = 'TTZ'
    wildcard_root_files = os.path.join(path_to_files, '*' + random_sample + '*', 'central', '*.root')
    single_root_file = glob.glob(wildcard_root_files)[0]
    channelInTree = info_dict['channelInTree']
    sampleParameters1 = os.path.join(tth_folder, 'sampleParameters_dm1.json')
    sampleParameters2 = os.path.join(tth_folder, 'sampleParameters_dm2.json')
    global_settings = universal.read_settings('global')
    json_dicts1 = universal.read_parameters(sampleParameters1)
    json_dicts2 = universal.read_parameters(sampleParameters2)
    folderName = random_sample
    sampleDict = dl.findSample(
        folderName, json_dicts1, json_dicts2)
    if sampleDict == {}:
        sampleDict = dl.advanced_SampleName(
            global_settings['bdtType'], folderName, []) # TTZ is just a random choice
    sampleName = sampleDict['sampleName']
    inputTree = str(os.path.join(
        channelInTree, 'sel/evtntuple', sampleName, 'evtTree'))
    trainvars = access_ttree(single_root_file, inputTree)
    trainvars = data_related_trainvars(trainvars)
    return trainvars


def data_related_trainvars(trainvars):
    '''Drops non-data-related trainvars like 'gen' and 'Weight'

    Parameters:
    ----------
    trainvars : list
        Not cleaned trainvariable list

    Returns:
    -------
    true_trainvars : list
        Updated trainvar list, that contains only data-related trainvars
    '''
    false_trainvars = ['gen', 'Weight']
    true_trainvars = []
    for trainvar in trainvars:
        do_not_include = 0
        for false_trainvar in false_trainvars:
            if false_trainvar in trainvar:
                do_not_include += 1
        if do_not_include == 0:
            true_trainvars.append(trainvar)
    return true_trainvars


def access_ttree(single_root_file, inputTree):
    '''Accesses the TTree and gets all trainvars from the branches

    Parameters:
    ----------
    single_root_file : str
        Path to the .root file the TTree is located in
    inputTree : str
        Path of the TTree

    Returns:
    -------
    trainvars : list
        List of all branch names in the .root file
    '''
    trainvars = []
    tfile = ROOT.TFile(single_root_file)
    ttree = tfile.Get(inputTree)
    branches = ttree.GetListOfBranches()
    for branch in branches:
        trainvars.append(branch.GetName())
    return trainvars


def relate_trainvar_name_and_number(feature_importances, trainvars):
    '''Relates the trainvar number and the name

    Parameters:
    ----------
    feature_importances : dict
        Dictionary containing the feature importances with f{int} as key name
        and the importance as the key value.
    trainvars : list
        List of trainvars

    Returns:
    -------
    named_dict : dict
        Dictionary with assinged names for the feature_importances
    '''
    named_dict = {}
    for key in feature_importances:
        feature_nr = int(str(key).strip('f'))
        var_name = trainvars[feature_nr]
        named_dict[var_name] = feature_importances[key]
    return named_dict


def drop_worst_parameters(named_feature_importances):
    '''Drops the worst performing training variable

    Parameters:
    ----------
    named_feature_importances : dict
        Contains the trainvar and the corresponding 'gain' score

    Returns:
    -------
    trainvars : list
        list of trainvars with the worst performing one removed
    '''
    worst_performing_value = 10000
    worst_performing_feature = ""
    for trainvar in named_feature_importances:
        value = named_feature_importances[trainvar]
        if value < worst_performing_value:
            worst_performing_feature = trainvar
            worst_performing_value = value
    trainvars = named_feature_importances.keys()
    index = trainvars.index(worst_performing_feature)
    del trainvars[index]
    return trainvars


def run_trainvar_optimization():
    global_settings = universal.read_settings('global')
    channel = global_settings['channel']
    trainvars = initialize_trainvars(channel)


# Write what sample with what path was used and which trainvars were the optimal
