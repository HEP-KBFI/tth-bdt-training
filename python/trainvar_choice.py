'''
Global functions for finding the optimal trainingvariables
'''
import json
import ROOT
from tthAnalysis.bdtHyperparameterOptimization import universal
from tthAnalysis.bdtTraining import data_loader as dl
from tthAnalysis.bdtTraining import xgb_tth as ttHxt
import matplotlib.pyplot as plt
import glob
import os
import pandas
import csv
import numpy as np


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
        trainvars = ttHxt.read_trainVars(trainvars_path)
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
    false_trainvars = [
        'gen', 'Weight', 'weight', 'lumi', 'event', 'mva', 'tau1_pt',
        'lep1_pt', 'lep2_pt', 'lep3_pt', 'htmiss', 'massL_FO',
        'decayMode', 'DecayMode', '_isTight', 'run', 'Raw', 'ID', 'ptmiss']
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
    return trainvars, worst_performing_feature


def plot_feature_importances(feature_importances, output_dir):
    output_path = os.path.join(output_dir, 'feature_importances.png')
    histogram_plot = pandas.Series(
        feature_importances,
        index=feature_importances.keys()
    ).sort_values().plot(kind='barh')
    histogram_plot.figure.savefig(
        output_path, bbox_inches='tight')


def read_from(): # think better solution -> read from file?
    keys = ['ttH', 'TTWJets', 'TTZ', 'TTTo2L2Nu', 'TTToSemiLeptonic']
    output = {
        'keys': keys
    }
    return output


def plot_auc_vs_nr_trainvars(auc_values, nr_trainvars, output_dir):
    plt.plot(nr_trainvars, auc_values)
    plt.ylabel('auc')
    plt.xlabel('nr_trainvars')
    plt.gca().invert_xaxis()
    plt.ylim(0.5, 1.0)
    plt.minorticks_on()
    plt.grid(b=True, which='major')
    plt.grid(b=True, which='minor', color='lightgray', linestyle='--')
    plt.title('AUC vs nr_trainvars')
    out_path = os.path.join(output_dir, 'auc_vs_nr_trainvars.png')
    plt.savefig(output_path)


def plot_data_correlation(data, trainvars, output_dir):
    output_path = os.path.join(output_dir, 'data_correlation.png')
    data = data[trainvars]
    plt.figure(figsize=(21, 16))
    plt.matshow(data.corr(), fignum=1, aspect='auto', cmap='inferno')
    plt.xticks(range(data.shape[1]), data.columns, rotation=90, fontsize=10)
    plt.yticks(range(data.shape[1]), data.columns, fontsize=10)
    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=14)
    plt.savefig(output_path, bbox_inches='tight')
    plt.close('all')


def write_worst_performing_features_to_file(
        worst_performing_features,
        output_dir
):
''' In the order of the drop'''
    output_path = os.path.join(output_dir, 'worst_performing_features.txt')
    with open(output_path, 'wt') as file:
        writer = csv.writer(file)
        writer.writerows(worst_performing_features)


def plot_distribution(result_dict, output_dir, nr_bins=70):
    data_dict = result_dict['data_dict']
    testing_processes = data_dict['testing_processes']
    training_processes = data_dict['training_processes']
    pred_train = result_dict['pred_train']
    pred_test = result_dict['pred_test']
    different_processes = list(set(training_processes))
    test_signal_probabilities = [item[1] for item in pred_test]
    train_signal_probabilities = [item[1] for item in pred_train]
    test_df = pandas.DataFrame(
        {
        'value': test_signal_probabilities,
        'process': testing_processes
        }
    )
    train_df = pandas.DataFrame(
        {
        'value': train_signal_probabilities,
        'process': training_processes
        }
    )
    step = 1. / nr_bins
    bins = np.arange(start=0, stop=1, step=step)
    test_path = os.path.join(output_dir, 'test_set_distribution.png')
    train_path = os.path.join(output_dir, 'train_set_distribution.png')
    alpha_value = 1. / (len(different_processes))
    for process in different_processes:
        plt.hist(
            test_df.loc[test_df['process'] == process, 'value'],
            label=process,
            alpha=alpha_value,
            histtype='bar',
            bins=bins
        )
    plt.legend()
    plt.xlabel('Signal probability')
    plt.ylabel('Count')
    ax = plt.gca()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    plt.savefig(test_path)
    plt.close('all')
    for process in different_processes:
        plt.hist(
            train_df.loc[train_df['process'] == process, 'value'],
            label=process,
            alpha=alpha_value,
            histtype='bar',
            bins=bins
        )
    plt.legend()
    plt.xlabel('Signal probability')
    plt.ylabel('Count')
    ax = plt.gca()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    plt.savefig(train_path)
    plt.close('all')


def plot_each_trainvar_distributions(data, trainvars, output_dir, bins=70):
    different_processes = list(set(data['process']))
    alpha_value = 1. / (len(different_processes))
    trainvars_distribution_dir = os.path.join(
        output_dir, 'trainvar_distributions')
    if not os.path.exists(trainvars_distribution_dir):
        os.makedirs(trainvars_distribution_dir)
    for trainvar in trainvars:
        out_path = os.path.join(trainvars_distribution_dir, trainvar + '.png')
        bins = plt.hist(data[trainvar])[1]
        plt.close('all')
        for process in different_processes:
            plt.hist(
                data.loc[data['process'] == process, trainvar],
                histtype='bar',
                label=process,
                alpha=alpha_value,
                bins=bins
            )
        ax = plt.gca()
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')
        plt.legend()
        plt.title(trainvar)
        plt.savefig(out_path)
        plt.close('all')


# plot_distribution(result_dict, "/home/laurits")