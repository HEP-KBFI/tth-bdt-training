import json
from pathlib import Path
import os
import ROOT
import pandas
import glob
from root_numpy import tree2array
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)



def load_data(
    inputPath,
    channelInTree,
    variables,
    bdtType,
    channel,
    keys,
    dataDir,
    masses = [],
    mass_randomization = 'default',
    sel = None
):
    print_info(
        inputPath, channelInTree, variables,
        bdtType, channel, keys, masses, mass_randomization
    )
    my_cols_list = variables + ['proces', 'key', 'target', "totalWeight"]
    data = pandas.DataFrame(columns=my_cols_list)
    sampleParameters1 = os.path.join(dataDir, 'sampleParameters_dm1.json')
    sampleParameters2 = os.path.join(dataDir, 'sampleParameters_dm2.json')
    json_dicts1 = read_parameters(sampleParameters1)
    json_dicts2 = read_parameters(sampleParameters2)
    for folderName in keys:
        data = data_mainLoop(
            folderName, json_dicts1, json_dicts2,
            channelInTree, masses, inputPath,
            bdtType, data, variables,
            mass_randomization, sel)
        folderProblem = signal_background_calc(data, bdtType, folderName)
        if folderProblem:
            print('Error: No data')
            continue
    return data


def data_mainLoop(
    folderName,
    json_dicts1,
    json_dicts2,
    channelInTree,
    masses,
    inputPath,
    bdtType,
    data,
    variables,
    mass_randomization,
    sel
):
    sampleDict = findSample(
        folderName, json_dicts1, json_dicts2)
    if sampleDict == {}:
        # print('Warning: No simple sample found')
        sampleDict = advanced_SampleName(
            bdtType, folderName, masses)
    sampleName = sampleDict['sampleName']
    target = sampleDict['target']
    print(':::::::::::::::::')
    print('inputPath:\t' + str(inputPath))
    print('folderName:\t' + str(folderName))
    print('channelInTree:\t' + str(channelInTree))
    newStructure_WildCardPath = os.path.join(
        inputPath, folderName + '*', 'central', '*.root') # Saswati datacards
    paths = glob.glob(newStructure_WildCardPath)
    if len(paths) == 0:
        oldStructure_WildCardPath = os.path.join(
            inputPath, folderName + '*', '*.root') # Ram's datacards
        paths = glob.glob(oldStructure_WildCardPath)
    inputTree = str(os.path.join(
        channelInTree, 'sel/evtntuple', sampleName, 'evtTree'))
    for path in paths:
        tree, tfile, problem = read_rootTree(path, inputTree)
        if problem:
            continue
        data, problem = get_chunkDF(
            tree, tfile, sampleName,
            folderName, target, variables,
            bdtType, mass_randomization,
            masses, data, sel, inputTree,
            path
        )
    return data


def read_rootTree(path, inputTree):
    problem = False
    try:
        tfile = ROOT.TFile(path)
    except:
        print('Error: No ".root" file with the path ', path)
        problem = True
    try:
        tree = tfile.Get(inputTree)
    except:
        print(inputTree, 'FAIL read inputTree', tfile)
        problem = True
    return tree, tfile, problem


def get_chunkDF(
    tree,
    tfile,
    sampleName,
    folderName,
    target,
    variables,
    bdtType,
    mass_randomization,
    masses,
    data,
    sel,
    inputTree,
    path
):
    newVariables = variables + ['evtWeight']
    problem = False
    if tree is not None:
        try:
            chunk_arr = tree2array(tree, selection=sel)
            chunk_df = pandas.DataFrame(
                chunk_arr, columns=newVariables)
            tfile.Close()
            chunk_df['process'] = sampleName
            chunk_df['key'] = folderName
            chunk_df['target'] = target
            chunk_df['totalWeight'] = chunk_df['evtWeight']
            case1 = mass_randomization != "oversampling"
            case2 = mass_randomization == "oversampling" and target == 1
            if case1 or case2:
                data = data.append(chunk_df, ignore_index=True)
            else:
                print('Error: empty path ', path)
        except:
            tfile.Close()
            problem = True
            print(inputTree, 'FAIL read inputTree', tfile)
    return data, problem


### Something more clever needed
def advanced_SampleName(bdtType, folderName, masses):
    if 'evtLevelSUM_HH_bb2l' in bdtType or 'evtLevelSUM_HH_bb1l' in bdtType:
        if 'signal_ggf' in folderName:
            sampleName = 'signal_ggf_spin0' if 'evtLevelSUM_HH_bb2l_res' in bdtType else 'signal_ggf_nonresonant_node'
            for mass in masses:
                if mass == 20:
                    sampleName = sampleName + '_' + 'sm' + '_'
                    break
                elif '_' + str(mass) + '_' in folderName:
                    sampleName = sampleName + '_' + str(mass) + '_'
                    break
            if '_2b2v' in folderName:
                sampleName = sampleName + 'hh_bbvv'
            target = 1
    elif 'HH' in bdtType:
        if 'signal_ggf_spin0' in folderName:
            sampleName = 'signal_ggf_spin0_'
            for mass in masses:
                if str(mass) in folderName:
                    sampleName = sampleName + str(mass)
            if '_4t' in folderName:
                sampleName = sampleName + '_hh_tttt'
            if '_4v' in folderName:
                sampleName = sampleName + '_hh_wwww'
            if '_2v2t' in folderName:
                sampleName = sampleName + '_hh_wwtt'
            target = 1
    if 'ttH' in folderName:
        if 'HH' in bdtType:
            target = 0
            sampleName = 'TTH'
        else:
            target = 1
            sampleName = 'ttH' # changed from 'signal'
    sampleDict = {
        'sampleName': sampleName,
        'target': target
    }
    return sampleDict


def signal_background_calc(data, bdtType, folderName):
    folderProblem = False
    if len(data) == 0:
        folderProblem = True
    if 'evtLevelSUM_HH_bb2l' in bdtType and folderName == 'TTTo2L2Nu':
        data.drop(data.tail(6000000).index, inplace = True)
    elif 'evtLevelSUM_HH_bb1l' in bdtType:
        if folderName == 'TTToSemiLeptonic_PSweights':
            data.drop(data.tail(24565062).index, inplace=True)
        if folderName == 'TTTo2L2Nu_PSweights':
            data.drop(data.tail(11089852).index, inplace=True) #12089852
        if folderName.find('signal') !=-1:
            if folderName.find('900') ==-1 and folderName.find('1000') ==-1:
                data.drop(data.tail(15000).index,inplace = True)
            if bdtType.find('nonres') != -1:
                data.drop(data.tail(20000).index,inplace = True)
        elif folderName == 'W':
            data.drop(data.tail(2933623).index, inplace=True)
    nS = len(data.loc[(data.target.values == 1) & (data.key.values==folderName) ])
    nB = len(data.loc[(data.target.values == 0) & (data.key.values==folderName) ])
    nNW = len(data.loc[(data['totalWeight'].values < 0) & (data.key.values==folderName)])
    print('Signal: ' + str(nS))
    print('Background: ' + str(nB))
    print('Event weight: ' + str(data.loc[
            (data.key.values==folderName)]['evtWeight'].sum()))
    print('Total data weight: ' + str(data.loc[
            (data.key.values==folderName)]['totalWeight'].sum()))
    print('events with -ve weights: ' + str(nNW))
    print(':::::::::::::::::')
    return folderProblem


def findSample(folderName, json_dicts1, json_dicts2):
    sampleDict = {}
    for json_dict1 in json_dicts1:
        if json_dict1['type'] in folderName:
            sampleDict = json_dict1
    if sampleDict == {}:
        for json_dict2 in json_dicts2: # in original e.g ['THQ_ctcvcp'] ???
            if folderName == json_dict2['type']:
                sampleDict = json_dict2
    return sampleDict


def print_info(
    inputPath,
    channelInTree,
    variables,
    bdtType,
    channel,
    keys,
    masses,
    mass_randomization
):
    print('In data_manager') 
    print(':::: load_data_2017() ::::')
    print('inputPath: ' + str(inputPath))
    print('channelInTree: ' + str(channelInTree))
    print('-----------------------------------')
    print('variables:')
    print_columns(variables)
    print('bdtType: ' + str(bdtType))
    print('channel: ' + str(channel))
    print('keys: ' + str(keys))
    print('masses: ' + str(masses))
    print('mass_randomization: ' + str(mass_randomization))


def print_columns(to_print):
    to_print = sorted(to_print)
    if len(to_print) % 2 != 0:
        to_print.append(' ')
    split = int(len(to_print)/2)
    l1 = to_print[0:split]
    l2 = to_print[split:]
    print('-----------------------------------')
    for one, two in zip(l1, l2):
        print('{0:<45s} {1}'.format(one, two))
    print('-----------------------------------')


def read_parameters(path):
    json_dicts = []
    with open(path, 'rt') as f:
        for line in f:
            json_dict = json.loads(line)
            json_dicts.append(json_dict)
    return json_dicts
