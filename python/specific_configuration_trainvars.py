from tthAnalysis.bdtTraining import xgb_tth as ttHxt
import os


def read_from(): # think better solution -> read from file?
    keys = ['ttH', 'TTWJets', 'TTZ', 'TTTo2L2Nu', 'TTToSemiLeptonic']
    output = {
        'keys': keys
    }
    return output


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
        Needed for compability with texpandvarshe other trainvars loading
    bdt_type : dummy argument
        Needed for compability with the other trainvars loading

    Returns:
    -------
    trainvars : list
        list of trainvars that are to be used in the optimization.
    '''
    cmssw_base_path = os.path.expandvars('$CMSSW_BASE')
    main_dir = os.path.join(
        cmssw_base_path,
        'src',
        'tthAnalysis',
        'bdtHyperparameterOptimization'
    )
    settings_dir = os.path.join(
        main_dir, 'data')
    trainvars_path = os.path.join(
        main_dir,
        'specific_configuration_trainvars.txt'
    )
    try:
        trainvars = ttHxt.read_trainVars(trainvars_path)
    except:
        print('Could not find trainvars')
        trainvars = ''
    return trainvars