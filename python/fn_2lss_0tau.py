from tthAnalysis.bdtTraining import xgb_tth as xt
import os
import numpy as np


def choose_trainVar(dataCard_dir, channel, trainvar, bdtType):
    try:
        trainVars_path = os.path.join(
            dataCard_dir, channel, 'ch_' + channel + '.txt')
        trainVars = xt.read_trainVars(trainVars_path)
    except:
        print("Could not find trainVars")
        trainVars = ''
    return trainVars


# norm_2lss_0tau = [66., 156., 96., 139]
def read_from():
    withFolder = False
    label = '2lss_0tau_NN_tHcat-correctBal_subcategories_2019Apr22'
    mom = '/afs/cern.ch/work/a/acarvalh/CMSSW_10_2_10/src/data_tth/'+label+'/' 
    bdtTypes = [
        'output_NN_2lss_ttH_tH_4cat_onlyTHQ_v4'
    ]
    keys = ['ttHJetToNonbb', 'TTWJets', 'TTZ', 'TTTo2L2Nu', 'TTToSemiLeptonic']
    # If there are subcategories construct the list of files to read based on their naming convention 
    cateDraw_type = ['ttH', 'tH', 'ttW']
    cateDraw_flavour = ['ee_bl', 'em_bl', 'mm_bl', 'ee_bt', 'em_bt', 'mm_bt']
    bdtTypes_exp = []
    for bdtType in bdtTypes:
        for catType in cateDraw_type:
            for cat_flavour in cateDraw_flavour:
                bdtTypes_exp += [ bdtType + '_' + catType + '_' + cat_flavour]
        for catType in ['rest']:
            for cat_flavour in ['bl', 'bt']:
                bdtTypes_exp += [ bdtType + '_' + catType + '_' + cat_flavour]
    if len(cateDraw_type) == 0:
        bdtTypes_exp = bdtTypes
    channelsTypes = ['2lss_0tau']
    ch_nickname = '2lss'
    originalBinning = 100
    nbinRegular = np.arange(10, 12)
    nbinQuant = np.arange(1, 20) 
    maxlim = 2.0
    output = {
        'withFolder'      : withFolder,
        'label'           : label,
        'mom'             : mom,
        'bdtTypes'        : bdtTypes_exp,
        'channelsTypes'   : channelsTypes,
        'originalBinning' : originalBinning,
        'nbinRegular'     : nbinRegular,
        'nbinQuant'       : nbinQuant,
        'maxlim'          : maxlim,
        'ch_nickname'     : ch_nickname,
        'keys'            : keys
    }
    return output


