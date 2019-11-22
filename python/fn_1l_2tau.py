from tthAnalysis.bdtTraining import xgb_tth as xt
import os
import numpy as np


def read_from():
    keys =  ['TTZ', 'ttH']
    output = {
        'keys': keys
    }
    return output


def choose_trainVar(dataCard_dir, channel, trainvar, bdtType):
    try:
        trainVars_path = os.path.join(
            dataCard_dir, channel, 'ch_' + channel + '.txt')
        trainVars = xt.read_trainVars(trainVars_path)
    except:
        trainVars = ''
    return trainVars