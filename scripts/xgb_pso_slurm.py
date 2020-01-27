'''
XGBoost for stuff.
Call with 'python'

Usage: slurm_tth_analysis.py
'''


import importlib
import numpy as np
import os
from tthAnalysis.bdtTraining import tth_data_handler as ttHxt
from tthAnalysis.bdtHyperparameterOptimization import universal
from tthAnalysis.bdtHyperparameterOptimization import pso_main as pm
from tthAnalysis.bdtHyperparameterOptimization import xgb_tools as xt
from tthAnalysis.bdtHyperparameterOptimization import slurm_main as sm
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)



def main():
    cmssw_base_path = os.path.expandvars('$CMSSW_BASE')
    main_dir = os.path.join(
        cmssw_base_path,
        'src',
        'tthAnalysis',
        'bdtHyperparameterOptimization'
    )
    settings_dir = os.path.join(
        main_dir, 'data')
    global_settings = universal.read_settings(settings_dir, 'global')
    channel = global_settings['channel']
    bdtType = global_settings['bdtType']
    trainvar = global_settings['trainvar']
    fnFile = '_'.join(['fn', channel])
    importString = "".join(['tthAnalysis.bdtTraining.', fnFile])
    cf = __import__(importString, fromlist=[''])
    nthread = global_settings['nthread']
    output_dir = os.path.expandvars(global_settings['output_dir'])
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    data, trainVars = ttHxt.tth_analysis_main(
        channel, bdtType, nthread,
        output_dir, trainvar, cf
    )
    data_dict = ttHxt.create_xgb_data_dict(data, trainVars, nthread)
    print("::::::: Reading parameters :::::::")
    param_file = os.path.join(
        settings_dir
        'xgb_parameters.json'
    )
    value_dicts = universal.read_parameters(param_file)
    pso_settings = pm.read_weights()
    parameter_dicts = xt.prepare_run_params(
        value_dicts, pso_settings['sample_size'])
    print("\n============ Starting hyperparameter optimization ==========\n")
    result_dict = pm.run_pso(
        data_dict, value_dicts, sm.run_iteration, parameter_dicts
    )
    print("\n============ Saving results ================\n")
    universal.save_results(result_dict, output_dir, plot_extras=True)
    sm.clear_from_files(global_settings)
    print("Results saved to " + str(output_dir))



if __name__ == '__main__':
    main()