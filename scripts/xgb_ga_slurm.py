'''
XGBoost for stuff.
Call with 'python'

Usage: ga_slurm_tth_analysis.py
'''
import os
import warnings
from tthAnalysis.bdtTraining import tth_data_handler as ttHxt
from tthAnalysis.bdtHyperparameterOptimization import universal
from tthAnalysis.bdtHyperparameterOptimization import ga_main as gm
from tthAnalysis.bdtHyperparameterOptimization import xgb_tools as xt
from tthAnalysis.bdtHyperparameterOptimization import slurm_main as sm
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)


def main():
    print('::::::: Reading GA settings & XGBoost parameters :::::::')
    global_settings = universal.read_settings('global')
    output_dir = os.path.expandvars(global_settings['output_dir'])
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    settings_dict = universal.read_settings('ga')
    settings_dict.update(global_settings)
    cmssw_base_path = os.path.expandvars('$CMSSW_BASE')
    param_file = os.path.join(
        cmssw_base_path,
        'src',
        'tthAnalysis',
        'bdtHyperparameterOptimization',
        'data',
        'xgb_parameters.json'
    )
    param_dict = universal.read_parameters(param_file)

    print('::::::: Loading data ::::::::')
    channel = global_settings['channel']
    bdtType = global_settings['bdtType']
    trainvar = global_settings['trainvar']
    fnFile = '_'.join(['fn', channel])
    importString = "".join(['tthAnalysis.bdtTraining.', fnFile])
    cf = __import__(importString, fromlist=[''])
    nthread = global_settings['nthread']
    data, trainVars = ttHxt.tth_analysis_main(
        channel, bdtType, nthread,
        output_dir, trainvar, cf
    )
    data_dict = ttHxt.create_xgb_data_dict(data, trainVars, nthread)

    print("\n============ Starting hyperparameter optimization ==========\n")
    result_dict = gm.evolution(
        settings_dict,
        data_dict,
        param_dict,
        xt.prepare_run_params,
        sm.run_iteration
    )

    print("\n============ Saving results ================\n")
    universal.save_results(result_dict, output_dir, plot_extras=True)
    sm.clear_from_files(global_settings)
    print("Results saved to " + str(output_dir))


if __name__ == '__main__':
    main()
