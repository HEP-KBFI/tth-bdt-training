'''
Call with 'python'

Usage: run_specific_configuration.py --parameter_file=PTH

Options:
    -p --parameter_file=PTH      Path to parameters to be run

'''
import importlib
import numpy as np
import os
from tthAnalysis.bdtTraining import tth_data_handler as ttHxt
from tthAnalysis.bdtHyperparameterOptimization import universal
from tthAnalysis.bdtHyperparameterOptimization import nn_tools as nnt
from tthAnalysis.bdtHyperparameterOptimization import xgb_tools as xt
from tthAnalysis.bdtTraining import specific_configuration_trainvars as sct
from tthAnalysis.bdtTraining import trainvar_choice as tc


def main(parameter_file):
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
    nthread = global_settings['nthread']
    output_dir = os.path.expandvars(global_settings['output_dir'])
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    universal.save_run_settings(output_dir)
    data, trainVars = ttHxt.tth_analysis_main(
        channel, bdtType, nthread,
        output_dir, trainvar, sct
    )
    tc.plot_data_correlation(data, trainVars, output_dir)
    parameter_dict = universal.read_parameters(parameter_file)[0]
    if global_settings['ml_method'] == 'xgb':
        data_dict = ttHxt.create_xgb_data_dict(data, trainVars, global_settings)
        score_dict, pred_train, pred_test, feature_importance = xt.parameter_evaluation(
            parameter_dict,
            data_dict,
            global_settings['nthread'],
            global_sttings['num_class']
        )
    elif global_settings['ml_method'] == 'nn':
        data_dict = ttHxt.create_nn_data_dict(data, trainVars, global_settings)
        score_dict, pred_train, pred_test, feature_importance = nnt.parameter_evaluation(
            parameter_dict,
            data_dict,
            global_settings['nthread'],
            global_sttings['num_class'],
            return_true_feature_importances=True
        )
    else:
        print('Unknown ml_method chosen in global_settings')
    print("\n============ Saving results ================\n")
    auc_info = universal.calculate_auc(
        result_dict['data_dict'],
        result_dict['pred_train'],
        result_dict['pred_test']
    )[-1]
    universal.plot_roc_curve(output_dir, auc_info)
    universal.save_feature_importances(result_dict, output_dir)
    universal.best_to_file(score_dict, output_dir, {})
    print("Results saved to " + str(output_dir))





if __name__ == '__main__':
    try:
        arguments = docopt.docopt(__doc__)
        parameter_file = arguments['--parameter_file']
        output_dir = arguments['--output_dir']
        main(parameter_file)
    except docopt.DocoptExit as e:
        print(e)