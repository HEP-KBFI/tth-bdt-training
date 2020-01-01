from tthAnalysis.bdtTraining import trainvar_choice as tc
from tthAnalysis.bdtHyperparameterOptimization import pso_main as pm
from tthAnalysis.bdtHyperparameterOptimization import universal
from tthAnalysis.bdtHyperparameterOptimization import xgb_tools as xt
from tthAnalysis.bdtHyperparameterOptimization import slurm_main as sm
from tthAnalysis.bdtTraining import xgb_tth as ttHxt
import os

def main():
    global_settings = universal.read_settings('global')
    channel = global_settings['channel']
    nthread = global_settings['nthread']
    bdtType = global_settings['bdtType']
    trainvar = global_settings['trainvar']
    output_dir = os.path.expandvars(global_settings['output_dir'])
    cmssw_base_path = os.path.expandvars('$CMSSW_BASE')
    param_file = os.path.join(
        cmssw_base_path,
        'src',
        'tthAnalysis',
        'bdtHyperparameterOptimization',
        'data',
        'xgb_parameters.json'
    )
    value_dicts = universal.read_parameters(param_file)
    pso_settings = pm.read_weights()
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    trainvars = tc.initialize_trainvars(channel)
    tc.write_new_trainvar_list(trainvars, output_dir)
    while len(trainvars) > 10:
        data, trainVars = ttHxt.tth_analysis_main(
            channel, bdtType, nthread,
            output_dir, trainvar, tc
        )
        data_dict = ttHxt.createDataSet(data, trainVars, nthread)
        print("::::::: Reading parameters :::::::")
        parameter_dicts = xt.prepare_run_params(
            value_dicts, pso_settings['sample_size'])
        print("\n============ Starting hyperparameter optimization ==========\n")
        result_dict = pm.run_pso(
            data_dict, value_dicts, sm.run_iteration, parameter_dicts
        )
        feature_importances = result_dict['feature_importances']
        if len(trainvars) > 10:
            trainvars = tc.drop_worst_parameters(feature_importances)
        if len(trainvars) > 10:
            sm.clear_from_files(global_settings)
        tc.write_new_trainvar_list(trainvars, output_dir)
        universal.save_feature_importances(result_dict, output_dir)
    print("\n============ Saving results ================\n")
    universal.save_results(result_dict, output_dir, plot_extras=True)
    sm.clear_from_files(global_settings)
    print("Results saved to " + str(output_dir))


if __name__ == '__main__':
    main()