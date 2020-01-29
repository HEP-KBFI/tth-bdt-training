from tthAnalysis.bdtTraining import trainvar_choice as tc
from tthAnalysis.bdtHyperparameterOptimization import pso_main as pm
from tthAnalysis.bdtHyperparameterOptimization import universal
from tthAnalysis.bdtHyperparameterOptimization import xgb_tools as xt
from tthAnalysis.bdtHyperparameterOptimization import slurm_main as sm
from tthAnalysis.bdtTraining import tth_data_handler as ttHxt
import os


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
    nthread = global_settings['nthread']
    bdtType = global_settings['bdtType']
    trainvar = global_settings['trainvar']
    output_dir = os.path.expandvars(global_settings['output_dir'])
    param_file = os.path.join(
        settings_dir
        'xgb_parameters.json'
    )
    value_dicts = universal.read_parameters(param_file)
    pso_settings = pm.read_weights(settings_dir)
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    universal.save_run_settings(output_dir)
    trainvars = tc.initialize_trainvars(channel)
    tc.write_new_trainvar_list(trainvars, output_dir)
    auc_values = []
    nr_trainvars = []
    plot_correlation = True
    worst_performing_features = []
    while len(trainvars) > 10:
        data, trainvars = ttHxt.tth_analysis_main(
            channel, bdtType, nthread,
            output_dir, trainvar, tc
        )
        print("Number trainvars:" + str(len(trainvars)))
        print("Trainvars: " + str(trainvars))
        data = ttHxt.convert_data_to_correct_format(data)
        if plot_correlation:
            tc.plot_data_correlation(data, trainvars, output_dir)
        data_dict = ttHxt.createDataSet(data, trainvars, global_settings)
        print("::::::: Reading parameters :::::::")
        parameter_dicts = xt.prepare_run_params(
            value_dicts, pso_settings['sample_size'])
        print("\n============ Starting hyperparameter optimization ==========\n")
        result_dict = pm.run_pso(
            data_dict, value_dicts, sm.run_iteration, parameter_dicts
        )
        auc_values.append(result_dict['best_test_auc'])
        nr_trainvars.append(len(trainvars))
        feature_importances = result_dict['feature_importances']
        if len(trainvars) > 10:
            trainvars, worst_performing_feature = tc.drop_worst_parameters(
                feature_importances)
            worst_performing_features.append(worst_performing_feature)
        if len(trainvars) > 10:
            sm.clear_from_files(global_settings)
        tc.write_new_trainvar_list(trainvars, output_dir)
        universal.save_feature_importances(result_dict, output_dir)
        plot_correlation = False
        print(
            "................. Number trainvars: "
            + str(len(trainvars)) 
            +  " ..........................."
        )
        print("Trainvars: " + str(trainvars))
    print("\n============ Saving results ================\n")
    tc.write_new_trainvar_list(trainvars, output_dir)
    tc.plot_auc_vs_nr_trainvars(auc_values, nr_trainvars, output_dir)
    tc.plot_feature_importances(feature_importances, output_dir)
    tc.plot_distribution(result_dict, output_dir)
    universal.save_results(result_dict, output_dir, plot_extras=True)
    sm.clear_from_files(global_settings)
    tc.write_worst_performing_features_to_file(
        worst_performing_features, output_dir)
    print("Results saved to " + str(output_dir))


if __name__ == '__main__':
    main()