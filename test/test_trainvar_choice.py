import os

from tthAnalysis.bdtTraining import trainvar_choice as tc
dir_path = os.path.dirname(os.path.realpath(__file__))
resources_dir = os.path.join(dir_path, 'resources')
tmp_dir = os.path.join(resources_dir, 'tmp')
if not os.path.exists(tmp_dir):
    os.makedirs(tmp_dir)

def test_write_new_trainvar_list():
    trainvars = ['foo', 'bar', 'baz']
    out_file = os.path.join(tmp_dir, 'optimization_trainvars.txt')
    tc.write_new_trainvar_list(trainvars, tmp_dir)
    count = len(open(out_file).readlines())
    assert count == 3


def test_choose_trainVar():
    datacard_dir = 'dummy'
    channel = 'dummy'
    trainvar = 'dummy'
    bdt_type = 'dummy'
    cmssw_base_path = os.path.expandvars('$CMSSW_BASE')
    main_dir = os.path.join(
        cmssw_base_path,
        'src',
        'tthAnalysis',
        'bdtHyperparameterOptimization'
    )
    global_settings_path = os.path.join(
        main_dir, 'data', 'global_settings.json')
    test_global_settings_path = os.path.join(
        main_dir, 'test', 'resources', 'global_settings.json')
    os.rename(global_settings_path, global_settings_path + '_')
    shutil.copy(test_global_settings_path, global_settings_path)
    trainvars = tc.choose_trainVar(datacard_dir, channel, trainvar, bdt_type)
    os.rename(global_settings_path + '_', global_settings_path)
    assert trainvars == ['foo', 'bar', 'baz']


def test_initialize_trainvars():
    channel = '3l_1tau'
    trainvars = tc.initialize_trainvars(channel)
    assert len(trainvars) > 0


def test_data_related_trainvars():
    testing_trainvars = ['genTrainvar', 'genWeight', 'trainVar']
    true_trainvars = tc.data_related_trainvars(testing_trainvars)
    assert len(true_trainvars) == 1


def test_dummy_delete_files():
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)