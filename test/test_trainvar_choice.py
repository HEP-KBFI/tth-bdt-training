import os

from tthAnalysis.bdtTraining import trainvar_choice as tc
dir_path = os.path.dirname(os.path.realpath(__file__))
resources_dir = os.path.join(dir_path, 'resources')
tmp_dir = os.path.join(resources_dir, 'tmp')
if not os.path.exists(tmp_dir):
    os.makedirs(tmp_dir)

def test_write_new_trainvar_list():
    trainvars = ['foo', 'bar', 'baz']
    out_file = os.path.join(tmp_dir, 'test_vars.txt')
    tc.write_new_trainvar_list(trainvars, out_file)
    count = len(open(out_file).readlines())
    assert count == 3


def test_initialize_trainvars():
    channel = '3l_1tau'
    trainvars = tc.initialize_trainvars(channel)
    assert len(trainvars) > 0


def test_data_related_trainvars():
    testing_trainvars = ['genTrainvar', 'genWeight', 'trainVar']
    true_trainvars = tc.data_related_trainvars(testing_trainvars)
    assert len(true_trainvars) == 1


def test_dummy_delete_files():
    if os.path.exists(tmp_folder):
        shutil.rmtree(tmp_folder)