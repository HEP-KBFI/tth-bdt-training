from tthAnalysis.bdtTraining.xgb_tth import read_trainVars
from tthAnalysis.bdtTraining.xgb_tth import to_oneDict
import os 
dir_path = os.path.dirname(os.path.realpath(__file__))
resources_dir = os.path.join(dir_path, 'resources')


def test_read_trainVars():
    path = os.path.join(resources_dir, 'variables.txt')
    result = read_trainVars(path)
    expected = ['foo', 'bar']
    assert result == expected


def test_to_oneDict():
    dict_list = [
        {'a': 1},
        {'b': 2},
        {'c': 3}
    ]
    expected = {
        'a': 1,
        'b': 2,
        'c': 3
    }
    result = to_oneDict(dict_list)
    assert result == expected
