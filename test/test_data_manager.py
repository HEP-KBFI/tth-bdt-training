from tthAnalysis.bdtTraining.data_loader import read_parameters
from tthAnalysis.bdtTraining.data_loader import findSample
import os
dir_path = os.path.dirname(os.path.realpath(__file__))
resources_dir = os.path.join(dir_path, 'resources')


def test_read_parameters():
    path = os.path.join(resources_dir, 'parameters.json')
    result = read_parameters(path)[0]
    expected = {'a': 1, 'b': 2, 'c': 3}
    assert result == expected


def test_findSample():
    folderName1 = "WWZ"
    folderName2 = "THQ_ctcvcp"
    folderName3 = "TTWW"
    json_dicts1 = [
        {"type": "TTT", "sampleName": "TT", "target": 0},
        {"type": "TTZ", "sampleName": "TTZ", "target": 0},
        {"type": "DY", "sampleName": "DY", "target": 0},
        {"type": "W", "sampleName": "W", "target": 0},
        {"type": "ZZ", "sampleName": "ZZ", "target": 0},
        {"type": "WW", "sampleName": "WW", "target": 0},
        {"type": "WZ", "sampleName": "WZ", "target": 0},
        {"type": "WWZ", "sampleName": "WW", "target": 0},
        {"type": "WpWp", "sampleName": "Other", "target": 0},
        {"type": "VH", "sampleName": "VH", "target": 0},
        {"type": "TTW", "sampleName": "TTW", "target": 0},
        {"type": "TTWW", "sampleName": "TTWW", "target": 0}
    ]
    json_dicts2 = [
        {
            "type": "THQ_ctcvcp",
            "sampleName": "tHq",
            "sampleNameF": "tHq",
            "target": 0
        },
        {
            "type": "THW_ctcvcp",
            "sampleName": "tHW",
            "sampleNameF": "tHW",
            "target": 0
        }
    ]
    result1 = findSample(folderName1, json_dicts1, json_dicts2)
    result2 = findSample(folderName2, json_dicts1, json_dicts2)
    result3 = findSample(folderName3, json_dicts1, json_dicts2)
    expected1 = {"type": "WWZ", "sampleName": "WW", "target": 0}
    expected2 = {
        "type": "THQ_ctcvcp",
        "sampleName": "tHq",
        "sampleNameF": "tHq",
        "target": 0
    }
    expected3 = {"type": "TTWW", "sampleName": "TTWW", "target": 0}
    assert result1 == expected1
    assert result2 == expected2
    assert result3 == expected3