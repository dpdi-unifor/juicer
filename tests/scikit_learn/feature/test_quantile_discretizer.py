from tests.scikit_learn import util
from juicer.scikit_learn.feature_operation import QuantileDiscretizerOperation
import pytest
import pandas as pd

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)


# QuantileDiscretizer
#
#
# # # # # # # # # # Success # # # # # # # # # #
def test_quantile_discretizer_kmeans_success():
    df = util.iris(['sepallength', 'sepalwidth'], size=10)
    arguments = {
        'parameters': {'attribute': ['sepallength', 'sepalwidth'],
                       'multiplicity': {'input data': 0},
                       'n_quantiles': 2, 'output_distribution': 'kmeans',
                       },
        'named_inputs': {
            'input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = QuantileDiscretizerOperation(**arguments)
    result = util.execute(instance.generate_code(),
                          {'df': df})
    print(result['out'])


def test_quantile_discretizer_uniform_success():
    df = util.iris(['sepallength', 'sepalwidth'], size=10)
    arguments = {
        'parameters': {'attribute': ['sepallength', 'sepalwidth'],
                       'multiplicity': {'input data': 0},
                       'n_quantiles': 2, 'output_distribution': 'uniform',
                       },
        'named_inputs': {
            'input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = QuantileDiscretizerOperation(**arguments)
    result = util.execute(instance.generate_code(),
                          {'df': df})
    print(result['out'])


def test_quantile_discretizer_quantile_success():
    df = util.iris(['sepallength', 'sepalwidth'], size=10)
    arguments = {
        'parameters': {'attribute': ['sepallength', 'sepalwidth'],
                       'multiplicity': {'input data': 0},
                       'n_quantiles': 2, 'output_distribution': 'quantile',
                       },
        'named_inputs': {
            'input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = QuantileDiscretizerOperation(**arguments)
    result = util.execute(instance.generate_code(),
                          {'df': df})
    print(result['out'])
