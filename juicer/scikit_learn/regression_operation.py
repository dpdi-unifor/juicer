# -*- coding: utf-8 -*-

from textwrap import dedent
from juicer.operation import Operation
from itertools import zip_longest
import pandas as pd
import re


# noinspection PyAbstractClass
class RegressionOperation(Operation):
    FEATURES_PARAM = 'features'
    LABEL_PARAM = 'label'
    PREDICTION_ATTR_PARAM = 'prediction'

    __slots__ = ('label', 'features', 'prediction')

    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)
        self.label = self.features = self.prediction = None
        self.output = named_outputs.get(
            'algorithm', 'regression_algorithm_{}'.format(self.order))

    def read_common_params(self, parameters):
        if not all([self.LABEL_PARAM in parameters,
                    self.FEATURES_PARAM in parameters]):
            msg = _("Parameters '{}' and '{}' must be informed for task {}")
            raise ValueError(msg.format(
                self.FEATURES_PARAM, self.LABEL_PARAM,
                self.__class__))
        else:
            self.label = parameters.get(self.LABEL_PARAM)[0]
            self.features = parameters.get(self.FEATURES_PARAM)[0]
            self.prediction = parameters.get(self.PREDICTION_ATTR_PARAM)[0]
            self.output = self.named_outputs.get(
                'algorithm', 'regression_algorithm_{}'.format(self.order))

    def get_output_names(self, sep=', '):
        return self.output

    def get_data_out_names(self, sep=','):
        return ''


class RegressionModelOperation(Operation):
    FEATURES_PARAM = 'features'
    LABEL_PARAM = 'label'
    PREDICTION_COL_PARAM = 'prediction'

    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)

        self.has_code = any([len(named_outputs) > 0 and len(named_inputs) == 2,
                             self.contains_results()])

        if self.has_code:
            self.algorithm = self.named_inputs['algorithm']
            self.input = self.named_inputs['train input data']

            if not all([self.FEATURES_PARAM in parameters,
                        self.LABEL_PARAM in parameters]):
                msg = _("Parameters '{}' and '{}' must be informed for task {}")
                raise ValueError(msg.format(
                    self.FEATURES_PARAM, self.LABEL_PARAM,
                    self.__class__.__name__))

            self.features = parameters[self.FEATURES_PARAM]
            self.label = parameters[self.LABEL_PARAM]
            self.prediction = parameters.get(self.PREDICTION_COL_PARAM,
                                             'prediction') or 'prediction'
            self.model = self.named_outputs.get(
                'model', 'model_{}'.format(self.order))
            self.output = self.named_outputs.get(
                'output data', 'out_task_{}'.format(self.order))

    @property
    def get_inputs_names(self):
        return ', '.join([self.named_inputs['train input data'],
                          self.named_inputs['algorithm']])

    def get_data_out_names(self, sep=','):
        return self.output

    def get_output_names(self, sep=', '):
        return sep.join([self.output, self.model])

    def generate_code(self):

        if self.has_code:
            code = """
            algorithm = {algorithm}
            {output_data} = {input}.copy()
            X_train = {input}['{features}'].values.tolist()
            if 'IsotonicRegression' in str(algorithm):
                X_train = np.ravel(X_train)
            y = {input}['{label}'].values.tolist()
            {model} = algorithm.fit(X_train, y)
            {output_data}['{prediction}'] = algorithm.predict(X_train).tolist()
            """.format(model=self.model, algorithm=self.algorithm,
                       input=self.named_inputs['train input data'],
                       output_data=self.output, prediction=self.prediction,
                       label=self.label[0], features=self.features[0])

            return dedent(code)


class GradientBoostingRegressorOperation(RegressionOperation):
    LEARNING_RATE_PARAM = 'learning_rate'
    N_ESTIMATORS_PARAM = 'n_estimators'
    MAX_DEPTH_PARAM = 'max_depth'
    MIN_SPLIT_PARAM = 'min_samples_split'
    MIN_LEAF_PARAM = 'min_samples_leaf'
    SEED_PARAM = 'seed'

    def __init__(self, parameters, named_inputs, named_outputs):
        RegressionOperation.__init__(self, parameters, named_inputs,
                                     named_outputs)

        self.name = 'regression.GradientBoostingRegressor'
        self.has_code = len(named_outputs) > 0

        if self.has_code:
            self.learning_rate = parameters.get(
                    self.LEARNING_RATE_PARAM, 0.1) or 0.1
            self.n_estimators = parameters.get(
                    self.N_ESTIMATORS_PARAM, 100) or 100
            self.max_depth = parameters.get(
                    self.MAX_DEPTH_PARAM, 3) or 3
            self.min_samples_split = parameters.get(
                    self.MIN_SPLIT_PARAM, 2) or 2
            self.min_samples_leaf = parameters.get(
                    self.MIN_LEAF_PARAM, 1) or 1
            self.seed = parameters.get(self.SEED_PARAM, 'None')

            vals = [self.learning_rate, self.n_estimators,
                    self.min_samples_split, self.min_samples_leaf]
            atts = [self.LEARNING_RATE_PARAM, self.N_ESTIMATORS_PARAM,
                    self.MIN_SPLIT_PARAM, self.MIN_LEAF_PARAM]
            for var, att in zip(vals, atts):
                if var <= 0:
                    raise ValueError(
                            _("Parameter '{}' must be x>0 for task {}").format(
                                    att, self.__class__))
            self.has_import = \
                "from sklearn.ensemble import GradientBoostingRegressor\n"

    def generate_code(self):
        code = dedent("""
        {output} = GradientBoostingRegressor(learning_rate={learning_rate},
        n_estimators={n_estimators}, max_depth={max_depth}, 
        min_samples_split={min_samples_split}, 
        min_samples_leaf={min_samples_leaf}, random_state={seed})""".format(
                output=self.output, learning_rate=self.learning_rate,
                n_estimators=self.n_estimators, max_depth=self.max_depth,
                seed=self.seed, min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf))
        return code


class HuberRegressorOperation(RegressionOperation):

    """
    Linear regression model that is robust to outliers.
    """

    EPSILON_PARAM = 'epsilon'
    MAX_ITER_PARAM = 'max_iter'
    ALPHA_PARAM = 'alpha'
    TOLERANCE_PARAM = 'tol'

    def __init__(self, parameters, named_inputs, named_outputs):
        RegressionOperation.__init__(self, parameters, named_inputs,
                                     named_outputs)
        self.parameters = parameters
        self.name = 'regression.HuberRegressor'
        self.has_code = len(self.named_outputs) > 0

        if self.has_code:
            self.epsilon = parameters.get(self.EPSILON_PARAM, 1.35) or 1.35
            self.max_iter = parameters.get(self.MAX_ITER_PARAM, 100) or 100
            self.alpha = parameters.get(self.ALPHA_PARAM, 0.0001) or 0.0001
            self.tol = parameters.get(self.TOLERANCE_PARAM, 0.00001) or 0.00001
            self.tol = abs(float(self.tol))

            vals = [self.max_iter, self.alpha]
            atts = [self.MAX_ITER_PARAM, self.ALPHA_PARAM]
            for var, att in zip(vals, atts):
                if var <= 0:
                    raise ValueError(
                            _("Parameter '{}' must be x>0 for task {}").format(
                                    att, self.__class__))

            if self.epsilon <= 1.0:
                raise ValueError(
                        _("Parameter '{}' must be x>1.0 for task {}").format(
                                self.EPSILON_PARAM, self.__class__))

            self.has_import = \
                "from sklearn.linear_model import HuberRegressor\n"

    def generate_code(self):
        code = dedent("""
            {output} = HuberRegressor(epsilon={epsilon},
                max_iter={max_iter}, alpha={alpha},
                tol={tol})
            """).format(output=self.output,
                        epsilon=self.epsilon,
                        alpha=self.alpha,
                        max_iter=self.max_iter,
                        tol=self.tol)

        return code


class IsotonicRegressionOperation(RegressionOperation):
    """
        Only univariate (single feature) algorithm supported
    """
    ISOTONIC_PARAM = 'isotonic'
    FEATURES_PARAM = 'features'
    LABEL_PARAM = 'label'
    PREDICTION_PARAM = 'prediction'
    Y_MIN_PARAM = 'y_min'
    Y_MAX_PARAM = 'y_max'
    OUT_OF_BOUNDS_PARAM = 'out_of_bounds'


    def __init__(self, parameters, named_inputs, named_outputs):
        RegressionOperation.__init__(self, parameters, named_inputs,
                                     named_outputs)
        self.parameters = parameters
        self.name = 'regression.IsotonicRegression'
        self.has_code = any([len(self.named_inputs) == 1, self.contains_results()])
        self.output = self.named_outputs.get(
            'output data', 'output_data_{}'.format(self.order))

        self.model = self.named_outputs.get(
            'model', 'model_{}'.format(self.order))

        self.input_port = self.named_inputs.get(
            'train input data', 'input_data_{}'.format(self.order))

        if self.has_code:
            self.isotonic = parameters.get(
                self.ISOTONIC_PARAM, True) in (1, '1', 'true', True)
            self.features = parameters['features']
            self.label = parameters.get(self.LABEL_PARAM, None)
            self.prediction = parameters.get(self.PREDICTION_PARAM, 'prediction')
            self.y_min = parameters.get(self.Y_MIN_PARAM, None)
            self.y_max = parameters.get(self.Y_MIN_PARAM, None)
            self.out_of_bounds = parameters.get(self.OUT_OF_BOUNDS_PARAM, "nan")

            self.treatment()

            self.has_import = \
                """
                import numpy as np
                import pandas as pd
                from sklearn.isotonic import IsotonicRegression
                """

    @property
    def get_data_out_names(self, sep=','):
        return self.output

    def get_output_names(self, sep=', '):
        return sep.join([self.output, self.model])

    def treatment(self):
        if len(self.features) >= 2:
            raise ValueError(
                _("Parameter '{}' must be x<2 for task {}").format(
                    self.FEATURES_PARAM, self.__class__))

    def generate_code(self):
        code = dedent("""
        {output_data} = {input_data}.copy()        
        X_train = np.array({input_data}[{columns}].values.tolist()).flatten()    
        y = np.array({input_data}[{label}].values.tolist()).flatten()
        if {min} != None and {max} != None:
            {model} = IsotonicRegression(min=float({min}), max=float({max}), increasing={isotonic}, 
            out_of_bounds='{bounds}')
        elif {min} != None:
            {model} = IsotonicRegression(min={min}, increasing={isotonic}, out_of_bounds='{bounds}')
        elif {max} != None:
            {model} = IsotonicRegression(max=float({max}), increasing={isotonic}, out_of_bounds='{bounds}')
        else:
            {model} = IsotonicRegression(increasing={isotonic}, out_of_bounds='{bounds}')
        {model}.fit(X_train, y)          
        {output_data}['{prediction}'] = {model}.predict(X_train).tolist()
        """).format(output_data=self.output,
                    isotonic=self.isotonic,
                    output=self.output,
                    model=self.model,
                    input_data=self.input_port,
                    min=self.y_min,
                    max=self.y_max,
                    bounds=self.out_of_bounds,
                    columns=self.features,
                    label=self.label,
                    prediction=self.prediction)
        return code


class LinearRegressionOperation(RegressionOperation):

    ALPHA_PARAM = 'alpha'
    ELASTIC_NET_PARAM = 'l1_ratio'
    NORMALIZE_PARAM = 'normalize'
    MAX_ITER_PARAM = 'max_iter'
    TOLERANCE_PARAM = 'tol'
    SEED_PARAM = 'seed'
    FEATURES_PARAM = 'features'
    LABEL_PARAM = 'label'
    PREDICTION_PARAM = 'prediction'

    def __init__(self, parameters, named_inputs, named_outputs):
        RegressionOperation.__init__(self, parameters, named_inputs,
                                     named_outputs)

        self.name = 'regression.LinearRegression'
        self.has_code = any([len(self.named_inputs) == 1, self.contains_results()])

        self.output = self.named_outputs.get(
            'output data', 'output_data_{}'.format(self.order))

        self.model = self.named_outputs.get(
            'model', 'model_{}'.format(self.order))

        self.input_port = self.named_inputs.get(
            'train input data', 'input_data_{}'.format(self.order))

        if self.has_code:
            self.alpha = float(parameters.get(self.ALPHA_PARAM, 1.0) or 1.0)
            self.elastic = float(parameters.get(self.ELASTIC_NET_PARAM,
                                          0.5) or 0.5)
            self.normalize = self.parameters.get(self.NORMALIZE_PARAM,
                                                 True) in (1, '1', 'true', True)
            self.max_iter = int(parameters.get(self.MAX_ITER_PARAM, 1000) or 1000)
            self.tol = float(self.parameters.get(
                    self.TOLERANCE_PARAM, 0.0001) or 0.0001)
            self.tol = abs(float(self.tol))
            self.seed = self.parameters.get(self.SEED_PARAM, 'None') or 'None'
            self.features = parameters['features']
            self.label = parameters.get(self.LABEL_PARAM, None)
            self.prediction = self.parameters.get(self.PREDICTION_PARAM, 'prediction')

            vals = [self.alpha, self.max_iter]
            atts = [self.ALPHA_PARAM, self.MAX_ITER_PARAM]
            for var, att in zip(vals, atts):
                if var <= 0:
                    raise ValueError(
                            _("Parameter '{}' must be x>0 for task {}").format(
                                    att, self.__class__))

            if 0 > self.elastic > 1:
                raise ValueError(
                        _("Parameter '{}' must be 0<=x<=1 for task {}").format(
                                self.ELASTIC_NET_PARAM, self.__class__))

            self.has_import = \
                """
                from sklearn.linear_model import ElasticNet
                import numpy as np
                """

    @property
    def get_data_out_names(self, sep=','):
        return self.output

    def get_output_names(self, sep=', '):
        return sep.join([self.output, self.model])

    def generate_code(self):
        code = dedent("""
        {output_data} = {input_data}.copy()
        X_train = {input_data}[{columns}].values.tolist()
        y = {input_data}[{label}].values.tolist()
        {model} = ElasticNet(alpha={alpha}, l1_ratio={elastic}, tol={tol}, max_iter={max_iter}, random_state={seed},
                             normalize={normalize})  
        {model}.fit(X_train, y)
        {output_data}['{prediction}'] = {model}.predict(X_train).tolist()
        """.format(output_data=self.output,
                   max_iter=self.max_iter,
                   alpha=self.alpha,
                   elastic=self.elastic,
                   seed=self.seed,
                   tol=self.tol,
                   normalize=self.normalize,
                   input_data=self.input_port,
                   prediction=self.prediction,
                   columns=self.features,
                   label=self.label,
                   model=self.model,
                   output=self.output))

        return code


class MLPRegressorOperation(Operation):

    HIDDEN_LAYER_SIZES_PARAM = 'hidden_layer_sizes'
    ACTIVATION_PARAM = 'activation'
    SOLVER_PARAM = 'solver'
    ALPHA_PARAM = 'alpha'
    MAX_ITER_PARAM = 'max_iter'
    TOLERANCE_PARAM = 'tol'
    SEED_PARAM = 'seed'

    SOLVER_PARAM_ADAM = 'adam'
    SOLVER_PARAM_LBFGS = 'lbfgs'
    SOLVER_PARAM_SGD = 'sgd'

    ACTIVATION_PARAM_ID = 'identity'
    ACTIVATION_PARAM_LOG = 'logistic'
    ACTIVATION_PARAM_TANH = 'tanh'
    ACTIVATION_PARAM_RELU = 'relu'

    def __init__(self, parameters,  named_inputs, named_outputs):
        Operation.__init__(self, parameters,  named_inputs,  named_outputs)

        self.has_code = len(named_outputs) > 0
        if self.has_code:
            self.output = \
                named_outputs.get('algorithm',
                                  'algorithm_tmp_{}'.format(self.order))

            self.hidden_layers = parameters.get(self.HIDDEN_LAYER_SIZES_PARAM,
                                                '(1,100,1)') or '(1,100,1)'
            self.hidden_layers = \
                self.hidden_layers.replace("(", "").replace(")", "")
            if not bool(re.match('(\d+,)+\d*', self.hidden_layers)):
                raise ValueError(
                        _("Parameter '{}' must be a tuple with the size "
                          "of each layer for task {}").format(
                                self.HIDDEN_LAYER_SIZES_PARAM, self.__class__))

            self.activation = parameters.get(
                    self.ACTIVATION_PARAM,
                    self.ACTIVATION_PARAM_RELU) or self.ACTIVATION_PARAM_RELU

            self.solver = parameters.get(
                    self.SOLVER_PARAM,
                    self.SOLVER_PARAM_ADAM) or self.SOLVER_PARAM_LINEAR

            self.alpha = parameters.get(self.ALPHA_PARAM, 0.0001) or 0.0001
            self.alpha = abs(float(self.alpha))

            self.max_iter = parameters.get(self.MAX_ITER_PARAM, 200) or 200
            self.max_iter = abs(int(self.max_iter))

            self.tol = parameters.get(self.TOLERANCE_PARAM, 0.0001) or 0.0001
            self.tol = abs(float(self.tol))

            self.seed = parameters.get(self.SEED_PARAM, 'None') or 'None'

            self.has_import = \
                "from sklearn.neural_network import MLPRegressor\n"

    def generate_code(self):
        """Generate code."""
        code = """
            {output} = MLPRegressor(hidden_layer_sizes=({hidden_layers}), 
            activation='{activation}', solver='{solver}', alpha={alpha}, 
            max_iter={max_iter}, random_state={seed}, tol={tol})
            """.format(tol=self.tol,
                       alpha=self.alpha,
                       activation=self.activation,
                       hidden_layers=self.hidden_layers,
                       max_iter=self.max_iter,
                       seed=self.seed,
                       solver=self.solver,
                       output=self.output)
        return dedent(code)


class RandomForestRegressorOperation(RegressionOperation):

    """
    A random forest is a meta estimator that fits a number of classifying
    decision trees on various sub-samples of the dataset and use averaging
    to improve the predictive accuracy and control over-fitting.
    """

    N_ESTIMATORS_PARAM = 'n_estimators'
    MAX_FEATURES_PARAM = 'max_features'
    MAX_DEPTH_PARAM = 'max_depth'
    MIN_SPLIT_PARAM = 'min_samples_split'
    MIN_LEAF_PARAM = 'min_samples_leaf'
    SEED_PARAM = 'seed'

    def __init__(self, parameters, named_inputs, named_outputs):
        RegressionOperation.__init__(self, parameters, named_inputs,
                                     named_outputs)
        self.parameters = parameters
        self.name = 'regression.RandomForestRegressor'
        self.has_code = len(self.named_outputs) > 0

        if self.has_code:
            self.n_estimators = parameters.get(
                    self.N_ESTIMATORS_PARAM, 10) or 10
            self.max_features = parameters.get(
                    self.MAX_FEATURES_PARAM, 'auto') or 'auto'
            self.max_depth = parameters.get(self.MAX_DEPTH_PARAM, 3) or 3
            self.min_samples_split = parameters.get(
                    self.MIN_SPLIT_PARAM, 2) or 2
            self.min_samples_leaf = parameters.get(
                    self.MIN_LEAF_PARAM, 1) or 1
            self.seed = parameters.get(self.SEED_PARAM, 'None')

            vals = [self.max_depth, self.n_estimators, self.min_samples_split,
                    self.min_samples_leaf]
            atts = [self.MAX_DEPTH_PARAM, self.N_ESTIMATORS_PARAM,
                    self.MIN_SPLIT_PARAM, self.MIN_LEAF_PARAM]
            for var, att in zip(vals, atts):
                if var <= 0:
                    raise ValueError(
                            _("Parameter '{}' must be x>0 for task {}").format(
                                    att, self.__class__))

            self.has_import = \
                "from sklearn.ensemble import RandomForestRegressor\n"

    def generate_code(self):
        code = dedent("""
            {output} = RandomForestRegressor(n_estimators={n_estimators},
                max_features='{max_features}',
                max_depth={max_depth},
                min_samples_split={min_samples_split},
                min_samples_leaf={min_samples_leaf},
                random_state={seed})
            """).format(output=self.output,
                        n_estimators=self.n_estimators,
                        max_features=self.max_features,
                        max_depth=self.max_depth,
                        min_samples_split=self.min_samples_split,
                        min_samples_leaf=self.min_samples_leaf,
                        seed=self.seed)

        return code


class SGDRegressorOperation(RegressionOperation):

    """
    Linear model fitted by minimizing a regularized empirical loss with
    Stochastic Gradient Descent.
    """

    ALPHA_PARAM = 'alpha'
    ELASTIC_PARAM = 'l1_ratio'
    MAX_ITER_PARAM = 'max_iter'
    TOLERANCE_PARAM = 'tol'
    SEED_PARAM = 'seed'

    def __init__(self, parameters, named_inputs, named_outputs):
        RegressionOperation.__init__(self, parameters, named_inputs,
                                     named_outputs)
        self.parameters = parameters
        self.name = 'regression.SGDRegressor'
        self.has_code = len(self.named_outputs) > 0

        if self.has_code:
            self.alpha = parameters.get(
                    self.ALPHA_PARAM, 0.0001) or 0.0001
            self.l1_ratio = parameters.get(
                    self.ELASTIC_PARAM, 0.15) or 0.15
            self.max_iter = parameters.get(self.MAX_ITER_PARAM, 1000) or 1000
            self.tol = parameters.get(
                    self.TOLERANCE_PARAM, 0.001) or 0.001
            self.tol = abs(float(self.tol))
            self.seed = parameters.get(self.SEED_PARAM, 'None')

            vals = [self.alpha, self.max_iter]
            atts = [self.ALPHA_PARAM, self.MAX_ITER_PARAM]
            for var, att in zip(vals, atts):
                if var <= 0:
                    raise ValueError(
                            _("Parameter '{}' must be x>0 for task {}").format(
                                    att, self.__class__))

            if 0 > self.l1_ratio > 1:
                raise ValueError(
                        _("Parameter '{}' must be 0<=x<=1 for task {}").format(
                                self.ELASTIC_PARAM, self.__class__))

            self.has_import = \
                "from sklearn.linear_model import SGDRegressor\n"

    def generate_code(self):
        code = dedent("""
            {output} = SGDRegressor(alpha={alpha},
                l1_ratio={l1_ratio}, max_iter={max_iter},
                tol={tol}, random_state={seed})
            """).format(output=self.output,
                        alpha=self.alpha,
                        l1_ratio=self.l1_ratio,
                        max_iter=self.max_iter,
                        tol=self.tol,
                        seed=self.seed)

        return code




class GeneralizedLinearRegressionOperation(RegressionOperation):

    FIT_INTERCEPT_ATTRIBUTE_PARAM = 'fit_intercept'
    NORMALIZE_ATTRIBUTE_PARAM = 'normalize'
    COPY_X_ATTRIBUTE_PARAM = 'copy_X'
    N_JOBS_ATTRIBUTE_PARAM = 'n_jobs'
    LABEL_ATTRIBUTE_PARAM = 'labels'
    FEATURES_ATTRIBUTE_PARAM = 'features_atr'
    ALIAS_ATTRIBUTE_PARAM = 'alias'

    def __init__(self, parameters, named_inputs, named_outputs):
        RegressionOperation.__init__(self, parameters,  named_inputs,  named_outputs)

        self.name = 'regression.GeneralizedLinearRegression'
        self.has_code = any([len(self.named_inputs) == 1, self.contains_results()])
        self.output = self.named_outputs.get(
            'output data', 'output_data_{}'.format(self.order))

        self.model = self.named_outputs.get(
            'model', 'model_{}'.format(self.order))

        self.input_port = self.named_inputs.get(
            'input data', 'input_data_{}'.format(self.order))

        self.fit_intercept = int(parameters.get(self.FIT_INTERCEPT_ATTRIBUTE_PARAM, 1))
        self.normalize = int(parameters.get(self.NORMALIZE_ATTRIBUTE_PARAM, 0))
        self.copy_X = int(parameters.get(self.COPY_X_ATTRIBUTE_PARAM, 1))
        self.n_jobs = int(parameters.get(self.N_JOBS_ATTRIBUTE_PARAM, 0))
        self.features_atr = parameters['features_atr']
        #import pdb
        #pdb.set_trace()
        self.label = parameters.get(self.LABEL_ATTRIBUTE_PARAM, None)
        self.alias = self.parameters.get(self.ALIAS_ATTRIBUTE_PARAM, 'prediction')
        if not all([self.LABEL_ATTRIBUTE_PARAM in parameters]):
            msg = _("Parameters '{}' must be informed for task {}")
            raise ValueError(msg.format(
                self.LABEL_ATTRIBUTE_PARAM,
                self.__class__.__name__))

        self.input_treatment()

        self.has_import = \
            """
            import numpy as np
            from sklearn import datasets, linear_model
            """

    @property
    def get_data_out_names(self, sep=','):
        return self.output

    def get_output_names(self, sep=', '):
        return sep.join([self.output, self.model])

    def input_treatment(self):
        if self.n_jobs < -1:
            raise ValueError(
                _("Parameter '{}' must be x>=-1 for task {}").format(
                    self.N_JOBS_ATTRIBUTE_PARAM, self.__class__))

        self.n_jobs = 1 if int(self.n_jobs) == 0 else int(self.n_jobs)

        self.fit_intercept = True if int(self.fit_intercept) == 1 else False

        self.normalize = True if int(self.normalize) == 1 else False

        self.copy_X = True if int(self.copy_X) == 1 else False


    def generate_code(self):
        """Generate code."""
        code = """
            {output_data} = {input_data}.copy()            
            X_train = {input_data}[{columns}].values.tolist()
            y = {input_data}[{label}].values.tolist()
            {model} = linear_model.LinearRegression(fit_intercept={fit_intercept}, normalize={normalize}, 
                                                    copy_X={copy_X}, n_jobs={n_jobs})
            {model}.fit(X_train, y)          
            {output_data}['{prediction}'] = {model}.predict(X_train).tolist()
            """.format(output=self.output,
                       fit_intercept=self.fit_intercept,
                       normalize=self.normalize,
                       copy_X=self.copy_X,
                       n_jobs=self.n_jobs,
                       model=self.model,
                       input_data=self.input_port,
                       label=self.label,
                       output_data=self.output,
                       prediction=self.alias,
                       columns=self.features_atr)

        return dedent(code)