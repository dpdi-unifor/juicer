from textwrap import dedent

from juicer.operation import Operation
from itertools import izip_longest
import re


class RegressionModelOperation(Operation):
    FEATURES_ATTRIBUTE_PARAM = 'features'
    LABEL_ATTRIBUTE_PARAM = 'label'
    PREDICTION_ATTRIBUTE_PARAM = 'prediction'

    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)

        self.has_code = len(named_inputs) == 2 and \
                        any([len(named_outputs) > 0, self.contains_results()])

        if self.has_code:
            if any([self.FEATURES_ATTRIBUTE_PARAM not in parameters,
                    self.LABEL_ATTRIBUTE_PARAM not in parameters]):
                msg = _("Parameters '{}' and '{}' must be informed for task {}")
                raise ValueError(msg.format(
                    self.FEATURES_ATTRIBUTE_PARAM, self.LABEL_ATTRIBUTE_PARAM,
                    self.__class__.__name__))

            self.label = parameters.get(self.LABEL_ATTRIBUTE_PARAM)[0]
            self.features = parameters.get(self.FEATURES_ATTRIBUTE_PARAM)[0]
            self.prediction = parameters.get(self.PREDICTION_ATTRIBUTE_PARAM,
                                             'prediction')

            self.algorithm = named_inputs.get('algorithm',
                                              'model_task_{}'.format(
                                                      self.order))
            self.perform_transformation = any(['output data' in named_outputs,
                                               self.contains_results()])
            self.output = self.named_outputs.get('output data',
                                                 'out_task_{}'.format(
                                                         self.order))
            self.model = self.named_outputs.get('model',
                                                'model_task_{}'.format(
                                                        self.order))

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
            {output_data} = {input}.copy()
            X_train = {input}['{features}'].values.tolist()
            if 'IsotonicRegression' in str(algorithm):
                X_train = np.ravel(X_train)
            y = {input}['{label}'].values.tolist()
            {model} = {algorithm}.fit(X_train, y)
            """.format(model=self.model, algorithm=self.algorithm,
                       input=self.named_inputs['train input data'],
                       output_data=self.output, prediction=self.prediction,
                       label=self.label, features=self.features)

            if self.perform_transformation:
                code += """
            {OUT} = {IN}.copy()
            {OUT}['{predCol}'] = {model}.predict(X_train).tolist()
            """.format(predCol=self.prediction, OUT=self.output,
                       model=self.model,
                       IN=self.named_inputs['train input data'])
            else:
                code += """
            {output} = None
            """.format(output=self.output)

            return dedent(code)


class AlgorithmOperation(Operation):
    def __init__(self, parameters, named_inputs, named_outputs, algorithm):
        super(AlgorithmOperation, self).__init__(
            parameters, named_inputs, named_outputs)

        model_in_ports = {'algorithm': 'algorithm'}
        if 'train input data' in named_inputs:
            model_in_ports['train input data'] = \
                named_inputs.get('train input data')

        self.algorithm = algorithm
        self.regression_model = RegressionModelOperation(
            parameters, model_in_ports, named_outputs)

        self.has_code = len(model_in_ports) == 2 and \
                        any([len(named_outputs) > 0, self.contains_results()])

    def generate_code(self):
        algorithm_code = self.algorithm.generate_code()
        model_code = self.regression_model.generate_code()
        return "\n".join([algorithm_code, model_code])

    def get_output_names(self, sep=','):
        output = self.named_outputs.get('output data',
                                        'out_task_{}'.format(self.order))
        models = self.named_outputs.get('model',
                                        'model_task_{}'.format(self.order))
        return sep.join([output, models])


class GradientBoostingRegressorOperation(Operation):
    LEARNING_RATE_PARAM = 'learning_rate'
    N_ESTIMATORS_PARAM = 'n_estimators'
    MAX_DEPTH_PARAM = 'max_depth'
    MIN_SPLIT_PARAM = 'min_samples_split'
    MIN_LEAF_PARAM = 'min_samples_leaf'
    SEED_PARAM = 'seed'

    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters, named_inputs,
                                     named_outputs)

        self.name = 'regression.GradientBoostingRegressor'
        self.has_code = len(named_outputs) > 0
        self.output = \
            named_outputs.get('algorithm',
                              'algorithm_tmp_{}'.format(self.order))

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


class HuberRegressorOperation(Operation):

    """
    Linear regression model that is robust to outliers.
    """

    EPSILON_PARAM = 'epsilon'
    MAX_ITER_PARAM = 'max_iter'
    ALPHA_PARAM = 'alpha'
    TOLERANCE_PARAM = 'tol'

    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters, named_inputs,
                                     named_outputs)
        self.parameters = parameters
        self.name = 'regression.HuberRegressor'
        self.has_code = len(self.named_outputs) > 0
        self.output = \
            named_outputs.get('algorithm',
                              'algorithm_tmp_{}'.format(self.order))

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


class IsotonicRegressionOperation(Operation):
    """
        Only univariate (single feature) algorithm supported
    """
    ISOTONIC_PARAM = 'isotonic'

    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters, named_inputs,
                                     named_outputs)
        self.parameters = parameters
        self.name = 'regression.IsotonicRegression'
        self.has_code = len(self.named_outputs) > 0
        self.output = \
            named_outputs.get('algorithm',
                              'algorithm_tmp_{}'.format(self.order))

        if self.has_code:
            self.isotonic = parameters.get(
                self.ISOTONIC_PARAM, True) in (1, '1', 'true', True)
            self.has_import = \
                "from sklearn.isotonic import IsotonicRegression\n"

    def generate_code(self):
        code = dedent("""
        {output} = IsotonicRegression(increasing={isotonic})
        """).format(output=self.output, isotonic=self.isotonic)
        return code


class LinearRegressionOperation(Operation):

    ALPHA_PARAM = 'alpha'
    ELASTIC_NET_PARAM = 'l1_ratio'
    NORMALIZE_PARAM = 'normalize'
    MAX_ITER_PARAM = 'max_iter'
    TOLERANCE_PARAM = 'tol'
    SEED_PARAM = 'seed'

    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters, named_inputs,
                                     named_outputs)

        self.name = 'regression.LinearRegression'
        self.has_code = len(named_outputs) > 0
        self.output = \
            named_outputs.get('algorithm',
                              'algorithm_tmp_{}'.format(self.order))

        if self.has_code:
            self.alpha = parameters.get(self.ALPHA_PARAM, 1.0) or 1.0
            self.elastic = parameters.get(self.ELASTIC_NET_PARAM,
                                          0.5) or 0.5
            self.normalize = self.parameters.get(self.NORMALIZE_PARAM,
                                                 True) in (1, '1', 'true', True)
            self.max_iter = parameters.get(self.MAX_ITER_PARAM, 1000) or 1000
            self.tol = self.parameters.get(
                    self.TOLERANCE_PARAM, 0.0001) or 0.0001
            self.tol = abs(float(self.tol))
            self.seed = self.parameters.get(self.SEED_PARAM, 'None') or 'None'

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
                "from sklearn.linear_model import ElasticNet\n"

    def generate_code(self):
        code = dedent("""
        {output} = ElasticNet(alpha={alpha}, l1_ratio={elastic}, tol={tol},
                              max_iter={max_iter}, random_state={seed},
                              normalize={normalize})""".format(
                output=self.output, max_iter=self.max_iter,
                alpha=self.alpha, elastic=self.elastic,
                seed=self.seed, tol=self.tol, normalize=self.normalize))
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


class RandomForestRegressorOperation(Operation):

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
        Operation.__init__(self, parameters, named_inputs,
                                     named_outputs)
        self.parameters = parameters
        self.name = 'regression.RandomForestRegressor'
        self.has_code = len(self.named_outputs) > 0
        self.output = \
            named_outputs.get('algorithm',
                              'algorithm_tmp_{}'.format(self.order))

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


class SGDRegressorOperation(Operation):

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
        Operation.__init__(self, parameters, named_inputs,
                                     named_outputs)
        self.parameters = parameters
        self.name = 'regression.SGDRegressor'
        self.has_code = len(self.named_outputs) > 0
        self.output = \
            named_outputs.get('algorithm',
                              'algorithm_tmp_{}'.format(self.order))

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


class GradientBoostingRegressorModelOperation(AlgorithmOperation):
    def __init__(self, parameters, named_inputs, named_outputs):
        algorithm = GradientBoostingRegressorOperation(
            parameters, named_inputs, {'algorithm': 'algorithm'})
        super(GradientBoostingRegressorModelOperation, self).__init__(
            parameters, named_inputs, named_outputs, algorithm)
        self.has_import = algorithm.has_import


class HuberRegressorModelOperation(AlgorithmOperation):
    def __init__(self, parameters, named_inputs, named_outputs):
        algorithm = HuberRegressorOperation(
            parameters, named_inputs, {'algorithm': 'algorithm'})
        super(HuberRegressorModelOperation, self).__init__(
            parameters, named_inputs, named_outputs, algorithm)
        self.has_import = algorithm.has_import


class IsotonicRegressionModelOperation(AlgorithmOperation):
    def __init__(self, parameters, named_inputs, named_outputs):
        algorithm = IsotonicRegressionOperation(
            parameters, named_inputs, {'algorithm': 'algorithm'})
        super(IsotonicRegressionModelOperation, self).__init__(
            parameters, named_inputs, named_outputs, algorithm)
        self.has_import = algorithm.has_import


class LinearRegressionModelOperation(AlgorithmOperation):
    def __init__(self, parameters, named_inputs, named_outputs):
        algorithm = LinearRegressionOperation(
            parameters, named_inputs, {'algorithm': 'algorithm'})
        super(LinearRegressionModelOperation, self).__init__(
            parameters, named_inputs, named_outputs, algorithm)
        self.has_import = algorithm.has_import


class MLPRegressorModelOperation(AlgorithmOperation):
    def __init__(self, parameters, named_inputs, named_outputs):
        algorithm = MLPRegressorOperation(
            parameters, named_inputs, {'algorithm': 'algorithm'})
        super(MLPRegressorModelOperation, self).__init__(
            parameters, named_inputs, named_outputs, algorithm)
        self.has_import = algorithm.has_import


class RandomForestRegressorModelOperation(AlgorithmOperation):
    def __init__(self, parameters, named_inputs, named_outputs):
        algorithm = RandomForestRegressorOperation(
            parameters, named_inputs, {'algorithm': 'algorithm'})
        super(RandomForestRegressorModelOperation, self).__init__(
            parameters, named_inputs, named_outputs, algorithm)
        self.has_import = algorithm.has_import


class SGDRegressorModelOperation(AlgorithmOperation):
    def __init__(self, parameters, named_inputs, named_outputs):
        algorithm = SGDRegressorOperation(
            parameters, named_inputs, {'algorithm': 'algorithm'})
        super(SGDRegressorModelOperation, self).__init__(
            parameters, named_inputs, named_outputs, algorithm)
        self.has_import = algorithm.has_import