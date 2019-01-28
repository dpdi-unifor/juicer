from textwrap import dedent
from juicer.operation import Operation
import re


class ClassificationModelOperation(Operation):

    LABEL_ATTRIBUTE_PARAM = 'label'
    FEATURES_ATTRIBUTE_PARAM = 'features'
    PREDICTION_ATTRIBUTE_PARAM = 'prediction'

    def __init__(self, parameters,  named_inputs, named_outputs):
        Operation.__init__(self, parameters,  named_inputs,  named_outputs)

        self.has_code = len(self.named_inputs) == 2

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
        return ''

    def get_output_names(self, sep=', '):
        return sep.join([self.output, self.model])

    def generate_code(self):
        """Generate code."""
        code = """
            X = {input}['{features}'].values.tolist()
            y = {input}['{label}'].values.tolist()
            {model} = {algorithm}.fit(X, y)
            """.format(model=self.model, label=self.label,
                       input=self.named_inputs['train input data'],
                       features=self.features,
                       algorithm=self.algorithm)

        if self.perform_transformation:
            code += """
            {OUT} = {IN}.copy()
            {OUT}['{predCol}'] = {model}.predict(X).tolist()
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
        self.classification_model = ClassificationModelOperation(
            parameters, model_in_ports, named_outputs)

        self.has_code = len(model_in_ports) == 2 and \
                        any([len(named_outputs) > 0, self.contains_results()])

    def generate_code(self):
        algorithm_code = self.algorithm.generate_code()
        model_code = self.classification_model.generate_code()
        return "\n".join([algorithm_code, model_code])

    def get_output_names(self, sep=','):
        output = self.named_outputs.get('output data',
                                        'out_task_{}'.format(self.order))
        models = self.named_outputs.get('model',
                                        'model_task_{}'.format(self.order))
        return sep.join([output, models])


class DecisionTreeModelOperation(AlgorithmOperation):
    def __init__(self, parameters, named_inputs, named_outputs):
        algorithm = DecisionTreeClassifierOperation(
            parameters, named_inputs, {'algorithm': 'algorithm'})
        super(DecisionTreeModelOperation, self).__init__(
            parameters, named_inputs, named_outputs, algorithm)
        self.has_import = algorithm.has_import


class GBTModelOperation(AlgorithmOperation):
    def __init__(self, parameters, named_inputs, named_outputs):
        algorithm = GBTClassifierOperation(
            parameters, named_inputs, {'algorithm': 'algorithm'})
        super(GBTModelOperation, self).__init__(
            parameters, named_inputs, named_outputs, algorithm)
        self.has_import = algorithm.has_import


class KNNModelOperation(AlgorithmOperation):
    def __init__(self, parameters, named_inputs, named_outputs):
        algorithm = GBTClassifierOperation(
            parameters, named_inputs, {'algorithm': 'algorithm'})
        super(KNNModelOperation, self).__init__(
            parameters, named_inputs, named_outputs, algorithm)
        self.has_import = algorithm.has_import


class LogisticRegressionModelOperation(AlgorithmOperation):
    def __init__(self, parameters, named_inputs, named_outputs):
        algorithm = LogisticRegressionOperation(
            parameters, named_inputs, {'algorithm': 'algorithm'})
        super(LogisticRegressionModelOperation, self).__init__(
            parameters, named_inputs, named_outputs, algorithm)
        self.has_import = algorithm.has_import


class MLPClassifierModelOperation(AlgorithmOperation):
    def __init__(self, parameters, named_inputs, named_outputs):
        algorithm = MLPClassifierOperation(
            parameters, named_inputs, {'algorithm': 'algorithm'})
        super(MLPClassifierModelOperation, self).__init__(
            parameters, named_inputs, named_outputs, algorithm)
        self.has_import = algorithm.has_import


class NaiveBayesModelOperation(AlgorithmOperation):
    def __init__(self, parameters, named_inputs, named_outputs):
        algorithm = NaiveBayesClassifierOperation(
            parameters, named_inputs, {'algorithm': 'algorithm'})
        super(NaiveBayesModelOperation, self).__init__(
            parameters, named_inputs, named_outputs, algorithm)
        self.has_import = algorithm.has_import


class PerceptronModelOperation(AlgorithmOperation):
    def __init__(self, parameters, named_inputs, named_outputs):
        algorithm = PerceptronClassifierOperation(
            parameters, named_inputs, {'algorithm': 'algorithm'})
        super(PerceptronModelOperation, self).__init__(
            parameters, named_inputs, named_outputs, algorithm)
        self.has_import = algorithm.has_import


class RandomForestModelOperation(AlgorithmOperation):
    def __init__(self, parameters, named_inputs, named_outputs):
        algorithm = RandomForestClassifierOperation(
            parameters, named_inputs, {'algorithm': 'algorithm'})
        super(RandomForestModelOperation, self).__init__(
            parameters, named_inputs, named_outputs, algorithm)
        self.has_import = algorithm.has_import


class SvmModelOperation(AlgorithmOperation):
    def __init__(self, parameters, named_inputs, named_outputs):
        algorithm = SvmClassifierOperation(
            parameters, named_inputs, {'algorithm': 'algorithm'})
        super(SvmModelOperation, self).__init__(
            parameters, named_inputs, named_outputs, algorithm)
        self.has_import = algorithm.has_import


class DecisionTreeClassifierOperation(Operation):

    MAX_DEPTH_PARAM = 'max_depth'
    MIN_SPLIT_PARAM = 'min_samples_split'
    MIN_LEAF_PARAM = 'min_samples_leaf'
    MIN_WEIGHT_PARAM = 'min_weight'
    SEED_PARAM = 'seed'

    def __init__(self, parameters,  named_inputs, named_outputs):
        Operation.__init__(self, parameters,  named_inputs,  named_outputs)

        self.has_code = len(self.named_outputs) > 0
        if self.has_code:
            self.output = \
                named_outputs.get('algorithm',
                                  'algorithm_tmp_{}'.format(self.order))

            self.min_split = parameters.get(self.MIN_SPLIT_PARAM, 2) or 2
            self.min_leaf = parameters.get(self.MIN_LEAF_PARAM, 1) or 1
            self.max_depth = parameters.get(self.MAX_DEPTH_PARAM,
                                            'None') or 'None'
            self.min_weight = parameters.get(self.MIN_WEIGHT_PARAM, 0.0) or 0.0
            self.seed = parameters.get(self.SEED_PARAM, 'None') or 'None'

            vals = [self.min_split, self.min_leaf]
            atts = [self.MIN_SPLIT_PARAM, self.MIN_LEAF_PARAM]
            for var, att in zip(vals, atts):
                if var <= 0:
                    raise ValueError(
                            _("Parameter '{}' must be x>0 for task {}").format(
                                    att, self.__class__))

            if self.min_weight < 0:
                raise ValueError(
                        _("Parameter '{}' must be x>=0 for task {}").format(
                                self.MIN_WEIGHT_PARAM, self.__class__))

            if self.max_depth is not 'None' and self.max_depth <= 0:
                raise ValueError(
                        _("Parameter '{}' must be x>0 for task {}").format(
                                self.MAX_DEPTH_PARAM, self.__class__))

            self.has_import = \
                "from sklearn.tree import DecisionTreeClassifier\n"

    def generate_code(self):
        """Generate code."""
        code = """
        {algorithm} = DecisionTreeClassifier(max_depth={max_depth}, 
        min_samples_split={min_split}, min_samples_leaf={min_leaf}, 
        min_weight_fraction_leaf={min_weight}, random_state={seed})
        """.format(algorithm=self.output,
                   min_split=self.min_split,
                   min_leaf=self.min_leaf, min_weight=self.min_weight,
                   seed=self.seed, max_depth=self.max_depth)
        return dedent(code)


class GBTClassifierOperation(Operation):

    LEARNING_RATE_PARAM = 'learning_rate'
    N_ESTIMATORS_PARAM = 'n_estimators'
    MAX_DEPTH_PARAM = 'max_depth'
    MIN_SPLIT_PARAM = 'min_samples_split'
    MIN_LEAF_PARAM = 'min_samples_leaf'
    LOSS_PARAM = 'loss'
    SEED_PARAM = 'seed'

    LOSS_PARAM_DEV = 'deviance'
    LOSS_PARAM_EXP = 'exponencial'

    def __init__(self, parameters,  named_inputs, named_outputs):
        Operation.__init__(self, parameters,  named_inputs,  named_outputs)

        self.has_code = len(self.named_outputs) > 0
        if self.has_code:
            self.output = \
                named_outputs.get('algorithm',
                                  'algorithm_tmp_{}'.format(self.order))

            self.max_depth = parameters.get(
                    self.MAX_DEPTH_PARAM, 3) or 3
            self.min_split = parameters.get(self.MIN_SPLIT_PARAM, 2) or 2
            self.min_leaf = parameters.get(self.MIN_LEAF_PARAM, 1) or 1
            self.n_estimators = parameters.get(self.N_ESTIMATORS_PARAM,
                                               100) or 100
            self.learning_rate = parameters.get(self.LEARNING_RATE_PARAM,
                                                0.1) or 0.1
            self.loss = \
                parameters.get(self.LOSS_PARAM, self.LOSS_PARAM_DEV) or \
                self.LOSS_PARAM_DEV
            self.seed = parameters.get(self.SEED_PARAM, 'None') or 'None'

            vals = [self.min_split, self.min_leaf, self.learning_rate,
                    self.n_estimators]
            atts = [self.MIN_SPLIT_PARAM, self.MIN_LEAF_PARAM,
                    self.LEARNING_RATE_PARAM, self.N_ESTIMATORS_PARAM]
            for var, att in zip(vals, atts):
                if var <= 0:
                    raise ValueError(
                            _("Parameter '{}' must be x>0 for task {}").format(
                                    att, self.__class__))

            self.has_import = \
                "from sklearn.ensemble import GradientBoostingClassifier\n"

    def generate_code(self):
        """Generate code."""
        code = """ 
        {algorithm} = GradientBoostingClassifier(loss='{loss}',
        learning_rate={learning_rate}, n_estimators={n_estimators},
        min_samples_split={min_split}, max_depth={max_depth},
        min_samples_leaf={min_leaf}, random_state={seed})
        """.format(algorithm=self.output, loss=self.loss,
                   n_estimators=self.n_estimators, min_leaf=self.min_leaf,
                   min_split=self.min_split, learning_rate=self.learning_rate,
                   max_depth=self.max_depth, seed=self.seed)
        return dedent(code)


class KNNClassifierOperation(Operation):

    K_PARAM = 'k'

    def __init__(self, parameters,  named_inputs, named_outputs):
        Operation.__init__(self, parameters,  named_inputs,  named_outputs)

        self.has_code = len(named_outputs) > 0

        if self.has_code:
            if self.K_PARAM not in parameters:
                raise ValueError(
                    _("Parameter '{}' must be informed for task {}")
                    .format(self.K_PARAM, self.__class__))

            self.k = self.parameters.get(self.K_PARAM, 1) or 1
            self.output = \
                named_outputs.get('algorithm',
                                  'algorithm_tmp_{}'.format(self.order))
            if self.k <= 0:
                raise ValueError(
                        _("Parameter '{}' must be x>0 for task {}").format(
                                self.K_PARAM, self.__class__))

            self.has_import = \
                "from sklearn.neighbors import KNeighborsClassifier\n"

    def generate_code(self):
        """Generate code."""
        code = """
        {algorithm} = KNeighborsClassifier(n_neighbors={K})
        """.format(K=self.k, algorithm=self.output)

        return dedent(code)


class LogisticRegressionOperation(Operation):

    TOLERANCE_PARAM = 'tol'
    REGULARIZATION_PARAM = 'regularization'
    MAX_ITER_PARAM = 'max_iter'
    SEED_PARAM = 'seed'
    SOLVER_PARAM = 'solver'

    SOLVER_PARAM_NEWTON = 'newton-cg'
    SOLVER_PARAM_LBFGS = 'lbfgs'
    SOLVER_PARAM_LINEAR = 'liblinear'
    SOLVER_PARAM_SAG = 'sag'
    SOLVER_PARAM_SAGa = 'saga'

    def __init__(self, parameters,  named_inputs, named_outputs):
        Operation.__init__(self, parameters,  named_inputs,  named_outputs)

        self.has_code = len(named_outputs) > 0
        if self.has_code:
            self.output = \
                named_outputs.get('algorithm',
                                  'algorithm_tmp_{}'.format(self.order))

            self.tol = self.parameters.get(self.TOLERANCE_PARAM,
                                           0.0001) or 0.0001
            self.tol = abs(float(self.tol))
            self.regularization = self.parameters.get(self.REGULARIZATION_PARAM,
                                                      1.0) or 1.0
            self.max_iter = self.parameters.get(self.MAX_ITER_PARAM,
                                                100) or 100
            self.seed = self.parameters.get(self.SEED_PARAM, 'None') or 'None'
            self.solver = self.parameters.get(
                    self.SOLVER_PARAM, self.SOLVER_PARAM_LINEAR)\
                or self.SOLVER_PARAM_LINEAR

            vals = [self.regularization, self.max_iter]
            atts = [self.REGULARIZATION_PARAM, self.MAX_ITER_PARAM]
            for var, att in zip(vals, atts):
                if var <= 0:
                    raise ValueError(
                            _("Parameter '{}' must be x>0 for task {}").format(
                                    att, self.__class__))

            self.has_import = \
                "from sklearn.linear_model import LogisticRegression\n"

    def generate_code(self):
        """Generate code."""
        code = """
            {algorithm} = LogisticRegression(tol={tol}, C={C}, 
                max_iter={max_iter}, solver='{solver}', random_state={seed})
            """.format(tol=self.tol,
                       C=self.regularization,
                       max_iter=self.max_iter,
                       seed=self.seed,
                       solver=self.solver,
                       algorithm=self.output)
        return dedent(code)


class MLPClassifierOperation(Operation):

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
                "from sklearn.neural_network import MLPClassifier\n"

    def generate_code(self):
        """Generate code."""
        code = """
            {output} = MLPClassifier(hidden_layer_sizes=({hidden_layers}), 
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


class NaiveBayesClassifierOperation(Operation):

    ALPHA_PARAM = 'alpha'
    CLASS_PRIOR_PARAM = 'class_prior'
    MODEL_TYPE_PARAM = 'type'

    MODEL_TYPE_PARAM_B = 'Bernoulli'
    MODEL_TYPE_PARAM_G = 'GaussianNB'
    MODEL_TYPE_PARAM_M = 'Multinomial'

    def __init__(self, parameters,  named_inputs, named_outputs):
        Operation.__init__(self, parameters,  named_inputs,  named_outputs)

        self.has_code = len(named_outputs) > 0
        if self.has_code:
            self.output = \
                named_outputs.get('algorithm',
                                  'algorithm_tmp_{}'.format(self.order))

            self.class_prior = parameters.get(self.CLASS_PRIOR_PARAM,
                                              'None') or 'None'
            if self.class_prior != "None":
                self.class_prior = '[' + self.class_prior + ']'

            self.smoothing = parameters.get(self.ALPHA_PARAM, 1.0) or 1.0
            self.model_type = parameters.get(
                    self.MODEL_TYPE_PARAM,
                    self.MODEL_TYPE_PARAM_M) or self.MODEL_TYPE_PARAM_M

            if self.smoothing <= 0:
                raise ValueError(
                        _("Parameter '{}' must be x>0 for task {}").format(
                                'smoothing', self.__class__))

            if self.model_type == self.MODEL_TYPE_PARAM_M:
                self.has_import = \
                    "from sklearn.naive_bayes import MultinomialNB\n"
            elif self.model_type == self.MODEL_TYPE_PARAM_B:
                self.has_import = \
                    "from sklearn.naive_bayes import BernoulliNB\n"
            else:
                self.has_import = \
                    "from sklearn.naive_bayes import GaussianNB\n"

    def generate_code(self):
        """Generate code."""
        if self.model_type == self.MODEL_TYPE_PARAM_M:
            code = """
        {output} = MultinomialNB(alpha={alpha}, class_prior={prior})
        """.format(output=self.output, prior=self.class_prior,
                   alpha=self.smoothing)
        elif self.model_type == self.MODEL_TYPE_PARAM_B:
            code = """
        {output} = BernoulliNB(alpha={smoothing}, class_prior={prior})
        """.format(output=self.output, smoothing= self.smoothing,
                   prior=self.class_prior)
        else:
            code = """
        {output} = GaussianNB(prior={prior})  
        """.format(prior=self.class_prior, output=self.output)

        return dedent(code)


class PerceptronClassifierOperation(Operation):

    ALPHA_PARAM = 'alpha'
    TOLERANCE_PARAM = 'tol'
    SHUFFLE_PARAM = 'shuffle'
    SEED_PARAM = 'seed'
    PENALTY_PARAM = 'penalty'
    MAX_ITER_PARAM = 'max_iter'

    PENALTY_PARAM_EN = 'elasticnet'
    PENALTY_PARAM_L1 = 'l1'
    PENALTY_PARAM_L2 = 'l2'
    PENALTY_PARAM_NONE = 'None'

    def __init__(self, parameters,  named_inputs, named_outputs):
        Operation.__init__(self, parameters,  named_inputs,  named_outputs)

        self.has_code = any([len(named_outputs) > 0,
                             self.contains_results(), self.named_inputs > 0])
        if self.has_code:
            self.output = \
                named_outputs.get('algorithm',
                                  'algorithm_tmp_{}'.format(self.order))

            self.max_iter = parameters.get(self.MAX_ITER_PARAM, 1000) or 1000
            self.alpha = parameters.get(self.ALPHA_PARAM, 0.0001) or 0.0001
            self.tol = parameters.get(self.TOLERANCE_PARAM, 0.001) or 0.001
            self.tol = abs(float(self.tol))
            self.shuffle = parameters.get(self.SHUFFLE_PARAM, False) or False
            self.seed = parameters.get(self.SEED_PARAM, 'None') or 'None'
            self.penalty = parameters.get(
                    self.PENALTY_PARAM,
                    self.PENALTY_PARAM_NONE) or self.PENALTY_PARAM_NONE

            vals = [self.max_iter, self.alpha]
            atts = [self.MAX_ITER_PARAM, self.ALPHA_PARAM]
            for var, att in zip(vals, atts):
                if var <= 0:
                    raise ValueError(
                            _("Parameter '{}' must be x>0 for task {}").format(
                                    att, self.__class__))

            self.has_import = \
                "from sklearn.linear_model import Perceptron\n"

    def generate_code(self):
        """Generate code."""
        code = """
        {output} = Perceptron(tol={tol}, alpha={alpha}, max_iter={max_iter},
        shuffle={shuffle}, random_state={seed}, penalty='{penalty}')
        """.format(tol=self.tol,
                   alpha=self.alpha,
                   max_iter=self.max_iter,
                   shuffle=self.shuffle,
                   penalty=self.penalty,
                   seed=self.seed,
                   output=self.output)
        return dedent(code)


class RandomForestClassifierOperation(Operation):

    N_ESTIMATORS_PARAM = 'n_estimators'
    MAX_DEPTH_PARAM = 'max_depth'
    MIN_SPLIT_PARAM = 'min_samples_split'
    MIN_LEAF_PARAM = 'min_samples_leaf'
    SEED_PARAM = 'seed'

    def __init__(self, parameters,  named_inputs, named_outputs):
        Operation.__init__(self, parameters,  named_inputs,  named_outputs)

        self.has_code = True
        if self.has_code:
            self.name = 'classifier.RandomForestClassifierOperation'
            self.output = \
                named_outputs.get('algorithm',
                                  'algorithm_tmp_{}'.format(self.order))
            self.seed = parameters.get(self.SEED_PARAM, 'None') or 'None'
            self.min_split = parameters.get(self.MIN_SPLIT_PARAM, 2) or 2
            self.min_leaf = parameters.get(self.MIN_LEAF_PARAM, 1) or 1
            self.max_depth = parameters.get(self.MAX_DEPTH_PARAM,
                                            'None') or 'None'
            self.n_estimators = parameters.get(self.N_ESTIMATORS_PARAM,
                                               10) or 10

            vals = [self.min_split, self.min_leaf, self.n_estimators]
            atts = [self.MIN_SPLIT_PARAM, self.MIN_LEAF_PARAM,
                    self.N_ESTIMATORS_PARAM]
            for var, att in zip(vals, atts):
                if var <= 0:
                    raise ValueError(
                            _("Parameter '{}' must be x>0 for task {}").format(
                                    att, self.__class__))

            if self.max_depth is not 'None' and self.max_depth <= 0:
                raise ValueError(
                        _("Parameter '{}' must be x>0 for task {}").format(
                                self.MAX_DEPTH_PARAM, self.__class__))

            self.has_import = \
                "from sklearn.ensemble import RandomForestClassifier\n"

    def generate_code(self):
        """Generate code."""
        code = """
        {output} = RandomForestClassifier(n_estimators={n_estimators}, 
        max_depth={max_depth},  min_samples_split={min_split}, 
        min_samples_leaf={min_leaf}, random_state={seed})
        """.format(output=self.output,
                   n_estimators=self.n_estimators,
                   max_depth=self.max_depth, min_split=self.min_split,
                   min_leaf=self.min_leaf, seed=self.seed)

        return dedent(code)


class SvmClassifierOperation(Operation):

    PENALTY_PARAM = 'c'
    KERNEL_PARAM = 'kernel'
    DEGREE_PARAM = 'degree'
    TOLERANCE_PARAM = 'tol'
    MAX_ITER_PARAM = 'max_iter'
    SEED_PARAM = 'seed'

    KERNEL_PARAM_LINEAR = 'linear'
    KERNEL_PARAM_RBF = 'rbf'
    KERNEL_PARAM_POLY = 'poly'
    KERNEL_PARAM_SIG = 'sigmoid'

    def __init__(self, parameters,  named_inputs, named_outputs):
        Operation.__init__(self, parameters,  named_inputs,  named_outputs)

        self.has_code = any([len(named_outputs) > 0,
                             self.contains_results(), self.named_inputs > 0])
        if self.has_code:
            self.name = 'classifier.SvmClassifierOperation'

            self.max_iter = parameters.get(self.MAX_ITER_PARAM, -1)
            self.tol = parameters.get(self.TOLERANCE_PARAM, 0.001) or 0.001
            self.tol = abs(float(self.tol))
            self.seed = parameters.get(self.SEED_PARAM, 'None') or 'None'
            self.degree = parameters.get(self.DEGREE_PARAM, 3) or 3
            self.kernel = parameters.get(
                    self.KERNEL_PARAM,
                    self.KERNEL_PARAM_RBF) or self.KERNEL_PARAM_RBF
            self.c = parameters.get(self.PENALTY_PARAM, 1.0) or 1.0

            vals = [self.degree, self.c]
            atts = [self.DEGREE_PARAM, self.PENALTY_PARAM]
            for var, att in zip(vals, atts):
                if var <= 0:
                    raise ValueError(
                            _("Parameter '{}' must be x>0 for task {}").format(
                                    att, self.__class__))

            self.has_import = \
                "from sklearn.svm import SVC\n"

    def generate_code(self):
        """Generate code."""
        code = """
        {algorithm} = SVC(tol={tol}, C={c}, max_iter={max_iter}, 
                       degree={degree}, kernel='{kernel}', random_state={seed})
        """.format(tol=self.tol, c=self.c, max_iter=self.max_iter,
                   degree=self.degree, kernel=self.kernel, seed=self.seed,
                   algorithm=self.named_outputs['algorithm'])
        return dedent(code)

