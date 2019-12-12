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

        if not self.has_code:
            raise ValueError(
                _("Parameters '{}' and '{}' must be informed for task {}")
                .format('train input data',  'algorithm', self.__class__))

        self.model = named_outputs.get('model',
                                       'model_task_{}'.format(self.order))

        if not self.has_code and len(self.named_outputs) > 0:
            raise ValueError(
                _('Model is being used, but at least one input is missing'))

        self.perform_transformation = 'output data' in self.named_outputs
        if not self.perform_transformation:
            self.output = 'task_{}'.format(self.order)
        else:
            self.output = self.named_outputs['output data']

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
                       algorithm=self.named_inputs['algorithm'],
                       features=self.features)

        if self.perform_transformation:
            code += """
            {OUT} = {IN}
            
            {OUT}['{predCol}'] = {model}.predict(X).tolist()
            """.format(predCol=self.prediction, OUT=self.output,
                       model=self.model,
                       IN=self.named_inputs['train input data'])
        else:
            code += """
            {output} = None
            """.format(output=self.output)

        return dedent(code)


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
        {output} = DecisionTreeClassifier(max_depth={max_depth}, 
        min_samples_split={min_split}, min_samples_leaf={min_leaf}, 
        min_weight_fraction_leaf={min_weight}, random_state={seed})
        """.format(output=self.output, min_split=self.min_split,
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
        {output} = GradientBoostingClassifier(loss='{loss}',
        learning_rate={learning_rate}, n_estimators={n_estimators},
        min_samples_split={min_split}, max_depth={max_depth},
        min_samples_leaf={min_leaf}, random_state={seed})
        """.format(output=self.output, loss=self.loss,
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
        {output} = KNeighborsClassifier(n_neighbors={K})
        """.format(K=self.k, output=self.output)
        return dedent(code)


class LogisticRegressionOperation(Operation):
    LABEL_PARAM = 'label'
    PREDICTION_PARAM = 'prediction'
    FEATURES_PARAM = 'features'
    TOLERANCE_PARAM = 'tol'
    REGULARIZATION_PARAM = 'regularization'
    MAX_ITER_PARAM = 'max_iter'
    SEED_PARAM = 'seed'
    SOLVER_PARAM = 'solver'
    PENALTY_PARAM = 'penalty'
    DUAL_PARAM = 'dual'
    FIT_INTERCEPT_PARAM = 'fit_intercept'
    INTERCEPT_SCALING_PARAM = 'intercept_scaling'
    MULTI_CLASS_PARAM = 'multi_class'
    N_JOBS_PARAM = 'n_jobs'
    L1_RATIO_PARAM = 'l1_ratio'
    VERBOSE_PARAM = 'verbose'
    MULTI_CLASS_PARAM = 'multi_class'

    SOLVER_PARAM_NEWTON = 'newton-cg'
    SOLVER_PARAM_LBFGS = 'lbfgs'
    SOLVER_PARAM_LINEAR = 'liblinear'
    SOLVER_PARAM_SAG = 'sag'
    SOLVER_PARAM_SAGa = 'saga'

    def __init__(self, parameters,  named_inputs, named_outputs):
        Operation.__init__(self, parameters,  named_inputs,  named_outputs)

        self.has_code = len(named_outputs) > 0
        if self.has_code:
            self.output = self.named_outputs.get(
            'output data', 'output_data_{}'.format(self.order))

            self.input_port = self.named_inputs.get(
            'train input data', 'input_data_{}'.format(self.order))

            self.model = self.named_outputs.get(
            'model', 'model_{}'.format(self.order))

            if self.LABEL_PARAM not in parameters:
                msg = _("Parameters '{}' must be informed for task {}")
                raise ValueError(msg.format(
                    self.LABEL_PARAM,
                    self.__class__.__name__))
            else: self.label = parameters.get(self.LABEL_PARAM, None)

            if self.FEATURES_PARAM not in parameters:
                msg = _("Parameters '{}' must be informed for task {}")
                raise ValueError(msg.format(
                    self.FEATURES_PARAM,
                    self.__class__.__name__))
            else: self.features = parameters.get(self.FEATURES_PARAM, None)

            self.prediction_column = parameters.get(self.PREDICTION_PARAM,
                                             'prediction')

            self.tol = float(self.parameters.get(self.TOLERANCE_PARAM,
                                           0.0001) or 0.0001)
            self.tol = abs(float(self.tol))
            self.regularization = float(self.parameters.get(self.REGULARIZATION_PARAM,
                                                      1.0)) or 1.0
            self.max_iter = int(self.parameters.get(self.MAX_ITER_PARAM,
                                                100)) or 100

            seed_ = self.parameters.get(self.SEED_PARAM, None)
            self.seed = int(seed_) if seed_ is not None else 'None'
            
            self.solver = self.parameters.get(
                    self.SOLVER_PARAM, self.SOLVER_PARAM_LINEAR)\
                or self.SOLVER_PARAM_LINEAR

            self.penalty = parameters.get(self.PENALTY_PARAM,
                                             'l2')

            self.dual = int(parameters.get(self.DUAL_PARAM, 0)) == 1
            self.fit_intercept = int(parameters.get(self.FIT_INTERCEPT_PARAM, 1)) == 1
            self.intercept_scaling = float(parameters.get(self.INTERCEPT_SCALING_PARAM, 1.0))
            self.verbose = int(parameters.get(self.VERBOSE_PARAM, 0))

            n_jobs_ = parameters.get(self.N_JOBS_PARAM, None)
            self.n_jobs = int(n_jobs_) if n_jobs_ is not None else 'None'

            l1_ratio_param_ = parameters.get(self.L1_RATIO_PARAM, None)
            if(l1_ratio_param_ is not None):
                self.l1_ratio = float(l1_ratio_param_)
                if(self.l1_ratio < 0 or self.l1_ratio > 1):
                    raise ValueError(
                            _("Parameter 'l1_ratio' must be 0 <= x <= 1 for task {}").format(
                                    self.__class__))
            else:
                self.l1_ratio = 'None'


            self.multi_class = parameters.get(self.MULTI_CLASS_PARAM, 'ovr') or 'ovr'

            vals = [self.regularization, self.max_iter, self.n_jobs, self.verbose]
            atts = [self.REGULARIZATION_PARAM, self.MAX_ITER_PARAM, self.N_JOBS_PARAM, self.VERBOSE_PARAM]
            for var, att in zip(vals, atts):
                if var <= 0:
                    raise ValueError(
                            _("Parameter '{}' must be x>0 for task {}").format(
                                    att, self.__class__))

            solver_dict = {
                'newton-cg': ['l2', 'none'],
                'lbfgs'    : ['l2', 'none'],
                'liblinear': ['l1'],
                'sag'      : ['l2', 'none'],
                'saga'     : ['l2', 'none', 'l1', 'elasticnet']
            }
            if(self.penalty not in solver_dict[self.solver]):
                raise ValueError(
                    ("For '{}' solver, the penalty type must be in {} for task {}").format(
                        self.solver, str(solver_dict[self.solver]), self.__class__))

            self.has_import = \
                "from sklearn.linear_model import LogisticRegression\n"

    @property
    def get_data_out_names(self, sep=','):
        return self.output

    def get_output_names(self, sep=', '):
        return sep.join([self.output, self.model])

    def generate_code(self):
        """Generate code."""
        code = """
            {model} = LogisticRegression(tol={tol}, C={C}, max_iter={max_iter},
            solver='{solver}', random_state={seed}, penalty='{penalty}', dual={dual},
            fit_intercept={fit_intercept}, intercept_scaling={intercept_scaling}, 
            multi_class='{multi_class}', verbose={verbose}, n_jobs={n_jobs},
            l1_ratio={l1_ratio})

            X_train = {input}[{features}].values.tolist()
            y = {input}[{label}].values.tolist()
            {model}.fit(X_train, y)

            {output} = {input}.copy()
            {output}['{prediction_column}'] = {model}.predict(X_train).tolist()
            """.format(tol=self.tol, C=self.regularization, max_iter=self.max_iter,
                       seed=self.seed, solver=self.solver, penalty=self.penalty,
                       dual=self.dual, fit_intercept=self.fit_intercept, 
                       intercept_scaling=self.intercept_scaling, multi_class=self.multi_class,
                       verbose=self.verbose, n_jobs=self.n_jobs, l1_ratio=self.l1_ratio, 
                       model=self.model, input=self.input_port, label=self.label,
                       output=self.output, prediction_column=self.prediction_column, 
                       features=self.features)
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
        {output} = MultinomialNB(alpha={alpha}, prior={prior})
        """.format(output=self.output, prior=self.class_prior,
                   alpha=self.smoothing)
        elif self.model_type == self.MODEL_TYPE_PARAM_B:
            code = """
        {output} = BernoulliNB(alpha={smoothing}, prior={prior})
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

        self.has_code = len(named_outputs) > 0
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
    LABEL_PARAM = 'label'
    PREDICTION_PARAM = 'prediction'
    FEATURES_PARAM = 'features'
    N_ESTIMATORS_PARAM = 'n_estimators'
    MAX_DEPTH_PARAM = 'max_depth'
    MIN_SPLIT_PARAM = 'min_samples_split'
    MIN_LEAF_PARAM = 'min_samples_leaf'
    SEED_PARAM = 'seed'
    CRITERION_PARAM = 'criterion'
    MIN_WEIGHT_FRACTION_LEAF_PARAM = 'min_weight_fraction_leaf'
    MAX_FEATURES_PARAM = 'max_features'
    MAX_LEAF_NODES_PARAM = 'max_leaf_nodes'
    MIN_IMPURITY_DECREASE_PARAM = 'min_impurity_decrease'
    BOOTSTRAP_PARAM = 'bootstrap'
    OOB_SCORE_PARAM = 'oob_score'
    N_JOBS_PARAM = 'n_jobs'
    VERBOSE_PARAM = 'verbose'
    CCP_ALPHA_PARAM = 'ccp_alpha'
    MAX_SAMPLES_PARAM = 'max_samples'

    def __init__(self, parameters,  named_inputs, named_outputs):
        Operation.__init__(self, parameters,  named_inputs,  named_outputs)

        self.has_code = True
        if self.has_code:
            self.output = self.named_outputs.get(
            'output data', 'output_data_{}'.format(self.order))

            self.input_port = self.named_inputs.get(
            'train input data', 'input_data_{}'.format(self.order))

            self.model = self.named_outputs.get(
                'model', 'model_{}'.format(self.order))

            if self.LABEL_PARAM not in parameters:
                msg = _("Parameters '{}' must be informed for task {}")
                raise ValueError(msg.format(
                    self.LABEL_PARAM,
                    self.__class__.__name__))
            else: self.label = parameters.get(self.LABEL_PARAM, None)

            if self.FEATURES_PARAM not in parameters:
                msg = _("Parameters '{}' must be informed for task {}")
                raise ValueError(msg.format(
                    self.FEATURES_PARAM,
                    self.__class__.__name__))
            else: self.features = parameters.get(self.FEATURES_PARAM, None)

            self.prediction_column = parameters.get(self.PREDICTION_PARAM,
                                             'prediction')

            self.seed = parameters.get(self.SEED_PARAM, 'None') or 'None'
            self.min_split = int(parameters.get(self.MIN_SPLIT_PARAM, 2) or 2)
            self.min_leaf = int(parameters.get(self.MIN_LEAF_PARAM, 1) or 1)
            
            self.max_depth = self.__positive_or_none_param(parameters, self.MAX_DEPTH_PARAM)
            self.n_estimators = int(parameters.get(self.N_ESTIMATORS_PARAM,
                                               10) or 10)

            self.criterion = parameters.get(self.CRITERION_PARAM, 'gini')

            self.min_weight_fraction_leaf = float(parameters.get(self.MIN_WEIGHT_FRACTION_LEAF_PARAM, 0.0))
            if(self.min_weight_fraction_leaf < 0.0 or self.min_weight_fraction_leaf > 0.5):
                raise ValueError(
                    _("Parameter '{}' must be x>=0.0 and x<=0.5 for task {}").format(
                            self.MIN_WEIGHT_FRACTION_LEAF_PARAM, self.__class__))


            self.max_features = self.__positive_or_none_param(parameters, self.MAX_FEATURES_PARAM)
            self.max_leaf_nodes = self.__positive_or_none_param(parameters, self.MAX_LEAF_NODES_PARAM)

            self.min_impurity_decrease = float(parameters.get(self.MIN_IMPURITY_DECREASE_PARAM, 0.0))

            self.bootstrap = int(parameters.get(self.BOOTSTRAP_PARAM, 1)) == 1
            self.oob_score = int(parameters.get(self.OOB_SCORE_PARAM, 0)) == 1

            self.n_jobs = self.__positive_or_none_param(parameters, self.N_JOBS_PARAM)
            self.verbose = int(parameters.get(self.VERBOSE_PARAM, 0)) or 0

            self.ccp_alpha = float(parameters.get(self.CCP_ALPHA_PARAM, 0.0))
            
            max_samples_ = parameters.get(self.MAX_SAMPLES_PARAM, None)
            if max_samples_ is not None:
                max_samples_ = float(max_samples_)
                if max_samples_ <= 0.0 or max_samples_ >= 100.0:
                    raise ValueError(
                            _("Parameter '{}' must be x>0 and x<100, or empty, case \
                             you want to use a fully sample for task {}").format(
                                    self.MAX_SAMPLES_PARAM, self.__class__))
                else:
                    self.max_samples = max_samples_/100.0
            else:
                self.max_samples = 'None'


            vals = [self.verbose, self.min_impurity_decrease, self.ccp_alpha]
            atts = [self.VERBOSE_PARAM, self.MIN_IMPURITY_DECREASE_PARAM, self.CCP_ALPHA_PARAM]
            for var, att in zip(vals, atts):
                if var < 0:
                    raise ValueError(
                            _("Parameter '{}' must be x>=0 for task {}").format(
                                    att, self.__class__))

            vals = [self.min_split, self.min_leaf, self.n_estimators]
            atts = [self.MIN_SPLIT_PARAM, self.MIN_LEAF_PARAM,
                    self.N_ESTIMATORS_PARAM]
            for var, att in zip(vals, atts):
                if var <= 0:
                    raise ValueError(
                            _("Parameter '{}' must be x>0 for task {}").format(
                                    att, self.__class__))

            self.has_import = \
                "from sklearn.ensemble import RandomForestClassifier\n"

    @property
    def get_data_out_names(self, sep=','):
        return self.output

    def get_output_names(self, sep=', '):
        return sep.join([self.output, self.model])

    def __positive_or_none_param(self, parameters, param_name):
        returned_param = None
        param = parameters.get(param_name, None)
        if param is not None:
            returned_param = int(param)
            if returned_param < 0:
                raise ValueError(
                        _("Parameter '{}' must be x>=0  for task {}").format(
                                param_name, self.__class__))
        else:
            returned_param = 'None'
        return returned_param

    def generate_code(self):

        """Generate code."""
        code = """
        {model} = RandomForestClassifier(n_estimators={n_estimators}, 
        max_depth={max_depth},  min_samples_split={min_split}, 
        min_samples_leaf={min_leaf}, random_state={seed},
        criterion='{criterion}', min_weight_fraction_leaf={min_weight_fraction_leaf},
        max_features={max_features}, max_leaf_nodes={max_leaf_nodes}, 
        min_impurity_decrease={min_impurity_decrease}, bootstrap={bootstrap},
        oob_score={oob_score}, n_jobs={n_jobs}, verbose={verbose},
        ccp_alpha={ccp_alpha}, max_samples={max_samples})

        X_train = {input}[{features}].values.tolist()
        y = {input}[{label}].values.tolist()
        {model}.fit(X_train, y)

        {output} = {input}.copy()
        {output}['{prediction_column}'] = {model}.predict(X_train).tolist()
        """.format(output=self.output, model=self.model, input=self.input_port,
                   n_estimators=self.n_estimators, max_depth=self.max_depth, 
                   min_split=self.min_split, min_leaf=self.min_leaf, seed=self.seed,
                   criterion=self.criterion, min_weight_fraction_leaf=self.min_weight_fraction_leaf,
                   max_features=self.max_features, max_leaf_nodes=self.max_leaf_nodes,
                   min_impurity_decrease=self.min_impurity_decrease, bootstrap=self.bootstrap,
                   oob_score=self.oob_score, n_jobs=self.n_jobs, verbose=self.verbose,
                   ccp_alpha=self.ccp_alpha, max_samples=self.max_samples, features=self.features,
                   prediction_column=self.prediction_column, label=self.label)

        return dedent(code)

class SvmClassifierOperation(Operation):

    PENALTY_PARAM = 'c'
    KERNEL_PARAM = 'kernel'
    DEGREE_PARAM = 'degree'
    TOLERANCE_PARAM = 'tol'
    MAX_ITER_PARAM = 'max_iter'
    SEED_PARAM = 'seed'
    LABEL_PARAM = 'label'
    FEATURES_PARAM = 'features'
    PREDICTION_PARAM = 'prediction'
    GAMMA_PARAM = 'gamma'
    COEF0_PARAM = 'coef0'
    SHRINKING_PARAM = 'shrinking'
    PROBABILITY_PARAM = 'probability'
    CACHE_SIZE_PARAM = 'cache_size'
    DECISION_FUNCTION_SHAPE_PARAM = 'decision_function_shape'

    KERNEL_PARAM_LINEAR = 'linear'
    KERNEL_PARAM_RBF = 'rbf'
    KERNEL_PARAM_POLY = 'poly'
    KERNEL_PARAM_SIG = 'sigmoid'

    def __init__(self, parameters,  named_inputs, named_outputs):
        Operation.__init__(self, parameters,  named_inputs,  named_outputs)

        self.has_code = True
        if self.has_code:
            self.output = self.named_outputs.get(
            'output data', 'output_data_{}'.format(self.order))

            self.input_port = self.named_inputs.get(
            'train input data', 'input_data_{}'.format(self.order))

            self.model = self.named_outputs.get(
            'model', 'model_{}'.format(self.order))

            if self.LABEL_PARAM not in parameters:
                msg = _("Parameters '{}' must be informed for task {}")
                raise ValueError(msg.format(
                    self.LABEL_PARAM,
                    self.__class__.__name__))
            else: self.label = parameters.get(self.LABEL_PARAM, None)

            if self.FEATURES_PARAM not in parameters:
                msg = _("Parameters '{}' must be informed for task {}")
                raise ValueError(msg.format(
                    self.FEATURES_PARAM,
                    self.__class__.__name__))
            else: self.features = parameters.get(self.FEATURES_PARAM, None)

            self.prediction = parameters.get(self.PREDICTION_PARAM,
                                             'prediction')
            self.max_iter = int(parameters.get(self.MAX_ITER_PARAM, -1))
            self.tol = float(parameters.get(self.TOLERANCE_PARAM, 0.001) or 0.001)
            self.tol = abs(float(self.tol))
            self.seed = parameters.get(self.SEED_PARAM, 'None') or 'None'
            self.degree = int(parameters.get(self.DEGREE_PARAM, 3) or 3)
            self.kernel = parameters.get(
                    self.KERNEL_PARAM,
                    self.KERNEL_PARAM_RBF) or self.KERNEL_PARAM_RBF
            self.c = float(parameters.get(self.PENALTY_PARAM, 1.0) or 1.0)

            gamma_ = parameters.get(self.GAMMA_PARAM, None)
            self.gamma = "'"+gamma_+"'" if gamma_ == 'auto' else (float(gamma_) if gamma_ is not None else "'auto'")
            self.coef0 = float(parameters.get(self.COEF0_PARAM, 0.0) or 0.0)
            self.shrinking = int(parameters.get(self.SHRINKING_PARAM, 1)) == 1
            self.probability = int(parameters.get(self.PROBABILITY_PARAM, 0)) == 1
            self.cache_size = float(parameters.get(self.CACHE_SIZE_PARAM, 200.0) or 200.0)
            self.decision_function_shape = parameters.get(self.DECISION_FUNCTION_SHAPE_PARAM, 'ovr') or 'ovr'           

            vals = [self.degree, self.c, self.cache_size]
            atts = [self.DEGREE_PARAM, self.PENALTY_PARAM, self.CACHE_SIZE_PARAM]
            for var, att in zip(vals, atts):
                if var <= 0:
                    raise ValueError(
                            _("Parameter '{}' must be x>0 for task {}").format(
                                    att, self.__class__))

            self.has_import = \
                "from sklearn.svm import SVC\n"

    @property
    def get_data_out_names(self, sep=','):
        return self.output

    def get_output_names(self, sep=', '):
        return sep.join([self.output, self.model])

    def generate_code(self):
        """Generate code."""
        code = """
        {model} = SVC(tol={tol}, C={c}, max_iter={max_iter}, 
                       degree={degree}, kernel='{kernel}', random_state={seed},
                       gamma={gamma}, coef0={coef0}, probability={prob},
                       cache_size={cache},shrinking={shrinking}, 
                       decision_function_shape='{decision_func_shape}',
                       class_weight=None, verbose=False)

        X_train = {input}[{features}].values.tolist()
        y = {input}[{label}].values.tolist()
        {model}.fit(X_train, y)

        {output} = {input}.copy()
        {output}['{prediction_column}'] = {model}.predict(X_train).tolist()
        
        """.format(tol=self.tol, c=self.c, max_iter=self.max_iter,
                   degree=self.degree, kernel=self.kernel, seed=self.seed,
                   gamma=self.gamma, coef0=self.coef0, prob=self.probability,
                   cache=self.cache_size, shrinking=self.shrinking, 
                   decision_func_shape=self.decision_function_shape,
                   model=self.model, input=self.input_port, label=self.label,
                   features=self.features, prediction_column=self.prediction,
                   output=self.output)
        return dedent(code)
