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
    CRITERION_PARAM = 'criterion'
    SPLITTER_PARAM = 'splitter'
    MAX_FEATURES_PARAM = 'max_features'
    MAX_LEAF_NODES_PARAM = 'max_leaf_nodes'
    MIN_IMPURITY_DECREASE_PARAM = 'min_impurity_decrease'
    CLASS_WEIGHT_PARAM = 'class_weight'
    PRESORT_PARAM = 'presort'
    LABEL_PARAM = 'label'
    PREDICTION_PARAM = 'prediction'
    FEATURES_PARAM = 'features'

    def __init__(self, parameters,  named_inputs, named_outputs):
        Operation.__init__(self, parameters,  named_inputs,  named_outputs)

        self.has_code = any([len(self.named_inputs) == 1, self.contains_results()])
        self.output = self.named_outputs.get(
            'output data', 'output_data_{}'.format(self.order))

        self.model = self.named_outputs.get(
            'model', 'model_{}'.format(self.order))

        self.input_port = self.named_inputs.get(
            'train input data', 'input_data_{}'.format(self.order))
        if self.has_code:
            self.min_split = int(parameters.get(self.MIN_SPLIT_PARAM, 2) or 2)
            self.min_leaf = int(parameters.get(self.MIN_LEAF_PARAM, 1) or 1)
            self.max_depth = parameters.get(self.MAX_DEPTH_PARAM, None) or None
            self.min_weight = float(parameters.get(self.MIN_WEIGHT_PARAM, 0.0) or 0.0)
            self.seed = parameters.get(self.SEED_PARAM, None) or None
            self.criterion = parameters.get(self.CRITERION_PARAM, 'gini') or 'gini'
            self.splitter = parameters.get(self.SPLITTER_PARAM, 'best') or 'best'
            self.max_features = parameters.get(self.MAX_FEATURES_PARAM, None) or None
            self.max_leaf_nodes = parameters.get(self.MAX_LEAF_NODES_PARAM, None) or None
            self.min_impurity_decrease = float(parameters.get(self.MIN_IMPURITY_DECREASE_PARAM, 0) or 0)
            self.class_weight = parameters.get(self.CLASS_WEIGHT_PARAM, None) or None
            self.presort = int(parameters.get(self.PRESORT_PARAM, 0) or 0)
            self.features = parameters['features']
            self.label = parameters.get(self.LABEL_PARAM, None)
            self. prediction = self.parameters.get(self.PREDICTION_PARAM, 'prediction')

            vals = [self.min_split, self.min_leaf]
            atts = [self.MIN_SPLIT_PARAM, self.MIN_LEAF_PARAM]
            for var, att in zip(vals, atts):
                if var <= 0:
                    raise ValueError(
                            _("Parameter '{}' must be x>0 for task {}").format(
                                    att, self.__class__))

            self.has_import = \
                "from sklearn.tree import DecisionTreeClassifier\n"

            self.input_treatment()

    @property
    def get_data_out_names(self, sep=','):
        return self.output

    def get_output_names(self, sep=', '):
        return sep.join([self.output, self.model])

    def input_treatment(self):
        self.presort = True if int(self.presort) == 1 else False

        if self.min_weight < 0:
            raise ValueError(
                _("Parameter '{}' must be x>=0 for task {}").format(
                    self.MIN_WEIGHT_PARAM, self.__class__))

        if self.max_depth is not None and self.max_depth <= 0:
            raise ValueError(
                _("Parameter '{}' must be x>0 for task {}").format(
                    self.MAX_DEPTH_PARAM, self.__class__))

        if self.max_leaf_nodes is not None and self.max_leaf_nodes != '0':
            self.max_leaf_nodes = int(self.max_leaf_nodes)
        else:
            self.max_leaf_nodes = None

        if self.seed is not None and self.seed != '0':
            self.seed = int(self.seed)
        else:
            self.seed = None

    def generate_code(self):
        """Generate code."""
        code = """
            {output_data} = {input_data}.copy()            
            X_train = {input_data}[{columns}].values.tolist()
            y = {input_data}[{label}].values.tolist()
            {model} = DecisionTreeClassifier(max_depth={max_depth}, min_samples_split={min_split}, 
                                             min_samples_leaf={min_leaf}, min_weight_fraction_leaf={min_weight}, 
                                             random_state={seed}, criterion='{criterion}', splitter='{splitter}', 
                                             max_features={max_features}, max_leaf_nodes={max_leaf_nodes}, 
                                             min_impurity_decrease={min_impurity_decrease}, class_weight={class_weight},
                                             presort={presort})
            {model}.fit(X_train, y)          
            {output_data}['{prediction}'] = {model}.predict(X_train).tolist()
            """.format(output_data=self.output,
                       prediction=self.prediction,
                       columns=self.features,
                       model=self.model,
                       input_data=self.input_port,
                       label=self.label,
                       min_split=self.min_split,
                       min_leaf=self.min_leaf,
                       min_weight=self.min_weight,
                       seed=self.seed,
                       max_depth=self.max_depth,
                       criterion=self.criterion,
                       splitter=self.splitter,
                       max_features=self.max_features,
                       max_leaf_nodes=self.max_leaf_nodes,
                       min_impurity_decrease=self.min_impurity_decrease,
                       class_weight=self.class_weight,
                       presort=self.presort)
        return dedent(code)


class GBTClassifierOperation(Operation):

    LEARNING_RATE_PARAM = 'learning_rate'
    N_ESTIMATORS_PARAM = 'n_estimators'
    MAX_DEPTH_PARAM = 'max_depth'
    MIN_SPLIT_PARAM = 'min_samples_split'
    MIN_LEAF_PARAM = 'min_samples_leaf'
    LOSS_PARAM = 'loss'
    SEED_PARAM = 'seed'
    SUBSAMPLE_PARAM = 'subsample'
    CRITERION_PARAM = 'criterion'
    MIN_WEIGHT_FRACTION_LEAF_PARAM = 'min_weight_fraction_leaf'
    MIN_IMPURITY_DECREASE_PARAM = 'min_impurity_decrease'
    INIT_PARAM = 'init'
    MAX_FEATURES_PARAM = 'max_features'
    VERBOSE_PARAM = 'verbose'
    MAX_LEAF_NODES_PARAM = 'max_leaf_nodes'
    WARM_START_PARAM = 'warm_start'
    PRESORT_PARAM = 'presort'
    VALIDATION_FRACTION_PARAM = 'validation_fraction'
    N_ITER_NO_CHANGE_PARAM = 'n_iter_no_change'
    TOL_PARAM = 'tol'
    LABEL_PARAM = 'label'
    PREDICTION_PARAM = 'prediction'
    FEATURES_PARAM = 'features'

    LOSS_PARAM_DEV = 'deviance'
    LOSS_PARAM_EXP = 'exponencial'

    def __init__(self, parameters,  named_inputs, named_outputs):
        Operation.__init__(self, parameters,  named_inputs,  named_outputs)

        self.has_code = any([len(self.named_inputs) == 1, self.contains_results()])

        self.output = self.named_outputs.get(
            'output data', 'output_data_{}'.format(self.order))

        self.model = self.named_outputs.get(
            'model', 'model_{}'.format(self.order))

        self.input_port = self.named_inputs.get(
            'train input data', 'input_data_{}'.format(self.order))
        if self.has_code:
            self.max_depth = int(parameters.get(self.MAX_DEPTH_PARAM, 3) or 3)
            self.min_split = int(parameters.get(self.MIN_SPLIT_PARAM, 2) or 2)
            self.min_leaf = int(parameters.get(self.MIN_LEAF_PARAM, 1) or 1)
            self.n_estimators = int(parameters.get(self.N_ESTIMATORS_PARAM, 100) or 100)
            self.learning_rate = float(parameters.get(self.LEARNING_RATE_PARAM, 0.1) or 0.1)
            self.loss = \
                parameters.get(self.LOSS_PARAM, self.LOSS_PARAM_DEV) or \
                self.LOSS_PARAM_DEV
            self.seed = parameters.get(self.SEED_PARAM, None) or None
            self.subsample = float(parameters.get(self.LEARNING_RATE_PARAM, 1.0) or 1.0)
            self.criterion = parameters.get(self.CRITERION_PARAM, 'friedman_mse') or 'friedman_mse'
            self.min_weight_fraction_leaf = float(parameters.get(self.MIN_WEIGHT_FRACTION_LEAF_PARAM, 0) or 0)
            self.min_impurity_decrease = float(parameters.get(self.MIN_IMPURITY_DECREASE_PARAM, 0) or 0)
            self.init = parameters.get(self.INIT_PARAM, 'None') or 'None'
            self.max_features = parameters.get(self.MAX_FEATURES_PARAM, None) or None
            self.verbose = int(parameters.get(self.VERBOSE_PARAM, 0) or 0)
            self.max_leaf_nodes = parameters.get(self.MAX_LEAF_NODES_PARAM, None) or None
            self.warm_start = int(parameters.get(self.WARM_START_PARAM, 0) or 0)
            self.presort = parameters.get(self.PRESORT_PARAM, 'auto') or 'auto'
            self.validation_fraction = float(parameters.get(self.VALIDATION_FRACTION_PARAM, 0.1) or 0.1)
            self.n_iter_no_change = parameters.get(self.N_ITER_NO_CHANGE_PARAM, None) or None
            self.tol = float(parameters.get(self.LEARNING_RATE_PARAM, 1e-4) or 1e-4)
            self.features = parameters['features']
            self.label = parameters.get(self.LABEL_PARAM, None)
            self. prediction = self.parameters.get(self.PREDICTION_PARAM, 'prediction')

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

            self.input_treatment()

    @property
    def get_data_out_names(self, sep=','):
        return self.output

    def get_output_names(self, sep=', '):
        return sep.join([self.output, self.model])

    def input_treatment(self):
        if self.max_leaf_nodes is not None and self.max_leaf_nodes != '0':
            self.max_leaf_nodes = int(self.max_leaf_nodes)
        else:
            self.max_leaf_nodes = None
        if self.seed is not None and self.seed != '0':
            self.seed = int(self.seed)
        else:
            self.seed = None
        if self.n_iter_no_change is not None and self.n_iter_no_change != '0':
            self.n_iter_no_change = int(self.n_iter_no_change)
        else:
            self.n_iter_no_change = None
        self.warm_start = True if int(self.warm_start) == 1 else False
        if self.validation_fraction < 0 or self.validation_fraction > 1:
            raise ValueError(
                _("Parameter '{}' must be 0 <= x =< 1 for task {}").format(
                    self.VALIDATION_FRACTION_PARAM, self.__class__))

    def generate_code(self):
        """Generate code."""
        code = """ 
            {output_data} = {input_data}.copy()            
            X_train = {input_data}[{columns}].values.tolist()
            y = {input_data}[{label}].values.tolist()
            {model} = GradientBoostingClassifier(loss='{loss}', learning_rate={learning_rate}, 
                                                  n_estimators={n_estimators}, min_samples_split={min_split},
                                                  max_depth={max_depth}, min_samples_leaf={min_leaf}, 
                                                  random_state={seed}, subsample={subsample}, criterion='{criterion}',
                                                  min_weight_fraction_leaf={min_weight_fraction_leaf}, 
                                                  min_impurity_decrease={min_impurity_decrease}, init={init},
                                                  max_features={max_features}, verbose={verbose}, 
                                                  max_leaf_nodes={max_leaf_nodes}, warm_start={warm_start}, 
                                                  presort='{presort}', validation_fraction={validation_fraction}, 
                                                  n_iter_no_change={n_iter_no_change}, tol={tol})
            {model}.fit(X_train, y)          
            {output_data}['{prediction}'] = {model}.predict(X_train).tolist()
            """.format(output_data=self.output,
                       prediction=self.prediction,
                       columns=self.features,
                       model=self.model,
                       input_data=self.input_port,
                       label=self.label,
                       loss=self.loss,
                       n_estimators=self.n_estimators,
                       min_leaf=self.min_leaf,
                       min_split=self.min_split,
                       learning_rate=self.learning_rate,
                       max_depth=self.max_depth,
                       seed=self.seed,
                       subsample=self.subsample,
                       criterion=self.criterion,
                       min_weight_fraction_leaf=self.min_weight_fraction_leaf,
                       min_impurity_decrease=self.min_impurity_decrease,
                       init=self.init,
                       max_features=self.max_features,
                       verbose=self.verbose,
                       max_leaf_nodes=self.max_leaf_nodes,
                       warm_start=self.warm_start,
                       presort=self.presort,
                       validation_fraction=self.validation_fraction,
                       n_iter_no_change=self.n_iter_no_change,
                       tol=self.tol)
        return dedent(code)


class KNNClassifierOperation(Operation):

    K_PARAM = 'n_neighbors'
    WEIGHTS_PARAM = 'weights'
    ALGORITHM_PARAM = 'algorithm'
    LEAF_SIZE_PARAM = 'leaf_size'
    P_PARAM = 'p'
    METRIC_PARAM = 'metric'
    METRIC_PARAMS_PARAM = 'metric_params'
    N_JOBS_PARAM = 'n_jobs'
    LABEL_PARAM = 'label'
    PREDICTION_PARAM = 'prediction'
    FEATURES_PARAM = 'features'

    def __init__(self, parameters,  named_inputs, named_outputs):
        Operation.__init__(self, parameters,  named_inputs,  named_outputs)

        self.has_code = any([len(self.named_inputs) == 1, self.contains_results()])

        self.output = self.named_outputs.get(
            'output data', 'output_data_{}'.format(self.order))

        self.model = self.named_outputs.get(
            'model', 'model_{}'.format(self.order))

        self.input_port = self.named_inputs.get(
            'train input data', 'input_data_{}'.format(self.order))

        if self.has_code:
            if self.K_PARAM not in parameters:
                raise ValueError(
                    _("Parameter '{}' must be informed for task {}")
                    .format(self.K_PARAM, self.__class__))

            self.n_neighbors = int(self.parameters.get(self.K_PARAM, 5) or 5)
            self.weights = self.parameters.get(self.WEIGHTS_PARAM, 'uniform') or 'uniform'
            self.algorithm = self.parameters.get(self.ALGORITHM_PARAM, 'auto') or 'auto'
            self.leaf_size = int(self.parameters.get(self.LEAF_SIZE_PARAM, 30) or 30)
            self.p = int(self.parameters.get(self.P_PARAM, 2) or 2)
            self.metric = self.parameters.get(self.METRIC_PARAM, 'minkowski') or 'minkowski'
            self.metric_params = self.parameters.get(self.METRIC_PARAMS_PARAM, None) or None
            self.n_jobs = self.parameters.get(self.N_JOBS_PARAM, None) or None
            self.features = parameters['features']
            self.label = parameters.get(self.LABEL_PARAM, None)
            self.prediction = self.parameters.get(self.PREDICTION_PARAM, 'prediction')

            if self.n_neighbors <= 0:
                raise ValueError(
                        _("Parameter '{}' must be x>0 for task {}").format(
                                self.K_PARAM, self.__class__))

            self.has_import = \
                "from sklearn.neighbors import KNeighborsClassifier\n"

            self.input_treatment()

    @property
    def get_data_out_names(self, sep=','):
        return self.output

    def get_output_names(self, sep=', '):
        return sep.join([self.output, self.model])

    def input_treatment(self):
        if self.n_jobs is not None:
            self.n_jobs = int(self.n_jobs)
        else:
            self.n_jobs = None

    def generate_code(self):
        """Generate code."""
        code = """
            {output_data} = {input_data}.copy()            
            X_train = {input_data}[{features}].values.tolist()
            y = {input_data}[{label}].values.tolist()
            {model} = KNeighborsClassifier(n_neighbors={n_neighbors}, weights='{weights}', algorithm='{algorithm}', 
                                           leaf_size={leaf_size}, p={p}, metric='{metric}', 
                                           metric_params={metric_params}, n_jobs={n_jobs})
            {model}.fit(X_train, y)          
            {output_data}['{prediction}'] = {model}.predict(X_train).tolist()
            """.format(n_neighbors=self.n_neighbors,
                       output_data=self.output,
                       model=self.model,
                       input_data=self.input_port,
                       prediction=self.prediction,
                       features=self.features,
                       label=self.label,
                       weights=self.weights,
                       algorithm=self.algorithm,
                       leaf_size=self.leaf_size,
                       p=self.p,
                       metric=self.metric,
                       metric_params=self.metric_params,
                       n_jobs=self.n_jobs)
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
            {output} = LogisticRegression(tol={tol}, C={C}, max_iter={max_iter},
            solver='{solver}', random_state={seed})
            """.format(tol=self.tol,
                       C=self.regularization,
                       max_iter=self.max_iter,
                       seed=self.seed,
                       solver=self.solver,
                       output=self.output)
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
    FIT_PRIOR_PARAM = 'fit_prior'
    VAR_SMOOTHING_PARAM = 'var_smoothing'
    PRIORS_PARAM = 'priors'
    BINARIZE_PARAM = 'binarize'
    MODEL_TYPE_PARAM = 'type'
    PREDICTION_PARAM = 'prediction'
    LABEL_PARAM = 'label'
    FEATURES_PARAM = 'features'

    MODEL_TYPE_PARAM_B = 'Bernoulli'
    MODEL_TYPE_PARAM_G = 'GaussianNB'
    MODEL_TYPE_PARAM_M = 'Multinomial'

    def __init__(self, parameters,  named_inputs, named_outputs):
        Operation.__init__(self, parameters,  named_inputs,  named_outputs)

        self.has_code = any([len(self.named_inputs) == 1, self.contains_results()])

        self.output = self.named_outputs.get(
            'output data', 'output_data_{}'.format(self.order))

        self.model = self.named_outputs.get(
            'model', 'model_{}'.format(self.order))

        self.input_port = self.named_inputs.get(
            'train input data', 'input_data_{}'.format(self.order))

        if self.has_code:
            self.class_prior = parameters.get(self.CLASS_PRIOR_PARAM, 'None') or 'None'
            self.alpha = float(parameters.get(self.ALPHA_PARAM, 1.0) or 1.0)
            self.fit_prior = int(parameters.get(self.FIT_PRIOR_PARAM, 1) or 1)
            self.var_smoothing = float(parameters.get(self.VAR_SMOOTHING_PARAM, 1e-9) or 1e-9)
            self.priors = parameters.get(self.PRIORS_PARAM, 'None') or 'None'
            self.binarize = float(parameters.get(self.BINARIZE_PARAM, 0) or 0)
            self.features = parameters['features']
            self.label = parameters.get(self.LABEL_PARAM, None)
            self.prediction = self.parameters.get(self.PREDICTION_PARAM, 'prediction')
            self.smoothing = float(parameters.get(self.ALPHA_PARAM, 1.0) or 1.0)
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

    @property
    def get_data_out_names(self, sep=','):
        return self.output

    def get_output_names(self, sep=', '):
        return sep.join([self.output, self.model])

    def input_treatment(self):
        self.fit_prior = True if int(self.fit_prior) == 1 else False

        if self.class_prior != "None":
            self.class_prior = '[' + self.class_prior + ']'

        if self.priors != "None":
            self.priors = '[' + self.priors + ']'

    def generate_code(self):
        """Generate code."""
        if self.model_type == self.MODEL_TYPE_PARAM_M:
            code = """
                {output_data} = {input_data}.copy()            
                X_train = {input_data}[{features}].values.tolist()
                y = {input_data}[{label}].values.tolist()
                {model} = MultinomialNB(alpha={alpha}, class_prior={class_prior}, fit_prior={fit_prior})
                {model}.fit(X_train, y)          
                {output_data}['{prediction}'] = {model}.predict(X_train).tolist()
                """.format(output_data=self.output,
                           model=self.model,
                           input_data=self.input_port,
                           prediction=self.prediction,
                           features=self.features,
                           label=self.label,
                           class_prior=self.class_prior,
                           alpha=self.alpha,
                           fit_prior=self.fit_prior)
        elif self.model_type == self.MODEL_TYPE_PARAM_B:
            code = """
                {output_data} = {input_data}.copy()            
                X_train = {input_data}[{features}].values.tolist()
                y = {input_data}[{label}].values.tolist()
                {model} = BernoulliNB(alpha={alpha}, class_prior={class_prior}, fit_prior={fit_prior}, 
                                            binarize={binarize})
                {model}.fit(X_train, y)          
                {output_data}['{prediction}'] = {model}.predict(X_train).tolist()
                """.format(output_data=self.output,
                           model=self.model,
                           input_data=self.input_port,
                           prediction=self.prediction,
                           features=self.features,
                           label=self.label,
                           alpha=self.alpha,
                           class_prior=self.class_prior,
                           fit_prior=self.fit_prior,
                           binarize=self.binarize)
        else:
            code = """
                {output_data} = {input_data}.copy()            
                X_train = {input_data}[{features}].values.tolist()
                y = {input_data}[{label}].values.tolist()
                {model} = GaussianNB(priors={priors}, var_smoothing={var_smoothing})  
                {model}.fit(X_train, y)          
                {output_data}['{prediction}'] = {model}.predict(X_train).tolist()
                """.format(output_data=self.output,
                           model=self.model,
                           input_data=self.input_port,
                           prediction=self.prediction,
                           features=self.features,
                           label=self.label,
                           priors=self.priors,
                           var_smoothing=self.var_smoothing)

        return dedent(code)


class PerceptronClassifierOperation(Operation):

    ALPHA_PARAM = 'alpha'
    TOLERANCE_PARAM = 'tol'
    SHUFFLE_PARAM = 'shuffle'
    SEED_PARAM = 'seed'
    PENALTY_PARAM = 'penalty'
    MAX_ITER_PARAM = 'max_iter'
    FIT_INTERCEPT_PARAM = 'fit_intercept'
    VERBOSE_PARAM = 'verbose'
    ETA0_PARAM = 'eta0'
    N_JOBS_PARAM = 'n_jobs'
    EARLY_STOPPING_PARAM = 'early_stopping'
    VALIDATION_FRACTION_PARAM = 'validation_fraction'
    N_ITER_NO_CHANGE_PARAM = 'n_iter_no_change'
    CLASS_WEIGHT_PARAM = 'class_weight'
    WARM_START_PARAM = 'warm_start'
    PREDICTION_PARAM = 'prediction'
    LABEL_PARAM = 'label'
    FEATURES_PARAM = 'features'

    PENALTY_PARAM_EN = 'elasticnet'
    PENALTY_PARAM_L1 = 'l1'
    PENALTY_PARAM_L2 = 'l2'
    PENALTY_PARAM_NONE = 'None'

    def __init__(self, parameters,  named_inputs, named_outputs):
        Operation.__init__(self, parameters,  named_inputs,  named_outputs)

        self.has_code = any([len(self.named_inputs) == 1, self.contains_results()])

        self.output = self.named_outputs.get(
            'output data', 'output_data_{}'.format(self.order))

        self.model = self.named_outputs.get(
            'model', 'model_{}'.format(self.order))

        self.input_port = self.named_inputs.get(
            'train input data', 'input_data_{}'.format(self.order))

        if self.has_code:
            self.output = \
                named_outputs.get('algorithm',
                                  'algorithm_tmp_{}'.format(self.order))

            self.max_iter = int(parameters.get(self.MAX_ITER_PARAM, 1000) or 1000)
            self.alpha = float(parameters.get(self.ALPHA_PARAM, 0.0001) or 0.0001)
            self.tol = parameters.get(self.TOLERANCE_PARAM, 0.001) or 0.001
            self.tol = abs(float(self.tol))
            self.shuffle = int(parameters.get(self.SHUFFLE_PARAM, 0) or 0)
            self.seed = parameters.get(self.SEED_PARAM, 'None') or 'None'
            self.penalty = parameters.get(
                    self.PENALTY_PARAM,
                    self.PENALTY_PARAM_NONE) or self.PENALTY_PARAM_NONE
            self.fit_intercept = int(parameters.get(self.FIT_INTERCEPT_PARAM, 1) or 1)
            self.verbose = int(parameters.get(self.VERBOSE_PARAM, 0) or 0)
            self.eta0 = float(parameters.get(self.ETA0_PARAM, 1) or 1)
            self.n_jobs = parameters.get(self.N_JOBS_PARAM, None) or None
            self.early_stopping = int(parameters.get(self.EARLY_STOPPING_PARAM, 0) or 0)
            self.validation_fraction = float(parameters.get(self.VALIDATION_FRACTION_PARAM, 0.1) or 0.1)
            self.n_iter_no_change = int(parameters.get(self.N_ITER_NO_CHANGE_PARAM, 5) or 5)
            self.class_weight = parameters.get(self.CLASS_WEIGHT_PARAM, None) or None
            self.warm_start = int(parameters.get(self.WARM_START_PARAM, 0) or 0)
            self.features = parameters['features']
            self.label = parameters.get(self.LABEL_PARAM, None)
            self.prediction = self.parameters.get(self.PREDICTION_PARAM, 'prediction')

            vals = [self.max_iter, self.alpha]
            atts = [self.MAX_ITER_PARAM, self.ALPHA_PARAM]
            for var, att in zip(vals, atts):
                if var <= 0:
                    raise ValueError(
                            _("Parameter '{}' must be x>0 for task {}").format(
                                    att, self.__class__))

            self.has_import = \
                "from sklearn.linear_model import Perceptron\n"

            self.input_treatment()

    @property
    def get_data_out_names(self, sep=','):
        return self.output

    def get_output_names(self, sep=', '):
        return sep.join([self.output, self.model])

    def input_treatment(self):
        if self.n_jobs is not None and self.n_jobs != '0':
            self.n_jobs = int(self.n_jobs)
        else:
            self.n_jobs = None

        self.warm_start = True if int(self.warm_start) == 1 else False
        self.shuffle = True if int(self.shuffle) == 1 else False
        self.early_stopping = True if int(self.early_stopping) == 1 else False

        if self.validation_fraction < 0 or self.validation_fraction > 1:
            raise ValueError(
                _("Parameter '{}' must be 0 <= x =< 1 for task {}").format(
                    self.VALIDATION_FRACTION_PARAM, self.__class__))

    def generate_code(self):
        """Generate code."""
        code = """
            {output_data} = {input_data}.copy()            
            X_train = {input_data}[{features}].values.tolist()
            y = {input_data}[{label}].values.tolist()
            {model} = Perceptron(tol={tol}, alpha={alpha}, max_iter={max_iter}, shuffle={shuffle}, random_state={seed},
                                 penalty='{penalty}', fit_intercept={fit_intercept}, verbose={verbose}, eta0={eta0},
                                 n_jobs={n_jobs}, early_stopping={early_stopping}, 
                                 validation_fraction={validation_fraction}, n_iter_no_change={n_iter_no_change}, 
                                 class_weight={class_weight}, warm_start={warm_start})
            {model}.fit(X_train, y)          
            {output_data}['{prediction}'] = {model}.predict(X_train).tolist()
            """.format(tol=self.tol,
                       alpha=self.alpha,
                       max_iter=self.max_iter,
                       shuffle=self.shuffle,
                       penalty=self.penalty,
                       seed=self.seed,
                       output_data=self.output,
                       model=self.model,
                       input_data=self.input_port,
                       prediction=self.prediction,
                       features=self.features,
                       label=self.label,
                       fit_intercept=self.fit_intercept,
                       verbose=self.verbose,
                       eta0=self.eta0,
                       n_jobs=self.n_jobs,
                       early_stopping=self.early_stopping,
                       validation_fraction=self.validation_fraction,
                       n_iter_no_change=self.n_iter_no_change,
                       class_weight=self.class_weight,
                       warm_start=self.warm_start)
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
        """.format(output=self.output, n_estimators=self.n_estimators,
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

        self.has_code = True
        if self.has_code:
            self.output = \
                named_outputs.get('algorithm',
                                  'algorithm_tmp_{}'.format(self.order))

            self.max_iter = int(parameters.get(self.MAX_ITER_PARAM, -1))
            self.tol = float(parameters.get(self.TOLERANCE_PARAM, 0.001) or 0.001)
            self.tol = abs(float(self.tol))
            self.seed = parameters.get(self.SEED_PARAM, 'None') or 'None'
            self.degree = int(parameters.get(self.DEGREE_PARAM, 3) or 3)
            self.kernel = parameters.get(
                    self.KERNEL_PARAM,
                    self.KERNEL_PARAM_RBF) or self.KERNEL_PARAM_RBF
            self.c = float(parameters.get(self.PENALTY_PARAM, 1.0) or 1.0)

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
        {output} = SVC(tol={tol}, C={c}, max_iter={max_iter}, 
                       degree={degree}, kernel='{kernel}', random_state={seed})
        """.format(tol=self.tol, c=self.c, max_iter=self.max_iter,
                   degree=self.degree, kernel=self.kernel, seed=self.seed,
                   output=self.output)
        return dedent(code)
