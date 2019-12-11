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
            X = {input}['{features}'].to_numpy().tolist()
            y = {input}['{label}'].to_numpy().tolist()
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
            X_train = {input_data}[{columns}].to_numpy().tolist()
            y = {input_data}[{label}].to_numpy().tolist()
            y = np.reshape(y, len(y))
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
    MAX_LEAF_NODES_PARAM = 'max_leaf_nodes'
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
            self.max_leaf_nodes = parameters.get(self.MAX_LEAF_NODES_PARAM, None) or None
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
                """
                import numpy as np
                from sklearn.ensemble import GradientBoostingClassifier
                """

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
        if self.validation_fraction < 0 or self.validation_fraction > 1:
            raise ValueError(
                _("Parameter '{}' must be 0 <= x =< 1 for task {}").format(
                    self.VALIDATION_FRACTION_PARAM, self.__class__))

    def generate_code(self):
        """Generate code."""
        code = """ 
            {output_data} = {input_data}.copy()            
            X_train = {input_data}[{columns}].to_numpy().tolist()
            y = {input_data}[{label}].to_numpy().tolist()
            y = np.reshape(y, len(y))
            {model} = GradientBoostingClassifier(loss='{loss}', learning_rate={learning_rate}, 
                                                  n_estimators={n_estimators}, min_samples_split={min_split},
                                                  max_depth={max_depth}, min_samples_leaf={min_leaf}, 
                                                  random_state={seed}, subsample={subsample}, criterion='{criterion}',
                                                  min_weight_fraction_leaf={min_weight_fraction_leaf}, 
                                                  min_impurity_decrease={min_impurity_decrease}, init={init},
                                                  max_features={max_features}, verbose=False, 
                                                  max_leaf_nodes={max_leaf_nodes}, warm_start=False, 
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
                       max_leaf_nodes=self.max_leaf_nodes,
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
                """
                import numpy as np
                from sklearn.neighbors import KNeighborsClassifier
                """

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
            X_train = {input_data}[{features}].to_numpy().tolist()
            y = {input_data}[{label}].to_numpy().tolist()
            y = np.reshape(y, len(y))
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
    BATCH_SIZE_PARAM = 'batch_size'
    LEARNING_RATE_PRAM = 'learning_rate'
    LEARNING_RATE_INIT_PRAM = 'learning_rate_init'
    POWER_T_PARAM = 'power_t'
    SHUFFLE_PARAM = 'shuffle'
    MOMENTUM_PARAM = 'momentum'
    NESTEROVS_MOMENTUM_PARAM = 'nesterovs_momentum'
    EARLY_STOPPING_PARAM = 'early_stopping'
    VALIDATION_FRACTION_PARAM = 'validation_fraction'
    BETA_1_PARAM = 'beta1'
    BETA_2_PARAM = 'beta2'
    EPSILON_PARAM = 'epsilon'
    N_ITER_NO_CHANGE_PARAM = 'n_iter_no_change'
    MAX_FUN_PARAM = 'max_fun'
    FEATURES_PARAM = 'features'
    LABEL_PARAM = 'label'
    PREDICTION_PARAM = 'prediction'

    SOLVER_PARAM_ADAM = 'adam'
    SOLVER_PARAM_LBFGS = 'lbfgs'
    SOLVER_PARAM_SGD = 'sgd'

    ACTIVATION_PARAM_ID = 'identity'
    ACTIVATION_PARAM_LOG = 'logistic'
    ACTIVATION_PARAM_TANH = 'tanh'
    ACTIVATION_PARAM_RELU = 'relu'

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
            self.add_functions_required = ""
            self.hidden_layers = parameters.get(self.HIDDEN_LAYER_SIZES_PARAM, '(1,100,1)') or '(1,100,1)'
            self.hidden_layers = \
                self.hidden_layers.replace("(", "").replace(")", "")
            self.activation = parameters.get(
                    self.ACTIVATION_PARAM,
                    self.ACTIVATION_PARAM_RELU) or self.ACTIVATION_PARAM_RELU
            self.solver = parameters.get(
                    self.SOLVER_PARAM,
                    self.SOLVER_PARAM_ADAM) or self.SOLVER_PARAM_ADAM
            self.alpha = parameters.get(self.ALPHA_PARAM, 0.0001) or 0.0001
            self.alpha = abs(float(self.alpha))
            self.max_iter = parameters.get(self.MAX_ITER_PARAM, 200) or 200
            self.max_iter = abs(int(self.max_iter))
            self.tol = parameters.get(self.TOLERANCE_PARAM, 0.0001) or 0.0001
            self.tol = abs(float(self.tol))
            self.seed = parameters.get(self.SEED_PARAM, None) or None

            self.batch_size = parameters.get(self.BATCH_SIZE_PARAM, 'auto') or 'auto'
            self.learning_rate = parameters.get(self.LEARNING_RATE_PRAM, 'constant') or 'constant'
            self.learning_rate_init = float(parameters.get(self.LEARNING_RATE_INIT_PRAM, 0.001) or 0.001)
            self.power_t = float(parameters.get(self.POWER_T_PARAM, 0.5) or 0.5)
            self.shuffle = int(parameters.get(self.SHUFFLE_PARAM, 1) or 1)
            self.momentum = float(parameters.get(self.MOMENTUM_PARAM, 0.9) or 0.9)
            self.nesterovs_momentum = int(parameters.get(self.NESTEROVS_MOMENTUM_PARAM, 1) or 1)
            self.early_stopping = int(parameters.get(self.EARLY_STOPPING_PARAM, 0) or 0)
            self.validation_fraction = float(parameters.get(self.VALIDATION_FRACTION_PARAM, 0.1) or 0.1)
            self.beta1 = float(parameters.get(self.BETA_1_PARAM, 0.9) or 0.9)
            self.beta2 = float(parameters.get(self.BETA_2_PARAM, 0.999) or 0.999)
            self.epsilon = float(parameters.get(self.EPSILON_PARAM, 1e-8) or 1e-8)
            self.n_iter_no_change = int(parameters.get(self.N_ITER_NO_CHANGE_PARAM, 10) or 10)
            self.max_fun = int(parameters.get(self.MAX_FUN_PARAM, 15000) or 15000)
            self.features = parameters['features']
            self.label = parameters.get(self.LABEL_PARAM, None)
            self.prediction = self.parameters.get(self.PREDICTION_PARAM, 'prediction')

            self.has_import = \
                """
                import numpy as np
                from sklearn.neural_network import MLPClassifier
                """

            self.input_treatment()

    @property
    def get_data_out_names(self, sep=','):
        return self.output

    def get_output_names(self, sep=', '):
        return sep.join([self.output, self.model])

    def input_treatment(self):
        self.shuffle = True if int(self.shuffle) == 1 else False
        self.nesterovs_momentum = True if int(self.nesterovs_momentum) == 1 else False
        self.early_stopping = True if int(self.early_stopping) == 1 else False

        if self.seed is not None:
            self.seed = int(self.seed)

        if self.batch_size != 'auto':
            self.batch_size = int(self.batch_size)

        if self.momentum < 0 or self.momentum > 1:
            raise ValueError(
                _("Parameter '{}' must be x between 0 and 1 for task {}").format(
                    self.MOMENTUM_PARAM, self.__class__))

        if self.validation_fraction < 0 or self.validation_fraction > 1:
            raise ValueError(
                _("Parameter '{}' must be x between 0 and 1 for task {}").format(
                    self.VALIDATION_FRACTION_PARAM, self.__class__))

        if self.beta1 < 0 or self.beta1 >= 1:
            raise ValueError(
                _("Parameter '{}' must be in [0, 1) for task {}").format(
                    self.BETA_1_PARAM, self.__class__))

        if self.beta2 < 0 or self.beta2 >= 1:
            raise ValueError(
                _("Parameter '{}' must be in [0, 1) for task {}").format(
                    self.BETA_2_PARAM, self.__class__))

        if not bool(re.match('(\d+,)+\d*', self.hidden_layers)):
            raise ValueError(
                _("Parameter '{}' must be a tuple with the size "
                  "of each layer for task {}").format(
                    self.HIDDEN_LAYER_SIZES_PARAM, self.__class__))

        functions_required = ["""hidden_layers_sizes={hidden_layers}""".format(hidden_layers=self.hidden_layers)]

        self.activation = """activation='{activation}'""".format(activation=self.activation)
        functions_required.append(self.activation)

        self.solver = """solver='{solver}'""".format(solver=self.solver)
        functions_required.append(self.solver)

        self.alpha = """alpha={alpha}""".format(alpha=self.alpha)
        functions_required.append(self.alpha)

        self.max_iter = """max_iter={max_iter}""".format(max_iter=self.max_iter)
        functions_required.append(self.max_iter)

        self.tol = """tol={tol}""".format(tol=self.tol)
        functions_required.append(self.tol)

        self.seed = """seed={seed}""".format(seed=self.seed)
        functions_required.append(self.seed)

        if self.solver != 'lbfgs':
            self.batch_size = """batch_size='{batch_size}'""".format(batch_size=self.batch_size)
            functions_required.append(self.batch_size)

        if self.solver == 'sgd':
            self.learning_rate = """learning_rate='{learning_rate}'""".format(learning_rate=self.learning_rate)
            functions_required.append(self.learning_rate)

            self.momentum = """momentum={momentum}""".format(momentum=self.momentum)
            functions_required.append(self.momentum)

            self.power_t = """power_t={power_t}""".format(power_t=self.power_t)
            functions_required.append(self.power_t)

            if self.momentum > 0:
                self.nesterovs_momentum = """nesterovs_momentum={nesterovs_momentum}""".format(
                    nesterovs_momentum=self.nesterovs_momentum)
                functions_required.append(self.nesterovs_momentum)
        if self.solver == 'sgd' or self.solver == 'adam':
            self.learning_rate_init = """learning_rate_init={learning_rate_init}""".format(
                learning_rate_init=self.learning_rate_init)
            functions_required.append(self.learning_rate_init)

            self.shuffle = """shuffle={shuffle}""".format(shuffle=self.shuffle)
            functions_required.append(self.shuffle)

            self.early_stopping = """early_stopping={early_stopping}""".format(early_stopping=self.early_stopping)
            functions_required.append(self.early_stopping)

            self.n_iter_no_change = """n_iter_no_change={n_iter_no_change}""".format(
                n_iter_no_change=self.n_iter_no_change)
            functions_required.append(self.n_iter_no_change)
        if self.early_stopping == 1:
            self.validation_fraction = """validation_fraction={validation_fraction}""".format(
                validation_fraction=self.validation_fraction)
            functions_required.append(self.validation_fraction)
        if self.solver == 'adam':
            self.beta1 = """beta1={beta1}""".format(beta1=self.beta1)
            functions_required.append(self.beta1)

            self.beta2 = """beta2={beta2}""".format(beta2=self.beta2)
            functions_required.append(self.beta2)

            self.epsilon = """epsilon={epsilon}""".format(epsilon=self.epsilon)
            functions_required.append(self.epsilon)
        if self.solver == 'lbfgs':
            self.max_fun = """max_fun={max_fun}""".format(max_fun=self.max_fun)
            functions_required.append(self.max_fun)

    def generate_code(self):
        """Generate code."""
        code = """
            {output_data} = {input_data}.copy()            
            X_train = {input_data}[{columns}].to_numpy().tolist()
            y = {input_data}[{label}].to_numpy().tolist()
            y = np.reshape(y, len(y))
            {model} = MLPClassifier({add_functions_required})
            {model}.fit(X_train, y)          
            {output_data}['{prediction}'] = {model}.predict(X_train).tolist()
            """.format(output_data=self.output,
                       prediction=self.prediction,
                       columns=self.features,
                       model=self.model,
                       input_data=self.input_port,
                       label=self.label,
                       add_functions_required=self.add_functions_required)
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

            if self.model_type == self.MODEL_TYPE_PARAM_M:
                self.has_import = \
                    """
                    import numpy as np
                    from sklearn.naive_bayes import MultinomialNB
                    """
            elif self.model_type == self.MODEL_TYPE_PARAM_B:
                self.has_import = \
                    """
                    import numpy as np
                    from sklearn.naive_bayes import BernoulliNB
                    """
            else:
                self.has_import = \
                    """
                    import numpy as np
                    from sklearn.naive_bayes import GaussianNB
                    """
            self.input_treatment()

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

        if self.smoothing <= 0:
            raise ValueError(
                _("Parameter '{}' must be x>0 for task {}").format(
                    'smoothing', self.__class__))

    def generate_code(self):
        """Generate code."""
        if self.model_type == self.MODEL_TYPE_PARAM_M:
            code = """
                {output_data} = {input_data}.copy()            
                X_train = {input_data}[{features}].to_numpy().tolist()
                y = {input_data}[{label}].to_numpy().tolist()
                y = np.reshape(y, len(y))
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
                X_train = {input_data}[{features}].to_numpy().tolist()
                y = {input_data}[{label}].to_numpy().tolist()
                y = np.reshape(y, len(y))
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
                X_train = {input_data}[{features}].to_numpy().tolist()
                y = {input_data}[{label}].to_numpy().tolist()
                y = np.reshape(y, len(y))
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
            """
            import numpy as np
            from sklearn.linear_model import Perceptron
            """

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
            X_train = {input_data}[{features}].to_numpy().tolist()
            y = {input_data}[{label}].to_numpy().tolist()
            y = np.reshape(y, len(y))
            if {early_stopping} == 1:
                {model} = Perceptron(tol={tol}, alpha={alpha}, max_iter={max_iter}, shuffle={shuffle}, 
                                      random_state={seed}, penalty='{penalty}', fit_intercept={fit_intercept}, 
                                      verbose={verbose}, eta0={eta0}, n_jobs={n_jobs}, early_stopping={early_stopping}, 
                                      validation_fraction={validation_fraction}, n_iter_no_change={n_iter_no_change}, 
                                      class_weight={class_weight}, warm_start=False)
            else:
                {model} = Perceptron(tol={tol}, alpha={alpha}, max_iter={max_iter}, shuffle={shuffle}, 
                                      random_state={seed}, penalty='{penalty}', fit_intercept={fit_intercept}, 
                                      verbose={verbose}, eta0={eta0}, n_jobs={n_jobs}, early_stopping={early_stopping}, 
                                      n_iter_no_change={n_iter_no_change}, class_weight={class_weight}, 
                                      warm_start=False)
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
                       class_weight=self.class_weight)
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
