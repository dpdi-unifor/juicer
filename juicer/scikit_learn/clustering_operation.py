# -*- coding: utf-8 -*-
from textwrap import dedent
from juicer.operation import Operation


class ClusteringModelOperation(Operation):
    FEATURES_PARAM = 'features'
    ALIAS_PARAM = 'prediction'

    def __init__(self, parameters,  named_inputs, named_outputs):
        Operation.__init__(self, parameters,  named_inputs,  named_outputs)

        self.has_code = len(self.named_inputs) == 2
        if self.has_code:

            if self.FEATURES_PARAM in parameters:
                self.features = parameters.get(self.FEATURES_PARAM)[0]
            else:
                raise \
                    ValueError(_("Parameter '{}' must be informed for task {}")
                               .format(self.FEATURES_PARAM, self.__class__))

            self.model = self.named_outputs.get('model',
                                                'model_{}'.format(self.output))

            self.perform_transformation = 'output data' in self.named_outputs
            if not self.perform_transformation:
                self.output = 'task_{}'.format(self.order)
            else:
                self.output = self.named_outputs['output data']
                self.alias = parameters.get(self.ALIAS_PARAM, 'prediction')

    @property
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
        {model} = {algorithm}.fit(X)
        """.format(model=self.model, features=self.features,
                   input=self.named_inputs['train input data'],
                   algorithm=self.named_inputs['algorithm'])

        if self.perform_transformation:
            code += """
        y = {algorithm}.predict(X)
        {OUT} = {IN}
        {OUT}['{predCol}'] = y
        """.format(OUT=self.output, model=self.model,
                   IN=self.named_inputs['train input data'],
                   predCol=self.alias, algorithm=self.named_inputs['algorithm'])
        else:
            code += """
        {output} = None
        """.format(output=self.output)

        return dedent(code)


class AgglomerativeClusteringOperation(Operation):
    FEATURES_PARAM = 'attributes'
    ALIAS_PARAM = 'alias'

    N_CLUSTERS_PARAM = 'number_of_clusters'
    LINKAGE_PARAM = 'linkage'
    AFFINITY_PARAM = 'affinity'

    AFFINITY_PARAM_EUCL = 'euclidean'
    AFFINITY_PARAM_L1 = 'l1'
    AFFINITY_PARAM_L2 = 'l2'
    AFFINITY_PARAM_MA = 'manhattan'
    AFFINITY_PARAM_COS = 'cosine'

    LINKAGE_PARAM_WARD = 'ward'
    LINKAGE_PARAM_COMP = 'complete'
    LINKAGE_PARAM_AVG = 'average'

    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)

        self.has_code = len(named_inputs) > 0 and any([self.contains_results(),
                                                       len(named_outputs) > 0])
        if self.has_code:
            self.output = named_outputs.get(
                    'output data', 'output_data_{}'.format(self.order))

            if self.FEATURES_PARAM not in parameters:
                raise \
                    ValueError(_("Parameter '{}' must be informed for task {}")
                               .format(self.FEATURES_PARAM, self.__class__))

            self.features = parameters.get(self.FEATURES_PARAM)[0]
            self.alias = parameters.get(self.ALIAS_PARAM, 'cluster')

            self.n_clusters = parameters.get(self.N_CLUSTERS_PARAM, 2) or 2
            self.linkage = parameters.get(
                    self.LINKAGE_PARAM,
                    self.LINKAGE_PARAM_WARD) or self.LINKAGE_PARAM_WARD
            self.affinity = parameters.get(
                    self.AFFINITY_PARAM,
                    self.AFFINITY_PARAM_EUCL) or self.AFFINITY_PARAM_EUCL

            if self.n_clusters <= 0:
                raise ValueError(
                        _("Parameter '{}' must be x>0 for task {}").format(
                                self.N_CLUSTERS_PARAM, self.__class__))

            self.has_import = \
                "from sklearn.cluster import AgglomerativeClustering\n"

    def generate_code(self):
        """Generate code."""
        code = """
        {output} = {input}.copy()
        
        X = {output}['{features}'].values.tolist()
        clt = AgglomerativeClustering(n_clusters={n_clusters},
            linkage='{linkage}', affinity='{affinity}')
        {output}['{alias}'] = clt.fit_predict(X)
        """.format(input=self.named_inputs['input data'], output=self.output,
                   features=self.features, alias=self.alias,
                   n_clusters=self.n_clusters,
                   affinity=self.affinity, linkage=self.linkage)

        return dedent(code)


class DBSCANClusteringOperation(Operation):
    EPS_PARAM = 'eps'
    MIN_SAMPLES_PARAM = 'min_samples'
    FEATURES_PARAM = 'features'
    PREDICTION_PARAM = 'prediction'
    METRIC_PARAM = 'metric'
    METRIC_PARAMS_PARAM = 'metric_params'
    ALGORITHM_PARAM = 'algorithm'
    LEAF_SIZE_PARAM = 'leaf_size'
    P_PARAM = 'p'
    N_JOBS_PARAM = 'n_jobs'

    def __init__(self, parameters, named_inputs,
                 named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)

        self.has_code = any([len(self.named_inputs) == 1, self.contains_results()])
        self.output = self.named_outputs.get(
            'output data', 'output_data_{}'.format(self.order))

        self.model = self.named_outputs.get(
            'model', 'model_{}'.format(self.order))

        self.input_port = self.named_inputs.get(
            'train input data', 'input_data_{}'.format(self.order))
        if self.has_code:
            self.eps = float(parameters.get(self.EPS_PARAM, 0.5) or 0.5)
            self.min_samples = int(parameters.get(self.MIN_SAMPLES_PARAM, 5) or 5)
            self.features = parameters['features']
            self.prediction = self.parameters.get(self.PREDICTION_PARAM, 'prediction')
            self.metric = parameters.get(self.METRIC_PARAM, 'euclidean')
            self.metric_params = parameters.get(self.METRIC_PARAMS_PARAM, None) #DICT
            self.algorithm = parameters.get(self.ALGORITHM_PARAM, 'auto')
            self.leaf_size = int(parameters.get(self.LEAF_SIZE_PARAM, 30) or 30)
            self.p = parameters.get(self.P_PARAM, None)
            self.n_jobs = parameters.get(self.N_JOBS_PARAM, None)

            if self.FEATURES_PARAM in parameters:
                self.features = parameters.get(self.FEATURES_PARAM)[0]
            else:
                raise \
                    ValueError(_("Parameter '{}' must be informed for task {}")
                               .format(self.FEATURES_PARAM, self.__class__))
            self.prediction = parameters.get(self.PREDICTION_PARAM, 'cluster')

            vals = [self.eps, self.min_samples]
            atts = [self.EPS_PARAM, self.MIN_SAMPLES_PARAM]
            for var, att in zip(vals, atts):
                if var <= 0:
                    raise ValueError(
                            _("Parameter '{}' must be x>0 for task {}").format(
                                    att, self.__class__))

            self.has_import = \
                "from sklearn.cluster import DBSCAN\n"
            self.input_treatment()

    @property
    def get_data_out_names(self, sep=','):
        return self.output

    def get_output_names(self, sep=', '):
        return sep.join([self.output, self.model])

    def input_treatment(self):
        if self.p is not None and self.p != '0':
            self.p = float(self.p)
        else:
            self.p = None

        if self.n_jobs is not None and self.n_jobs != '0':
            self.n_jobs = int(self.n_jobs)
        else:
            self.n_jobs = None

    def generate_code(self):
        """Generate code."""
        code = """
        {output_data} = {input_data}.copy()
        print({columns})
        X_train = {input_data}[{columns}].to_numpy().tolist()
        if '{algorithm}' == 'kd_tree' or '{algorithm}' == 'ball_tree':
            {model} = DBSCAN(eps={eps}, min_samples={min_samples}, metric='{metric}', metric_params={metric_params}, 
                             algorithm='{algorithm}', leaf_size={leaf_size}, p={p}, n_jobs={n_jobs})
        else:
            {model} = DBSCAN(eps={eps}, min_samples={min_samples}, metric='{metric}', metric_params={metric_params}, 
                             algorithm='{algorithm}', p={p}, n_jobs={n_jobs})
        {output_data}['{prediction}'] = {model}.fit_predict(X_train)
        """.format(eps=self.eps,
                   min_samples=self.min_samples,
                   output_data=self.output,
                   model=self.model,
                   input_data=self.input_port,
                   prediction=self.prediction,
                   columns=self.features,
                   metric=self.metric,
                   metric_params=self.metric_params,
                   algorithm=self.algorithm,
                   leaf_size=self.leaf_size,
                   p=self.p,
                   n_jobs=self.n_jobs)

        return dedent(code)


class GaussianMixtureClusteringOperation(Operation):
    MAX_ITER_PARAM = 'max_iter'
    TOL_PARAM = 'tol'
    PREDICTION_PARAM = 'prediction'
    FEATURES_PARAM = 'features'
    N_COMPONENTS_PARAM = 'n_components'
    COVARIANCE_TYPE_PARAM = 'covariance_type'
    REG_COVAR_PARAM = 'reg_covar'
    N_INIT_PARAM = 'n_init'
    INIT_PARAMS_PARAM = 'init_params'
    WEIGHTS_INIT_PARAM = 'weights_init'
    MEANS_INIT_PARAM = 'means_init'
    PRECISIONS_INIT_PARAM = 'precisions_init'
    RANDOM_STATE_PARAM = 'random_state'

    def __init__(self, parameters, named_inputs,
                 named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)

        self.has_code = any([len(self.named_inputs) == 1, self.contains_results()])
        self.output = self.named_outputs.get(
            'output data', 'output_data_{}'.format(self.order))

        self.model = self.named_outputs.get(
            'model', 'model_{}'.format(self.order))

        self.input_port = self.named_inputs.get(
            'train input data', 'input_data_{}'.format(self.order))
        if self.has_code:
            self.features = parameters['features']
            self.prediction = self.parameters.get(self.PREDICTION_PARAM, 'prediction')
            self.n_components = int(parameters.get(self.N_COMPONENTS_PARAM, 1) or 1)
            self.covariance_type = parameters.get(self.COVARIANCE_TYPE_PARAM, 'full')
            self.tol = float(parameters.get(self.TOL_PARAM, 0.001) or 0.001)
            self.tol = abs(self.tol)
            self.reg_covar = float(parameters.get(self.REG_COVAR_PARAM, 0.000001))
            self.max_iter = int(parameters.get(self.MAX_ITER_PARAM, 100) or 100)
            self.n_init = int(parameters.get(self.N_INIT_PARAM, 1))
            self.init_params = parameters.get(self.INIT_PARAMS_PARAM, 'kmeans')
            self.weights_init = parameters.get(self.WEIGHTS_INIT_PARAM, None)
            self.means_init = parameters.get(self.MEANS_INIT_PARAM, None)
            self.precisions_init = parameters.get(self.PRECISIONS_INIT_PARAM, None)
            self.random_state = parameters.get(self.RANDOM_STATE_PARAM, None)

            vals = [self.n_components, self.max_iter]
            atts = [self.N_COMPONENTS_PARAM, self.MAX_ITER_PARAM]
            for var, att in zip(vals, atts):
                if var <= 0:
                    raise ValueError(
                            _("Parameter '{}' must be x>0 for task {}").format(
                                    att, self.__class__))

            self.has_import = \
                "from sklearn.mixture import GaussianMixture\n"
            self.input_treatment()

    @property
    def get_data_out_names(self, sep=','):
        return self.output

    def get_output_names(self, sep=', '):
        return sep.join([self.output, self.model])

    def input_treatment(self):
        if self.random_state is not None and self.random_state != '0':
            self.random_state = int(self.random_state)
        else:
            self.random_state = None

    def generate_code(self):
        """Generate code."""
        code = """
        {output_data} = {input_data}.copy()
        X_train = {input_data}[{columns}].to_numpy().tolist()
        {model} = GaussianMixture(n_components={k}, max_iter={iter}, tol={tol}, covariance_type='{covariance_type}', 
                                  reg_covar={reg_covar}, n_init={n_init}, init_params='{init_params}', 
                                  weights_init={weights_init}, means_init={means_init}, 
                                  precisions_init={precisions_init}, random_state={random_state})
        {output_data}['{prediction}'] = {model}.fit_predict(X_train)
        """.format(k=self.n_components,
                   iter=self.max_iter,
                   tol=self.tol,
                   output_data=self.output,
                   model=self.model,
                   input_data=self.input_port,
                   prediction=self.prediction,
                   columns=self.features,
                   covariance_type=self.covariance_type,
                   reg_covar=self.reg_covar,
                   n_init=self.n_init,
                   init_params=self.init_params,
                   weights_init=self.weights_init,
                   means_init=self.means_init,
                   precisions_init=self.precisions_init,
                   random_state=self.random_state)

        return dedent(code)


class KMeansClusteringOperation(Operation):

    N_CLUSTERS_PARAM = 'n_clusters'
    INIT_PARAM = 'init'
    MAX_ITER_PARAM = 'max_iter'
    TOLERANCE_PARAM = 'tolerance'
    TYPE_PARAM = 'type'
    SEED_PARAM = 'seed'
    N_INIT_PARAM = 'n_init'
    N_INIT_MB_PARAM = 'n_init_mb'
    PRECOMPUTE_DISTANCES_PARAM = 'precompute_distances'
    VERBOSE_PARAM = 'verbose'
    COPY_X_PARAM = 'copy_x'
    N_JOBS_PARAM = 'n_jobs'
    ALGORITHM_PARAM = 'algorithm'
    BATCH_SIZE_PARAM = 'batch_size'
    COMPUTE_LABELS_PARAM = 'compute_labels'
    TOL_PARAM = 'tol'
    MAX_NO_IMPROVEMENT_PARAM = 'max_no_improvement'
    INIT_SIZE_PARAM = 'init_size'
    REASSIGNMENT_RATIO_PARAM = 'reassignment_ratio'
    PREDICTION_PARAM = 'prediction'
    FEATURES_PARAM = 'features'

    INIT_PARAM_RANDOM = 'random'
    INIT_PARAM_KM = 'K-Means++'
    TYPE_PARAM_KMEANS = 'K-Means'
    TYPE_PARAM_MB = 'Mini-Batch K-Means'

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
            self.features = parameters['features']
            self.prediction = self.parameters.get(self.PREDICTION_PARAM, 'prediction')
            self.n_init = int(parameters.get(self.N_INIT_PARAM, 10) or 10)
            self.precompute_distances = parameters.get(self.PRECOMPUTE_DISTANCES_PARAM, 'auto') or 'auto'
            self.copy_x = int(parameters.get(self.COPY_X_PARAM, 1) or 1)
            self.n_jobs = parameters.get(self.N_JOBS_PARAM, None) or None
            self.algorithm = parameters.get(self.ALGORITHM_PARAM, 'auto') or 'auto'
            self.verbose = int(parameters.get(self.VERBOSE_PARAM, 0) or 0)
            self.n_init_mb = int(parameters.get(self.N_INIT_MB_PARAM, 3) or 3)
            self.reassignment_ratio = float(parameters.get(self.REASSIGNMENT_RATIO_PARAM, 0.01) or 0.01)
            self.tol = float(parameters.get(self.TOL_PARAM, 0.0) or 0.0)
            self.compute_labels = int(parameters.get(self.COMPUTE_LABELS_PARAM, 1) or 1)
            self.max_no_improvement = int(parameters.get(self.MAX_NO_IMPROVEMENT_PARAM, 10) or 10)
            self.init_size = parameters.get(self.INIT_SIZE_PARAM, None) or None
            self.batch_size = int(parameters.get(self.BATCH_SIZE_PARAM, 100) or 100)
            self.n_clusters = int(parameters.get(self.N_CLUSTERS_PARAM, 8) or 8)
            self.max_iter = int(parameters.get(self.MAX_ITER_PARAM, 100) or 100)
            self.init_mode = parameters.get(
                    self.INIT_PARAM, self.INIT_PARAM_KM) or self.INIT_PARAM_KM
            self.init_mode = self.init_mode.lower()
            self.tolerance = parameters.get(self.TOLERANCE_PARAM, 1e-4)
            self.tolerance = abs(float(self.tolerance))
            self.seed = parameters.get(self.SEED_PARAM, None) or None
            self.type = parameters.get(
                    self.TYPE_PARAM,
                    self.TYPE_PARAM_KMEANS) or self.TYPE_PARAM_KMEANS

            vals = [self.n_clusters, self.max_iter]
            atts = [self.N_CLUSTERS_PARAM, self.MAX_ITER_PARAM]
            for var, att in zip(vals, atts):
                if var <= 0:
                    raise ValueError(
                            _("Parameter '{}' must be x>0 for task {}").format(
                                    att, self.__class__))

            if self.type.lower() == "k-means":
                self.has_import = \
                    "from sklearn.cluster import KMeans\n"
            else:
                self.has_import = \
                    "from sklearn.cluster import MiniBatchKMeans\n"
            self.input_treatment()

    @property
    def get_data_out_names(self, sep=','):
        return self.output

    def get_output_names(self, sep=', '):
        return sep.join([self.output, self.model])

    def input_treatment(self):
        self.copy_x = True if int(self.copy_x) == 1 else False

        self.compute_labels = True if int(self.compute_labels) == 1 else False

        if self.seed is not None and self.seed != '0':
            self.seed = int(self.seed)
        else:
            self.seed = None

        if self.n_jobs is not None and self.n_jobs != '0':
            self.n_jobs = int(self.n_jobs)
        else:
            self.n_jobs = None

        if self.init_size is not None and self.init_size != '0':
            self.init_size = int(self.init_size)
        else:
            self.init_size = None

        if self.precompute_distances == 'true':
            self.precompute_distances = True
        elif self.precompute_distances == 'false':
            self.precompute_distances = False

    def generate_code(self):
        """Generate code."""
        if self.type.lower() == "k-means":
            code = """
            {output_data} = {input_data}.copy()
            X_train = {input_data}[{columns}].to_numpy().tolist()
            if '{precompute_distances}' == 'auto':
                {model} = KMeans(n_clusters={k}, init='{init}', max_iter={max_iter}, tol={tol}, random_state={seed}, 
                                  n_init={n_init}, precompute_distances='{precompute_distances}', copy_x={copy_x}, 
                                  n_jobs={n_jobs}, algorithm='{algorithm}', verbose={verbose})
            else:
                {model} = KMeans(n_clusters={k}, init='{init}', max_iter={max_iter}, tol={tol}, random_state={seed}, 
                                  n_init={n_init}, precompute_distances={precompute_distances}, copy_x={copy_x}, 
                                  n_jobs={n_jobs}, algorithm='{algorithm}', verbose={verbose})
            {output_data}['{prediction}'] = {model}.fit_predict(X_train)
            """.format(k=self.n_clusters,
                       max_iter=self.max_iter,
                       tol=self.tolerance,
                       init=self.init_mode,
                       output_data=self.output,
                       seed=self.seed,
                       model=self.model,
                       input_data=self.input_port,
                       prediction=self.prediction,
                       columns=self.features,
                       n_init=self.n_init,
                       precompute_distances=self.precompute_distances,
                       copy_x=self.copy_x,
                       n_jobs=self.n_jobs,
                       algorithm=self.algorithm,
                       verbose=self.verbose)
        else:
            code = """
            {output_data} = {input_data}.copy()
            X_train = {input_data}[{columns}].to_numpy().tolist()
            {model} = MiniBatchKMeans(n_clusters={k}, init='{init}', max_iter={max_iter}, tol={tol}, 
                                       random_state={seed}, verbose={verbose}, n_init={n_init}, 
                                       reassignment_ratio={reassignment_ratio}, compute_labels={compute_labels}, 
                                       max_no_improvement={max_no_improvement}, init_size={init_size}, 
                                       batch_size={batch_size})
            {output_data}['{prediction}'] = {model}.fit_predict(X_train)
            """.format(k=self.n_clusters,
                       max_iter=self.max_iter,
                       tol=self.tol,
                       init=self.init_mode,
                       output_data=self.output,
                       seed=self.seed,
                       model=self.model,
                       input_data=self.input_port,
                       prediction=self.prediction,
                       columns=self.features,
                       verbose=self.verbose,
                       n_init=self.n_init_mb,
                       reassignment_ratio=self.reassignment_ratio,
                       compute_labels=self.compute_labels,
                       max_no_improvement=self.max_no_improvement,
                       init_size=self.init_size,
                       batch_size=self.batch_size)
        return dedent(code)


class LdaClusteringOperation(Operation):
    N_COMPONENTES_PARAM = 'n_components'
    ALPHA_PARAM = 'doc_topic_pior'
    ETA_PARAM = 'topic_word_prior'
    LEARNING_METHOD_PARAM = 'learning_method'
    MAX_ITER_PARAM = 'max_iter'
    SEED_PARAM = 'seed'

    LEARNING_METHOD_ON = 'online'
    LEARNING_METHOD_BA = 'batch'

    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)

        self.has_code = len(named_outputs) > 0
        if self.has_code:
            self.output = named_outputs.get(
                    'algorithm', 'clustering_algorithm_{}'.format(self.order))
            self.n_clusters = parameters.get(
                    self.N_COMPONENTES_PARAM, 10) or self.N_COMPONENTES_PARAM
            self.learning_method = parameters.get(
                    self.LEARNING_METHOD_PARAM,
                    self.LEARNING_METHOD_ON) or self.LEARNING_METHOD_ON
            self.max_iter = parameters.get(self.MAX_ITER_PARAM, 10) or 10

            self.doc_topic_pior = \
                parameters.get(self.ALPHA_PARAM, 'None') or 'None'
            self.topic_word_prior = parameters.get(self.ETA_PARAM,
                                                   'None') or 'None'

            self.seed = parameters.get(self.SEED_PARAM, 'None') or 'None'

            if self.learning_method not in [self.LEARNING_METHOD_ON,
                                            self.LEARNING_METHOD_BA]:
                raise ValueError(
                    _('Invalid optimizer value {} for class {}').format(
                        self.learning_method, self.__class__))

            vals = [self.n_clusters, self.max_iter]
            atts = [self.N_COMPONENTES_PARAM, self.MAX_ITER_PARAM]
            for var, att in zip(vals, atts):
                if var <= 0:
                    raise ValueError(
                            _("Parameter '{}' must be x>0 for task {}").format(
                                    att, self.__class__))

            self.has_import = \
                "from sklearn.decomposition import LatentDirichletAllocation\n"

    def generate_code(self):
        """Generate code."""
        code = """
        {output} = LatentDirichletAllocation(n_components={n_components}, 
        doc_topic_prior={doc_topic_prior}, topic_word_prior={topic_word_prior}, 
        learning_method='{learning_method}', max_iter={max_iter})
        """.format(n_components=self.n_clusters, max_iter=self.max_iter,
                   doc_topic_prior=self.doc_topic_pior,
                   topic_word_prior=self.topic_word_prior,
                   learning_method=self.learning_method,
                   output=self.output)

        return dedent(code)