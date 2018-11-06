from textwrap import dedent

from juicer.operation import Operation


# TODO: https://spark.apache.org/docs/2.2.0/ml-features.html#vectorassembler
class FeatureAssemblerOperation(Operation):

    ATTRIBUTES_PARAM = 'attributes'
    ALIAS_PARAM = 'alias'

    def __init__(self, parameters,  named_inputs, named_outputs):
        Operation.__init__(self, parameters,  named_inputs,  named_outputs)
        self.has_code = len(self.named_inputs) == 1
        if self.has_code:
            if self.ATTRIBUTES_PARAM not in parameters:
                raise ValueError(
                    _("Parameters '{}' must be informed for task {}")
                    .format(self.ATTRIBUTES_PARAM, self.__class__))

            self.alias = parameters.get(self.ALIAS_PARAM, 'FeatureField')
            self.output = self.named_outputs.get('output data',
                                                 'output_data_{}'.format(
                                                         self.order))

    def generate_code(self):
        code = """
        cols = {cols}
        {output} = {input}
        {output}['{alias}'] = {input}[cols].values.tolist()
        """.format(output=self.output, alias=self.alias,
                   input=self.named_inputs['input data'],
                   cols=self.parameters[self.ATTRIBUTES_PARAM])

        return dedent(code)


class MinMaxScalerOperation(Operation):
    """
    Transforms features by scaling each feature to a given range.

    This estimator scales and translates each feature individually
    such that it is in the given range on the training set, i.e.
    between zero and one.

    This transformation is often used as an alternative to zero mean,
    unit variance scaling.
    """

    ALIAS_PARAM = 'alias'
    ATTRIBUTE_PARAM = 'attribute'
    MIN_PARAM = 'min'
    MAX_PARAM = 'max'

    def __init__(self, parameters,  named_inputs, named_outputs):
        Operation.__init__(self, parameters,  named_inputs,  named_outputs)

        self.has_code = len(self.named_inputs) == 1
        if self.has_code:

            if self.ATTRIBUTE_PARAM not in parameters:
                raise ValueError(
                    _("Parameters '{}' must be informed for task {}")
                    .format(self.ATTRIBUTE_PARAM, self.__class__))

            self.output = self.named_outputs.get(
                    'output data', 'output_data_{}'.format(self.order))
            self.attribute = parameters[self.ATTRIBUTE_PARAM][0]
            self.alias = parameters.get(self.ALIAS_PARAM,
                                        self.attribute+'_norm')

            self.min = parameters.get(self.MIN_PARAM, 0) or 0
            self.max = parameters.get(self.MAX_PARAM, 1) or 1

    def generate_code(self):
        """Generate code."""
        code = """
        {output} = {input}
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler(feature_range=({min},{max}))
        X_train = {input}['{att}'].values.tolist()
        {output}['{alias}'] = scaler.fit_transform(X_train).tolist()
        """.format(output=self.output,
                   input=self.named_inputs['input data'],
                   att=self.attribute, alias=self.alias,
                   min=self.min, max=self.max)
        return dedent(code)


class MaxAbsScalerOperation(Operation):
    """
    Scale each feature by its maximum absolute value.

    This estimator scales and translates each feature individually
    such that the maximal absolute value of each feature in the training
    set will be 1.0. It does not shift/center the data, and thus does not
     destroy any sparsity.
    """

    ALIAS_PARAM = 'alias'
    ATTRIBUTE_PARAM = 'attribute'

    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)

        self.has_code = len(self.named_inputs) == 1
        if self.has_code:

            if self.ATTRIBUTE_PARAM not in parameters:
                raise ValueError(
                        _("Parameters '{}' must be informed for task {}")
                        .format(self.ATTRIBUTE_PARAM, self.__class__))

            self.output = self.named_outputs.get(
                    'output data', 'output_data_{}'.format(self.order))
            self.attribute = parameters[self.ATTRIBUTE_PARAM][0]
            self.alias = parameters.get(self.ALIAS_PARAM,
                                        self.attribute + '_norm')

    def generate_code(self):
        """Generate code."""
        code = """
        {output} = {input}
        from sklearn.preprocessing import MaxAbsScaler
        scaler = MaxAbsScaler()
        X_train = {input}['{att}'].values.tolist()
        {output}['{alias}'] = scaler.fit_transform(X_train).tolist()
        """.format(output=self.output,
                   input=self.named_inputs['input data'],
                   att=self.attribute, alias=self.alias)
        return dedent(code)


class StandardScalerOperation(Operation):
    """
    Standardize features by removing the mean and scaling to unit variance

    Centering and scaling happen independently on each feature by computing the
    relevant statistics on the samples in the training set. Mean and standard
    deviation are then stored to be used on later data using the transform
    method.

    Standardization of a dataset is a common requirement for many machine
    learning estimators: they might behave badly if the individual feature
    do not more or less look like standard normally distributed data.
    """

    ALIAS_PARAM = 'alias'
    ATTRIBUTE_PARAM = 'attribute'

    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)

        self.has_code = len(self.named_inputs) == 1
        if self.has_code:
            if self.ATTRIBUTE_PARAM not in parameters:
                raise ValueError(
                        _("Parameters '{}' must be informed for task {}")
                        .format(self.ATTRIBUTE_PARAM, self.__class__))

            self.output = self.named_outputs.get(
                    'output data', 'output_data_{}'.format(self.order))
            self.attribute = parameters[self.ATTRIBUTE_PARAM][0]
            self.alias = parameters.get(self.ALIAS_PARAM,
                                        self.attribute + '_norm')

    def generate_code(self):
        """Generate code."""
        code = """
        {output} = {input}
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_train = {input}['{att}'].values.tolist()
        {output}['{alias}'] = scaler.fit_transform(X_train).tolist()
        """.format(output=self.output,
                   input=self.named_inputs['input data'],
                   att=self.attribute, alias=self.alias)
        return dedent(code)


class QuantileDiscretizerOperation(Operation):
    """
    Transform features using quantiles information.

    This method transforms the features to follow a uniform or a
    normal distribution. Therefore, for a given feature, this transformation
    tends to spread out the most frequent values.
    """

    ALIAS_PARAM = 'alias'
    ATTRIBUTE_PARAM = 'attribute'
    N_QUANTILES_PARAM = 'n_quantiles'
    DISTRIBUITION_PARAM = 'output_distribution'
    SEED_PARAM = 'seed'

    DISTRIBUITION_PARAM_NORMAL = 'normal'
    DISTRIBUITION_PARAM_UNIFORM = 'uniform'

    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)

        self.has_code = len(self.named_inputs) == 1
        if self.has_code:

            if self.ATTRIBUTE_PARAM not in parameters:
                raise ValueError(
                        _("Parameters '{}' must be informed for task {}")
                        .format('attributes', self.__class__))

            self.output = self.named_outputs.get(
                    'output data', 'output_data_{}'.format(self.order))
            self.attribute = parameters[self.ATTRIBUTE_PARAM][0]
            self.alias = parameters.get(self.ALIAS_PARAM,
                                        self.attribute + '_norm')
            self.n_quantiles = parameters.get(
                    self.N_QUANTILES_PARAM, 1000) or 1000
            self.output_distribution = parameters.get(
                    self.DISTRIBUITION_PARAM, self.DISTRIBUITION_PARAM_UNIFORM)\
                or self.DISTRIBUITION_PARAM_UNIFORM
            self.seed = parameters.get(self.SEED_PARAM, 'None') or 'None'

            if self.n_quantiles <= 0:
                raise ValueError(
                        _("Parameter '{}' must be x>0 for task {}").format(
                                self.N_QUANTILES_PARAM, self.__class__))

    def generate_code(self):
        """Generate code."""
        code = """
        {output} = {input}
        from sklearn.preprocessing import QuantileTransformer
        qt = QuantileTransformer(n_quantiles={n_quantiles},
            output_distribution='{output_distribution}', random_state={seed})
        X_train = {input}['{att}'].values.tolist()
        {output}['{alias}'] = qt.fit_transform(X_train).toarray().tolist()
        """.format(output=self.output,
                   input=self.named_inputs['input data'],
                   att=self.attribute, alias=self.alias, seed=self.seed,
                   n_quantiles=self.n_quantiles,
                   output_distribution=self.output_distribution)
        return dedent(code)


class OneHotEncoderOperation(Operation):
    """
    Encode categorical integer features using a one-hot aka one-of-K scheme.

    This encoding is needed for feeding categorical data to many
    scikit-learn estimators, notably linear models and SVMs with
    the standard kernels.
    """

    ALIAS_PARAM = 'alias'
    ATTRIBUTE_PARAM = 'attributes'

    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)

        self.has_code = len(self.named_inputs) == 1
        if self.has_code:

            if self.ATTRIBUTE_PARAM not in parameters:
                raise ValueError(
                        _("Parameters '{}' must be informed for task {}")
                        .format('attributes', self.__class__))

            self.output = self.named_outputs.get(
                    'output data', 'output_data_{}'.format(self.order))
            self.attribute = parameters[self.ATTRIBUTE_PARAM][0]
            self.alias = parameters.get(self.ALIAS_PARAM,
                                        self.attribute + '_norm')

    def generate_code(self):
        """Generate code."""
        code = """
        {output} = {input}
        from sklearn.preprocessing import OneHotEncoder
        enc = OneHotEncoder()
        X_train = {input}['{att}'].values.tolist()
        {output}['{alias}'] = enc.fit_transform(X_train).toarray().tolist()
        """.format(output=self.output,
                   input=self.named_inputs['input data'],
                   att=self.attribute, alias=self.alias)
        return dedent(code)


class PCAOperation(Operation):
    """
    Principal component analysis (PCA)

    Linear dimensionality reduction using Singular Value
    Decomposition of the data to project it to a lower dimensional space.
    """

    ALIAS_PARAM = 'alias'
    ATTRIBUTE_PARAM = 'attribute'
    N_COMPONENTS = 'k'

    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)

        self.has_code = len(self.named_inputs) == 1
        if self.has_code:

            if self.ATTRIBUTE_PARAM not in parameters:
                raise ValueError(
                        _("Parameters '{}' must be informed for task {}")
                        .format(self.ATTRIBUTE_PARAM, self.__class__))
            self.attributes = parameters[self.ATTRIBUTE_PARAM]
            self.n_components = parameters[self.N_COMPONENTS]

            self.output = self.named_outputs.get(
                    'output data',
                    'output_data_{}'.format(self.order))
            self.alias = parameters.get(self.ALIAS_PARAM, 'pca_feature')
            if self.n_components <= 0:
                raise ValueError(
                        _("Parameter '{}' must be x>0 for task {}").format(
                                self.N_COMPONENTS, self.__class__))

    def generate_code(self):
        """Generate code."""
        code = """
        {output} = {input}
        from sklearn.decomposition import PCA
        pca = PCA(n_components={n_comp})
        X_train = {input}['{att}'].values.tolist()
        {output}['{alias}'] = pca.fit_transform(X_train).tolist()
        """.format(output=self.output,
                   input=self.named_inputs['input data'],
                   att=self.attributes[0], alias=self.alias,
                   n_comp=self.n_components)
        return dedent(code)