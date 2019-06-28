# -*- coding: utf-8 -*-
from gettext import gettext
from textwrap import dedent
from juicer.operation import Operation
import re
from ast import parse
from juicer.util.template_util import *


class Convolution1D(Operation):
    FILTERS_PARAM = 'filters'
    KERNEL_SIZE_PARAM = 'kernel_size'
    STRIDES_PARAM = 'strides'
    INPUT_SHAPE_PARAM = 'input_shape'
    PADDING_PARAM = 'padding'
    DATA_FORMAT_PARAM = 'data_format'
    DILATION_RATE_PARAM = 'dilation_rate'
    ACTIVATION_PARAM = 'activation'
    USE_BIAS_PARAM = 'use_bias'
    TRAINABLE_PARAM = 'trainable'
    KERNEL_INITIALIZER_PARAM = 'kernel_initializer'
    BIAS_INITIALIZER_PARAM = 'bias_initializer'
    KERNEL_REGULARIZER_PARAM = 'kernel_regularizer'
    BIAS_REGULARIZER_PARAM = 'bias_regularizer'
    ACTIVITY_REGULARIZER_PARAM = 'activity_regularizer'
    KERNEL_CONSTRAINT_PARAM = 'kernel_constraint'
    BIAS_CONSTRAINT_PARAM = 'bias_constraint'

    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)
        self.output = named_outputs.get('output data',
                                        'out_task_{}'.format(self.order))

        if self.FILTERS_PARAM not in parameters or \
                self.KERNEL_SIZE_PARAM not in parameters or \
                self.STRIDES_PARAM not in parameters:
            raise ValueError(gettext('Parameter {} {} {} are required').format(
                self.FILTERS_PARAM, self.KERNEL_SIZE_PARAM, self.STRIDES_PARAM)
            )

        self.filters = parameters.get(self.FILTERS_PARAM)
        self.kernel_size = parameters.get(self.KERNEL_SIZE_PARAM)
        self.strides = parameters.get(self.STRIDES_PARAM)
        self.input_shape = parameters.get(self.INPUT_SHAPE_PARAM, None)
        self.padding = parameters.get(self.PADDING_PARAM)
        self.data_format = parameters.get(self.DATA_FORMAT_PARAM, None)
        self.dilation_rate = parameters.get(self.DILATION_RATE_PARAM, None) or \
                             None
        self.activation = parameters.get(self.ACTIVATION_PARAM, None)
        self.trainable = parameters.get(self.TRAINABLE_PARAM)
        self.use_bias = parameters.get(self.USE_BIAS_PARAM)
        self.kernel_initializer = parameters.get(self.KERNEL_INITIALIZER_PARAM,
                                                 None)
        self.bias_initializer = parameters.get(self.BIAS_INITIALIZER_PARAM,
                                               None)
        self.kernel_regularizer = parameters.get(self.KERNEL_REGULARIZER_PARAM,
                                                 None)
        self.bias_regularizer = parameters.get(self.BIAS_REGULARIZER_PARAM,
                                               None)
        self.activity_regularizer = parameters.get(self.
                                                   ACTIVITY_REGULARIZER_PARAM,
                                                   None)
        self.kernel_constraint = parameters.get(self.KERNEL_CONSTRAINT_PARAM,
                                                None)
        self.bias_constraint = parameters.get(self.BIAS_CONSTRAINT_PARAM, None)\
                              

        self.add_functions_required = ""
        self.task_name = self.parameters.get('task').get('name')
        self.parent = ""
        self.var_name = ""
        self.has_code = True

        self.treatment()

        self.import_code = {'layer': 'Conv1D',
                            'callbacks': [],
                            'model': None,
                            'preprocessing_image': None,
                            'others': None}

    def treatment(self):
        self.parent = convert_parents_to_variable_name(self.parameters
                                                       .get('parents', []))
        self.var_name = convert_variable_name(self.task_name)
        self.task_name = self.var_name
        if self.parent:
            self.parent = '({})'.format(self.parent[0])
        else:
            self.parent = ''

        self.kernel_size = get_int_or_tuple(self.kernel_size)
        self.strides = get_int_or_tuple(self.strides)
        self.dilation_rate = get_int_or_tuple(self.dilation_rate)

        if self.filters < 0:
            raise ValueError(gettext('Parameter {} is invalid').format(
                self.FILTERS_PARAM))

        if self.kernel_size is False:
            raise ValueError(gettext('Parameter {} is invalid').format(
                self.KERNEL_SIZE_PARAM))

        if self.strides is False:
            raise ValueError(gettext('Parameter {} is invalid').format(
                self.STRIDES_PARAM))

        if self.dilation_rate is False:
            raise ValueError(gettext('Parameter {} is invalid').format(
                self.DILATION_RATE_PARAM))

        self.use_bias = True if int(self.use_bias) == 1 else False
        self.trainable = True if int(self.trainable) == 1 else False

        functions_required = []

        self.filters = """filters={filters}""".format(filters=self.filters)
        functions_required.append(self.filters)

        self.kernel_size = """kernel_size={kernel_size}""" \
            .format(kernel_size=self.kernel_size)
        functions_required.append(self.kernel_size)

        self.strides = """strides={strides}""".format(strides=self.strides)
        functions_required.append(self.strides)

        self.use_bias = """use_bias={use_bias}""".format(use_bias=self.use_bias)
        functions_required.append(self.use_bias)

        if self.input_shape is not None:
            self.input_shape = get_int_or_tuple(self.input_shape)
            if self.input_shape is False:
                raise ValueError(gettext('Parameter {} is invalid').format(
                    self.INPUT_SHAPE_PARAM))
            self.input_shape = """input_shape='{input_shape}'""" \
                .format(input_shape=self.input_shape)
            functions_required.append(self.input_shape)

        if self.padding is not None:
            self.padding = """padding='{padding}'""" \
                .format(padding=self.padding)
            functions_required.append(self.padding)

        if self.data_format is not None:
            self.data_format = """data_format='{data_format}'""" \
                .format(data_format=self.data_format)
            functions_required.append(self.data_format)

        if self.dilation_rate is not None:
            self.dilation_rate = """dilation_rate={dilation_rate}""" \
                .format(dilation_rate=self.dilation_rate)
            functions_required.append(self.dilation_rate)

        if self.activation is not None:
            self.activation = """activation='{activation}'""" \
                .format(activation=self.activation)
            functions_required.append(self.activation)

        if self.kernel_initializer is not None:
            self.kernel_initializer = """kernel_initializer=
            '{kernel_initializer}'""" \
                .format(kernel_initializer=self.kernel_initializer)
            functions_required.append(self.kernel_initializer)

        if self.bias_initializer is not None:
            self.bias_initializer = """bias_initializer='{bias_initializer}'""" \
                .format(bias_initializer=self.bias_initializer)
            functions_required.append(self.bias_initializer)

        if self.kernel_regularizer is not None:
            self.kernel_regularizer = """kernel_regularizer=
            '{kernel_regularizer}'""" \
                .format(kernel_regularizer=self.kernel_regularizer)
            functions_required.append(self.kernel_regularizer)

        if self.bias_regularizer is not None:
            self.bias_regularizer = """bias_regularizer=
            '{bias_regularizer}'""" \
                .format(bias_regularizer=self.bias_regularizer)
            functions_required.append(self.bias_regularizer)

        if self.activity_regularizer is not None:
            self.activity_regularizer = """activity_regularizer=
            '{activity_regularizer}'""" \
                .format(activity_regularizer=self.activity_regularizer)
            functions_required.append(self.activity_regularizer)

        if self.kernel_constraint is not None:
            self.kernel_constraint = """kernel_constraint=
            '{kernel_constraint}'""" \
                .format(kernel_constraint=self.kernel_constraint)
            functions_required.append(self.kernel_constraint)

        if self.bias_constraint is not None:
            self.bias_constraint = """bias_constraint='{bias_constraint}'""" \
                .format(bias_constraint=self.bias_constraint)
            functions_required.append(self.bias_constraint)

        self.add_functions_required = ',\n    '.join(functions_required)
        if self.add_functions_required:
            self.add_functions_required = ',\n    ' + self.add_functions_required

    def generate_code(self):
        return dedent(
            """
            {var_name} = Conv1D(
                name='{name}'{add_functions_required}
            ){parent}
            {var_name}.trainable = {trainable}
            """
        ).format(var_name=self.var_name,
                 name=self.task_name,
                 add_functions_required=self.add_functions_required,
                 trainable=self.trainable,
                 parent=self.parent)


class Convolution2D(Operation):
    FILTERS_PARAM = 'filters'
    KERNEL_SIZE_PARAM = 'kernel_size'
    STRIDES_PARAM = 'strides'
    INPUT_SHAPE_PARAM = 'input_shape'
    PADDING_PARAM = 'padding'
    DATA_FORMAT_PARAM = 'data_format'
    DILATION_RATE_PARAM = 'dilation_rate'
    ACTIVATION_PARAM = 'activation'
    USE_BIAS_PARAM = 'use_bias'
    TRAINABLE_PARAM = 'trainable'
    KERNEL_INITIALIZER_PARAM = 'kernel_initializer'
    BIAS_INITIALIZER_PARAM = 'bias_initializer'
    KERNEL_REGULARIZER_PARAM = 'kernel_regularizer'
    BIAS_REGULARIZER_PARAM = 'bias_regularizer'
    ACTIVITY_REGULARIZER_PARAM = 'activity_regularizer'
    KERNEL_CONSTRAINT_PARAM = 'kernel_constraint'
    BIAS_CONSTRAINT_PARAM = 'bias_constraint'
    WEIGHTS_PARAM = 'weights'

    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)
        self.output = named_outputs.get('output data',
                                        'out_task_{}'.format(self.order))

        if self.FILTERS_PARAM not in parameters or \
                self.KERNEL_SIZE_PARAM not in parameters or \
                self.STRIDES_PARAM not in parameters:
            raise ValueError(gettext('Parameter {} {} {} are required').format(
                self.FILTERS_PARAM, self.KERNEL_SIZE_PARAM, self.STRIDES_PARAM)
            )

        self.filters = parameters.get(self.FILTERS_PARAM)
        self.kernel_size = parameters.get(self.KERNEL_SIZE_PARAM)
        self.strides = parameters.get(self.STRIDES_PARAM)
        self.input_shape = parameters.get(self.INPUT_SHAPE_PARAM, None)
        self.padding = parameters.get(self.PADDING_PARAM)
        self.data_format = parameters.get(self.DATA_FORMAT_PARAM, None)
        self.dilation_rate = parameters.get(self.DILATION_RATE_PARAM, None)
        self.activation = parameters.get(self.ACTIVATION_PARAM, None)
        self.trainable = parameters.get(self.TRAINABLE_PARAM)
        self.use_bias = parameters.get(self.USE_BIAS_PARAM)
        self.kernel_initializer = parameters.get(self.KERNEL_INITIALIZER_PARAM,
                                                 None)
        self.bias_initializer = parameters.get(self.BIAS_INITIALIZER_PARAM,
                                               None)
        self.kernel_regularizer = parameters.get(self.KERNEL_REGULARIZER_PARAM,
                                                 None)
        self.bias_regularizer = parameters.get(self.BIAS_REGULARIZER_PARAM,
                                               None)
        self.activity_regularizer = parameters.get(self.
                                                   ACTIVITY_REGULARIZER_PARAM,
                                                   None)
        self.kernel_constraint = parameters.get(self.KERNEL_CONSTRAINT_PARAM,
                                                None)
        self.bias_constraint = parameters.get(self.BIAS_CONSTRAINT_PARAM, None)
        self.weights = parameters.get(self.WEIGHTS_PARAM, None)

        self.add_functions_required = ""
        self.task_name = self.parameters.get('task').get('name')
        self.parent = ""
        self.var_name = ""
        self.has_code = True

        self.treatment()

        self.import_code = {'layer': 'Conv2D',
                            'callbacks': [],
                            'model': None,
                            'preprocessing_image': None,
                            'others': None}

    def treatment(self):
        self.parent = convert_parents_to_variable_name(self.parameters
                                                       .get('parents', []))
        self.var_name = convert_variable_name(self.task_name)
        self.task_name = self.var_name
        if self.parent:
            self.parent = '({})'.format(self.parent[0])
        else:
            self.parent = ''

        self.kernel_size = get_int_or_tuple(self.kernel_size)
        self.strides = get_int_or_tuple(self.strides)
        self.dilation_rate = get_int_or_tuple(self.dilation_rate)

        try:
            self.filters = int(self.filters)
            if self.filters < 0:
                raise ValueError(gettext('Parameter {} is invalid').format(
                    self.FILTERS_PARAM))
        except:
            pass # The user possibly is using a var defined in PythonCode Layer

        if self.kernel_size is False:
            raise ValueError(gettext('Parameter {} is invalid').format(
                self.KERNEL_SIZE_PARAM))

        if self.strides is False:
            raise ValueError(gettext('Parameter {} is invalid').format(
                self.STRIDES_PARAM))

        if self.dilation_rate is False:
            raise ValueError(gettext('Parameter {} is invalid').format(
                self.DILATION_RATE_PARAM))

        self.use_bias = True if int(self.use_bias) == 1 else False
        self.trainable = True if int(self.trainable) == 1 else False

        functions_required = []

        self.filters = """filters={filters}""".format(filters=self.filters)
        functions_required.append(self.filters)

        self.kernel_size = """kernel_size={kernel_size}""" \
            .format(kernel_size=self.kernel_size)
        functions_required.append(self.kernel_size)

        self.strides = """strides={strides}""".format(strides=self.strides)
        functions_required.append(self.strides)

        self.use_bias = """use_bias={use_bias}""".format(use_bias=self.use_bias)
        functions_required.append(self.use_bias)

        if self.input_shape is not None:
            self.input_shape = get_int_or_tuple(self.input_shape)
            if self.input_shape is False:
                raise ValueError(gettext('Parameter {} is invalid').format(
                    self.INPUT_SHAPE_PARAM))
            self.input_shape = """input_shape='{input_shape}'""" \
                .format(input_shape=self.input_shape)
            functions_required.append(self.input_shape)

        if self.padding is not None:
            self.padding = """padding='{padding}'""" \
                .format(padding=self.padding)
            functions_required.append(self.padding)

        if self.data_format is not None:
            self.data_format = """data_format='{data_format}'""" \
                .format(data_format=self.data_format)
            functions_required.append(self.data_format)

        if self.dilation_rate is not None:
            self.dilation_rate = """dilation_rate={dilation_rate}""" \
                .format(dilation_rate=self.dilation_rate)
            functions_required.append(self.dilation_rate)

        if self.activation is not None:
            self.activation = """activation='{activation}'""" \
                .format(activation=self.activation)
            functions_required.append(self.activation)

        if self.kernel_initializer is not None:
            self.kernel_initializer = """kernel_initializer=
            '{kernel_initializer}'""" \
                .format(kernel_initializer=self.kernel_initializer)
            functions_required.append(self.kernel_initializer)

        if self.bias_initializer is not None:
            self.bias_initializer = """bias_initializer='{bias_initializer}'""" \
                .format(bias_initializer=self.bias_initializer)
            functions_required.append(self.bias_initializer)

        if self.kernel_regularizer is not None:
            self.kernel_regularizer = """kernel_regularizer=
            '{kernel_regularizer}'""" \
                .format(kernel_regularizer=self.kernel_regularizer)
            functions_required.append(self.kernel_regularizer)

        if self.bias_regularizer is not None:
            self.bias_regularizer = """bias_regularizer=
            '{bias_regularizer}'""" \
                .format(bias_regularizer=self.bias_regularizer)
            functions_required.append(self.bias_regularizer)

        if self.activity_regularizer is not None:
            self.activity_regularizer = """activity_regularizer=
            '{activity_regularizer}'""" \
                .format(activity_regularizer=self.activity_regularizer)
            functions_required.append(self.activity_regularizer)

        if self.kernel_constraint is not None:
            self.kernel_constraint = """kernel_constraint=
            '{kernel_constraint}'""" \
                .format(kernel_constraint=self.kernel_constraint)
            functions_required.append(self.kernel_constraint)

        if self.bias_constraint is not None:
            self.bias_constraint = """bias_constraint='{bias_constraint}'""" \
                .format(bias_constraint=self.bias_constraint)
            functions_required.append(self.bias_constraint)

        if self.weights is not None and self.weights.strip():
            if convert_to_list(self.weights):
                self.weights = """weights={weights}"""\
                    .format(weights=self.weights)
                functions_required.append(self.weights)

        self.add_functions_required = ',\n    '.join(functions_required)
        if self.add_functions_required:
            self.add_functions_required = ',\n    ' + self.add_functions_required

    def generate_code(self):
        return dedent(
            """
            {var_name} = Conv2D(
                name='{name}'{add_functions_required}
            ){parent}
            {var_name}.trainable = {trainable}
            """
        ).format(var_name=self.var_name,
                 name=self.task_name,
                 add_functions_required=self.add_functions_required,
                 trainable=self.trainable,
                 parent=self.parent)


class SeparableConv1D(Operation):
    FILTERS_PARAM = 'filters'
    KERNEL_SIZE_PARAM = 'kernel_size'
    STRIDES_PARAM = 'strides'
    INPUT_SHAPE_PARAM = 'input_shape'
    PADDING_PARAM = 'padding'
    DATA_FORMAT_PARAM = 'data_format'
    DILATION_RATE_PARAM = 'dilation_rate'
    DEPTH_MULTIPLIER_PARAM = 'depth_multiplier'
    ACTIVATION_PARAM = 'activation'
    USE_BIAS_PARAM = 'use_bias'
    DEPTHWISE_INITIALIZER_PARAM = 'depthwise_initializer'
    POINTWISE_INITIALIZER_PARAM = 'pointwise_initializer'
    BIAS_INITIALIZER_PARAM = 'bias_initializer'
    DEPTHWISE_REGULARIZER_PARAM = 'depthwise_regularizer'
    POINTWISE_REGULARIZER_PARAM = 'pointwise_regularizer'
    BIAS_REGULARIZER_PARAM = 'bias_regularizer'
    ACTIVITY_REGULARIZER_PARAM = 'activity_regularizer'
    DEPTHWISE_CONSTRAINT_PARAM = 'depthwise_constraint'
    POINTWISE_CONSTRAINT_PARAM = 'pointwise_constraint'
    BIAS_CONSTRAINT_PARAM = 'bias_constraint'
    TRAINABLE_PARAM = 'trainable'

    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)
        self.output = named_outputs.get('output data',
                                        'out_task_{}'.format(self.order))

        if self.FILTERS_PARAM not in parameters or \
                self.KERNEL_SIZE_PARAM not in parameters or \
                self.STRIDES_PARAM not in parameters:
            raise ValueError(gettext('Parameter {} {} {} are required').format(
                self.FILTERS_PARAM, self.KERNEL_SIZE_PARAM, self.STRIDES_PARAM)
            )

        self.filters = abs(parameters.get(self.FILTERS_PARAM))
        self.kernel_size = parameters.get(self.KERNEL_SIZE_PARAM)
        self.strides = parameters.get(self.STRIDES_PARAM)
        self.input_shape = parameters.get(self.INPUT_SHAPE_PARAM, None)
        self.padding = parameters.get(self.PADDING_PARAM)
        self.data_format = parameters.get(self.DATA_FORMAT_PARAM, None)
        self.dilation_rate = parameters.get(self.DILATION_RATE_PARAM, None) or \
                             None
        self.depth_multiplier = abs(parameters.get(self.DEPTH_MULTIPLIER_PARAM,
                                                   None))
        self.activation = parameters.get(self.ACTIVATION_PARAM, None)
        self.use_bias = parameters.get(self.USE_BIAS_PARAM)
        self.depthwise_initializer = parameters.get(self.
                                                    DEPTHWISE_INITIALIZER_PARAM,
                                                    None)
        self.pointwise_initializer = parameters.get(self.
                                                    POINTWISE_INITIALIZER_PARAM,
                                                    None)
        self.bias_initializer = parameters.get(self.BIAS_INITIALIZER_PARAM,
                                               None)
        self.depthwise_regularizer = parameters.get(self.
                                                    DEPTHWISE_REGULARIZER_PARAM,
                                                    None)
        self.pointwise_regularizer = parameters.get(self.
                                                    POINTWISE_REGULARIZER_PARAM,
                                                    None)
        self.bias_regularizer = parameters.get(self.BIAS_REGULARIZER_PARAM,
                                               None)
        self.activity_regularizer = parameters.get(self.
                                                   ACTIVITY_REGULARIZER_PARAM,
                                                   None)
        self.depthwise_constraint = parameters.get(self.
                                                   DEPTHWISE_CONSTRAINT_PARAM,
                                                   None)
        self.pointwise_constraint = parameters.get(self.
                                                   POINTWISE_CONSTRAINT_PARAM,
                                                   None)
        self.bias_constraint = parameters.get(self.BIAS_CONSTRAINT_PARAM, None)\
                              
        self.trainable = parameters.get(self.TRAINABLE_PARAM)

        self.add_functions_required = ""
        self.task_name = self.parameters.get('task').get('name')
        self.parent = ""
        self.var_name = ""
        self.has_code = True

        self.treatment()

        self.import_code = {'layer': 'SeparableConv1D',
                            'callbacks': [],
                            'model': None,
                            'preprocessing_image': None,
                            'others': None}

    def treatment(self):
        self.parent = convert_parents_to_variable_name(self.parameters
                                                       .get('parents', []))
        self.var_name = convert_variable_name(self.task_name)
        self.task_name = self.var_name
        if self.parent:
            self.parent = '({})'.format(self.parent[0])
        else:
            self.parent = ''

        self.kernel_size = get_int_or_tuple(self.kernel_size)
        self.strides = get_int_or_tuple(self.strides)
        self.dilation_rate = get_int_or_tuple(self.dilation_rate)

        if self.filters < 0:
            raise ValueError(gettext('Parameter {} is invalid').format(
                self.FILTERS_PARAM))

        if self.kernel_size is False:
            raise ValueError(gettext('Parameter {} is invalid').format(
                self.KERNEL_SIZE_PARAM))

        if self.strides is False:
            raise ValueError(gettext('Parameter {} is invalid').format(
                self.STRIDES_PARAM))

        if self.dilation_rate is False:
            raise ValueError(gettext('Parameter {} is invalid').format(
                self.DILATION_RATE_PARAM))

        self.use_bias = True if int(self.use_bias) == 1 else False
        self.trainable = True if int(self.trainable) == 1 else False

        functions_required = []

        self.filters = """filters={filters}""".format(filters=self.filters)
        functions_required.append(self.filters)

        self.kernel_size = """kernel_size={kernel_size}""" \
            .format(kernel_size=self.kernel_size)
        functions_required.append(self.kernel_size)

        self.strides = """strides={strides}""".format(strides=self.strides)
        functions_required.append(self.strides)

        if self.input_shape is not None:
            self.input_shape = get_int_or_tuple(self.input_shape)
            if self.input_shape is False:
                raise ValueError(gettext('Parameter {} is invalid').format(
                    self.INPUT_SHAPE_PARAM))
            self.input_shape = """input_shape='{input_shape}'""" \
                .format(input_shape=self.input_shape)
            functions_required.append(self.input_shape)

        if self.padding is not None:
            self.padding = """padding='{padding}'""" \
                .format(padding=self.padding)
            functions_required.append(self.padding)

        if self.data_format is not None:
            self.data_format = """data_format='{data_format}'""" \
                .format(data_format=self.data_format)
            functions_required.append(self.data_format)

        if self.dilation_rate is not None:
            self.dilation_rate = """dilation_rate={dilation_rate}""" \
                .format(dilation_rate=self.dilation_rate)
            functions_required.append(self.dilation_rate)

        if self.depth_multiplier is not None:
            self.depth_multiplier = """depth_multiplier={depth_multiplier}""" \
                .format(depth_multiplier=self.depth_multiplier)
            functions_required.append(self.depth_multiplier)

        if self.activation is not None:
            self.activation = """activation='{activation}'""" \
                .format(activation=self.activation)
            functions_required.append(self.activation)

        self.use_bias = """use_bias={use_bias}""".format(use_bias=self.use_bias)
        functions_required.append(self.use_bias)

        if self.depthwise_initializer is not None:
            self.depthwise_initializer = """depthwise_initializer=
            '{depthwise_initializer}'""".format(depthwise_initializer=self
                                                .depthwise_initializer)
            functions_required.append(self.depthwise_initializer)

        if self.pointwise_initializer is not None:
            self.pointwise_initializer = """pointwise_initializer=
            '{pointwise_initializer}'""".format(pointwise_initializer=self
                                                .pointwise_initializer)
            functions_required.append(self.pointwise_initializer)

        if self.bias_initializer is not None:
            self.bias_initializer = """bias_initializer='{bias_initializer}'"""\
                .format(bias_initializer=self.bias_initializer)
            functions_required.append(self.bias_initializer)

        if self.depthwise_regularizer is not None:
            self.depthwise_regularizer = """depthwise_regularizer=
            '{depthwise_regularizer}'""" \
                .format(depthwise_regularizer=self.depthwise_regularizer)
            functions_required.append(self.depthwise_regularizer)

        if self.pointwise_regularizer is not None:
            self.pointwise_regularizer = """pointwise_regularizer=
            '{pointwise_regularizer}'""".format(pointwise_regularizer=self.
                                                pointwise_regularizer)
            functions_required.append(self.pointwise_regularizer)

        if self.bias_regularizer is not None:
            self.bias_regularizer = """bias_regularizer=
            '{bias_regularizer}'""".format(bias_regularizer=self.
                                           bias_regularizer)
            functions_required.append(self.bias_regularizer)

        if self.activity_regularizer is not None:
            self.activity_regularizer = """activity_regularizer=
            '{activity_regularizer}'""" \
                .format(activity_regularizer=self.activity_regularizer)
            functions_required.append(self.activity_regularizer)

        if self.depthwise_constraint is not None:
            self.depthwise_constraint = """depthwise_constraint=
            '{depthwise_constraint}'""".format(depthwise_constraint=self.
                                               depthwise_constraint)
            functions_required.append(self.depthwise_constraint)

        if self.pointwise_constraint is not None:
            self.pointwise_constraint = """pointwise_constraint=
            '{pointwise_constraint}'""".format(pointwise_constraint=self.
                                               pointwise_constraint)
            functions_required.append(self.pointwise_constraint)

        if self.bias_constraint is not None:
            self.bias_constraint = """bias_constraint='{bias_constraint}'""" \
                .format(bias_constraint=self.bias_constraint)
            functions_required.append(self.bias_constraint)

        self.add_functions_required = ',\n    '.join(functions_required)
        if self.add_functions_required:
            self.add_functions_required = ',\n    ' + self.add_functions_required

    def generate_code(self):
        return dedent(
            """
            {var_name} = SeparableConv1D(
                name='{name}'{add_functions_required}
            ){parent}
            """
        ).format(var_name=self.var_name,
                 name=self.task_name,
                 add_functions_required=self.add_functions_required,
                 parent=self.parent)


class SeparableConv2D(Operation):
    FILTERS_PARAM = 'filters'
    KERNEL_SIZE_PARAM = 'kernel_size'
    STRIDES_PARAM = 'strides'
    INPUT_SHAPE_PARAM = 'input_shape'
    PADDING_PARAM = 'padding'
    DATA_FORMAT_PARAM = 'data_format'
    DILATION_RATE_PARAM = 'dilation_rate'
    DEPTH_MULTIPLIER_PARAM = 'depth_multiplier'
    ACTIVATION_PARAM = 'activation'
    USE_BIAS_PARAM = 'use_bias'
    DEPTHWISE_INITIALIZER_PARAM = 'depthwise_initializer'
    POINTWISE_INITIALIZER_PARAM = 'pointwise_initializer'
    BIAS_INITIALIZER_PARAM = 'bias_initializer'
    DEPTHWISE_REGULARIZER_PARAM = 'depthwise_regularizer'
    POINTWISE_REGULARIZER_PARAM = 'pointwise_regularizer'
    BIAS_REGULARIZER_PARAM = 'bias_regularizer'
    ACTIVITY_REGULARIZER_PARAM = 'activity_regularizer'
    DEPTHWISE_CONSTRAINT_PARAM = 'depthwise_constraint'
    POINTWISE_CONSTRAINT_PARAM = 'pointwise_constraint'
    BIAS_CONSTRAINT_PARAM = 'bias_constraint'

    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)
        self.output = named_outputs.get('output data',
                                        'out_task_{}'.format(self.order))

        if self.FILTERS_PARAM not in parameters or \
                self.KERNEL_SIZE_PARAM not in parameters or \
                self.STRIDES_PARAM not in parameters:
            raise ValueError(gettext('Parameter {} {} {} are required').format(
                self.FILTERS_PARAM, self.KERNEL_SIZE_PARAM, self.STRIDES_PARAM)
            )

        self.filters = abs(parameters.get(self.FILTERS_PARAM))
        self.kernel_size = parameters.get(self.KERNEL_SIZE_PARAM)
        self.strides = parameters.get(self.STRIDES_PARAM)
        self.input_shape = parameters.get(self.INPUT_SHAPE_PARAM, None)
        self.padding = parameters.get(self.PADDING_PARAM)
        self.data_format = parameters.get(self.DATA_FORMAT_PARAM, None)
        self.dilation_rate = parameters.get(self.DILATION_RATE_PARAM, None) or \
                             None
        self.depth_multiplier = abs(parameters.get(self.DEPTH_MULTIPLIER_PARAM,
                                                   None))
        self.activation = parameters.get(self.ACTIVATION_PARAM, None)
        self.use_bias = parameters.get(self.USE_BIAS_PARAM)
        self.depthwise_initializer = parameters.get(self.
                                                    DEPTHWISE_INITIALIZER_PARAM,
                                                    None)
        self.pointwise_initializer = parameters.get(self.
                                                    POINTWISE_INITIALIZER_PARAM,
                                                    None)
        self.bias_initializer = parameters.get(self.BIAS_INITIALIZER_PARAM,
                                               None)
        self.depthwise_regularizer = parameters.get(self.
                                                    DEPTHWISE_REGULARIZER_PARAM,
                                                    None)
        self.pointwise_regularizer = parameters.get(self.
                                                    POINTWISE_REGULARIZER_PARAM,
                                                    None)
        self.bias_regularizer = parameters.get(self.BIAS_REGULARIZER_PARAM,
                                               None)
        self.activity_regularizer = parameters.get(self.
                                                   ACTIVITY_REGULARIZER_PARAM,
                                                   None)
        self.depthwise_constraint = parameters.get(self.
                                                   DEPTHWISE_CONSTRAINT_PARAM,
                                                   None)
        self.pointwise_constraint = parameters.get(self.
                                                   POINTWISE_CONSTRAINT_PARAM,
                                                   None)
        self.bias_constraint = parameters.get(self.BIAS_CONSTRAINT_PARAM, None) \
                              
        self.trainable = parameters.get(self.TRAINABLE_PARAM)

        self.add_functions_required = ""
        self.task_name = self.parameters.get('task').get('name')
        self.parent = ""
        self.var_name = ""
        self.has_code = True

        self.treatment()

        self.import_code = {'layer': 'SeparableConv2D',
                            'callbacks': [],
                            'model': None,
                            'preprocessing_image': None,
                            'others': None}

    def treatment(self):
        self.parent = convert_parents_to_variable_name(self.parameters
                                                       .get('parents', []))
        self.var_name = convert_variable_name(self.task_name)
        self.task_name = self.var_name
        if self.parent:
            self.parent = '({})'.format(self.parent[0])
        else:
            self.parent = ''

        self.kernel_size = get_int_or_tuple(self.kernel_size)
        self.strides = get_int_or_tuple(self.strides)
        self.dilation_rate = get_int_or_tuple(self.dilation_rate)

        if self.filters < 0:
            raise ValueError(gettext('Parameter {} is invalid').format(
                self.FILTERS_PARAM))

        if self.kernel_size is False:
            raise ValueError(gettext('Parameter {} is invalid').format(
                self.KERNEL_SIZE_PARAM))

        if self.strides is False:
            raise ValueError(gettext('Parameter {} is invalid').format(
                self.STRIDES_PARAM))

        if self.dilation_rate is False:
            raise ValueError(gettext('Parameter {} is invalid').format(
                self.DILATION_RATE_PARAM))

        self.use_bias = True if int(self.use_bias) == 1 else False
        self.trainable = True if int(self.trainable) == 1 else False

        functions_required = []

        self.filters = """filters={filters}""".format(filters=self.filters)
        functions_required.append(self.filters)

        self.kernel_size = """kernel_size={kernel_size}""" \
            .format(kernel_size=self.kernel_size)
        functions_required.append(self.kernel_size)

        self.strides = """strides={strides}""".format(strides=self.strides)
        functions_required.append(self.strides)

        if self.input_shape is not None:
            self.input_shape = get_int_or_tuple(self.input_shape)
            if self.input_shape is False:
                raise ValueError(gettext('Parameter {} is invalid').format(
                    self.INPUT_SHAPE_PARAM))
            self.input_shape = """input_shape='{input_shape}'""" \
                .format(input_shape=self.input_shape)
            functions_required.append(self.input_shape)

        if self.padding is not None:
            self.padding = """padding='{padding}'""" \
                .format(padding=self.padding)
            functions_required.append(self.padding)

        if self.data_format is not None:
            self.data_format = """data_format='{data_format}'""" \
                .format(data_format=self.data_format)
            functions_required.append(self.data_format)

        if self.dilation_rate is not None:
            self.dilation_rate = """dilation_rate={dilation_rate}""" \
                .format(dilation_rate=self.dilation_rate)
            functions_required.append(self.dilation_rate)

        if self.depth_multiplier is not None:
            self.depth_multiplier = """depth_multiplier={depth_multiplier}""" \
                .format(depth_multiplier=self.depth_multiplier)
            functions_required.append(self.depth_multiplier)

        if self.activation is not None:
            self.activation = """activation='{activation}'""" \
                .format(activation=self.activation)
            functions_required.append(self.activation)

        self.use_bias = """use_bias={use_bias}""".format(use_bias=self.use_bias)
        functions_required.append(self.use_bias)

        if self.depthwise_initializer is not None:
            self.depthwise_initializer = """depthwise_initializer=
            '{depthwise_initializer}'""".format(depthwise_initializer=self
                                                .depthwise_initializer)
            functions_required.append(self.depthwise_initializer)

        if self.pointwise_initializer is not None:
            self.pointwise_initializer = """pointwise_initializer=
            '{pointwise_initializer}'""".format(pointwise_initializer=self
                                                .pointwise_initializer)
            functions_required.append(self.pointwise_initializer)

        if self.bias_initializer is not None:
            self.bias_initializer = """bias_initializer='{bias_initializer}'""" \
                .format(bias_initializer=self.bias_initializer)
            functions_required.append(self.bias_initializer)

        if self.depthwise_regularizer is not None:
            self.depthwise_regularizer = """depthwise_regularizer=
            '{depthwise_regularizer}'""" \
                .format(depthwise_regularizer=self.depthwise_regularizer)
            functions_required.append(self.depthwise_regularizer)

        if self.pointwise_regularizer is not None:
            self.pointwise_regularizer = """pointwise_regularizer=
            '{pointwise_regularizer}'""".format(pointwise_regularizer=self.
                                                pointwise_regularizer)
            functions_required.append(self.pointwise_regularizer)

        if self.bias_regularizer is not None:
            self.bias_regularizer = """bias_regularizer=
            '{bias_regularizer}'""".format(bias_regularizer=self.
                                           bias_regularizer)
            functions_required.append(self.bias_regularizer)

        if self.activity_regularizer is not None:
            self.activity_regularizer = """activity_regularizer=
            '{activity_regularizer}'""" \
                .format(activity_regularizer=self.activity_regularizer)
            functions_required.append(self.activity_regularizer)

        if self.depthwise_constraint is not None:
            self.depthwise_constraint = """depthwise_constraint=
            '{depthwise_constraint}'""".format(depthwise_constraint=self.
                                               depthwise_constraint)
            functions_required.append(self.depthwise_constraint)

        if self.pointwise_constraint is not None:
            self.pointwise_constraint = """pointwise_constraint=
            '{pointwise_constraint}'""".format(pointwise_constraint=self.
                                               pointwise_constraint)
            functions_required.append(self.pointwise_constraint)

        if self.bias_constraint is not None:
            self.bias_constraint = """bias_constraint='{bias_constraint}'""" \
                .format(bias_constraint=self.bias_constraint)
            functions_required.append(self.bias_constraint)

        self.add_functions_required = ',\n    '.join(functions_required)
        if self.add_functions_required:
            self.add_functions_required = ',\n    ' + self.add_functions_required

    def generate_code(self):
        return dedent(
            """
            {var_name} = SeparableConv2D(
                name='{name}'{add_functions_required}
            ){parent}
            """
        ).format(var_name=self.var_name,
                 name=self.task_name,
                 add_functions_required=self.add_functions_required,
                 parent=self.parent)


class DepthwiseConv2D(Operation):
    KERNEL_SIZE_PARAM = 'kernel_size'
    STRIDES_PARAM = 'strides'
    INPUT_SHAPE_PARAM = 'input_shape'
    PADDING_PARAM = 'padding'
    DEPTH_MULTIPLIER_PARAM = 'depth_multiplier'
    DATA_FORMAT_PARAM = 'data_format'
    ACTIVATION_PARAM = 'activation'
    USE_BIAS_PARAM = 'use_bias'
    DEPTHWISE_INITIALIZER_PARAM = 'depthwise_initializer'
    BIAS_INITIALIZER_PARAM = 'bias_initializer'
    DEPTHWISE_REGULARIZER_PARAM = 'depthwise_regularizer'
    BIAS_REGULARIZER_PARAM = 'bias_regularizer'
    ACTIVITY_REGULARIZER_PARAM = 'activity_regularizer'
    DEPTHWISE_CONSTRAINT_PARAM = 'depthwise_constraint'
    BIAS_CONSTRAINT_PARAM = 'bias_constraint'

    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)
        self.output = named_outputs.get('output data',
                                        'out_task_{}'.format(self.order))

        if self.KERNEL_SIZE_PARAM not in parameters or \
                self.STRIDES_PARAM not in parameters:
            raise ValueError(gettext('Parameter {} {} are required').format(
                self.KERNEL_SIZE_PARAM, self.STRIDES_PARAM)
            )

        self.kernel_size = parameters.get(self.KERNEL_SIZE_PARAM)
        self.strides = parameters.get(self.STRIDES_PARAM)
        self.input_shape = parameters.get(self.INPUT_SHAPE_PARAM, None)
        self.padding = parameters.get(self.PADDING_PARAM)
        self.data_format = parameters.get(self.DATA_FORMAT_PARAM, None)
        self.depth_multiplier = abs(parameters.get(self.DEPTH_MULTIPLIER_PARAM,
                                                   None))
        self.activation = parameters.get(self.ACTIVATION_PARAM, None)
        self.use_bias = parameters.get(self.USE_BIAS_PARAM)
        self.depthwise_initializer = parameters.get(self.
                                                    DEPTHWISE_INITIALIZER_PARAM,
                                                    None)
        self.bias_initializer = parameters.get(self.BIAS_INITIALIZER_PARAM,
                                               None)
        self.depthwise_regularizer = parameters.get(self.
                                                    DEPTHWISE_REGULARIZER_PARAM,
                                                    None)
        self.bias_regularizer = parameters.get(self.BIAS_REGULARIZER_PARAM,
                                               None)
        self.activity_regularizer = parameters.get(self.
                                                   ACTIVITY_REGULARIZER_PARAM,
                                                   None)
        self.depthwise_constraint = parameters.get(self.
                                                   DEPTHWISE_CONSTRAINT_PARAM,
                                                   None)
        self.bias_constraint = parameters.get(self.BIAS_CONSTRAINT_PARAM, None) \
                              

        self.add_functions_required = ""
        self.task_name = self.parameters.get('task').get('name')
        self.parent = ""
        self.var_name = ""
        self.has_code = True

        self.treatment()

        self.import_code = {'layer': 'DepthwiseConv2D',
                            'callbacks': [],
                            'model': None,
                            'preprocessing_image': None,
                            'others': None}

    def treatment(self):
        self.parent = convert_parents_to_variable_name(self.parameters
                                                       .get('parents', []))
        self.var_name = convert_variable_name(self.task_name)
        self.task_name = self.var_name
        if self.parent:
            self.parent = '({})'.format(self.parent[0])
        else:
            self.parent = ''

        self.kernel_size = get_int_or_tuple(self.kernel_size)
        self.strides = get_int_or_tuple(self.strides)
        self.dilation_rate = get_int_or_tuple(self.dilation_rate)

        if self.kernel_size is False:
            raise ValueError(gettext('Parameter {} is invalid').format(
                self.KERNEL_SIZE_PARAM))

        if self.strides is False:
            raise ValueError(gettext('Parameter {} is invalid').format(
                self.STRIDES_PARAM))

        self.use_bias = True if int(self.use_bias) == 1 else False

        functions_required = []

        self.kernel_size = """kernel_size={kernel_size}""" \
            .format(kernel_size=self.kernel_size)
        functions_required.append(self.kernel_size)

        self.strides = """strides={strides}""".format(strides=self.strides)
        functions_required.append(self.strides)

        if self.input_shape is not None:
            self.input_shape = get_int_or_tuple(self.input_shape)
            if self.input_shape is False:
                raise ValueError(gettext('Parameter {} is invalid').format(
                    self.INPUT_SHAPE_PARAM))
            self.input_shape = """input_shape='{input_shape}'""" \
                .format(input_shape=self.input_shape)
            functions_required.append(self.input_shape)

        if self.padding is not None:
            self.padding = """padding='{padding}'""" \
                .format(padding=self.padding)
            functions_required.append(self.padding)

        if self.depth_multiplier is not None:
            self.depth_multiplier = """depth_multiplier={depth_multiplier}""" \
                .format(depth_multiplier=self.depth_multiplier)
            functions_required.append(self.depth_multiplier)

        if self.data_format is not None:
            self.data_format = """data_format='{data_format}'""" \
                .format(data_format=self.data_format)
            functions_required.append(self.data_format)

        if self.activation is not None:
            self.activation = """activation='{activation}'""" \
                .format(activation=self.activation)
            functions_required.append(self.activation)

        self.use_bias = """use_bias={use_bias}""".format(use_bias=self.use_bias)
        functions_required.append(self.use_bias)

        if self.depthwise_initializer is not None:
            self.depthwise_initializer = """depthwise_initializer=
            '{depthwise_initializer}'""".format(depthwise_initializer=self
                                                .depthwise_initializer)
            functions_required.append(self.depthwise_initializer)

        if self.bias_initializer is not None:
            self.bias_initializer = """bias_initializer='{bias_initializer}'""" \
                .format(bias_initializer=self.bias_initializer)
            functions_required.append(self.bias_initializer)

        if self.depthwise_regularizer is not None:
            self.depthwise_regularizer = """depthwise_regularizer=
            '{depthwise_regularizer}'""" \
                .format(depthwise_regularizer=self.depthwise_regularizer)
            functions_required.append(self.depthwise_regularizer)

        if self.bias_regularizer is not None:
            self.bias_regularizer = """bias_regularizer=
            '{bias_regularizer}'""".format(bias_regularizer=self.
                                           bias_regularizer)
            functions_required.append(self.bias_regularizer)

        if self.activity_regularizer is not None:
            self.activity_regularizer = """activity_regularizer=
            '{activity_regularizer}'""" \
                .format(activity_regularizer=self.activity_regularizer)
            functions_required.append(self.activity_regularizer)

        if self.depthwise_constraint is not None:
            self.depthwise_constraint = """depthwise_constraint=
            '{depthwise_constraint}'""".format(depthwise_constraint=self.
                                               depthwise_constraint)
            functions_required.append(self.depthwise_constraint)

        if self.bias_constraint is not None:
            self.bias_constraint = """bias_constraint='{bias_constraint}'""" \
                .format(bias_constraint=self.bias_constraint)
            functions_required.append(self.bias_constraint)

        self.add_functions_required = ',\n    '.join(functions_required)
        if self.add_functions_required:
            self.add_functions_required = ',\n    ' + self.add_functions_required

    def generate_code(self):
        return dedent(
            """
            {var_name} = DepthwiseConv2D(
                name='{name}'{add_functions_required}
            ){parent}
            """
        ).format(var_name=self.var_name,
                 name=self.task_name,
                 add_functions_required=self.add_functions_required,
                 parent=self.parent)


class Conv2DTranspose(Operation):
    FILTERS_PARAM = 'filters'
    KERNEL_SIZE_PARAM = 'kernel_size'
    STRIDES_PARAM = 'strides'
    INPUT_SHAPE_PARAM = 'input_shape'
    PADDING_PARAM = 'padding'
    OUTPUT_PADDING_PARAM = 'output_padding'
    DATA_FORMAT_PARAM = 'data_format'
    DILATION_RATE_PARAM = 'dilation_rate'
    ACTIVATION_PARAM = 'activation'
    USE_BIAS_PARAM = 'use_bias'
    KERNEL_INITIALIZER_PARAM = 'kernel_initializer'
    BIAS_INITIALIZER_PARAM = 'bias_initializer'
    KERNEL_REGULARIZER_PARAM = 'kernel_regularizer'
    BIAS_REGULARIZER_PARAM = 'bias_regularizer'
    ACTIVITY_REGULARIZER_PARAM = 'activity_regularizer'
    KERNEL_CONSTRAINT_PARAM = 'kernel_constraint'
    BIAS_CONSTRAINT_PARAM = 'bias_constraint'

    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)
        self.output = named_outputs.get('output data',
                                        'out_task_{}'.format(self.order))

        if self.FILTERS_PARAM not in parameters or \
                self.KERNEL_SIZE_PARAM not in parameters or \
                self.STRIDES_PARAM not in parameters:
            raise ValueError(gettext('Parameter {} {} {} are required').format(
                self.FILTERS_PARAM, self.KERNEL_SIZE_PARAM, self.STRIDES_PARAM)
            )

        self.filters = abs(parameters.get(self.FILTERS_PARAM))
        self.kernel_size = parameters.get(self.KERNEL_SIZE_PARAM)
        self.strides = parameters.get(self.STRIDES_PARAM)
        self.input_shape = parameters.get(self.INPUT_SHAPE_PARAM, None)
        self.padding = parameters.get(self.PADDING_PARAM)
        self.output_padding = parameters.get(self.OUTPUT_PADDING_PARAM)
        self.data_format = parameters.get(self.DATA_FORMAT_PARAM, None)
        self.dilation_rate = parameters.get(self.DILATION_RATE_PARAM, None) or \
                             None
        self.activation = parameters.get(self.ACTIVATION_PARAM, None)
        self.use_bias = parameters.get(self.USE_BIAS_PARAM)
        self.kernel_initializer = parameters.get(self.KERNEL_INITIALIZER_PARAM,
                                                 None)
        self.bias_initializer = parameters.get(self.BIAS_INITIALIZER_PARAM,
                                               None)
        self.kernel_regularizer = parameters.get(self.KERNEL_REGULARIZER_PARAM,
                                                 None)
        self.bias_regularizer = parameters.get(self.BIAS_REGULARIZER_PARAM,
                                               None)
        self.activity_regularizer = parameters.get(self.
                                                   ACTIVITY_REGULARIZER_PARAM,
                                                   None)
        self.kernel_constraint = parameters.get(self.KERNEL_CONSTRAINT_PARAM,
                                                None)
        self.bias_constraint = parameters.get(self.BIAS_CONSTRAINT_PARAM, None)\
                              

        self.add_functions_required = ""
        self.task_name = self.parameters.get('task').get('name')
        self.parent = ""
        self.var_name = ""
        self.has_code = True

        self.treatment()

        self.import_code = {'layer': 'Conv2DTranspose',
                            'callbacks': [],
                            'model': None,
                            'preprocessing_image': None,
                            'others': None}

    def treatment(self):
        self.parent = convert_parents_to_variable_name(self.parameters
                                                       .get('parents', []))
        self.var_name = convert_variable_name(self.task_name)
        self.task_name = self.var_name
        if self.parent:
            self.parent = '({})'.format(self.parent[0])
        else:
            self.parent = ''

        self.kernel_size = get_int_or_tuple(self.kernel_size)
        self.strides = get_int_or_tuple(self.strides)
        self.dilation_rate = get_int_or_tuple(self.dilation_rate)
        self.output_padding = get_int_or_tuple(self.output_padding)

        if self.filters < 0:
            raise ValueError(gettext('Parameter {} is invalid').format(
                self.FILTERS_PARAM))

        if self.kernel_size is False:
            raise ValueError(gettext('Parameter {} is invalid').format(
                self.KERNEL_SIZE_PARAM))

        if self.strides is False:
            raise ValueError(gettext('Parameter {} is invalid').format(
                self.STRIDES_PARAM))

        if self.dilation_rate is False:
            raise ValueError(gettext('Parameter {} is invalid').format(
                self.DILATION_RATE_PARAM))

        if self.output_padding is False:
            raise ValueError(gettext('Parameter {} is invalid').format(
                self.OUTPUT_PADDING_PARAM))

        self.use_bias = True if int(self.use_bias) == 1 else False

        functions_required = []

        self.filters = """filters={filters}""".format(filters=self.filters)
        functions_required.append(self.filters)

        self.kernel_size = """kernel_size={kernel_size}""" \
            .format(kernel_size=self.kernel_size)
        functions_required.append(self.kernel_size)

        self.strides = """strides={strides}""".format(strides=self.strides)
        functions_required.append(self.strides)

        if self.input_shape is not None:
            self.input_shape = get_int_or_tuple(self.input_shape)
            if self.input_shape is False:
                raise ValueError(gettext('Parameter {} is invalid').format(
                    self.INPUT_SHAPE_PARAM))
            self.input_shape = """input_shape='{input_shape}'""" \
                .format(input_shape=self.input_shape)
            functions_required.append(self.input_shape)

        if self.padding is not None:
            self.padding = """padding='{padding}'""" \
                .format(padding=self.padding)
            functions_required.append(self.padding)

        if self.output_padding is not None:
            self.output_padding = """output_padding={output_padding}""" \
                .format(output_padding=self.output_padding)
            functions_required.append(self.output_padding)

        if self.data_format is not None:
            self.data_format = """data_format='{data_format}'""" \
                .format(data_format=self.data_format)
            functions_required.append(self.data_format)

        if self.dilation_rate is not None:
            self.dilation_rate = """dilation_rate={dilation_rate}""" \
                .format(dilation_rate=self.dilation_rate)
            functions_required.append(self.dilation_rate)

        if self.activation is not None:
            self.activation = """activation='{activation}'""" \
                .format(activation=self.activation)
            functions_required.append(self.activation)

        self.use_bias = """use_bias={use_bias}""".format(use_bias=self.use_bias)
        functions_required.append(self.use_bias)

        if self.kernel_initializer is not None:
            self.kernel_initializer = """kernel_initializer=
            '{kernel_initializer}'""" \
                .format(kernel_initializer=self.kernel_initializer)
            functions_required.append(self.kernel_initializer)

        if self.bias_initializer is not None:
            self.bias_initializer = """bias_initializer='{bias_initializer}'""" \
                .format(bias_initializer=self.bias_initializer)
            functions_required.append(self.bias_initializer)

        if self.kernel_regularizer is not None:
            self.kernel_regularizer = """kernel_regularizer=
            '{kernel_regularizer}'""" \
                .format(kernel_regularizer=self.kernel_regularizer)
            functions_required.append(self.kernel_regularizer)

        if self.bias_regularizer is not None:
            self.bias_regularizer = """bias_regularizer=
            '{bias_regularizer}'""" \
                .format(bias_regularizer=self.bias_regularizer)
            functions_required.append(self.bias_regularizer)

        if self.activity_regularizer is not None:
            self.activity_regularizer = """activity_regularizer=
            '{activity_regularizer}'""" \
                .format(activity_regularizer=self.activity_regularizer)
            functions_required.append(self.activity_regularizer)

        if self.kernel_constraint is not None:
            self.kernel_constraint = """kernel_constraint=
            '{kernel_constraint}'""" \
                .format(kernel_constraint=self.kernel_constraint)
            functions_required.append(self.kernel_constraint)

        if self.bias_constraint is not None:
            self.bias_constraint = """bias_constraint='{bias_constraint}'""" \
                .format(bias_constraint=self.bias_constraint)
            functions_required.append(self.bias_constraint)

        self.add_functions_required = ',\n    '.join(functions_required)
        if self.add_functions_required:
            self.add_functions_required = ',\n    ' + self.add_functions_required

    def generate_code(self):
        return dedent(
            """
            {var_name} = Conv2DTranspose(
                name='{name}'{add_functions_required}
            ){parent}
            """
        ).format(var_name=self.var_name,
                 name=self.task_name,
                 add_functions_required=self.add_functions_required,
                 parent=self.parent)


class Conv3D(Operation):
    FILTERS_PARAM = 'filters'
    KERNEL_SIZE_PARAM = 'kernel_size'
    STRIDES_PARAM = 'strides'
    INPUT_SHAPE_PARAM = 'input_shape'
    PADDING_PARAM = 'padding'
    DATA_FORMAT_PARAM = 'data_format'
    DILATION_RATE_PARAM = 'dilation_rate'
    ACTIVATION_PARAM = 'activation'
    USE_BIAS_PARAM = 'use_bias'
    KERNEL_INITIALIZER_PARAM = 'kernel_initializer'
    BIAS_INITIALIZER_PARAM = 'bias_initializer'
    KERNEL_REGULARIZER_PARAM = 'kernel_regularizer'
    BIAS_REGULARIZER_PARAM = 'bias_regularizer'
    ACTIVITY_REGULARIZER_PARAM = 'activity_regularizer'
    KERNEL_CONSTRAINT_PARAM = 'kernel_constraint'
    BIAS_CONSTRAINT_PARAM = 'bias_constraint'

    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)
        self.output = named_outputs.get('output data',
                                        'out_task_{}'.format(self.order))

        if self.FILTERS_PARAM not in parameters or \
                self.KERNEL_SIZE_PARAM not in parameters or \
                self.STRIDES_PARAM not in parameters:
            raise ValueError(gettext('Parameter {} {} {} are required').format(
                self.FILTERS_PARAM, self.KERNEL_SIZE_PARAM, self.STRIDES_PARAM)
            )

        self.filters = abs(parameters.get(self.FILTERS_PARAM))
        self.kernel_size = parameters.get(self.KERNEL_SIZE_PARAM)
        self.strides = parameters.get(self.STRIDES_PARAM)
        self.input_shape = parameters.get(self.INPUT_SHAPE_PARAM, None)
        self.padding = parameters.get(self.PADDING_PARAM)
        self.data_format = parameters.get(self.DATA_FORMAT_PARAM, None)
        self.dilation_rate = parameters.get(self.DILATION_RATE_PARAM, None) or \
                             None
        self.activation = parameters.get(self.ACTIVATION_PARAM, None)
        self.use_bias = parameters.get(self.USE_BIAS_PARAM)
        self.kernel_initializer = parameters.get(self.KERNEL_INITIALIZER_PARAM,
                                                 None)
        self.bias_initializer = parameters.get(self.BIAS_INITIALIZER_PARAM,
                                               None)
        self.kernel_regularizer = parameters.get(self.KERNEL_REGULARIZER_PARAM,
                                                 None)
        self.bias_regularizer = parameters.get(self.BIAS_REGULARIZER_PARAM,
                                               None)
        self.activity_regularizer = parameters.get(self.
                                                   ACTIVITY_REGULARIZER_PARAM,
                                                   None)
        self.kernel_constraint = parameters.get(self.KERNEL_CONSTRAINT_PARAM,
                                                None)
        self.bias_constraint = parameters.get(self.BIAS_CONSTRAINT_PARAM, None)\
                              

        self.add_functions_required = ""
        self.task_name = self.parameters.get('task').get('name')
        self.parent = ""
        self.var_name = ""
        self.has_code = True

        self.treatment()

        self.import_code = {'layer': 'Conv3D',
                            'callbacks': [],
                            'model': None,
                            'preprocessing_image': None,
                            'others': None}

    def treatment(self):
        self.parent = convert_parents_to_variable_name(self.parameters
                                                       .get('parents', []))
        self.var_name = convert_variable_name(self.task_name)
        self.task_name = self.var_name
        if self.parent:
            self.parent = '({})'.format(self.parent[0])
        else:
            self.parent = ''

        self.kernel_size = get_int_or_tuple(self.kernel_size)
        self.strides = get_int_or_tuple(self.strides)
        self.dilation_rate = get_int_or_tuple(self.dilation_rate)

        if self.filters < 0:
            raise ValueError(gettext('Parameter {} is invalid').format(
                self.FILTERS_PARAM))

        if self.kernel_size is False:
            raise ValueError(gettext('Parameter {} is invalid').format(
                self.KERNEL_SIZE_PARAM))

        if self.strides is False:
            raise ValueError(gettext('Parameter {} is invalid').format(
                self.STRIDES_PARAM))

        if self.dilation_rate is False:
            raise ValueError(gettext('Parameter {} is invalid').format(
                self.DILATION_RATE_PARAM))

        self.use_bias = True if int(self.use_bias) == 1 else False

        functions_required = []

        self.filters = """filters={filters}""".format(filters=self.filters)
        functions_required.append(self.filters)

        self.kernel_size = """kernel_size={kernel_size}""" \
            .format(kernel_size=self.kernel_size)
        functions_required.append(self.kernel_size)

        self.strides = """strides={strides}""".format(strides=self.strides)
        functions_required.append(self.strides)

        if self.input_shape is not None:
            self.input_shape = get_int_or_tuple(self.input_shape)
            if self.input_shape is False:
                raise ValueError(gettext('Parameter {} is invalid').format(
                    self.INPUT_SHAPE_PARAM))
            self.input_shape = """input_shape='{input_shape}'""" \
                .format(input_shape=self.input_shape)
            functions_required.append(self.input_shape)

        if self.padding is not None:
            self.padding = """padding='{padding}'""" \
                .format(padding=self.padding)
            functions_required.append(self.padding)

        if self.data_format is not None:
            self.data_format = """data_format='{data_format}'""" \
                .format(data_format=self.data_format)
            functions_required.append(self.data_format)

        if self.dilation_rate is not None:
            self.dilation_rate = """dilation_rate={dilation_rate}""" \
                .format(dilation_rate=self.dilation_rate)
            functions_required.append(self.dilation_rate)

        if self.activation is not None:
            self.activation = """activation='{activation}'""" \
                .format(activation=self.activation)
            functions_required.append(self.activation)

        self.use_bias = """use_bias={use_bias}""".format(use_bias=self.use_bias)
        functions_required.append(self.use_bias)

        if self.kernel_initializer is not None:
            self.kernel_initializer = """kernel_initializer=
            '{kernel_initializer}'""" \
                .format(kernel_initializer=self.kernel_initializer)
            functions_required.append(self.kernel_initializer)

        if self.bias_initializer is not None:
            self.bias_initializer = """bias_initializer='{bias_initializer}'""" \
                .format(bias_initializer=self.bias_initializer)
            functions_required.append(self.bias_initializer)

        if self.kernel_regularizer is not None:
            self.kernel_regularizer = """kernel_regularizer=
            '{kernel_regularizer}'""" \
                .format(kernel_regularizer=self.kernel_regularizer)
            functions_required.append(self.kernel_regularizer)

        if self.bias_regularizer is not None:
            self.bias_regularizer = """bias_regularizer=
            '{bias_regularizer}'""" \
                .format(bias_regularizer=self.bias_regularizer)
            functions_required.append(self.bias_regularizer)

        if self.activity_regularizer is not None:
            self.activity_regularizer = """activity_regularizer=
            '{activity_regularizer}'""" \
                .format(activity_regularizer=self.activity_regularizer)
            functions_required.append(self.activity_regularizer)

        if self.kernel_constraint is not None:
            self.kernel_constraint = """kernel_constraint=
            '{kernel_constraint}'""" \
                .format(kernel_constraint=self.kernel_constraint)
            functions_required.append(self.kernel_constraint)

        if self.bias_constraint is not None:
            self.bias_constraint = """bias_constraint='{bias_constraint}'""" \
                .format(bias_constraint=self.bias_constraint)
            functions_required.append(self.bias_constraint)

        self.add_functions_required = ',\n    '.join(functions_required)
        if self.add_functions_required:
            self.add_functions_required = ',\n    ' + self.add_functions_required

    def generate_code(self):
        return dedent(
            """
            {var_name} = Conv3D(
                name='{name}'{add_functions_required}
            ){parent}
            """
        ).format(var_name=self.var_name,
                 name=self.task_name,
                 add_functions_required=self.add_functions_required,
                 parent=self.parent)


class Conv3DTranspose(Operation):
    FILTERS_PARAM = 'filters'
    KERNEL_SIZE_PARAM = 'kernel_size'
    STRIDES_PARAM = 'strides'
    INPUT_SHAPE_PARAM = 'input_shape'
    PADDING_PARAM = 'padding'
    OUTPUT_PADDING_PARAM = 'output_padding'
    DATA_FORMAT_PARAM = 'data_format'
    DILATION_RATE_PARAM = 'dilation_rate'
    ACTIVATION_PARAM = 'activation'
    USE_BIAS_PARAM = 'use_bias'
    KERNEL_INITIALIZER_PARAM = 'kernel_initializer'
    BIAS_INITIALIZER_PARAM = 'bias_initializer'
    KERNEL_REGULARIZER_PARAM = 'kernel_regularizer'
    BIAS_REGULARIZER_PARAM = 'bias_regularizer'
    ACTIVITY_REGULARIZER_PARAM = 'activity_regularizer'
    KERNEL_CONSTRAINT_PARAM = 'kernel_constraint'
    BIAS_CONSTRAINT_PARAM = 'bias_constraint'

    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)
        self.output = named_outputs.get('output data',
                                        'out_task_{}'.format(self.order))

        if self.FILTERS_PARAM not in parameters or \
                self.KERNEL_SIZE_PARAM not in parameters or \
                self.STRIDES_PARAM not in parameters:
            raise ValueError(gettext('Parameter {} {} {} are required').format(
                self.FILTERS_PARAM, self.KERNEL_SIZE_PARAM, self.STRIDES_PARAM)
            )

        self.filters = abs(parameters.get(self.FILTERS_PARAM))
        self.kernel_size = parameters.get(self.KERNEL_SIZE_PARAM)
        self.strides = parameters.get(self.STRIDES_PARAM)
        self.input_shape = parameters.get(self.INPUT_SHAPE_PARAM, None)
        self.padding = parameters.get(self.PADDING_PARAM)
        self.output_padding = parameters.get(self.OUTPUT_PADDING_PARAM)
        self.data_format = parameters.get(self.DATA_FORMAT_PARAM, None)
        self.dilation_rate = parameters.get(self.DILATION_RATE_PARAM, None) or \
                             None
        self.activation = parameters.get(self.ACTIVATION_PARAM, None)
        self.use_bias = parameters.get(self.USE_BIAS_PARAM)
        self.kernel_initializer = parameters.get(self.KERNEL_INITIALIZER_PARAM,
                                                 None)
        self.bias_initializer = parameters.get(self.BIAS_INITIALIZER_PARAM,
                                               None)
        self.kernel_regularizer = parameters.get(self.KERNEL_REGULARIZER_PARAM,
                                                 None)
        self.bias_regularizer = parameters.get(self.BIAS_REGULARIZER_PARAM,
                                               None)
        self.activity_regularizer = parameters.get(self.
                                                   ACTIVITY_REGULARIZER_PARAM,
                                                   None)
        self.kernel_constraint = parameters.get(self.KERNEL_CONSTRAINT_PARAM,
                                                None)
        self.bias_constraint = parameters.get(self.BIAS_CONSTRAINT_PARAM, None) \
                              

        self.add_functions_required = ""
        self.task_name = self.parameters.get('task').get('name')
        self.parent = ""
        self.var_name = ""
        self.has_code = True

        self.treatment()

        self.import_code = {'layer': 'Conv3DTranspose',
                            'callbacks': [],
                            'model': None,
                            'preprocessing_image': None,
                            'others': None}

    def treatment(self):
        self.parent = convert_parents_to_variable_name(self.parameters
                                                       .get('parents', []))
        self.var_name = convert_variable_name(self.task_name)
        self.task_name = self.var_name
        if self.parent:
            self.parent = '({})'.format(self.parent[0])
        else:
            self.parent = ''

        self.kernel_size = get_int_or_tuple(self.kernel_size)
        self.strides = get_int_or_tuple(self.strides)
        self.dilation_rate = get_int_or_tuple(self.dilation_rate)
        self.output_padding = get_int_or_tuple(self.output_padding)

        if self.filters < 0:
            raise ValueError(gettext('Parameter {} is invalid').format(
                self.FILTERS_PARAM))

        if self.kernel_size is False:
            raise ValueError(gettext('Parameter {} is invalid').format(
                self.KERNEL_SIZE_PARAM))

        if self.strides is False:
            raise ValueError(gettext('Parameter {} is invalid').format(
                self.STRIDES_PARAM))

        if self.dilation_rate is False:
            raise ValueError(gettext('Parameter {} is invalid').format(
                self.DILATION_RATE_PARAM))

        if self.output_padding is False:
            raise ValueError(gettext('Parameter {} is invalid').format(
                self.OUTPUT_PADDING_PARAM))

        self.use_bias = True if int(self.use_bias) == 1 else False

        functions_required = []

        self.filters = """filters={filters}""".format(filters=self.filters)
        functions_required.append(self.filters)

        self.kernel_size = """kernel_size={kernel_size}""" \
            .format(kernel_size=self.kernel_size)
        functions_required.append(self.kernel_size)

        self.strides = """strides={strides}""".format(strides=self.strides)
        functions_required.append(self.strides)

        if self.input_shape is not None:
            self.input_shape = get_int_or_tuple(self.input_shape)
            if self.input_shape is False:
                raise ValueError(gettext('Parameter {} is invalid').format(
                    self.INPUT_SHAPE_PARAM))
            self.input_shape = """input_shape='{input_shape}'""" \
                .format(input_shape=self.input_shape)
            functions_required.append(self.input_shape)

        if self.padding is not None:
            self.padding = """padding='{padding}'""" \
                .format(padding=self.padding)
            functions_required.append(self.padding)

        if self.output_padding is not None:
            self.output_padding = """output_padding={output_padding}""" \
                .format(output_padding=self.output_padding)
            functions_required.append(self.output_padding)

        if self.data_format is not None:
            self.data_format = """data_format='{data_format}'""" \
                .format(data_format=self.data_format)
            functions_required.append(self.data_format)

        if self.dilation_rate is not None:
            self.dilation_rate = """dilation_rate={dilation_rate}""" \
                .format(dilation_rate=self.dilation_rate)
            functions_required.append(self.dilation_rate)

        if self.activation is not None:
            self.activation = """activation='{activation}'""" \
                .format(activation=self.activation)
            functions_required.append(self.activation)

        self.use_bias = """use_bias={use_bias}""".format(use_bias=self.use_bias)
        functions_required.append(self.use_bias)

        if self.kernel_initializer is not None:
            self.kernel_initializer = """kernel_initializer=
            '{kernel_initializer}'""" \
                .format(kernel_initializer=self.kernel_initializer)
            functions_required.append(self.kernel_initializer)

        if self.bias_initializer is not None:
            self.bias_initializer = """bias_initializer='{bias_initializer}'""" \
                .format(bias_initializer=self.bias_initializer)
            functions_required.append(self.bias_initializer)

        if self.kernel_regularizer is not None:
            self.kernel_regularizer = """kernel_regularizer=
            '{kernel_regularizer}'""" \
                .format(kernel_regularizer=self.kernel_regularizer)
            functions_required.append(self.kernel_regularizer)

        if self.bias_regularizer is not None:
            self.bias_regularizer = """bias_regularizer=
            '{bias_regularizer}'""" \
                .format(bias_regularizer=self.bias_regularizer)
            functions_required.append(self.bias_regularizer)

        if self.activity_regularizer is not None:
            self.activity_regularizer = """activity_regularizer=
            '{activity_regularizer}'""" \
                .format(activity_regularizer=self.activity_regularizer)
            functions_required.append(self.activity_regularizer)

        if self.kernel_constraint is not None:
            self.kernel_constraint = """kernel_constraint=
            '{kernel_constraint}'""" \
                .format(kernel_constraint=self.kernel_constraint)
            functions_required.append(self.kernel_constraint)

        if self.bias_constraint is not None:
            self.bias_constraint = """bias_constraint='{bias_constraint}'""" \
                .format(bias_constraint=self.bias_constraint)
            functions_required.append(self.bias_constraint)

        self.add_functions_required = ',\n    '.join(functions_required)
        if self.add_functions_required:
            self.add_functions_required = ',\n    ' + self.add_functions_required

    def generate_code(self):
        return dedent(
            """
            {var_name} = Conv3DTranspose(
                name='{name}'{add_functions_required}
            ){parent}
            """
        ).format(var_name=self.var_name,
                 name=self.task_name,
                 add_functions_required=self.add_functions_required,
                 parent=self.parent)


class Cropping1D(Operation):
    CROPPING_PARAM = 'cropping'
    INPUT_SHAPE_PARAM = 'input_shape'

    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)
        self.output = named_outputs.get('output data',
                                        'out_task_{}'.format(self.order))

        if self.CROPPING_PARAM not in parameters:
            raise ValueError(gettext('Parameter {} are required').format(
                self.CROPPING_PARAM)
            )

        self.cropping = abs(parameters.get(self.CROPPING_PARAM))
        self.input_shape = parameters.get(self.INPUT_SHAPE_PARAM, None)

        self.add_functions_required = ""
        self.task_name = self.parameters.get('task').get('name')
        self.parent = ""
        self.var_name = ""
        self.has_code = True

        self.treatment()

        self.import_code = {'layer': 'Cropping1D',
                            'callbacks': [],
                            'model': None,
                            'preprocessing_image': None,
                            'others': None}

    def treatment(self):
        self.parent = convert_parents_to_variable_name(self.parameters
                                                       .get('parents', []))
        self.var_name = convert_variable_name(self.task_name)
        self.task_name = self.var_name
        if self.parent:
            self.parent = '({})'.format(self.parent[0])
        else:
            self.parent = ''

        self.cropping = tuple_of_tuples(self.cropping)

        if self.cropping is False:
            raise ValueError(gettext('Parameter {} is invalid').format(
                self.CROPPING_PARAM))

        functions_required = []

        self.cropping = """cropping={cropping}""".format(cropping=self.cropping)
        functions_required.append(self.cropping)

        if self.input_shape is not None:
            self.input_shape = get_int_or_tuple(self.input_shape)
            if self.input_shape is False:
                raise ValueError(gettext('Parameter {} is invalid').format(
                    self.INPUT_SHAPE_PARAM))
            self.input_shape = """input_shape='{input_shape}'""" \
                .format(input_shape=self.input_shape)
            functions_required.append(self.input_shape)

        self.add_functions_required = ',\n    '.join(functions_required)
        if self.add_functions_required:
            self.add_functions_required = ',\n    ' + self.add_functions_required

    def generate_code(self):
        return dedent(
            """
            {var_name} = Cropping1D(
                name='{name}'{add_functions_required}
            ){parent}
            """
        ).format(var_name=self.var_name,
                 name=self.task_name,
                 add_functions_required=self.add_functions_required,
                 parent=self.parent)


class Cropping2D(Operation):
    CROPPING_PARAM = 'cropping'
    INPUT_SHAPE_PARAM = 'input_shape'
    DATA_FORMAT_PARAM = 'data_format'

    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)
        self.output = named_outputs.get('output data',
                                        'out_task_{}'.format(self.order))

        if self.CROPPING_PARAM not in parameters:
            raise ValueError(gettext('Parameter {} are required').format(
                self.CROPPING_PARAM)
            )

        self.cropping = abs(parameters.get(self.CROPPING_PARAM))
        self.input_shape = parameters.get(self.INPUT_SHAPE_PARAM, None)
        self.data_format = parameters.get(self.DATA_FORMAT_PARAM, None)

        self.add_functions_required = ""
        self.task_name = self.parameters.get('task').get('name')
        self.parent = ""
        self.var_name = ""
        self.has_code = True

        self.treatment()

        self.import_code = {'layer': 'Cropping2D',
                            'callbacks': [],
                            'model': None,
                            'preprocessing_image': None,
                            'others': None}

    def treatment(self):
        self.parent = convert_parents_to_variable_name(self.parameters
                                                       .get('parents', []))
        self.var_name = convert_variable_name(self.task_name)
        self.task_name = self.var_name
        if self.parent:
            self.parent = '({})'.format(self.parent[0])
        else:
            self.parent = ''

        self.cropping = tuple_of_tuples(self.cropping)

        if self.cropping is False:
            raise ValueError(gettext('Parameter {} is invalid').format(
                self.CROPPING_PARAM))

        functions_required = []

        self.cropping = """cropping={cropping}""".format(cropping=self.cropping)
        functions_required.append(self.cropping)

        if self.input_shape is not None:
            self.input_shape = get_int_or_tuple(self.input_shape)
            if self.input_shape is False:
                raise ValueError(gettext('Parameter {} is invalid').format(
                    self.INPUT_SHAPE_PARAM))
            self.input_shape = """input_shape='{input_shape}'""" \
                .format(input_shape=self.input_shape)
            functions_required.append(self.input_shape)

        if self.data_format is not None:
            self.data_format = """data_format='{data_format}'""" \
                .format(data_format=self.data_format)
            functions_required.append(self.data_format)

        self.add_functions_required = ',\n    '.join(functions_required)
        if self.add_functions_required:
            self.add_functions_required = ',\n    ' + self.add_functions_required

    def generate_code(self):
        return dedent(
            """
            {var_name} = Cropping2D(
                name='{name}'{add_functions_required}
            ){parent}
            """
        ).format(var_name=self.var_name,
                 name=self.task_name,
                 add_functions_required=self.add_functions_required,
                 parent=self.parent)


class Cropping3D(Operation):
    CROPPING_PARAM = 'cropping'
    INPUT_SHAPE_PARAM = 'input_shape'
    DATA_FORMAT_PARAM = 'data_format'

    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)
        self.output = named_outputs.get('output data',
                                        'out_task_{}'.format(self.order))

        if self.CROPPING_PARAM not in parameters:
            raise ValueError(gettext('Parameter {} are required').format(
                self.CROPPING_PARAM)
            )

        self.cropping = abs(parameters.get(self.CROPPING_PARAM))
        self.input_shape = parameters.get(self.INPUT_SHAPE_PARAM, None)
        self.data_format = parameters.get(self.DATA_FORMAT_PARAM, None)

        self.add_functions_required = ""
        self.task_name = self.parameters.get('task').get('name')
        self.parent = ""
        self.var_name = ""
        self.has_code = True

        self.treatment()

        self.import_code = {'layer': 'Cropping3D',
                            'callbacks': [],
                            'model': None,
                            'preprocessing_image': None,
                            'others': None}

    def treatment(self):
        self.parent = convert_parents_to_variable_name(self.parameters
                                                       .get('parents', []))
        self.var_name = convert_variable_name(self.task_name)
        self.task_name = self.var_name
        if self.parent:
            self.parent = '({})'.format(self.parent[0])
        else:
            self.parent = ''

        self.cropping = tuple_of_tuples(self.cropping)

        if self.cropping is False:
            raise ValueError(gettext('Parameter {} is invalid').format(
                self.CROPPING_PARAM))

        functions_required = []

        self.cropping = """cropping={cropping}""".format(cropping=self.cropping)
        functions_required.append(self.cropping)

        if self.input_shape is not None:
            self.input_shape = get_int_or_tuple(self.input_shape)
            if self.input_shape is False:
                raise ValueError(gettext('Parameter {} is invalid').format(
                    self.INPUT_SHAPE_PARAM))
            self.input_shape = """input_shape='{input_shape}'""" \
                .format(input_shape=self.input_shape)
            functions_required.append(self.input_shape)

        if self.data_format is not None:
            self.data_format = """data_format='{data_format}'""" \
                .format(data_format=self.data_format)
            functions_required.append(self.data_format)

        self.add_functions_required = ',\n    '.join(functions_required)
        if self.add_functions_required:
            self.add_functions_required = ',\n    ' + self.add_functions_required

    def generate_code(self):
        return dedent(
            """
            {var_name} = Cropping3D(
                name='{name}'{add_functions_required}
            ){parent}
            """
        ).format(var_name=self.var_name,
                 name=self.task_name,
                 add_functions_required=self.add_functions_required,
                 parent=self.parent)
