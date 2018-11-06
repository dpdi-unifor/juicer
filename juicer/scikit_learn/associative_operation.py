# -*- coding: utf-8 -*-
from textwrap import dedent
from juicer.operation import Operation


class FrequentItemSetOperation(Operation):
    """FP-growth"""

    MIN_SUPPORT_PARAM = 'min_support'
    ATTRIBUTE_PARAM = 'attribute'
    CONFIDENCE_PARAM = 'confidence'

    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)

        self.has_code = all([len(named_inputs) == 1,
                             self.contains_results() or len(named_outputs) > 0])
        if self.has_code:

            if self.MIN_SUPPORT_PARAM not in parameters:
                raise ValueError(
                        _("Parameter '{}' must be informed for task {}").format(
                                self.MIN_SUPPORT_PARAM, self.__class__))

            self.perform_genRules = 'rules output' in self.named_outputs \
                                    or self.contains_results()

            self.column = parameters.get(self.ATTRIBUTE_PARAM, [''])[0]
            self.confidence = float(parameters.get(self.CONFIDENCE_PARAM, 0.9))
            self.min_support = float(parameters.get(self.MIN_SUPPORT_PARAM))
            if self.min_support < .0001 or self.min_support > 1.0:
                raise ValueError('Support must be greater or equal '
                                 'to 0.0001 and smaller than 1.0')

            self.output = named_outputs.get('output data',
                                            'output_data_{}'.format(
                                                         self.order))
            self.rules_output = named_outputs.get('rules output',
                                                  'rules_{}'.format(
                                                           self.order))

            self.has_import = "from fim import fpgrowth\n" + \
                              "from juicer.scikit_learn.library." \
                              "rules_generator import RulesGenerator\n"

    def get_output_names(self, sep=', '):
        return sep.join([self.output,
                         self.rules_output])

    def generate_code(self):
        """Generate code."""

        if not len(self.column) > 1:
            self.column = "{input}.columns[0]" \
                .format(input=self.named_inputs['input data'])
        else:
            self.column = "'{}'".format(self.column)

        code = """
        col = {col}
        transactions = {input}[col].values.tolist()
        min_support = 100 * {min_support}
        
        result = fpgrowth(transactions, target="s",
          supp=min_support, report="s")
        
        {output} = pd.DataFrame(result, columns=['itemsets', 'support'])
        """.format(output=self.output, col=self.column,
                   input=self.named_inputs['input data'],
                   min_support=self.min_support)

        if self.perform_genRules:
            code += """
        
        # generating rules
        col_item = 'itemsets'
        col_freq = 'support'
        rg = RulesGenerator(min_conf={min_conf}, max_len=-1)
        {rules} = rg.get_rules({output}, col_item, col_freq)    
        """.format(min_conf=self.confidence, output=self.output,
                   rules=self.rules_output)
        else:
            code += """
        {rules} = None
        """.format(rules=self.rules_output)
        return dedent(code)


class SequenceMiningOperation(Operation):

    MIN_SUPPORT_PARAM = 'min_support'
    ATTRIBUTE_PARAM = 'attribute'
    MAX_LENGTH_PARAM = 'max_length'

    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)

        self.has_code = all([len(named_inputs) == 1,
                             self.contains_results() or len(named_outputs) > 0])
        if self.has_code:
            if self.MIN_SUPPORT_PARAM not in parameters:
                raise ValueError(
                        _("Parameter '{}' must be informed for task {}").format(
                            self.MIN_SUPPORT_PARAM, self.__class__))

            self.column = parameters.get(self.ATTRIBUTE_PARAM, [''])[0]
            self.output = self.named_outputs.get('output data',
                                                 'output_data_{}'.format(
                                                         self.order))

            self.min_support = float(parameters.get(self.MIN_SUPPORT_PARAM))
            if self.min_support < .0001 or self.min_support > 1.0:
                raise ValueError('Support must be greater or equal '
                                 'to 0.0001 and smaller than 1.0')

            self.max_length = abs(int(parameters.get(self.MAX_LENGTH_PARAM,
                                                     10))) or 10
            self.has_import = \
                "from juicer.scikit_learn.library." \
                "prefix_span import PrefixSpan\n"

    def generate_code(self):
        """Generate code."""

        if not len(self.column) > 1:
            self.column = "{input}.columns[0]" \
                .format(input=self.named_inputs['input data'])
        else:
            self.column = "'{}'".format(self.column)

        code = """
        col = {col}
        transactions = {input}[col].values.tolist()
        min_support = {min_support}
        max_length = {max_length}

        span = PrefixSpan(transactions)
        span.run(min_support, max_length)
        result = span.get_patest_text_operations.pyatterns()

        {output} = pd.DataFrame(result, columns=['itemsets', 'support'])
        """.format(output=self.output, col=self.column,
                   input=self.named_inputs['input data'],
                   min_support=self.min_support,
                   max_length=self.max_length)

        return dedent(code)


class AssociationRulesOperation(Operation):
    """AssociationRulesOperation.
    """

    MAX_COUNT_PARAM = 'rules_count'
    CONFIDENCE_PARAM = 'confidence'

    ITEMSET_ATTR_PARAM = 'item_col'
    SUPPORT_ATTR_PARAM = 'support_col'
    SUPPORT_ATTR_PARAM_VALUE = 'support'

    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)

        self.has_code = all([len(named_inputs) == 1,
                             self.contains_results() or len(named_outputs) > 0])
        if self.has_code:
            self.output = named_outputs.get('output data',
                                            'output_data_{}'.format(self.order))

            self.confidence = float(parameters.get(self.CONFIDENCE_PARAM, 0.5))
            if self.confidence < .0001 or self.confidence > 1.0:
                raise ValueError('Confidence must be greater or equal '
                                 'to 0.0001 and smaller than 1.0')

            self.has_import = \
                "from juicer.scikit_learn.library.rules_generator " \
                "import RulesGenerator\n"

            self.support_col = \
                parameters.get(self.SUPPORT_ATTR_PARAM,
                               [self.SUPPORT_ATTR_PARAM_VALUE])[0]
            self.items_col = parameters.get(self.ITEMSET_ATTR_PARAM, [''])[0]
            self.max_rules = parameters.get(self.MAX_COUNT_PARAM, -1) or -1

    def generate_code(self):
        """Generate code."""

        if len(self.items_col) == 0:
            self.items_col = "{input}.columns[0]" \
                .format(input=self.named_inputs['input data'])
        else:
            self.items_col = "'{}'".format(self.items_col)

        code = """
        col_item = {items}
        col_freq = "{freq}"
        
        rg = RulesGenerator(min_conf={min_conf}, max_len={max_len})
        {rules} = rg.get_rules({input}, col_item, col_freq)   
        """.format(min_conf=self.confidence, rules=self.output,
                   input=self.named_inputs['input data'],
                   items=self.items_col, freq=self.support_col,
                   max_len=self.max_rules)

        return dedent(code)