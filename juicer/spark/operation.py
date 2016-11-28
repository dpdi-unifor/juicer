# coding=utf-8
import ast
import json
import logging
import time
from random import random
from textwrap import dedent

from expression import Expression
from juicer.dist.metadata import MetadataGet

log = logging.getLogger()
log.setLevel(logging.DEBUG)


class Operation:
    """ Defines an operation in Lemonade """

    def __init__(self, parameters, inputs, outputs):
        self.parameters = parameters
        self.inputs = inputs
        self.outputs = outputs

        # Indicate if operation generates code or not. Some operations, e.g.
        # Comment, does not generate code
        self.has_code = len(self.inputs) > 0 or len(self.outputs) > 0

    @property
    def get_inputs_names(self):
        return ', '.join(self.inputs)

    def get_output_names(self, sep=", "):
        result = ''
        if len(self.outputs) > 0:
            result = sep.join(self.outputs)
        elif len(self.inputs) > 0:
            result = '{}_tmp'.format(self.inputs[0])
        else:
            raise ValueError(
                "Operation has neither input nor output: {}".format(
                    self.__class__))
        return result

    def test_null_operation(self):
        """
        Test if an operation is null, i.e, does nothing.
        An operation does nothing if it has zero inputs or outputs.
        """
        return any([len(self.outputs) == 0, len(self.inputs) == 0])


class DataReader(Operation):
    """
    Reads a database.
    Parameters:
    - Limonero database ID
    """
    DATA_SOURCE_ID_PARAM = 'data_source'
    HEADER_PARAM = 'header'
    SEPARATOR_PARAM = 'separator'
    INFER_SCHEMA_PARAM = 'infer_schema'

    def __init__(self, parameters, inputs, outputs):
        Operation.__init__(self, parameters, inputs, outputs)
        if self.DATA_SOURCE_ID_PARAM in parameters:
            self.database_id = parameters[self.DATA_SOURCE_ID_PARAM]
            self.header = bool(parameters.get(self.HEADER_PARAM, False))
            self.sep = parameters.get(self.SEPARATOR_PARAM, ',')
            self.infer_schema = bool(parameters.get(self.INFER_SCHEMA_PARAM, True))

            metadata_obj = MetadataGet('123456')
            self.metadata = metadata_obj.get_metadata(self.database_id)
        else:
            raise ValueError(
                "Parameter '{}' must be informed for task {}".format(
                    self.DATA_SOURCE_ID_PARAM, self.__class__))

    def generate_code(self):

        # For now, just accept CSV files.
        # Should we create a dict with the CSV info at Limonero?
        # such as header and sep.
        #print "\n\n",self.metadata,"\n\n"
        code = ''
        if len(self.outputs) == 1:
            if self.metadata['format'] == 'CSV':
                code = """{} = spark.read.csv('{}',
                header={}, sep='{}', inferSchema={})""".format(
                    self.outputs[0], self.metadata['url'],
                    self.header, self.sep, self.infer_schema)

            elif self.metadata['format'] == 'PARQUET_FILE':
                # TO DO
                pass
            elif self.metadata['format'] == 'JSON_FILE':
                # TO DO
                pass
        else:
            code = ''

        return dedent(code)


class RandomSplit(Operation):
    """
    Randomly splits the Data Frame into two data frames.
    Parameters:
    - List with two weights for the two new data frames.
    - Optional seed in case of deterministic random operation
    ('0' means no seed).
    """
    SEED_PARAM = 'seed'
    WEIGHTS_PARAM = 'weights'

    def __init__(self, parameters, inputs, outputs):
        Operation.__init__(self, parameters, inputs, outputs)
        value = float(parameters.get(self.WEIGHTS_PARAM, 50))
        self.weights = [value, 100 - value]
        self.seed = parameters.get(self.SEED_PARAM, int(random() * time.time()))

    def generate_code(self):
        if len(self.inputs) == 1:
            output1 = self.outputs[0] if len(
                self.outputs) else '{}_tmp1'.format(
                self.inputs[0])
            output2 = self.outputs[1] if len(
                self.outputs) == 2 else '{}_tmp2'.format(
                self.inputs[0])
            code = """{0}, {1} = {2}.randomSplit({3}, {4})""".format(
                output1, output2, self.inputs[0],
                json.dumps(self.weights), self.seed)
        else:
            code = ""
        return dedent(code)


class AddRows(Operation):
    """
    Return a new DataFrame containing all rows in this frame and another frame.
    Takes no parameters. 
    """

    def __init__(self, parameters, inputs, outputs):
        Operation.__init__(self, parameters, inputs, outputs)
        self.parameters = parameters

    def generate_code(self):
        output = self.outputs[0] if len(self.outputs) else '{}_tmp'.format(
            self.inputs[0])

        if len(self.inputs) == 2:
            code = "{0} = {1}.unionAll({2})".format(output,
                                                    self.inputs[0],
                                                    self.inputs[1])
        else:
            code = ""
        return dedent(code)


class Sort(Operation):
    """ 
    Returns a new DataFrame sorted by the specified column(s).
    Parameters:
    - The list of columns to be sorted.
    - A list indicating whether the sort order is ascending for the columns.
    Condition: the list of columns should have the same size of the list of 
               boolean to indicating if it is ascending sorting.
    """
    ATTRIBUTES_PARAM = 'attributes'
    ASCENDING_PARAM = 'ascending'

    def __init__(self, parameters, inputs, outputs):
        Operation.__init__(self, parameters, inputs, outputs)
        if self.ATTRIBUTES_PARAM in parameters:
            self.attributes = parameters.get(self.ATTRIBUTES_PARAM)
        else:
            raise ValueError(
                "Parameter '{}' must be informed for task {}".format(
                    self.ATTRIBUTES_PARAM, self.__class__))

        self.ascending = map(lambda x: int(x),
                             parameters.get(self.ASCENDING_PARAM,
                                            [1] * len(self.attributes)))

    def generate_code(self):
        output = self.outputs[0] if len(self.outputs) else '{}_tmp'.format(
            self.inputs[0])
        code = "{0} = {1}.orderBy({2}, ascending={3})".format(
            output, self.inputs[0],
            json.dumps(self.attributes),
            json.dumps(self.ascending))

        return dedent(code)


class Distinct(Operation):
    """
    Returns a new DataFrame containing the distinct rows in this DataFrame.
    Parameters: attributes to consider during operation (keys)
    """
    ATTRIBUTES_PARAM = 'attributes'

    def __init__(self, parameters, inputs, outputs):
        Operation.__init__(self, parameters, inputs, outputs)
        if self.ATTRIBUTES_PARAM in parameters:
            self.attributes = parameters.get(self.ATTRIBUTES_PARAM)
        else:
            self.attributes = []

    def generate_code(self):
        output = self.outputs[0] if len(self.outputs) else '{}_tmp'.format(
            self.inputs[0])
        if len(self.inputs) == 1:
            if self.attributes:
                code = "{} = {}.dropDuplicates(subset={})".format(
                    output, self.inputs[0], json.dumps(self.attributes))
            else:
                code = "{} = {}.dropDuplicates()".format(output, self.inputs[0])
        else:
            code = ""

        return dedent(code)


class Sample(Operation):
    """
    Returns a sampled subset of this DataFrame.
    Parameters:
    - withReplacement -> can elements be sampled multiple times
                        (replaced when sampled out)
    - fraction -> fraction of the data frame to be sampled.
        without replacement: probability that each element is chosen; 
            fraction must be [0, 1]
        with replacement: expected number of times each element is chosen; 
            fraction must be >= 0
    - seed -> seed for random operation.
    """
    FRACTION_PARAM = 'fraction'
    SEED_PARAM = 'seed'
    WITH_REPLACEMENT_PARAM = 'withReplacement'

    def __init__(self, parameters, inputs, outputs):
        Operation.__init__(self, parameters, inputs, outputs)
        if self.FRACTION_PARAM in parameters:
            self.withReplacement = parameters.get(self.WITH_REPLACEMENT_PARAM,
                                                  False)
            self.fraction = float(parameters[self.FRACTION_PARAM])
            if not (0 <= self.fraction <= 100):
                msg = "Parameter '{}' must be in range [0, 100] for task {}" \
                    .format(self.FRACTION_PARAM, __name__)
                raise ValueError(msg)
            if self.fraction > 1.0:
                self.fraction *= 0.01

            self.seed = parameters.get(self.SEED_PARAM,
                                       int(random() * time.time()))
        else:
            raise ValueError(
                "Parameter '{}' must be informed for task {}".format(
                    self.FRACTION_PARAM, self.__class__))

    def generate_code(self):
        if len(self.outputs) > 0:
            output = self.outputs[0]
        else:
            output = '{}_tmp'.format(self.inputs[0])

        code = "{} = {}.sample(withReplacement={}, fraction={}, seed={})" \
            .format(output, self.inputs[0], self.withReplacement,
                    self.fraction, self.seed)

        return dedent(code)


class Intersection(Operation):
    """
    Returns a new DataFrame containing rows only in both this frame 
    and another frame.
    """

    def __init__(self, parameters, inputs, outputs):
        Operation.__init__(self, parameters, inputs, outputs)
        self.parameters = parameters

    def generate_code(self):
        if len(self.inputs) == 2:
            output = self.outputs[0] if len(self.outputs) else '{}_tmp'.format(
                self.inputs[0])

            code = "{} = {}.intersect({})".format(
                output, self.inputs[0], self.inputs[1])
        else:
            code = ''
        return dedent(code)


class Difference(Operation):
    """
    Returns a new DataFrame containing rows in this frame but not in another
    frame.
    """

    def __init__(self, parameters, inputs, outputs):
        Operation.__init__(self, parameters, inputs, outputs)

    def generate_code(self):
        code = "{} = {}.subtract({})".format(
            self.outputs[0], self.inputs[0], self.inputs[1])
        return dedent(code)


class Join(Operation):
    """
    Joins with another DataFrame, using the given join expression.
    The expression must be defined as a string parameter.
    """
    KEEP_RIGHT_KEYS_PARAM = 'keep_right_keys'
    MATCH_CASE_PARAM = 'match_case'
    JOIN_TYPE_PARAM = 'join_type'
    LEFT_ATTRIBUTES_PARAM = 'left_attributes'
    RIGHT_ATTRIBUTES_PARAM = 'right_attributes'

    def __init__(self, parameters, inputs, outputs):
        Operation.__init__(self, parameters, inputs, outputs)
        self.keep_right_keys = parameters.get(self.KEEP_RIGHT_KEYS_PARAM, False)
        self.match_case = parameters.get(self.MATCH_CASE_PARAM, False)
        self.join_type = parameters.get(self.JOIN_TYPE_PARAM, 'inner')

        if not all([self.LEFT_ATTRIBUTES_PARAM in parameters,
                    self.RIGHT_ATTRIBUTES_PARAM in parameters]):
            raise ValueError(
                "Parameters '{}' and {} must be informed for task {}".format(
                    self.LEFT_ATTRIBUTES_PARAM, self.RIGHT_ATTRIBUTES_PARAM,
                    self.__class__))
        else:
            self.left_attributes = parameters.get(self.LEFT_ATTRIBUTES_PARAM)
            self.right_attributes = parameters.get(self.RIGHT_ATTRIBUTES_PARAM)

    def generate_code(self):
        on_clause = zip(self.left_attributes, self.right_attributes)
        join_condition = ', '.join([
                                       '{}.{} == {}.{}'.format(self.inputs[0],
                                                               pair[0],
                                                               self.inputs[1],
                                                               pair[1]) for pair
                                       in on_clause])

        output = self.outputs[0] if len(self.outputs) else '{}_tmp'.format(
            self.inputs[0])

        code = """
            cond_{0} = [{1}]
            {0} = {2}.join({3}, on=cond_{0}, how='{4}')""".format(
            output, join_condition, self.inputs[0], self.inputs[1],
            self.join_type)

        # TO-DO: Convert str False to boolean for evaluation
        if self.keep_right_keys == "False":
            for column in self.right_attributes:
                code += """.drop({}.{})""".format(self.inputs[1], column)

        return dedent(code)


class ReadCSV(Operation):
    """
    Reads a CSV file without HDFS.
    The purpose of this operation is to read files in
    HDFS without using the Limonero API.
    """

    def __init__(self, parameters, inputs, outputs):
        Operation.__init__(self, parameters, inputs, outputs)
        self.url = parameters['url']
        try:
            self.header = parameters['header']
        except KeyError:
            self.header = "True"
        try:
            self.separator = parameters['separator']
        except KeyError:
            self.separator = ";"

    def generate_code(self):
        code = """{} = spark.read.csv('{}',
            header={}, sep='{}' ,inferSchema=True)""".format(
            self.outputs[0], self.url, self.header, self.separator)
        return dedent(code)


class Drop(Operation):
    """
    Returns a new DataFrame that drops the specified column.
    Nothing is done if schema doesn't contain the given column name(s).
    The only parameters is the name of the columns to be removed.
    """

    def __init__(self, parameters, inputs, outputs):
        Operation.__init__(self, parameters, inputs, outputs)
        self.column = parameters['column']

    def generate_code(self):
        code = """{} = {}.drop('{}')""".format(
            self.outputs[0], self.inputs[0], self.column)
        return dedent(code)


# class Transformation(Operation):
#     """
#     Returns a new DataFrame applying the expression to the specified column.
#     Parameters:
#         - Alias: new column name. If the name is the same of an existing,
#         replace it.
#         - Expression: json describing the transformation expression
#     """
#
#     def __init__(self, parameters, inputs, outputs):
#         Operation.__init__(self, inputs, outputs)
#         self.alias = parameters['alias']
#         self.json_expression = parameters['expression']
#
#     def generate_code(self):
#         # Builds the expression and identify the target column
#         expression = Expression(self.json_expression)
#         built_expression = expression.parsed_expression
#         # Builds the code
#         code = """{} = {}.withColumn('{}', {})""".format(self.outputs[0],
#                                                          self.inputs[0],
#                                                          self.alias,
#                                                          built_expression)
#         return dedent(code)

class Transformation(Operation):
    """
    Returns a new DataFrame applying the expression to the specified column.
    Parameters:
        - Alias: new column name. If the name is the same of an existing,
        replace it.
        - Expression: json describing the transformation expression
    """
    ALIAS_PARAM = 'alias'
    EXPRESSION_PARAM = 'expression'

    def __init__(self, parameters, inputs, outputs):
        Operation.__init__(self, parameters, inputs, outputs)
        if all(['alias' in parameters, 'expression' in parameters]):
            self.alias = parameters['alias']
            self.json_expression = json.loads(parameters['expression'])['tree']
        else:
            raise ValueError(
                "Parameters '{}' and {} must be informed for task {}".format(
                    self.ALIAS_PARAM, self.EXPRESSION_PARAM, self.__class__))

    def generate_code(self):
        if len(self.inputs) > 0:
            # Builds the expression and identify the target column
            expression = Expression(self.json_expression)
            built_expression = expression.parsed_expression
            if len(self.outputs) > 0:
                output = self.outputs[0]
            else:
                output = '{}_tmp'.format(self.inputs[0])

            # Builds the code
            code = """{} = {}.withColumn('{}', {})""".format(output,
                                                             self.inputs[0],
                                                             self.alias,
                                                             built_expression)
        else:
            code = ''
        return dedent(code)



class Select(Operation):
    """
    Projects a set of expressions and returns a new DataFrame.
    Parameters:
    - The list of columns selected.
    """
    ATTRIBUTES_PARAM = 'attributes'
    ASCENDING_PARAM = 'ascending'

    def __init__(self, parameters, inputs, outputs):
        Operation.__init__(self, parameters, inputs, outputs)
        if self.ATTRIBUTES_PARAM in parameters:
            self.attributes = parameters.get(self.ATTRIBUTES_PARAM)
        else:
            raise ValueError(
                "Parameter '{}' must be informed for task {}".format(
                    self.ATTRIBUTES_PARAM, self.__class__))

    def generate_code(self):
        output = self.outputs[0] if len(self.outputs) else '{}_tmp'.format(
            self.inputs[0])

        code = """{} = {}.select({})""".format(
            output, self.inputs[0],
            ', '.join(['"{}"'.format(x) for x in self.attributes]))
        return dedent(code)


class Aggregation(Operation):
    """
    Compute aggregates and returns the result as a DataFrame.
    Parameters:
        - Expression: a single dict mapping from string to string, then the key
        is the column to perform aggregation on, and the value is the aggregate
        function. The available aggregate functions are avg, max, min, sum,
        count.
    """
    ATTRIBUTES_PARAM = 'attributes'
    FUNCTION_PARAM = 'function'

    def __init__(self, parameters, inputs, outputs):
        Operation.__init__(self, parameters, inputs, outputs)
        self.group_by = map(lambda x: str(x), parameters['group_by'])
        self.columns = map(lambda x: str(x), parameters['columns'])
        self.function = map(lambda x: str(x), parameters['functions'])
        self.names = map(lambda x: str(x), parameters['new_names'])

        if not all([self.ATTRIBUTES_PARAM in parameters,
                    self.FUNCTION_PARAM in parameters]):
            raise ValueError(
                "Parameters '{}' and {} must be informed for task {}".format(
                    self.ATTRIBUTES_PARAM, self.FUNCTION_PARAM, self.__class__))
        self.attributes = parameters.get(self.ATTRIBUTES_PARAM)
        self.function = parameters['function']

    def generate_code(self):
        elements = []
        for i in range(0, len(self.columns)):
            content = '''{}('{}').alias('{}')'''.format(self.function[i],
                                                        self.columns[i],
                                                        self.names[i])
            elements.append(content)
        code = '''{} = {}.groupBy({}).agg({})'''.format(
            self.outputs[0], self.inputs[0], self.group_by, ', '.join(elements))


'''

def generate_code(self):
    info = {self.attributes: self.function}
    output = self.outputs[0] if len(self.outputs) else '{}_tmp'.format(
        self.inputs[0])
    if len(self.inputs) == 1:
        code = """{} = {}.groupBy('{}').agg({})""".format(
            output, self.inputs[0], self.attributes, json.dumps(info))
    else:
        code = ""
    return dedent(code)
'''


class Filter(Operation):
    """
    Filters rows using the given condition.
    Parameters:
        - The expression (==, <, >)
    """

    def __init__(self, parameters, inputs, outputs):
        Operation.__init__(self, parameters, inputs, outputs)
        self.expression = parameters['expression']

    def generate_code(self):
        code = """{} = {}.filter('{}')""".format(
            self.outputs[0], self.inputs[0], self.expression)
        return dedent(code)


class DatetimeToBins(Operation):
    """
    """

    def __init__(self, parameters, inputs, outputs):
        Operation.__init__(self, parameters, inputs, outputs)
        self.target_column = parameters['target_column']
        self.new_column = parameters['new_column']
        self.group_size = parameters['group_size']

    def generate_code(self):
        code = '''
            from bins import *
            {} = datetime_to_bins({}, {}, '{}', '{}')
            '''.format(self.outputs[0], self.inputs[0], self.group_size,
                       self.target_column, self.new_column)
        return dedent(code)


class Save(Operation):
    """
    Saves the content of the DataFrame at the specified path
    and generate the code to call the Limonero API.
    Parameters:
        - Database name
        - Path for storage
        - Storage ID
        - Database tags
        - Workflow that generated the database
    """
    NAME_PARAM = 'name'
    PATH_PARAM = 'url'
    STORAGE_ID_PARAM = 'storage_id'
    FORMAT_PARAM = 'format'
    TAGS_PARAM = 'tags'
    OVERWRITE_MODE_PARAM = 'mode'
    HEADER_PARAM = 'header'

    MODE_ERROR = 'error'
    MODE_APPEND = 'append'
    MODE_OVERWRITE = 'overwrite'
    MODE_IGNORE = 'ignore'

    FORMAT_PARQUET = 'PARQUET'
    FORMAT_CSV = 'CSV'
    FORMAT_JSON = 'JSON'

    def __init__(self, parameters, inputs, outputs):
        Operation.__init__(self, parameters, inputs, outputs)

        self.name = parameters.get(self.NAME_PARAM)
        self.format = parameters.get(self.FORMAT_PARAM)
        self.url = parameters.get(self.PATH_PARAM)
        self.storage_id = parameters.get(self.STORAGE_ID_PARAM)
        self.tags = ast.literal_eval(parameters.get(self.TAGS_PARAM, '[]'))

        self.workflow = parameters.get('workflow', '')

        self.mode = parameters.get(self.OVERWRITE_MODE_PARAM, self.MODE_ERROR)
        self.header = parameters.get(self.HEADER_PARAM, True)


    def generate_code(self):

        code_save = ''
        if self.format == self.FORMAT_CSV:
            code_save = """{}.write.csv('{}', header={}, mode='{}')""".format(
                self.inputs[0], self.url, self.header, self.mode)
        elif self.format == self.FORMAT_PARQUET:
            pass
        elif self.format == self.FORMAT_JSON:
            pass
    
        code = dedent(code_save)


        if not (self.workflow == ''):
            code_api = """
                from metadata import MetadataPost
                types_names = {{
                'IntegerType': "INTEGER",
                'StringType': "TEXT",
                'LongType': "LONG",
                'DoubleType': "DOUBLE",
                'TimestampType': "DATETIME",
                }}
                schema = []
                for att in {0}.schema:
                schema.append({{
                    'name': att.name,
                    'dataType': types_names[str(att.dataType)],
                    'nullable': att.nullable,
                    'metadata': att.metadata,
                }})
                parameters = {{
                'name': "{1}",
                'format': "{2}",
                'storage_id': {3},
                'provenience': str("{4}"),
                'description': "{5}",
                'user_id': "{6}",
                'user_login': "{7}",
                'user_name': "{8}",
                'workflow_id': "{9}",
                'url': "{10}",
                }}
                instance = MetadataPost('{11}', schema, parameters)
                """.format(self.inputs[0], self.name, self.format, self.storage_id,
                       str(json.dumps(self.workflow)).replace("\"", "'"),
                       self.workflow['workflow']['name'],
                       self.workflow['user']['id'],
                       self.workflow['user']['login'],
                       self.workflow['user']['name'],
                       self.workflow['workflow']['id'], self.url, "123456"
                       )
            code += dedent(code_api)

        return code


class NoOp(Operation):
    """ Null operation """

    def __init__(self, parameters, inputs, outputs):
        Operation.__init__(self, parameters, inputs, outputs)
        self.parameters = parameters
        self.has_code = False


class CleanMissing(Operation):
    """
    Clean missing fields from data set
    Parameters:
        - attributes: list of attributes to evaluate
        - cleaning_mode: what to do with missing values. Possible values include
          * "VALUE": replace by parameter "value",
          * "MEDIAN": replace by median value
          * "MODE": replace by mode value
          * "MEAN": replace by mean value
          * "REMOVE_ROW": remove entire row
          * "REMOVE_COLUMN": remove entire column
        - value: optional, used to replace missing values
    """
    ATTRIBUTES_PARAM = 'attributes'
    CLEANING_MODE_PARAM = 'cleaning_mode'
    VALUE_PARAMETER = 'value'
    MIN_MISSING_RATIO_PARAM = 'min_missing_ratio'
    MAX_MISSING_RATIO_PARAM = 'max_missing_ratio'

    VALUE = 'VALUE'
    MEAN = 'MEAN'
    MODE = 'MODE'
    MEDIAN = 'MEDIAN'
    REMOVE_ROW = 'REMOVE_ROW'
    REMOVE_COLUMN = 'REMOVE_COLUMN'

    def __init__(self, parameters, inputs, outputs):
        Operation.__init__(self, parameters, inputs, outputs)
        if self.ATTRIBUTES_PARAM in parameters:
            self.attributes = parameters.get(self.ATTRIBUTES_PARAM)
        else:
            raise ValueError(
                "Parameter '{}' must be informed for task {}".format(
                    self.ATTRIBUTES_PARAM, self.__class__))
        self.cleaning_mode = parameters.get(self.CLEANING_MODE_PARAM,
                                            self.REMOVE_ROW)

        self.value = parameters.get(self.VALUE_PARAMETER)
        self.min_missing_ratio = float(
            parameters.get(self.MIN_MISSING_RATIO_PARAM))
        self.max_missing_ratio = float(
            parameters.get(self.MAX_MISSING_RATIO_PARAM))

        # In this case, nothing will be generated besides create reference to
        # data frame
        if (self.value is None and self.cleaning_mode == self.VALUE) or len(
                self.inputs) == 0:
            self.has_code = False

    def generate_code(self):
        if not self.has_code:
            return "{} = {}".format(self.outputs[0], self.inputs[0])

        output = self.outputs[0] if len(self.outputs) else '{}_tmp'.format(
            self.inputs[0])
        pre_code = []
        partial = []
        attrs_json = json.dumps(self.attributes)

        if any([self.min_missing_ratio, self.max_missing_ratio]):
            self.min_missing_ratio = self.min_missing_ratio or 0.0
            self.max_missing_ratio = self.max_missing_ratio or 1.0

            # Based on http://stackoverflow.com/a/35674589/1646932
            select_list = [
                "\n    (count('{0}') / count('*')).alias('{0}')".format(attr)
                for attr in self.attributes]
            pre_code.extend([
                "# Computes the ratio of missing values for each attribute",
                "ratio_{0} = {0}.select({1}).collect()".format(
                    self.inputs[0], ', '.join(select_list)), "",
                "attributes_{0} = [c for c in {1} "
                "\n        if {2} <= count_{0}[0][c] <= {3}]".format(
                    self.inputs[0], attrs_json, self.min_missing_ratio,
                    self.max_missing_ratio)
            ])
        else:
            pre_code.append(
                "attributes_{0} = {1}".format(self.inputs[0], attrs_json))

        if self.cleaning_mode == self.REMOVE_ROW:
            partial.append("""
                {0} = {1}.na.drop(how='any', subset=attributes_{1})""".format(
                output, self.inputs[0]))

        elif self.cleaning_mode == self.VALUE:
            value = ast.literal_eval(self.value)
            if not (isinstance(value, int) or isinstance(value, float)):
                value = '"{}"'.format(value)
            partial.append(
                "\n    {0} = {1}.na.fill(value={2}, subset=attributes_{1})".format(
                    output, self.inputs[0], value))

        elif self.cleaning_mode == self.REMOVE_COLUMN:
            # Based on http://stackoverflow.com/a/35674589/1646932"
            partial.append(
                "\n{0} = {1}.select("
                "[c for c in {1}.columns if c not in attributes_{1}])".format(
                    output, self.inputs[0]))

        elif self.cleaning_mode == self.MODE:
            # Based on http://stackoverflow.com/a/36695251/1646932
            partial.append("""
                md_replace_{1} = dict()
                for md_attr_{1} in attributes_{1}:
                    md_count_{1} = {0}.groupBy(md_attr_{1}).count()\\
                        .orderBy(desc('count')).limit(1)
                    md_replace_{1}[md_attr_{1}] = md_count_{1}.collect()[0][0]
             {0} = {1}.fillna(value=md_replace_{1})""".format(
                output, self.inputs[0])
            )

        elif self.cleaning_mode == self.MEDIAN:
            # See http://stackoverflow.com/a/31437177/1646932
            # But null values cause exception, so it needs to remove them
            partial.append("""
                mdn_replace_{1} = dict()
                for mdn_attr_{1} in attributes_{1}:
                    # Computes median value for column with relat. error = 10%
                    mdn_{1} = {1}.na.drop(subset=[mdn_attr_{1}])\\
                        .approxQuantile(mdn_attr_{1}, [.5], .1)
                    md_replace_{1}[mdn_attr_{1}] = mdn_{1}[0]
                {0} = {1}.fillna(value=mdn_replace_{1})""".format(
                output, self.inputs[0]))

        elif self.cleaning_mode == self.MEAN:
            partial.append("""
                avg_{1} = {1}.select([avg(c).alias(c) for c in attributes_{1}]).collect()
                values_{1} = dict([(c, avg_{1}[0][c]) for c in attributes_{1}])
                {0} = {1}.na.fill(value=values_{1})""".format(output,
                                                              self.inputs[0]))
        else:
            raise ValueError(
                "Parameter '{}' has an incorrect value '{}' in {}".format(
                    self.CLEANING_MODE_PARAM, self.cleaning_mode,
                    self.__class__))

        return '\n'.join(pre_code) + \
               "\nif len(attributes_{0}) > 0:".format(self.inputs[0]) + \
               '\n    '.join([dedent(line) for line in partial]).replace('\n',
                                                                         '\n    ') + \
               "\nelse:\n    {0} = {1}".format(output, self.inputs[0])


class AddColumns(Operation):
    """
    Merge two data frames, column-wise, similar to the command paste in Linux.
    Implementation based on post http://stackoverflow.com/a/40510320/1646932
    """

    def __init__(self, parameters, inputs, outputs):
        Operation.__init__(self, parameters, inputs, outputs)
        self.has_code = len(inputs) == 2

    def generate_code(self):
        if self.has_code:
            output = self.outputs[0] if len(self.outputs) else '{}_tmp'.format(
                self.inputs[0])
            code = """
                w_{0}_{1} = Window().orderBy()
                {0}_inx =  {0}.withColumn("_inx", rowNumber().over(w_{0}_{1}))
                {1}_inx =  {1}.withColumn("_inx", rowNumber().over(w_{0}_{1})
                {2} = {0}_indexed.join({1}_inx, {0}_inx._inx == {0}_inx._inx,
                                             'inner')
                    .drop({0}_inx._inx).drop({1}_inx._inx)
                """.format(self.inputs[0], self.inputs[1], output)
            return dedent(code)

        return ""


class Replace(Operation):
    """
    Replaces values of columns by specified value. Similar to Transformation
    operation.
    @deprecated
    """
    ATTRIBUTES_PARAM = 'attributes'

    def __init__(self, parameters, inputs, outputs):
        Operation.__init__(self, parameters, inputs, outputs)
        if self.ATTRIBUTES_PARAM in parameters:
            self.attributes = parameters.get(self.ATTRIBUTES_PARAM)
        else:
            raise ValueError(
                "Parameter '{}' must be informed for task {}".format(
                    self.ATTRIBUTES_PARAM, self.__class__))

    def generate_code(self):
        if self.has_code:
            output = self.outputs[0] if len(self.outputs) else '{}_tmp'.format(
                self.inputs[0])
            code = """ """

            return dedent(code)

        return ""


class PearsonCorrelation(Operation):
    """
    Calculates the correlation of two columns of a DataFrame as a double value.
    @deprecated: It should be used as a function in expressions
    """
    ATTRIBUTES_PARAM = 'attributes'

    def __init__(self, parameters, inputs, outputs):
        Operation.__init__(self, parameters, inputs, outputs)
        if self.ATTRIBUTES_PARAM in parameters:
            self.attributes = parameters.get(self.ATTRIBUTES_PARAM)
        else:
            raise ValueError(
                "Parameter '{}' must be informed for task {}".format(
                    self.ATTRIBUTES_PARAM, self.__class__))

    def generate_code(self):
        if len(self.inputs) == 1:
            output = self.outputs[0] if len(self.outputs) else '{}_tmp'.format(
                self.inputs[0])
            code = """{} = {}.corr('{}', '{}')""".format(
                output, self.inputs[0], self.attributes[0], self.attributes[1])
        else:
            code = ''

        return dedent(code)