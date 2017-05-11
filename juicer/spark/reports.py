from textwrap import dedent

from juicer.spark.vis_operation import HtmlVisModel


class BaseHtmlReport(object):
    pass


class EvaluateModelOperationReport(BaseHtmlReport):
    """ Report generated by class EvaluateModelOperation """

    @staticmethod
    def generate_visualization(**kwargs):
        evaluator = kwargs['evaluator']
        metric_value = kwargs['metric_value']
        title = kwargs['title']
        operation_id = kwargs['operation_id']
        task_id = kwargs['task_id']

        vis_model = dedent('''
            <div>
                <strong>{title}</strong>
                <dl>
                    <dt>{metric}</dl>
                    <dd>{value}</dd>
                </dl>
                <strong>Execution parameters</strong>
                {params}
            </div>
        ''').format(title=title, metric=evaluator.getMetricName(),
                    value=metric_value, params=evaluator.explainParams())

        return HtmlVisModel(
            vis_model, task_id, operation_id, 'EvaluateModelOperation', title,
            '[]', '', '', '', {})