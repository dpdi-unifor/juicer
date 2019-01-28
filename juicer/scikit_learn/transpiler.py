# -*- coding: utf-8 -*-

import juicer.scikit_learn.associative_operation as associative
import juicer.scikit_learn.classification_operation as classifiers
import juicer.scikit_learn.clustering_operation as clustering
import juicer.scikit_learn.data_operation as io
import juicer.scikit_learn.etl_operation as etl
import juicer.scikit_learn.feature_operation as feature_extraction
import juicer.scikit_learn.geo_operation as geo
import juicer.scikit_learn.model_operation as model
import juicer.scikit_learn.regression_operation as regression
import juicer.scikit_learn.text_operation as text_operations
import juicer.scikit_learn.vis_operation as vis_operation
import os
from juicer import operation
from juicer.transpiler import Transpiler


# noinspection SpellCheckingInspection
class ScikitLearnTranspiler(Transpiler):
    """
    Convert Lemonade workflow representation (JSON) into code to be run in
    Scikit-Learn.
    """

    def __init__(self, configuration, slug_to_op_id=None, port_id_to_port=None):
        super(ScikitLearnTranspiler, self).__init__(
            configuration, os.path.abspath(os.path.dirname(__file__)),
            slug_to_op_id, port_id_to_port)

        self._assign_operations()

    def get_context(self):
        dict_msgs = {
            'task_completed': _('Task completed'),
            'task_running': _('Task running'),
            'lemonade_task_completed': _('Lemonade task %s completed'),
            'lemonade_task_parents': _('Parents completed, submitting %s'),
            'lemonade_task_started': _('Lemonade task %s started'),
            'lemonade_task_afterbefore': _(
                "Submitting parent task {} before {}")}

        return {'dict_msgs': dict_msgs}

    def _assign_operations(self):
        etl_ops = {
            'add-columns': etl.AddColumnsOperation,
            'add-rows': etl.UnionOperation,
            'aggregation': etl.AggregationOperation,  # TODO: agg sem groupby
            'clean-missing': etl.CleanMissingOperation,
            'difference': etl.DifferenceOperation,
            'drop': etl.DropOperation,
            'execute-python': etl.ExecutePythonOperation,
            'execute-sql': etl.ExecuteSQLOperation,
            'filter-selection': etl.FilterOperation,
            'join': etl.JoinOperation,
            'projection': etl.SelectOperation,
            'remove-duplicated-rows': etl.DistinctOperation,
            'replace-value': etl.ReplaceValuesOperation,
            'sample': etl.SampleOrPartitionOperation,
            'set-intersection': etl.IntersectionOperation,
            'sort': etl.SortOperation,
            'split': etl.SplitOperation,
            'transformation': etl.TransformationOperation,
            # TODO in 'transformation': test others functions
        }

        data_ops = {
            'data-reader': io.DataReaderOperation,
            'data-writer': io.SaveOperation,
            'save': io.SaveOperation,
            # 'change-attribute': io.ChangeAttributesOperation,
        }

        geo_ops = {
            'read-shapefile': geo.ReadShapefileOperation,
            'stdbscan': geo.STDBSCANOperation,
            'within': geo.GeoWithinOperation,
        }

        ml_ops = {
            # ------ Associative ------#
            'association-rules': associative.AssociationRulesOperation,
            'frequent-item-set': associative.FrequentItemSetOperation,
            'sequence-mining': associative.SequenceMiningOperation,

            # ------ Feature Extraction Operations  ------#
            'feature-assembler': feature_extraction.FeatureAssemblerOperation,
            'min-max-scaler': feature_extraction.MinMaxScalerOperation,
            'max-abs-scaler': feature_extraction.MaxAbsScalerOperation,
            'one-hot-encoder': feature_extraction.OneHotEncoderOperation,
            'pca': feature_extraction.PCAOperation,
            'quantile-discretizer':
                feature_extraction.QuantileDiscretizerOperation,
            'standard-scaler': feature_extraction.StandardScalerOperation,
            # 'feature-indexer': #confirm vector and strings
            #     juicer.compss.feature_operation.FeatureIndexerOperation,
            #
            # ------ Model Operations  ------#
            'apply-model': model.ApplyModelOperation,
            'cross-validation': model.CrossValidationOperation,
            'evaluate-model': model.EvaluateModelOperation,
            'load-model': model.LoadModel,
            'save-model': model.SaveModel,

            # ------ Clustering      -----#
            'clustering-model': clustering.ClusteringModelOperation,
            'agglomerative-clustering':
                clustering.AgglomerativeClusteringOperation,
            'dbscan-clustering': clustering.DBSCANClusteringOperation,
            'gaussian-mixture-clustering':
                clustering.GaussianMixtureClusteringOperation,
            'gaussian-mixture-clustering-model':
                clustering.GaussianMixtureClusteringModelOperation,
            'k-means-clustering': clustering.KMeansClusteringOperation,
            'k-means-clustering-model':
                clustering.KMeansClusteringModelOperation,

            # ------ Classification  -----#
            'classification-model': classifiers.ClassificationModelOperation,
            'decision-tree-classifier':
                classifiers.DecisionTreeClassifierOperation,
            'decision-tree-classifier-model':
                classifiers.DecisionTreeClassifierModelOperation,
            'gbt-classifier': classifiers.GBTClassifierOperation,
            'gbt-classifier-model': classifiers.GBTClassifierModelOperation,
            'knn-classifier': classifiers.KNNClassifierOperation,
            'knn-classifier-model': classifiers.KNNClassifierModelOperation,
            'logistic-regression': classifiers.LogisticRegressionOperation,
            'logistic-regression-model':
                classifiers.LogisticRegressionModelOperation,
            'mlp-classifier': classifiers.MLPClassifierOperation,
            'mlp-classifier-model': classifiers.MLPClassifierModelOperation,
            'naive-bayes-classifier': classifiers.NaiveBayesClassifierOperation,
            'naive-bayes-classifier-model':
                classifiers.NaiveBayesClassifierModelOperation,
            'perceptron-classifier': classifiers.PerceptronClassifierOperation,
            'perceptron-classifier-model':
                classifiers.PerceptronClassifierModelOperation,
            'random-forest-classifier':
                classifiers.RandomForestClassifierOperation,
            'random-forest-classifier-model':
                classifiers.RandomForestClassifierModelOperation,
            'svm-classification': classifiers.SvmClassifierOperation,
            'svm-classification-model': classifiers.SvmClassifierModelOperation,

            # ------ Regression  -----#
            'regression-model': regression.RegressionModelOperation,
            'gbt-regressor': regression.GradientBoostingRegressorOperation,
            'gbt-regressor-model':
                regression.GradientBoostingRegressorModelOperation,
            'huber-regressor': regression.HuberRegressorOperation,
            'huber-regressor-model': regression.HuberRegressorModelOperation,
            'isotonic-regression':
                regression.IsotonicRegressionOperation,
            'isotonic-regression-model':
                regression.IsotonicRegressionModelOperation,
            'linear-regression': regression.LinearRegressionOperation,
            'linear-regression-model': regression.LinearRegressionModelOperation,
            'mlp-regressor': regression.MLPRegressorOperation,
            'mlp-regressor-model': regression.MLPRegressorModelOperation,
            'random-forest-regressor':
                regression.RandomForestRegressorOperation,
            'random-forest-regressor-model':
                regression.RandomForestRegressorModelOperation,
            'sgd-regressor': regression.SGDRegressorOperation,
            'sgd-regressor-model': regression.SGDRegressorModelOperation,
        }

        text_ops = {
            'generate-n-grams': text_operations.GenerateNGramsOperation,
            'remove-stop-words': text_operations.RemoveStopWordsOperation,
            'tokenizer': text_operations.TokenizerOperation,
            'word-to-vector': text_operations.WordToVectorOperation,
            'lda-clustering-model': text_operations.LdaClusteringOperation,
        }

        other_ops = {
            'comment': operation.NoOp,
        }

        ws_ops = {}

        vis_ops = {
            'publish-as-visualization':
                vis_operation.PublishVisualizationOperation,
            'bar-chart': vis_operation.BarChartOperation,
            'donut-chart': vis_operation.DonutChartOperation,
            'pie-chart': vis_operation.PieChartOperation,
            'area-chart': vis_operation.AreaChartOperation,
            'line-chart': vis_operation.LineChartOperation,
            'table-visualization': vis_operation.TableVisualizationOperation,
            'summary-statistics': vis_operation.SummaryStatisticsOperation,
            'plot-chart': vis_operation.ScatterPlotOperation,
            'scatter-plot': vis_operation.ScatterPlotOperation,
            'map-chart': vis_operation.MapOperation,
            'map': vis_operation.MapOperation
        }

        self.operations = {}
        for ops in [data_ops, etl_ops, geo_ops, ml_ops,
                    other_ops, text_ops, ws_ops, vis_ops]:
            self.operations.update(ops)
