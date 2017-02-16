# coding=utf-8

import codecs
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError
import gc
import importlib
import json
import logging
import multiprocessing
import signal
import socketio
import sys
import time
from io import StringIO

import datetime
import imp

import os
from juicer.runner import juicer_protocol
from juicer.runner.juicer_server import JuicerServer
from juicer.runner.minion_base import Minion
from juicer.spark.transpiler import SparkTranspiler
from juicer.util.string_importer import StringImporter
from juicer.workflow.workflow import Workflow

logging.basicConfig(
    format=('[%(levelname)s] %(asctime)s,%(msecs)05.1f '
            '(%(funcName)s:%(lineno)s) %(message)s'),
    datefmt='%H:%M:%S')
log = logging.getLogger()
log.setLevel(logging.INFO)

class SparkMinion(Minion):
    """
    Controls the execution of Spark code in Lemonade Juicer.
    """
    # Errors and messages
    MNN000 = ('MNN000', 'Success.')
    MNN001 = ('MNN001', 'Port output format not supported.')
    MNN002 = ('MNN002', 'Success getting data from task.')
    MNN003 = ('MNN003', 'State does not exists, processing app.')
    MNN004 = ('MNN004', 'Invalid port.')
    MNN005 = ('MNN005', 'Unable to retrieve data because a previous error.')
    MNN006 = ('MNN006', 'Invalid Python code or incorrect encoding: {}')
    MNN007 = ('MNN007', 'Job {} was canceled')
    MNN008 = ('MNN008', 'App {} was terminated')

    def __init__(self, redis_conn, workflow_id, app_id, config):
        Minion.__init__(self, redis_conn, workflow_id, app_id, config)

        self.execute_process = None
        self.ping_process = None
        self.module = None

        self.string_importer = StringImporter()
        self._state = {}
        self.transpiler = SparkTranspiler()
        self.config = config
        sys.meta_path.append(self.string_importer)
        self.tmp_dir = self.config.get('config', {}).get('tmp_dir', '/tmp')
        sys.path.append(self.tmp_dir)

        self.job_count = 0
        self.spark_session = None
        signal.signal(signal.SIGTERM, self._terminate)

        self.mgr = socketio.RedisManager(config['juicer']['servers']['redis_url'],
                'job_output')

        self.executor = ThreadPoolExecutor(max_workers=1)
        self.job_future = None

    def _emit_event(self, room, namespace):
        def emit_event(name, msg, status, identifier):
            self.mgr.emit(name,
                    data={'msg': msg, 'status': status, 'id': identifier},
                    room=room, namespace=namespace)
        return emit_event

    def _generate_output(self, msg, status=None, code=None):
        """
        Sends feedback about execution of this minion.
        """
        obj = {'message': msg, 'workflow_id': self.workflow_id,
                'app_id': self.app_id, 'code': code,
               'date': datetime.datetime.now().isoformat(),
               'status': status if status is not None else 'OK'}

        m = json.dumps(obj)
        self.state_control.push_app_output_queue(self.app_id, m)

    def ping(self):
        """ Pings redis to inform master this minion is online """
        log.info('Start ping')
        while True:
            self._perform_ping()
            time.sleep(5)

    def _perform_ping(self):
        status = {
            'status': 'READY', 'pid': os.getpid(),
        }
        self.state_control.set_minion_status(self.app_id,
                json.dumps(status), ex=30, nx=False)

    def execute(self):
        """
        Starts consuming jobs that must be processed by this minion.
        """
        while True:
            try:
                self._process_message_nb()
            except KeyError as ke:
                log.error('Message does not match any convention: %s', msg_type)

    def _process_message(self):
        self._process_message_nb()
        if self.job_future:
            self.job_future.result()

    def _process_message_nb(self):
        # Get next message
        msg_info = json.loads(
            self.state_control.pop_app_queue(self.app_id))

        # Sanity check: this minion should not process messages from another
        # workflow/app
        assert str(msg_info['workflow_id']) == self.workflow_id, \
                'Expected workflow_id=%s, got workflow_id=%s' % ( \
                self.workflow_id, msg_info['workflow_id'])
        
        assert str(msg_info['app_id']) == self.app_id, \
                'Expected app_id=%s, got app_id=%s' % ( \
                self.workflow_id, msg_info['app_id'])

        # Extract the message type
        msg_type = msg_info['type']
        self._generate_output('Processing message %s for app %s' %
                (msg_type, self.app_id))

        # Forward the message according to its purpose
        if msg_type == juicer_protocol.EXECUTE:
            log.info('Execute message received')
            job_id = msg_info['job_id']
            workflow = msg_info['workflow']
            # TODO: We should consider the case in which the spark session is
            # already instanciated and this new request asks for a different set
            # of configurations:
            # - Should we rebuild the context from scratch and execute all jobs so far?
            # - Should we ignore this part of the request and execute over the existing
            # (old configs) spark session?
            app_configs = msg_info.get('app_configs', {})

            if self.job_future:
                self.job_future.result()

            self.job_future = self._execute_future(job_id, workflow, app_configs)

        elif msg_type == juicer_protocol.DELIVER:
            log.info('Deliver message received')
            task_id = msg_info.get('task_id')
            output = msg_info.get('output')
            port = int(msg_info.get('port'))
            job_id = msg_info['job_id']
            workflow = msg_info['workflow']
            app_configs = msg_info.get('app_configs', {})
            
            if self.job_future:
                self.job_future.result()

            self.job_future = self._deliver_future(task_id,
                    output, port, job_id, workflow, app_configs)

        elif msg_type == juicer_protocol.TERMINATE:
            job_id = msg_info.get('job_id', None)
            if job_id:
                log.info('Terminate message received (job_id=%s)', job_id)
                self.cancel_job(job_id)
            else:
                log.info('Terminate message received (app=%s)', self.app_id)
                self.terminate()

        else:
            log.warn('Unknown message type %s', msg_type)
            self._generate_output('Unknown message type %s' % msg_type)

    def _execute_future(self, job_id, workflow, app_configs):
        return self.executor.submit(self._perform_execute,
                job_id, workflow, app_configs)

    def _perform_execute(self, job_id, workflow, app_configs):
        result = True
        try:
            loader = Workflow(workflow)
            
            # Mark job as running
            self._emit_event(room=job_id, namespace='/stand')(
                    name='update job', msg='Running job',
                    status='RUNNING', identifier=job_id)

            module_name = 'juicer_app_{}_{}_{}'.format(
                    self.workflow_id,
                    self.app_id,
                    job_id)

            generated_code_path = os.path.join(
                self.tmp_dir, module_name + '.py')

            with codecs.open(generated_code_path, 'w', 'utf8') as out:
                self.transpiler.transpile(
                    loader.workflow_data, loader.workflow_graph, {}, out)

            # Get rid of .pyc file if it exists
            if os.path.isfile('{}c'.format(generated_code_path)):
                os.remove('{}c'.format(generated_code_path))
            self.module = importlib.import_module(module_name)
            self.module = imp.reload(self.module)
            log.debug('Objects in memory after loading module: %s',
                      len(gc.get_objects()))

            # Starting execution. At this point, the transpiler have created a
            # module with a main function that receives a spark_session and the
            # current state (if any). We pass the current state to the execution
            # to avoid re-computing the same tasks over and over again, in case
            # of several partial workflow executions.
            new_state = self.module.main(
                    self.get_or_create_spark_session(loader, app_configs),
                    self._state,
                    self._emit_event(room=job_id, namespace='/stand'))
            
            # Mark job as completed
            self._emit_event(room=job_id, namespace='/stand')(
                    name='update job', msg='Job finished',
                    status='COMPLETED', identifier=job_id)

            # We update the state incrementally, i.e., new task results can be
            # overwritten but never lost.
            self._state.update(new_state)

        except UnicodeEncodeError as ude:
            msg = self.MNN006[1].format(ude)
            log.warn(msg)
            # Mark job as failed
            self._emit_event(room=job_id, namespace='/stand')(
                    name='update job', msg=msg,
                    status='ERROR', identifier=job_id)
            self._generate_output(self.MNN006[1], 'ERROR', self.MNN006[0])
            result = False

        except ValueError as ve:
            msg = 'Invalid message format: {}'.format(ve.message)
            log.warn(msg)
            self._emit_event(room=job_id, namespace='/stand')(
                    name='update job', msg=msg,
                    status='ERROR', identifier=job_id)
            self._generate_output(msg, 'ERROR')
            result = False

        except SyntaxError as se:
            msg = self.MNN006[1].format(se)
            log.warn(msg)
            self._emit_event(room=job_id, namespace='/stand')(
                    name='update job', msg=msg,
                    status='ERROR', identifier=job_id)
            self._generate_output(self.MNN006[1], 'ERROR', self.MNN006[0])
            result = False

        except Exception as ee:
            log.error(ee)
            self._emit_event(room=job_id, namespace='/stand')(
                    name='update job', msg=ee.message,
                    status='ERROR', identifier=job_id)
            self._generate_output(ee.message, 'ERROR', code=1000)
            result = False
        return result

    def is_spark_session_available(self):
        """
        Check whether the spark session is available, i.e., the spark session
        is set and not stopped.
        """
        return self.spark_session and \
                self.spark_session.sparkContext._jsc and \
                not self.spark_session.sparkContext._jsc.sc().isStopped()

    # noinspection PyUnresolvedReferences
    def get_or_create_spark_session(self, loader, app_configs):
        """
        Get an existing spark session (context) for this minion or create a new
        one. Ideally the spark session instanciation is done only once, in order
        to support partial workflow executions within the same context.
        """

        from pyspark.sql import SparkSession
        if not self.is_spark_session_available():

            if "HADOOP_HOME" in os.environ:
                app_configs['driver-library-path'] = \
                        '{}/lib/native/'.format(os.environ.get('HADOOP_HOME'))

            app_name = u'%s(workflow_id=%s,app_id=%s)' % (
                    loader.workflow_data.get('name', ''),
                    self.workflow_id, self.app_id)

            spark_builder = SparkSession.builder.appName(app_name)
            for option, value in app_configs.iteritems():
                spark_builder = spark_builder.config(option, value)

            self.spark_session = spark_builder.getOrCreate()
            self.spark_session.sparkContext.setLogLevel('INFO')

        return self.spark_session

    @staticmethod
    def _convert_to_csv(row):
        result = []
        for v in row:
            t = type(v)
            if t in [datetime.datetime]:
                result.append(v.isoformat())
            elif t in [unicode, str]:
                result.append('"{}"'.format(v))
            else:
                result.append(str(v))
        return ','.join(result)

    def _send_to_output(self, data):
        self.state_control.push_app_output_queue(
            self.app_id, json.dumps(data))

    def _deliver_future(self, task_id, output, port, job_id, workflow, app_configs):
        return self.executor.submit(self._perform_deliver, task_id, output,
                port, job_id, workflow, app_configs)

    def _perform_deliver(self, task_id, output, port, job_id, workflow, app_configs):
        result = True
        # FIXME: state must expire. How to identify this in the interface?
        # FIXME: Define how to identify the request.
        # FIXME: Define where to store generated data (Redis?)
        if task_id in self._state:
            self._read_dataframe_data(task_id, output, port)
        else:
            data = {'status': 'WARNING', 'code': self.MNN003[0],
                    'message': self.MNN003[1]}
            self._send_to_output(data)

            # FIXME: Report missing or process workflow until this task
            if self._perform_execute(job_id, workflow, app_configs):
                self._read_dataframe_data(task_id, output, port)
            else:
                data = {'status': 'ERROR', 'code': self.MNN005[0],
                        'message': self.MNN005[1]}
                self._send_to_output(data)
                result = False

        return result

    def _read_dataframe_data(self, task_id, output, port):
        result = True
        # Last position in state is the execution time, so it should be ignored
        port_idx = (len(self._state[task_id]) - 1) / 2
        if port_idx >= port:
            df = self._state[task_id][port * 2]
            partial_result = None if port == port_idx else self._state[task_id][port * 2 + 1]

            # In this case we already have partial data collected for the
            # particular task
            if partial_result:
                self.state_control.push_queue(
                    output, '\n'.join(partial_result))
                data = {'status': 'SUCCESS', 'code': self.MNN002[0],
                        'message': self.MNN002[1], 'output': output}
                self._send_to_output(data)

            # In this case we do not have partial data collected for the task
            # Then we must obtain it if the 'take' operation applies
            elif df is not None and hasattr(df, 'take'):
                # Evaluating if df has method "take" allows unit testing
                # instead of testing exact pyspark.sql.dataframe.Dataframe
                # type check.
                # FIXME define as a parameter?:
                result = df.take(100).rdd.map(
                    SparkMinion._convert_to_csv).collect()
                self.state_control.push_queue(
                    output, '\n'.join(result))
                data = {'status': 'SUCCESS', 'code': self.MNN002[0],
                        'message': self.MNN002[1], 'output': output}
                self._send_to_output(data)

            # In this case, do not make sense to request data for this
            # particular task output port
            else:
                data = {'status': 'ERROR', 'code': self.MNN001[0],
                        'message': self.MNN001[1]}
                self._send_to_output(data)
                result = False

        else:
            data = {'status': 'ERROR', 'code': self.MNN004[0],
                    'message': self.MNN004[1]}
            self._send_to_output(data)
            result = False

        return result

    def cancel_job(self, job_id):
        if self.job_future:
            while True:
                if self.is_spark_session_available():
                    self.spark_session.sparkContext.cancelAllJobs()
                try:
                    self.job_future.result(timeout=1)
                    break
                except TimeoutError as te:
                    pass

        msg = self.MNN007[1].format(self.app_id)
        log.info(msg)
        self._generate_output(msg, 'SUCCESS', self.MNN007[0])

    def _terminate(self, _signal, _frame):
        self.terminate()

    def terminate(self):
        """
        This is a handler that reacts to a sigkill signal. The most feasible
        scenario is when the JuicerServer is demanding the termination of this
        minion. In this case, we stop and release any allocated resource
        (spark_session) and kill the subprocesses managed in here.
        """
        if self.spark_session:
            self.spark_session.stop()
            self.spark_session.sparkContext.stop()
            self.spark_session = None
        if self.execute_process:
            os.kill(self.execute_process.pid, signal.SIGKILL)
        if self.ping_process:
            os.kill(self.ping_process.pid, signal.SIGKILL)

        msg = self.MNN008[1].format(self.app_id)
        log.info(msg)
        self._generate_output(msg, 'SUCCESS', self.MNN008[0])
        self.state_control.unset_minion_status(self.app_id)


    def process(self):
        log.info('Spark minion (workflow_id=%s,app_id=%s) started (pid=%s)',
                self.workflow_id, self.app_id, os.getpid())
        self.execute_process = multiprocessing.Process(
            name="minion", target=self.execute)
        self.execute_process.daemon = False

        self.ping_process = multiprocessing.Process(
            name="ping process", target=self.ping)
        self.ping_process.daemon = False

        self.execute_process.start()
        self.ping_process.start()

        # We join the following processes because this script only terminates by
        # explicitly receiving a SIGKILL signal.
        self.execute_process.join()
        self.ping_process.join()

        # TODO: clean state files in the temporary directory after joining
        # (maybe pack it and persist somewhere else ?)