import collections

import tensorflow as tf
from tensorflow.python.ops import control_flow_ops

import tensorflow.contrib.slim as slim

__all__ = ['create_clones',
           'deploy',
           'optimize_clones',
           'DeployedModel',
           'DeploymentConfig',
           'Clone',]

Clone = collections.namedtuple('Clone',
                               ['outputs',
                                'scope',
                                'device',])

DeployedModel = collections.namedtuple('DeplyedModel',
                                       ['train_op',
                                        'summary_op',
                                        'total_loss',
                                        'clones',])

_deployment_params = {'num_clones': 1,
                      'clone_on_cpu': False,
                      'fake_multiple_gpus': False,
                      'replica_id': 0,
                      'num_replica': 1,
                      'num_ps_tasks': 0,
                      'work_job_name': 'worker',
                      'ps_job_name': 'ps'}

def create_clones(config, model_fn, args=None, kwargs=None):

    clones = []
    args = args or []
    kwargs = kwargs or []
    with slim.arg_scope([slim.model_variable, slim.variable],
                        device=config.variables_device()):
        # Create clones
        for i in range(0, config.num_clones):
            with tf.name_scope(config.clone_scope(i)) as clone_scope:
                clone_device = config.clone_device(i)
                with tf.device(clone_device):
                    with tf.variable_scope(tf.get_variable_scope(),     # Returen current variable scope
                                           reuse=True if i > 0 else None):
                        outputs = model_fn(*args, **kwargs)
                    clones.append(Clone(outputs, clone_scope, clone_device))
    return clones


class DeploymentConfig(object):

    """
    An Instance of this class will be passed to deploy() to deploy the model
    """

    def __init__(self,
                 num_clones=1,
                 clone_on_cpu=False,
                 fake_multiple_gpus=False,      # When True, model will only be replicated once on single gup
                 replica_id=0,
                 num_replicas=1,                # When 1, model will be deployed via single process
                 num_ps_tasks=0,
                 worker_job_name='worker',
                 ps_job_name='ps'):             # parameter sever
        if num_replicas > 1:
            if num_ps_tasks < 1:
                raise ValueError('Replica is activated, but num_ps_tasks is not positive')
        if num_replicas > 1 or num_ps_tasks > 0:
            if not worker_job_name:
                raise ValueError('Replica is activated, but no work_job_name')
            if not ps_job_name:
                raise ValueError('No ps_job_name')
        if replica_id>=num_replicas:
            raise ValueError('replica_id must be less than num_replicas')
        self._num_clones = num_clones
        self._clone_on_cpu = clone_on_cpu
        self._fake_multiple_gpus = fake_multiple_gpus
        self._replica_id = replica_id
        self._num_replicas = num_replicas
        self._num_ps_tasks = num_ps_tasks
        self._ps_device = '/job:' + ps_job_name if num_ps_tasks > 0 else ''
        self._worker_device = '/job:' + worker_job_name if num_ps_tasks > 0 else ''

    @property
    def num_clones(self):
        return self._num_clones

    @property
    def clone_on_cpu(self):
        return self._clone_on_cpu

    @property
    def fake_multiple_gpus(self):
        return self._fake_multiple_gpus

    @property
    def replica_id(self):
        return self._replica_id

    @property
    def num_replicas(self):
        return self._num_replicas

    @property
    def num_ps_tasks(self):
        return self._num_ps_tasks

    @property
    def ps_device(self):
        return self._ps_device

    @property
    def worker_device(self):
        return self._worker_device

    def caching_device(self):

        if self._num_ps_tasks > 0:
            return lambda op: op.device
        else:
            return None

    