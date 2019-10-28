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
    kwargs = kwargs or {}
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


def _sum_clones_gradients(clone_grads):

    sum_grads = []
    for grad_and_vars in zip(*clone_grads):

        # grad_and_vars:
        # ((grad_var0_clone0, var0), ... (grad_varN_cloneN, varN))

        grads = []
        var = grad_and_vars[0][1]
        for g, v in grad_and_vars:
            assert v == var
            if g is not None:
                grads.append(g)
        if grads:
            if len(grads) > 1:
                sum_grad = tf.add_n(grads, name=var.op.name + '/sum_grads')
            else:
                sum_grad = grads[0]
            sum_grads.append((sum_grad, var))
    return sum_grads


def _add_gradients_summaries(grad_and_vars):
    """ Add histogram summaries to gradients """

    summaries = []
    for grad, var in grads_and_vars:
        if grad is not None:
            if isinstance(grad, tf.IndexedSlices):
                grad_values = grad.values
            else:
                grad_values = grad
            summaries.append(tf.summary.histogram(var.op.name + ':gradient',
                                                  grad_values))
            summaries.append(tf.summary.histogram(var.op.name + ':gradient_norm',
                                                  tf.global_norm([grad_values])))
        else:
            tf.logging.info('Var %s has no gradient', var.op.name)
    return summaries


def _gather_clone_loss(clone, num_clones, regularization_losses):
    """Gather the loss for a single clone
    """
    sum_loss = None
    clone_loss = None
    regularization_loss = None
    # Sum up losses on the clone device
    with tf.device(clone.device):
        all_losses = []
        clone_losses = tf.get_collection(tf.GraphKeys.LOSSES, clone.scope)
        if clone_losses:
            clone_loss = tf.add_n(clone_losses, name='clone_loss')
            if num_clones > 1:
                clone_loss = tf.div(clone_loss, 1.0 * num_clones,
                                    name='scaled_clone_loss')
            all_losses.append(clone_loss)
        if regularization_losses:
            regularization_loss = tf.add_n(regularization_losses,
                                           name='regularization_loss')
            all_losses.append(regularization_loss)
        if all_losses:
            sum_loss = tf.add_n(all_losses)
    if clone_loss is not None:
        tf.summary.scalar('clone_loss', clone_loss)
    if regularization_loss is not None:
        tf.summary.scalar('regularization_loss', regularization_loss)
    return sum_loss


def _optimize_clone(optimizer, clone, num_clone, regularization_losses,
                    **kwargs):
    sum_loss = _gather_clone_loss(clone, num_clone, regularization_losses)
    clone_grad = None
    if sum_loss is not None:
        with tf.device(clone.device):
            clone_grad = optimizer.compute_gradients(sum_loss, **kwargs)
        return sum_loss, clone_grad


def optimize_clones(clones, optimizer,
                     regularization_loss=None,
                     **kwargs):
    """
    Compute clone losses and gradients for clones list
    regularization is added to first clone
    """
    grads_and_vars = []
    clones_losses = []
    num_clones = len(clones)
    if regularization_loss is None:
        regularization_loss = tf.get_collection(
            tf.GraphKeys.REGULARIZATION_LOSSES)
    for clone in clones:
        with tf.name_scope(clone.scope):
            clone_loss, clone_grad = _optimize_clone(
                optimizer, clone, num_clones, regularization_loss, **kwargs
            )
            if clone_loss is not None:
                clones_losses.append(clone_loss)
                grads_and_vars.append(clone_grad)

            regularization_loss = None      # Only the regularization loss for the first clone
    total_loss = tf.add_n(clones_losses, name='total_loss')
    grads_and_vars = _sum_clones_gradients(grads_and_vars)
    return total_loss, grads_and_vars


def deploy(config,
           model_fn,
           args=None,
           kwargs=None,
           optimizer=None,
           summarize_gradients=False):
    """Deploys a Slim-constructed model across multiple clones.

        The deployment options are specified by the config object and support
        deploying one or several clones on different GPUs and one or several replicas
        of such clones.

        The argument `model_fn` is called `config.num_clones` times to create the
        model clones as `model_fn(*args, **kwargs)`.

        The optional argument `optimizer` is an `Optimizer` object.  If not `None`,
        the deployed model is configured for training with that optimizer.

        If `config` specifies deployment on multiple replicas then the default
        tensorflow device is set appropriatly for each call to `model_fn` and for the
        slim variable creation functions: model and global variables will be created
        on the `ps` device, the clone operations will be on the `worker` device.
        Args:
          config: A `DeploymentConfig` object.
          model_fn: A callable. Called as `model_fn(*args, **kwargs)`
          args: Optional list of arguments to pass to `model_fn`.
          kwargs: Optional list of keyword arguments to pass to `model_fn`.
          optimizer: Optional `Optimizer` object.  If passed the model is deployed
              for training with that optimizer.
          summarize_gradients: Whether or not add summaries to the gradients.
        """
    summaries = set(tf.get_collection(tf.GraphKeys.SUMMARIES))

    clones = create_clones(config, model_fn, args, kwargs)
    first_clone = clones[0]

    # Get update ops for fist clone. Eg. the updates for the batch_norm variable created by model_fn
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, first_clone.scope)
    train_op = None
    total_loss = None
    with tf.device(config.optimizer_device()):
        if optimizer:
            with tf.device(config.variables_device()):
                global_step = slim.get_or_create_global_step()

            total_loss, clones_gradients = optimize_clones(clones, optimizer)

            if clones_gradients:
                if summarize_gradients:
                    summeries |= set(_add_gradients_summaries(clones_gradients))

                grad_updates = optimizer.apply_gradients(clones_gradients,
                                                         global_step=global_step)
                update_ops.append(grad_updates)
                update_op = tf.group(*update_ops)
                train_op = control_flow_ops.with_dependencies([update_op], total_loss,
                                                              name='train_op')

            else:
                clones_losses = []
                regularization_losses = tf.get_collection(
                    tf.GraphKeys.REGULARIZATION_LOSSES
                )
                for clone in clones:
                    with tf.name_scope(clone.scope):
                        clone_loss = _gather_clone_loss(clone, len(clones),
                                                        regularization_losses)
                        if clone_loss is not None:
                            clones_losses.append(clone_loss)
                        regularization_losses = None
                if clones_losses:
                    total_loss = tf.add_n(clones_losses, name='total_loss')

            summaries |= set(tf.get_collection(tf.GraphKeys.SUMMARIES, first_clone.scope))

            if total_loss is not None:
                summaries.add(tf.summary.scalar('total_loss', total_loss))

            if summaries:
                summary_op = tf.summary.merge(list(summaries), name='summary_op')
            else:
                summary_op = None
        return DeployedModel(train_op, summary_op, total_loss, clones)


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

    def clone_device(self, clone_index):

        """
        Device used to create clone and all ops
        """
        if clone_index >= self._num_clones:
            raise ValueError('Clone index must be less than clone numbers')
        device = ''
        if self._num_ps_tasks > 0:
            device += self._worker_device
        if self._clone_on_cpu:
            device += '/device:CPU:0'
        else:
            if self._num_clones > 1 and not self._fake_multiple_gpus:
                device += '/device:GPU:%d' % clone_index
        return device

    def clone_scope(self, clone_index):

        """
        Name scope to create the clone
        """
        if clone_index >= self._num_clones:
            raise ValueError('Clone index must be less than clone numbers')
        scope = ''
        if self._num_clones > 1:
            scope = 'clone_%d' % clone_index
        return scope

    def optimizer_device(self):

        if self._num_ps_tasks > 0 or self._num_clones > 0:
            return self._worker_device + '/device:CPU:0'
        else:
            return ''

    def input_device(self):

        device = ''
        if self._num_ps_tasks > 0:
            device += self._worker_device
        device += '/device:CPU:0'
        return device

    def variables_device(self):

        device = ''
        if self._num_ps_tasks > 0:
            device += self._ps_device
        device += '/device:CPU:0'

        class _PSDeviceChooser(object):

            def __init__(self, device, tasks):
                self._device = device
                self._tasks = tasks
                self._task = 0

            def choose(self, op):
                if op.device:
                    return op.device
                node_def = op if isinstance(op, tf.NodeDef) else op.node_def
                if node_def.op == 'Variable':
                    t = self._task
                    self._task = (self._task + 1) % self._tasks
                    d = '%s/task:%d' (self._device, t)
                    return d
                else:
                    return op.device

        if not self._num_ps_tasks:
            return device
        else:
            chooser = _PSDeviceChooser(device, self._num_ps_tasks)
            return chooser.choose
