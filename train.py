import tensorflow as tf
import tensorflow.contrib.slim as slim

from nets import ssd_vgg
from tensorflow.python.ops import control_flow_ops
from preprocessing import ssd_vgg_preprocessing
from Utils import tf_utils
from datasets import dataset_factory
from deployment import model_deploy



DATA_FORMAT = 'NHWC'

# -------------------------------------
#           Flags of SSD Net
# -------------------------------------

tf.app.flags.DEFINE_float(
    'loss_alpha', 1, 'Alpha parameter in loss function')
tf.app.flags.DEFINE_float(
    'negative_ratio', 3, 'Negative ratio in loss function')
tf.app.flags.DEFINE_float(
    'match_threshold', 0.5, 'Matching threshold')

# -------------------------------------
#           General Flags
# -------------------------------------

tf.app.flags.DEFINE_string(
    'train_dir', 'D:\Pycharm\Projects\SSD_tensorflow\\tmp\\tfmodel\\',
    'Directory where checkpoints and event logs are written to.')
tf.app.flags.DEFINE_integer('num_clones', 1,
                            'Number of model clones to deploy.')
tf.app.flags.DEFINE_boolean('clone_on_cpu', False,
                            'Use CPUs to deploy clones.')
tf.app.flags.DEFINE_integer(
    'num_readers', 4,
    'The number of parallel readers that read data from the dataset.')
tf.app.flags.DEFINE_integer(
    'num_preprocessing_threads', 4,
    'The number of threads used to create the batches.')

tf.app.flags.DEFINE_integer(
    'log_every_n_steps', 10,
    'The frequency with which logs are print.')
tf.app.flags.DEFINE_integer(
    'save_summaries_secs', 10,
    'The frequency with which summaries are saved, in seconds.')
tf.app.flags.DEFINE_integer(
    'save_interval_secs', 600,
    'The frequency with which the model is saved, in seconds.')
tf.app.flags.DEFINE_float(
    'gpu_memory_fraction', 0.8, 'GPU memory fraction to use.')

# -------------------------------------
#          Optimization Flags
# -------------------------------------

tf.app.flags.DEFINE_float(
    'weight_decay', 0.00004, 'The weight decay on the model weights.')
tf.app.flags.DEFINE_string(
    'optimizer', 'rmsprop',
    'The name of the optimizer, one of "adadelta", "adagrad", "adam",'
    '"ftrl", "momentum", "sgd" or "rmsprop".')
tf.app.flags.DEFINE_float(
    'adadelta_rho', 0.95,
    'The decay rate for adadelta.')
tf.app.flags.DEFINE_float(
    'adagrad_initial_accumulator_value', 0.1,
    'Starting value for the AdaGrad accumulators.')
tf.app.flags.DEFINE_float(
    'adam_beta1', 0.9,
    'The exponential decay rate for the 1st moment estimates.')
tf.app.flags.DEFINE_float(
    'adam_beta2', 0.999,
    'The exponential decay rate for the 2nd moment estimates.')
tf.app.flags.DEFINE_float('opt_epsilon', 1.0, 'Epsilon term for the optimizer.')
tf.app.flags.DEFINE_float('ftrl_learning_rate_power', -0.5,
                          'The learning rate power.')
tf.app.flags.DEFINE_float(
    'ftrl_initial_accumulator_value', 0.1,
    'Starting value for the FTRL accumulators.')
tf.app.flags.DEFINE_float(
    'ftrl_l1', 0.0, 'The FTRL l1 regularization strength.')
tf.app.flags.DEFINE_float(
    'ftrl_l2', 0.0, 'The FTRL l2 regularization strength.')
tf.app.flags.DEFINE_float(
    'momentum', 0.9,
    'The momentum for the MomentumOptimizer and RMSPropOptimizer.')
tf.app.flags.DEFINE_float('rmsprop_momentum', 0.9, 'Momentum.')
tf.app.flags.DEFINE_float('rmsprop_decay', 0.9, 'Decay term for RMSProp.')

# -------------------------------------
#         Learning Rate Flags
# -------------------------------------

tf.app.flags.DEFINE_string(
    'learning_rate_decay_type',
    'exponential',
    'Specifies how the learning rate is decayed. One of "fixed", "exponential",'
    ' or "polynomial"')
tf.app.flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
tf.app.flags.DEFINE_float(
    'end_learning_rate', 0.0001,
    'The minimal end learning rate used by a polynomial decay learning rate.')
tf.app.flags.DEFINE_float(
    'label_smoothing', 0.0, 'The amount of label smoothing.')
tf.app.flags.DEFINE_float(
    'learning_rate_decay_factor', 0.94, 'Learning rate decay factor.')
tf.app.flags.DEFINE_float(
    'num_epochs_per_decay', 2.0,
    'Number of epochs after which learning rate decays.')
tf.app.flags.DEFINE_float(
    'moving_average_decay', None,
    'The decay to use for the moving average.'
    'If left as None, then moving averages are not used.')

# -------------------------------------
#              Dataset Flags
# -------------------------------------

tf.app.flags.DEFINE_string(
    'dataset_name', 'pascalvoc_2007', 'The name of the dataset to load.')
tf.app.flags.DEFINE_integer(
    'num_classes', 21, 'Number of classes to use in the dataset.')
tf.app.flags.DEFINE_string(
    'dataset_split_name', 'train', 'The name of the train/test split.')
tf.app.flags.DEFINE_string(
    'dataset_dir', None, 'The directory where the dataset files are stored.')
tf.app.flags.DEFINE_integer(
    'labels_offset', 0,
    'An offset for the labels in the dataset. This flag is primarily used to '
    'evaluate the VGG and ResNet architectures which do not use a background '
    'class for the ImageNet dataset.')
tf.app.flags.DEFINE_string(
    'model_name', 'ssd_vgg', 'The name of the architecture to train.')
tf.app.flags.DEFINE_string(
    'preprocessing_name', None, 'The name of the preprocessing to use. If left '
    'as `None`, then the model_name flag is used.')
tf.app.flags.DEFINE_integer(
    'batch_size', 32, 'The number of samples in each batch.')
tf.app.flags.DEFINE_integer(
    'train_image_size', None, 'Train image size')
tf.app.flags.DEFINE_integer('max_number_of_steps', None,
                            'The maximum number of training steps.')

# -------------------------------------
#            Fine-Tuning Flags
# -------------------------------------
tf.app.flags.DEFINE_string(
    'checkpoint_path', None,
    'The path to a checkpoint from which to fine-tune.')
tf.app.flags.DEFINE_string(
    'checkpoint_model_scope', None,
    'Model scope in the checkpoint. None if the same as the trained model.')
tf.app.flags.DEFINE_string(
    'checkpoint_exclude_scopes', None,
    'Comma-separated list of scopes of variables to exclude when temprestoring '
    'from a checkpoint.')
tf.app.flags.DEFINE_string(
    'trainable_scopes', None,
    'Comma-separated list of scopes to filter the set of variables to train.'
    'By default, None would train all the variables.')
tf.app.flags.DEFINE_boolean(
    'ignore_missing_vars', False,
    'When restoring a checkpoint would ignore missing variables.')

FLAGS = tf.app.flags.FLAGS

def main(_):
    # for name, value in FLAGS.__flags.items():
    #     print(name, ': ', value.value)
    if not FLAGS.dataset_dir:
        raise ValueError('Directory of dataset is not found')
    tf.logging.set_verbosity(tf.logging.DEBUG)
    tf.logging.debug("hahahaha %s" %FLAGS.dataset_dir )
    with tf.Graph().as_default():

        deploy_config = model_deploy.DeploymentConfig(
            num_clones=FLAGS.num_clones,
            clone_on_cpu=FLAGS.clone_on_cpu,
            replica_id=0,
            num_replicas=1,
            num_ps_tasks=0
        )
        with tf.device(deploy_config.variables_device()):
            global_step = slim.create_global_step()

        dataset = dataset_factory.get_dataset(
            FLAGS.dataset_name, FLAGS.dataset_split_name, FLAGS.dataset_dir
        )

        #SSD Net and anchors
        ssd_class = ssd_vgg.SSDNet
        ssd_params = ssd_class.default_parameters._replace(num_classes=FLAGS.num_classes)
        print("Class numbers", ssd_params.num_classes)
        ssd_net = ssd_class(ssd_params)
        ssd_shape = ssd_net.params.img_shape
        ssd_anchors = ssd_net.anchors(ssd_shape)

        # Preprocessing function
        image_preprocessing_fun = ssd_vgg_preprocessing.preprocess_image        # Need is_train = True
        tf_utils.print_configs(FLAGS.__flags, ssd_params, dataset.data_sources, FLAGS.train_dir)

        #--------------------------------------------
        #         Data provider and batches
        # -------------------------------------------
        with tf.device(deploy_config.input_device()):
            with tf.name_scope(FLAGS.dataset_name + '_data_provider'):
                provider = slim.dataset_data_provider.DatasetDataProvider(
                    dataset,
                    num_readers=FLAGS.num_readers,
                    common_queue_capacity=20*FLAGS.batch_size,
                    common_queue_min=10*FLAGS.batch_size,
                    shuffle=True
                )
            [image, shape, gtlabels, gtbboxes] = provider.get(['image',
                                                               'shape',
                                                               'object/label',
                                                               'object/bbox'])
            # Pre-processing image, labels and bboxes
            image, gtlabels, gtbboxes = image_preprocessing_fun(image, gtlabels,
                                                                gtbboxes, out_shape=ssd_shape,
                                                                data_format=DATA_FORMAT,
                                                                is_training=True)
            # Encode groundtruth labels and bboxes
            gtclasses, gtlocations, gtscores = ssd_net.bboxes_encode(gtlabels,
                                                                     gtbboxes,
                                                                     ssd_anchors)
            batch_shape = [1] + [len(ssd_anchors)] * 3          #[1, len, len, len]

            # Training batch and queue
            r = tf.train.batch(
                tf_utils.reshape_list([image, gtclasses, gtlocations, gtscores]),
                batch_size=FLAGS.batch_size,
                num_threads=FLAGS.num_preprocessing_threads,
                capacity=5 * FLAGS.batch_size
            )
            b_image, b_gtclasses, b_gtlocations, b_gtscores = \
                tf_utils.reshape_list(r, batch_shape)           #[1*image, N*gtclasses, N*gtlocations, N*gtstores]

            # Intermediate queueing: unique batch computation pipeline for all
            # GPUs running the training.
            batch_queue = slim.prefetch_queue.prefetch_queue(
                tf_utils.reshape_list([b_image, b_gtclasses, b_gtlocations, b_gtscores]),
                capacity=2 * deploy_config.num_clones
            )

        #--------------------------------------------------
        #                 Clone on every GPU
        #--------------------------------------------------
        def clone_fn(batch_queue):
            b_image, b_gtclasses, b_gtlocations, b_gtscores = \
                tf_utils.reshape_list(batch_queue.dequeue(), batch_shape)

            arg_scope = ssd_net.arg_scope(weight_decay=FLAGS.weight_decay,
                                          data_format=DATA_FORMAT)
            with slim.arg_scope(arg_scope):
                prediction, location, logits, end_points = \
                    ssd_net.net(b_image, is_training=True)

            ssd_net.losses(logits, location,
                           b_gtclasses, b_gtlocations, b_gtscores,
                           match_threshold=FLAGS.match_threshold,
                           negative_ratio=FLAGS.negative_ratio,
                           alpha=FLAGS.loss_alpha,
                           label_smoothing=FLAGS.label_smoothing)
            return end_points

        summaries = set(tf.get_collection(tf.GraphKeys.SUMMARIES))

        # ---------------------------------------------------------
        #                 Summary for first clone
        # ---------------------------------------------------------
        clones = model_deploy.create_clones(deploy_config, clone_fn, [batch_queue])
        first_clone_scope = deploy_config.clone_scope(0)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, first_clone_scope)

        end_points = clones[0].outputs
        for end_point in end_points:
            x = end_points[end_point]
            summaries.add(tf.summary.histogram('activations/' + end_point, x))
            summaries.add(tf.summary.scalar('sparsity/' + end_point,
                                            tf.nn.zero_fraction(x)))        # 零元素所占比例
        for loss in tf.get_collection(tf.GraphKeys.LOSSES, first_clone_scope):
            summaries.add(tf.summary.scalar(loss.op.name, loss))
        for loss in tf.get_collection('EXTRA_LOSSES', first_clone_scope):
            summaries.add(tf.summary.scalar(loss.op.name, loss))

        for variable in slim.get_model_variables():
            summaries.add(tf.summary.histogram(variable.op.name, variable))

        if FLAGS.moving_average_decay:
            moving_average_variables = slim.get_model_variables()
            variable_averages = tf.train.ExponentialMovingAverage(
                FLAGS.moving_average_decay, global_step
            )
        else:
            moving_average_variables, variable_averages = None, None

        # ----------------------------------------------
        #              Optimization procedure
        # ----------------------------------------------
        with tf.device(deploy_config.optimizer_device()):
            learning_rate = tf_utils.configure_learning_rate(FLAGS,
                                                             dataset.num_samples,
                                                             global_step)
            optimizer = tf_utils.configure_optimizer(FLAGS, learning_rate)
            summaries.add(tf.summary.scalar('learning_rate', learning_rate))

            if FLAGS.moving_average_decay:
                update_ops.append(variable_averages.apply(moving_average_variables))

            variables_to_train = tf_utils.get_variables_to_train(FLAGS)

            total_loss, clones_gradients = model_deploy.optimize_clones(
                clones,
                optimizer,
                var_list=variables_to_train
            )

            summaries.add(tf.summary.scalar('total_loss', total_loss))

            grad_updates = optimizer.apply_gradients(clones_gradients,
                                                     global_step=global_step)
            update_ops.append(grad_updates)
            update_op = tf.group(*update_ops)               # *Operations, not lists
            # All the operations needed to be ran prior
            train_tensor = control_flow_ops.with_dependencies([update_op], total_loss,
                                                              name='train_op')

            summaries |= set(tf.get_collection(tf.GraphKeys.SUMMARIES,
                                               first_clone_scope))

            summary_op = tf.summary.merge(list(summaries), name='summary_op')

            # ---------------------------------------------
            #               Now let's start!
            # ---------------------------------------------
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=FLAGS.gpu_memory_fraction,
                                        allow_growth=True)
            config = tf.ConfigProto(log_device_placement=False,
                                    gpu_options=gpu_options)
            saver = tf.train.Saver(max_to_keep=5,
                                   keep_checkpoint_every_n_hours=1.0,
                                   write_version=2,
                                   pad_step_number=False,
                                   name='Model_ssd_vgg')


            def train_step_fn(session, *args, **kwargs):
                total_loss, should_stop = slim.learning.train_step(session, *args, *kwargs)
                if train_step_fn.step % 2 == 0:
                    print('step: %s || loss: %f || gradient: '
                          %(str(train_step_fn.step), total_loss))

                train_step_fn.step += 1
                return [total_loss, should_stop]

            train_step_fn.step = 0


            slim.learning.train(
                train_tensor,
                logdir=FLAGS.train_dir,
                master='',
                is_chief=True,
                init_fn=tf_utils.get_init_fn(FLAGS),
                summary_op=summary_op,
                number_of_steps=FLAGS.max_number_of_steps,
                log_every_n_steps=FLAGS.log_every_n_steps,
                save_summaries_secs=FLAGS.save_summaries_secs,      # Save summaries in seconds
                saver=saver,
                save_interval_secs=FLAGS.save_interval_secs,        # Save checkpoints in seconds
                session_config=config,
                sync_optimizer=None,
                train_step_fn=train_step_fn
            )



if __name__ == '__main__':
    tf.app.run()