import shutil
import time
import tensorflow as tf
import os, sys
import logging

import config 
from make_prior_box import prior_box
from tf_dataloader import load_dataset 
from tf_build_ssd_model import SsdModel


def MultiStepWarmUpLR(initial_learning_rate, lr_steps, lr_rate,
                      warmup_steps=0., min_lr=0.,
                      name='MultiStepWarmUpLR'):
    """Multi-steps warm up learning rate scheduler."""
    assert warmup_steps <= lr_steps[0]
    assert min_lr <= initial_learning_rate
    lr_steps_value = [initial_learning_rate]
    for _ in range(len(lr_steps)):
        lr_steps_value.append(lr_steps_value[-1] * lr_rate)
    return PiecewiseConstantWarmUpDecay(
        boundaries=lr_steps, values=lr_steps_value, warmup_steps=warmup_steps,
        min_lr=min_lr)


class PiecewiseConstantWarmUpDecay(
        tf.keras.optimizers.schedules.LearningRateSchedule):
    """A LearningRateSchedule wiht warm up schedule.
    Modified from tf.keras.optimizers.schedules.PiecewiseConstantDecay"""

    def __init__(self, boundaries, values, warmup_steps, min_lr,
                 name=None):
        super(PiecewiseConstantWarmUpDecay, self).__init__()

        if len(boundaries) != len(values) - 1:
            raise ValueError(
                    "The length of boundaries should be 1 less than the"
                    "length of values")

        self.boundaries = boundaries
        self.values = values
        self.name = name
        self.warmup_steps = warmup_steps
        self.min_lr = min_lr

    def __call__(self, step):
        with tf.name_scope(self.name or "PiecewiseConstantWarmUp"):
            step = tf.cast(tf.convert_to_tensor(step), tf.float32)
            pred_fn_pairs = []
            warmup_steps = self.warmup_steps
            boundaries = self.boundaries
            values = self.values
            min_lr = self.min_lr

            pred_fn_pairs.append(
                (step <= warmup_steps,
                 lambda: min_lr + step * (values[0] - min_lr) / warmup_steps))
            pred_fn_pairs.append(
                (tf.logical_and(step <= boundaries[0],
                                step > warmup_steps),
                 lambda: tf.constant(values[0])))
            pred_fn_pairs.append(
                (step > boundaries[-1], lambda: tf.constant(values[-1])))

            for low, high, v in zip(boundaries[:-1], boundaries[1:],
                                    values[1:-1]):
                # Need to bind v here; can do this with lambda v=v: ...
                pred = (step > low) & (step <= high)
                pred_fn_pairs.append((pred, lambda: tf.constant(v)))

            # The default isn't needed here because our conditions are mutually
            # exclusive and exhaustive, but tf.case requires it.
            return tf.case(pred_fn_pairs, lambda: tf.constant(values[0]),
                           exclusive=True)

    def get_config(self):
        return {
                "boundaries": self.boundaries,
                "values": self.values,
                "warmup_steps": self.warmup_steps,
                "min_lr": self.min_lr,
                "name": self.name
        }


def hard_negative_mining(loss, class_truth, neg_ratio):
    """ Hard negative mining algorithm
        to pick up negative examples for back-propagation
        base on classification loss values
    Args:
        loss: list of classification losses of all default boxes (B, num_default)
        class_truth: classification targets (B, num_default)
        neg_ratio: negative / positive ratio
    Returns:
        class_loss: classification loss
        loc_loss: regression loss
    """
    # loss: B x N
    # class_truth: B x N
    pos_idx = class_truth > 0
    num_pos = tf.reduce_sum(tf.dtypes.cast(pos_idx, tf.int32), axis=1)
    num_neg = num_pos * neg_ratio

    rank = tf.argsort(loss, axis=1, direction='DESCENDING')
    rank = tf.argsort(rank, axis=1)
    neg_idx = rank < tf.expand_dims(num_neg, 1)

    return pos_idx, neg_idx


def MultiBoxLoss(num_class=3, neg_pos_ratio=3.0):
    def multi_loss(y_true, y_pred):
        """ Compute losses for SSD
               regression loss: smooth L1
               classification loss: cross entropy
           Args:
               y_true: [B,N,5]
               y_pred: [B,N,num_class]
               class_pred: outputs of classification heads (B,N, num_classes)
               loc_pred: outputs of regression heads (B,N, 4)
               class_truth: classification targets (B,N)
               loc_truth: regression targets (B,N, 4)
           Returns:
               class_loss: classification loss
               loc_loss: regression loss
       """
        num_batch = tf.shape(y_true)[0]
        num_prior = tf.shape(y_true)[1]
        loc_pred, class_pred = y_pred[..., :4], y_pred[..., 4:]
        loc_truth, class_truth = y_true[..., :4], tf.squeeze(y_true[..., 4:])

        cross_entropy = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')

        # compute classification losses without reduction
        temp_loss = cross_entropy(class_truth, class_pred)
        # 2. hard negative mining
        pos_idx, neg_idx = hard_negative_mining(temp_loss, class_truth, neg_pos_ratio)

        # classification loss will consist of positive and negative examples
        cross_entropy = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='sum')

        smooth_l1_loss = tf.keras.losses.Huber(reduction='sum')

        loss_class = cross_entropy(
            class_truth[tf.math.logical_or(pos_idx, neg_idx)],
            class_pred[tf.math.logical_or(pos_idx, neg_idx)])

        # localization loss only consist of positive examples (smooth L1)
        loss_loc = smooth_l1_loss(loc_truth[pos_idx],loc_pred[pos_idx])

        num_pos = tf.reduce_sum(tf.dtypes.cast(pos_idx, tf.float32))

        loss_class = loss_class / num_pos
        loss_loc = loss_loc / num_pos
        return loss_loc, loss_class

    return multi_loss



def main(_):
    global load_t1
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    logger = tf.get_logger()
    logger.disabled = True
    logger.setLevel(logging.FATAL)

    weights_dir = 'checkpoints/'
    if not os.path.exists(weights_dir):
        os.mkdir(weights_dir)

    logging.info("Load configuration...")
    cfg = config.cfg
    label_classes = cfg['labels_list']
    logging.info(f"Total image sample:{cfg['dataset_len']},Total classes number:"
                 f"{len(label_classes)},classes list:{label_classes}")

    logging.info("Compute prior boxes...")
    priors, num_cell = prior_box(cfg)
    logging.info(f"Prior boxes number:{len(priors)},default anchor box number per feature map cell:{num_cell}") # 4420, [3, 2, 2, 3]

    logging.info("Loading dataset...")
    train_dataset = load_dataset(cfg, priors, shuffle=True, train=True)
    # val_dataset = load_dataset(cfg, priors, shuffle=False, train=False)

    logging.info("Create Model...")
    try:
        model = SsdModel(cfg=cfg, num_cell=num_cell, training=True)
        model.summary()
        tf.keras.utils.plot_model(model, to_file=os.path.join(os.getcwd(), 'model.png'),
                                  show_shapes=True, show_layer_names=True)
    except Exception as e:
        logging.error(e)
        logging.info("Create network failed.")
        sys.exit()


    if cfg['resume']:
        # Training from latest weights
        paths = [os.path.join(weights_dir, path)
                 for path in os.listdir(weights_dir)]
        latest = sorted(paths, key=os.path.getmtime)[-1]
        model.load_weights(latest)
        init_epoch = int(os.path.splitext(latest)[0][-3:])

    else:
        init_epoch = -1


    steps_per_epoch = cfg['dataset_len'] // cfg['batch_size']

    logging.info(f"steps_per_epoch:{steps_per_epoch}")


    logging.info("Define optimizer and loss computation and so on...")

    learning_rate = MultiStepWarmUpLR(
        initial_learning_rate=cfg['init_lr'],
        lr_steps=[e * steps_per_epoch for e in cfg['lr_decay_epoch']],
        lr_rate=cfg['lr_rate'],
        warmup_steps=cfg['warmup_epoch'] * steps_per_epoch,
        min_lr=cfg['min_lr'])


    optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=cfg['momentum'], nesterov=True)

    multi_loss = MultiBoxLoss(num_class=len(label_classes), neg_pos_ratio=3)

    train_log_dir = 'logs/train'
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)


    @tf.function
    def train_step(inputs, labels):
        with tf.GradientTape() as tape:
            predictions = model(inputs, training=True)
            losses = {}
            losses['reg'] = tf.reduce_sum(model.losses)  #unused. Init for redefine network
            losses['loc'], losses['class'] = multi_loss(labels, predictions)
            total_loss = tf.add_n([l for l in losses.values()])

        grads = tape.gradient(total_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        return total_loss, losses

    for epoch in range(init_epoch+1,cfg['epoch']):
        try:
            start = time.time()
            avg_loss = 0.0
            for step, (inputs, labels) in enumerate(train_dataset.take(steps_per_epoch)):

                load_t0 = time.time()
                total_loss, losses = train_step(inputs, labels)
                avg_loss = (avg_loss * step + total_loss.numpy()) / (step + 1)
                load_t1 = time.time()
                batch_time = load_t1 - load_t0

                steps =steps_per_epoch*epoch+step
                with train_summary_writer.as_default():
                    tf.summary.scalar('loss/total_loss', total_loss, step=steps)
                    for k, l in losses.items():
                        tf.summary.scalar('loss/{}'.format(k), l, step=steps)
                    tf.summary.scalar('learning_rate', optimizer.lr(steps), step=steps)

                print(f"\rEpoch: {epoch + 1}/{cfg['epoch']} | Batch {step + 1}/{steps_per_epoch} | Batch time {batch_time:.3f} || Loss: {total_loss:.6f} | loc loss:{losses['loc']:.6f} | class loss:{losses['class']:.6f} ",end = '',flush=True)

            print(f"\nEpoch: {epoch + 1}/{cfg['epoch']}  | Epoch time {(load_t1 - start):.3f} || Average Loss: {avg_loss:.6f}")

            with train_summary_writer.as_default():
                tf.summary.scalar('loss/avg_loss',avg_loss,step=epoch)

            if (epoch + 1) % cfg['save_freq'] == 0:
                filepath = os.path.join(weights_dir, f'weights_epoch_{(epoch + 1):03d}.h5')
                model.save_weights(filepath)
                if os.path.exists(filepath):
                    print(f">>>>>>>>>>Save weights file at {filepath}<<<<<<<<<<")

        except KeyboardInterrupt:
            print('interrupted')
            exit(0)

if __name__ == '__main__':

    try:
        main(None)
    except SystemExit:
        pass


