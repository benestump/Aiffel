import shutil
import time

import tensorflow as tf
import os, sys

import logging
import argparse
import cv2
import numpy as np

import config 
from make_prior_box import prior_box
from tf_dataloader import load_dataset, _jaccard 
from tf_build_ssd_model import SsdModel

# hyperparameters
args = argparse.ArgumentParser()
args.add_argument('model_path', type=str, nargs='?', default='checkpoints/')
args.add_argument('img_path', type=str, nargs='?', default=None)
args.add_argument('camera', type=str, nargs='?', default=False)

args_config = args.parse_args()


def compute_nms(boxes, scores, nms_threshold=0.5, limit=200):
    """ Perform Non Maximum Suppression algorithm
        to eliminate boxes with high overlap
    Args:
        boxes: tensor (num_boxes, 4)
               of format (xmin, ymin, xmax, ymax)
        scores: tensor (num_boxes,)
        nms_threshold: NMS threshold
        limit: maximum number of boxes to keep
    Returns:
        idx: indices of kept boxes
    """
    if boxes.shape[0] == 0:
        return tf.constant([], dtype=tf.int32)
    selected = [0]
    idx = tf.argsort(scores, direction='DESCENDING')
    idx = idx[:limit]
    boxes = tf.gather(boxes, idx)

    iou = _jaccard(boxes, boxes)

    while True:
        row = iou[selected[-1]]
        next_indices = row <= nms_threshold

        iou = tf.where(
            tf.expand_dims(tf.math.logical_not(next_indices), 0),
            tf.ones_like(iou, dtype=tf.float32),
            iou)

        if not tf.math.reduce_any(next_indices):
            break

        selected.append(tf.argsort(
            tf.dtypes.cast(next_indices, tf.int32), direction='DESCENDING')[0].numpy())

    return tf.gather(idx, selected)


def decode_bbox_tf(pre, priors, variances=None):
    """Decode locations from predictions using prior to undo
    the encoding we did for offset regression at train time.
    Args:
        pre (tensor): location predictions for loc layers,
            Shape: [num_prior,4]
        prior (tensor): Prior boxes in center-offset form.
            Shape: [num_prior,4].
        variances: (list[float]) Variances of prior boxes
    Return:
        decoded bounding box predictions xmin, ymin, xmax, ymax
    """
    if variances is None:
        variances = [0.1, 0.2]
    centers = priors[:, :2] + pre[:, :2] * variances[0] * priors[:, 2:]
    sides = priors[:, 2:] * tf.math.exp(pre[:, 2:] * variances[1])

    return tf.concat([centers - sides / 2, centers + sides / 2], axis=1)


def parse_predict(predictions, priors, cfg):
    label_classes = cfg['labels_list']

    bbox_regressions, confs = tf.split(predictions[0], [4, -1], axis=-1)
    boxes = decode_bbox_tf(bbox_regressions, priors, cfg['variances'])


    confs = tf.math.softmax(confs, axis=-1)

    out_boxes = []
    out_labels = []
    out_scores = []

    for c in range(1, len(label_classes)):
        cls_scores = confs[:, c]

        score_idx = cls_scores > cfg['score_threshold']

        cls_boxes = boxes[score_idx]
        cls_scores = cls_scores[score_idx]

        nms_idx = compute_nms(cls_boxes, cls_scores, cfg['nms_threshold'], cfg['max_number_keep'])

        cls_boxes = tf.gather(cls_boxes, nms_idx)
        cls_scores = tf.gather(cls_scores, nms_idx)

        cls_labels = [c] * cls_boxes.shape[0]

        out_boxes.append(cls_boxes)
        out_labels.extend(cls_labels)
        out_scores.append(cls_scores)

    out_boxes = tf.concat(out_boxes, axis=0)
    out_scores = tf.concat(out_scores, axis=0)

    boxes = tf.clip_by_value(out_boxes, 0.0, 1.0).numpy()
    classes = np.array(out_labels)
    scores = out_scores.numpy()

    return boxes, classes, scores


def pad_input_image(img, max_steps):
    """pad image to suitable shape"""
    img_h, img_w, _ = img.shape

    img_pad_h = 0
    if img_h % max_steps > 0:
        img_pad_h = max_steps - img_h % max_steps

    img_pad_w = 0
    if img_w % max_steps > 0:
        img_pad_w = max_steps - img_w % max_steps

    padd_val = np.mean(img, axis=(0, 1)).astype(np.uint8)
    img = cv2.copyMakeBorder(img, 0, img_pad_h, 0, img_pad_w,
                             cv2.BORDER_CONSTANT, value=padd_val.tolist())
    pad_params = (img_h, img_w, img_pad_h, img_pad_w)

    return img, pad_params


def recover_pad_output(outputs, pad_params):
    """
        recover the padded output effect

    """
    img_h, img_w, img_pad_h, img_pad_w = pad_params

    recover_xy = np.reshape(outputs[0], [-1, 2, 2]) * \
                 [(img_pad_w + img_w) / img_w, (img_pad_h + img_h) / img_h]
    outputs[0] = np.reshape(recover_xy, [-1, 4])

    return outputs


def show_image(img, boxes, classes, scores, img_height, img_width, prior_index, class_list):
    """
    draw bboxes and labels
    out:boxes,classes,scores
    """
    # bbox

    x1, y1, x2, y2 = int(boxes[prior_index][0] * img_width), int(boxes[prior_index][1] * img_height), \
                     int(boxes[prior_index][2] * img_width), int(boxes[prior_index][3] * img_height)
    if classes[prior_index] == 1:
        color = (0, 255, 0)
    else:
        color = (0, 0, 255)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    # confidence

    if scores:
        score = "{:.4f}".format(scores[prior_index])
        class_name = class_list[classes[prior_index]]

        cv2.putText(img, '{} {}'.format(class_name, score),
                    (int(boxes[prior_index][0] * img_width), int(boxes[prior_index][1] * img_height) - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))

def main(_):
    global model
    cfg = config.cfg
    min_sizes=cfg['min_sizes']
    num_cell = [len(min_sizes[k]) for k in range(len(cfg['steps']))]

    try:
        model = SsdModel(cfg=cfg, num_cell=num_cell, training=False)

        paths = [os.path.join(args_config.model_path, path)
                 for path in os.listdir(args_config.model_path)]
        latest = sorted(paths, key=os.path.getmtime)[-1]
        model.load_weights(latest)
        print(f"model path : {latest}")

    except AttributeError as e:
        print('Please make sure there is at least one weights at {}'.format(args_config.model_path))

    if not args_config.camera:
        if not os.path.exists(args_config.img_path):
            print(f"Cannot find image path from {args_config.img_path}")
            exit()
        print("[*] Predict {} image.. ".format(args_config.img_path))
        img_raw = cv2.imread(args_config.img_path)
        img_raw = cv2.resize(img_raw, (320, 240))
        img_height_raw, img_width_raw, _ = img_raw.shape
        img = np.float32(img_raw.copy())

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # pad input image to avoid unmatched shape problem
        img, pad_params = pad_input_image(img, max_steps=max(cfg['steps']))
        img = img / 255.0 - 0.5
        print(img.shape)
        priors, _ = prior_box(cfg, image_sizes = (img.shape[0], img.shape[1]))
        priors = tf.cast(priors, tf.float32)

        predictions = model.predict(img[np.newaxis, ...])

        boxes, classes, scores = parse_predict(predictions, priors, cfg)

        print(f"scores:{scores}")
        # recover padding effect
        boxes = recover_pad_output(boxes, pad_params)

        # draw and save results
        save_img_path = os.path.join('assets/out_' + os.path.basename(args_config.img_path))


        for prior_index in range(len(boxes)):
            show_image(img_raw, boxes, classes, scores, img_height_raw, img_width_raw, prior_index,cfg['labels_list'])

        cv2.imwrite(save_img_path, img_raw)
        cv2.imshow('results', img_raw)
        if cv2.waitKey(0) == ord('q'):
            exit(0)

    else:
        capture = cv2.VideoCapture(args_config.img_path)
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

        priors, _ = prior_box(cfg, image_sizes=(240, 320))
        priors = tf.cast(priors, tf.float32)
        start = time.time()
        while True:
            _,frame = capture.read()
            if frame is None:
                print('No camera found')

            h,w,_ = frame.shape
            img = np.float32(frame.copy())

            img  = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            
            img = img / 255.0 - 0.5

            predictions = model(img[np.newaxis, ...])
            boxes, classes, scores = parse_predict(predictions, priors, cfg)

            for prior_index in range(len(classes)):
                show_image(frame, boxes, classes, scores, h, w, prior_index,cfg['labels_list'])
            # calculate fps
            fps_str = "FPS: %.2f" % (1 / (time.time() - start))
            start = time.time()
            cv2.putText(frame, fps_str, (25, 25),cv2.FONT_HERSHEY_DUPLEX, 0.75, (0, 255, 0), 2)

            # show frame
            cv2.imshow('frame', frame)
            if cv2.waitKey(1) == ord('q'):
                exit()

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    try:
        main(None)
    except Exception as e:
        print(e)
        exit()
