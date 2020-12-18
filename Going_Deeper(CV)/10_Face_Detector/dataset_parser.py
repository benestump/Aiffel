import os
import logging
import tensorflow as tf
import tqdm

rootPath = os.getenv('HOME')+'/aiffel/face_detector'

# Input: [x0, y0, w, h, blur, expression, illumination, invalid, occlusion, pose]
# Output: x0, y0, w, h
def get_box(data):
    x0 = int(data[0])
    y0 = int(data[1])
    w = int(data[2])
    h = int(data[3])
    return x0, y0, w, h

def parse_widerface(config_path):
    boxes_per_img = []
    with open(config_path) as fp:
        line = fp.readline()
        cnt = 1
        while line:
            num_of_obj = int(fp.readline())
            boxes = []
            for i in range(num_of_obj):
                obj_box = fp.readline().split(' ')
                x0, y0, w, h = get_box(obj_box)
                if w == 0:
                    # remove boxes with no width
                    continue
                if h == 0:
                    # remove boxes with no height
                    continue
                # Because our network is outputting 7x7 grid then it's not worth processing images with more than
                # 5 faces because it's highly probable they are close to each other.
                # You could remove this filter if you decide to switch to larger grid (like 14x14)
                # Don't worry about number of train data because even with this filter we have around 16k samples
                boxes.append([x0, y0, w, h])

            if num_of_obj == 0:
                obj_box = fp.readline().split(' ')
                x0, y0, w, h = get_box(obj_box)
                boxes.append([x0, y0, w, h])
            boxes_per_img.append((line.strip(), boxes))
            line = fp.readline()
            cnt += 1

    return boxes_per_img


def process_image(image_file):
    # image_string = open(image_file,'rb').read()
    image_string = tf.io.read_file(image_file)
    try:
        image_data = tf.image.decode_jpeg(image_string, channels=3)
        return 0, image_string, image_data
    except tf.errors.InvalidArgumentError:
        logging.info('{}: Invalid JPEG data or crop window'.format(image_file))
        return 1, image_string, None


def xywh_to_voc(file_name, boxes, image_data):
    shape = image_data.shape
    image_info = {}
    image_info['filename'] = file_name
    image_info['width'] = shape[1]
    image_info['height'] = shape[0]
    image_info['depth'] = 3

    difficult = []
    classes = []
    xmin, ymin, xmax, ymax = [], [], [], []

    for box in boxes:
        classes.append(1)
        difficult.append(0)
        xmin.append(box[0])
        ymin.append(box[1])
        xmax.append(box[0] + box[2])
        ymax.append(box[1] + box[3])
    image_info['class'] = classes
    image_info['xmin'] = xmin
    image_info['ymin'] = ymin
    image_info['xmax'] = xmax
    image_info['ymax'] = ymax
    image_info['difficult'] = difficult

    return image_info


def make_example(image_string, image_info_list):

    for info in image_info_list:
        filename = info['filename']
        width = info['width']
        height = info['height']
        depth = info['depth']
        classes = info['class']
        xmin = info['xmin']
        ymin = info['ymin']
        xmax = info['xmax']
        ymax = info['ymax']
        # difficult = info['difficult']

    if isinstance(image_string, type(tf.constant(0))):
        encoded_image = [image_string.numpy()]
    else:
        encoded_image = [image_string]

    base_name = [tf.compat.as_bytes(os.path.basename(filename))]

    example = tf.train.Example(features=tf.train.Features(feature={
        'filename':tf.train.Feature(bytes_list=tf.train.BytesList(value=base_name)),
        'height':tf.train.Feature(int64_list=tf.train.Int64List(value=[height])),
        'width':tf.train.Feature(int64_list=tf.train.Int64List(value=[width])),
        'classes':tf.train.Feature(int64_list=tf.train.Int64List(value=classes)),
        'x_mins':tf.train.Feature(float_list=tf.train.FloatList(value=xmin)),
        'y_mins':tf.train.Feature(float_list=tf.train.FloatList(value=ymin)),
        'x_maxes':tf.train.Feature(float_list=tf.train.FloatList(value=xmax)),
        'y_maxes':tf.train.Feature(float_list=tf.train.FloatList(value=ymax)),
        'image_raw':tf.train.Feature(bytes_list=tf.train.BytesList(value=encoded_image))
    }))
    return example


def main(argv):
    dataset_path = 'widerface'

    if not os.path.isdir(dataset_path):
        logging.info('Please define valid dataset path.')
    else:
        logging.info('Loading {}'.format(dataset_path))

    logging.info('Reading configuration...')

    for split in ['train', 'val']:
        output_file = rootPath+'/dataset/train_mask.tfrecord' if split == 'train' else rootPath+'/dataset/val_mask.tfrecord'

        with tf.io.TFRecordWriter(output_file) as writer:
            
            counter = 0
            skipped = 0
            anno_txt = 'wider_face_train_bbx_gt.txt' if split == 'train' else 'wider_face_val_bbx_gt.txt'
            file_path = 'WIDER_train' if split == 'train' else 'WIDER_val'
            for info in tqdm.tqdm(parse_widerface(os.path.join(dataset_path, 'wider_face_split', anno_txt))):
                image_file = os.path.join(dataset_path, file_path, 'images', info[0])

                error, image_string, image_data = process_image(image_file)
                boxes = xywh_to_voc(image_file, info[1], image_data)

                if not error:
                    tf_example = make_example(image_string, [boxes])

                    writer.write(tf_example.SerializeToString())
                    counter += 1

                else:
                    skipped += 1
                    logging.info('Skipped {:d} of {:d} images.'.format(skipped, len(img_list)))

        logging.info('Wrote {} images to {}'.format(counter, output_file))

if __name__ == '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    try:
        main(None)
    except SystemExit:
        pass

