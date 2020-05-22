import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import collections
import os
from os import makedirs, path as op
import sys
import glob
import six.moves.urllib as urllib
import tensorflow as tf
import tarfile
import cv2
from io import StringIO
import zipfile
import numpy as np
from collections import defaultdict
#from matplotlib import pyplot as plt
from PIL import ImageDraw, Image
import csv
sys.path.append("..")

from utils import label_map_util
from utils import visualization_utils as vis_util
import json

flags = tf.app.flags
flags.DEFINE_string('model_name', '', 'Path to frozen detection graph')
flags.DEFINE_string('path_to_label', '', 'Path to label file')
flags.DEFINE_string('test_image_path', '', 'Path to test imgs and output directory')
FLAGS = flags.FLAGS



def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)

def tf_od_pred():
    with detection_graph.as_default():
        out = {}
        with tf.Session(graph=detection_graph) as sess:
            # Definite input and output Tnsors for detection_graph
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            # Each box represents a part of the image where a particular object was detected.
            detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            # Each score represent how level of confidence for each of the objects.
            # Score is shown on the result image, together with the class label.
            detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
            detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')
            for image_path in test_imgs:
                image = Image.open(image_path)
                image_np = load_image_into_numpy_array(image)
                # the array based representation of the image will be used later in order to prepare the
                # result image with boxes and labels on it.
                # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                image_np_expanded = np.expand_dims(image_np, axis=0)
                # Actual detection.
                (boxes, scores, classes, num) = sess.run(
                  [detection_boxes, detection_scores, detection_classes, num_detections],
                  feed_dict={image_tensor: image_np_expanded})
                 # draw_bounding_box_on_image(image, boxes, )
                 # Visualization of the results of a detection.
                final_score = np.squeeze(scores) 
                count = 0
                for i in range(100):
                    if scores is None or final_score[i] > 0.5:
                            count = count + 1
                                      
                vis_image = vis_util.visualize_boxes_and_labels_on_image_array(
                         image_np,
                         np.squeeze(boxes),
                         np.squeeze(classes).astype(np.int32),
                         np.squeeze(scores),
                         category_index,
                         use_normalized_coordinates=True,
                         line_thickness=8)
                # insert information text to video frame
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(
                    image_np,
                    'Detected products count: ' + str(count),
                    (10, 35),
                    font,
                    0.8,
                    (0, 0xFF, 0xFF),
                    2,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    )
                #cv2.imshow('product detection', image_np)
                out[image_path.split('/')[-1]] = str(count) +" products found"
                #print(outpdict)
                #print("{} boxes in {} image tile!".format(len(boxes), image_path))
                image_pil = Image.fromarray(np.uint8(vis_image)).convert('RGB')
                with tf.gfile.Open(image_path, 'w') as fid:
                     image_pil.save(fid, 'PNG')
            
            #out['products'].append(outpdict)
            with open('image2products.json','w+') as jfile:
              json.dump(out, jfile)



if __name__ =='__main__':
    # load your own trained model inference graph. This inference graph was generated from
    # export_inference_graph.py under model directory, see `models/research/object_detection/`
    model_name = op.join(os.getcwd(), FLAGS.model_name)
    # Path to frozen detection graph.
    path_to_ckpt = op.join(model_name,  '/content/models/research/fine_tuned_model/frozen_inference_graph.pb')
    # Path to the label file
    path_to_label = op.join(os.getcwd(), '/content/Object_Detection-with-custom_data/data/annotations/label_map.pbtxt')
    #only train on buildings
    num_classes = 2
    #Directory to test images path
    test_image_path = op.join(os.getcwd(), '/content/Object_Detection-with-custom_data/test')
    test_imgs = glob.glob(test_image_path + "/*.JPG")
    #print(test_imgs)
    ############
    #Load the frozen tensorflow model
    #############

    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.io.gfile.GFile('/content/models/research/fine_tuned_model/frozen_inference_graph.pb', 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
    
    ############
    #Load the label file
    #############
    label_map = label_map_util.load_labelmap('/content/Object_Detection-with-custom_data/data/annotations/label_map.pbtxt')
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=num_classes, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)
    tf_od_pred()


