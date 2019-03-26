from styx_msgs.msg import TrafficLight
import tensorflow as tf
import numpy as np
import datetime

class TLClassifier(object):
    def __init__(self, is_site):
        # load classifier
        if is_site:
            PATH_TO_GRAPH = r'models/site/ssd_mobilenet_frozen_inference_graph.pb'
        else:
            PATH_TO_GRAPH = r'models/simulator/ssd_inception_frozen_inference_graph.pb'

        self.graph = tf.Graph()
        self.threshold = .5

        with self.graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(PATH_TO_GRAPH, 'rb') as fid:
                od_graph_def.ParseFromString(fid.read())
                tf.import_graph_def(od_graph_def, name='')

            self.image_tensor = self.graph.get_tensor_by_name('image_tensor:0')
            self.boxes = self.graph.get_tensor_by_name('detection_boxes:0')
            self.scores = self.graph.get_tensor_by_name('detection_scores:0')
            self.classes = self.graph.get_tensor_by_name('detection_classes:0')
            self.num_detections = self.graph.get_tensor_by_name(
                'num_detections:0')

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
        #config.log_device_placement = True  # to log device placement (on which device the operation ran)
                                            # (nothing gets printed in Jupyter, only if you run it standalone)
        self.sess = tf.Session(graph=self.graph, config=config)

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        time = 0
        with self.graph.as_default():            
            img_expand = np.expand_dims(image, axis=0)
            start = datetime.datetime.now()
            (boxes, scores, classes, num_detections) = self.sess.run(
                [self.boxes, self.scores, self.classes, self.num_detections],
                feed_dict={self.image_tensor: img_expand})
            end = datetime.datetime.now()
            time = end - start

        boxes = np.squeeze(boxes)
        scores = np.squeeze(scores)
        classes = np.squeeze(classes).astype(np.int32)

        #print('SCORES: ', scores[0])
        #print('CLASSES: ', classes[0])

        if scores[0] > self.threshold:
            if classes[0] == 1:
                print('Traffic Light: *** GREEN ***, Detection Speed: ', time.total_seconds())
                return TrafficLight.GREEN
            elif classes[0] == 2:
                print('Traffic Light: *** RED ***, Detection Speed: ', time.total_seconds())
                return TrafficLight.RED
            elif classes[0] == 3:
                print('Traffic Light: *** YELLOW ***, Detection Speed: ', time.total_seconds())
                return TrafficLight.YELLOW

        return TrafficLight.UNKNOWN