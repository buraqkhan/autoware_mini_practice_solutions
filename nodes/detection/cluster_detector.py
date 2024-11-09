#!/usr/bin/env python3

import rospy
import numpy as np

from shapely import MultiPoint
from tf2_ros import TransformListener, Buffer, TransformException
from numpy.lib.recfunctions import structured_to_unstructured
from ros_numpy import numpify, msgify

from sensor_msgs.msg import PointCloud2
from autoware_msgs.msg import DetectedObjectArray, DetectedObject
from std_msgs.msg import ColorRGBA, Header
from geometry_msgs.msg import Point32


BLUE80P = ColorRGBA(0.0, 0.0, 1.0, 0.8)

class ClusterDetector:
    def __init__(self):
        self.min_cluster_size = rospy.get_param('~min_cluster_size')
        self.output_frame = rospy.get_param('/detection/output_frame')
        self.transform_timeout = rospy.get_param('~transform_timeout')

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer)

        self.objects_pub = rospy.Publisher('detected_objects', DetectedObjectArray, queue_size=1, tcp_nodelay=True)
        rospy.Subscriber('points_clustered', PointCloud2, self.cluster_callback, queue_size=1, buff_size=2**24, tcp_nodelay=True)

        rospy.loginfo("%s - initialized", rospy.get_name())


    def cluster_callback(self, msg):
        data = numpify(msg)
        points = structured_to_unstructured(data[['x', 'y', 'z']], dtype=np.float32)

        if msg.header.frame_id != self.output_frame:
            try:
                transform = self.tf_buffer.lookup_transform(self.output_frame, msg.header.frame_id, 
                                                            msg.header.stamp, rospy.Duration(self.transform_timeout))
                tf_matrix = numpify(transform.transform).astype(np.float32).T
                points = np.concatenate((points, np.ones((points.shape[0],1))), axis=1)
                transformed_points = points @ tf_matrix

            except (TransformException, rospy.ROSTimeMovedBackwardsException) as e:
                rospy.logwarn("%s - %s", rospy.get_name(), e)
                return
            
        labels = data['label']
        
        header = Header()
        header.stamp = msg.header.stamp
        header.frame_id = self.output_frame

        detected_objects = DetectedObjectArray()
        detected_objects.header = header

        unique_labels = np.unique(labels)

        # iterating over the labels
        for label in unique_labels:

            mask = (labels == label)
            points3d = transformed_points[mask, :3]

            if points3d.shape[0] < self.min_cluster_size:
                continue

            centroid = points3d.mean(axis = 0)

            points2d = MultiPoint(points3d[:, :2])
            hull = points2d.convex_hull

            convex_hull_points = [Point32(x,y,centroid[2]) for x, y in hull.exterior.coords]

            detected_object = DetectedObject()
            detected_object.header = header
            detected_object.id = label
            detected_object.label = "unknown"
            detected_object.color = BLUE80P
            detected_object.valid = True
            detected_object.pose_reliable = True
            detected_object.velocity_reliable = False
            detected_object.acceleration_reliable = False
            detected_object.space_frame = self.output_frame
            detected_object.pose.position.x = centroid[0]
            detected_object.pose.position.y = centroid[1]
            detected_object.pose.position.z = centroid[2]

            detected_object.convex_hull.polygon.points = convex_hull_points
            detected_objects.objects.append(detected_object)

        self.objects_pub.publish(detected_objects)


    def run(self):
        rospy.spin()

if __name__ == '__main__':
    rospy.init_node('cluster_detector', log_level=rospy.INFO)
    node = ClusterDetector()
    node.run()