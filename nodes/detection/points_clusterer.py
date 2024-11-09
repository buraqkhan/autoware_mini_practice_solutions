#!/usr/bin/env python3

import rospy
import numpy as np

from numpy.lib.recfunctions import structured_to_unstructured, unstructured_to_structured
from ros_numpy import numpify, msgify
from sklearn.cluster import DBSCAN

from sensor_msgs.point_cloud2 import PointCloud2

class PointsClusterer:
    def __init__(self):
        self.cluster_epsilon = rospy.get_param('~cluster_epsilon')
        self.cluster_min_size = rospy.get_param('~cluster_min_size')

        self.clusterer = DBSCAN(self.cluster_epsilon, min_samples=self.cluster_min_size)

        self.clustered_publisher = rospy.Publisher('points_clustered', PointCloud2, queue_size=1, tcp_nodelay=True)

        rospy.Subscriber('points_filtered', PointCloud2, self.points_callback, queue_size=1, buff_size=2**24, tcp_nodelay=True)

    def points_callback(self, msg):
        data = numpify(msg)
        points = structured_to_unstructured(data[['x', 'y', 'z']], dtype=np.float32)
        labels = self.clusterer.fit_predict(points)

        assert points.shape[0] == labels.shape[0], f"{rospy.get_name()} Mismatch in # of labels and # of points."

        labeled_points = np.concatenate((points, labels[:, np.newaxis]), axis=1)
        remove_labels = np.argwhere(labeled_points[:, 3] == -1).tolist()
        labeled_points = np.delete(labeled_points, (remove_labels), axis=0)

        # convert labelled points to PointCloud2 format
        clusters = unstructured_to_structured(labeled_points, dtype=np.dtype([
            ('x', np.float32),
            ('y', np.float32),
            ('z', np.float32),
            ('label', np.int32)
        ]))

        # publish clustered points message
        cluster_msg = msgify(PointCloud2, clusters)
        cluster_msg.header.stamp = msg.header.stamp
        cluster_msg.header.frame_id = msg.header.frame_id
        self.clustered_publisher.publish(cluster_msg)

    def run(self):
        rospy.spin()

if __name__ == '__main__':
    rospy.init_node('points_clusterer')
    node = PointsClusterer()
    node.run()