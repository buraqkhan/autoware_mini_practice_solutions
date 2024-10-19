#!/usr/bin/env python3

import rospy
import numpy as np

from autoware_msgs.msg import Lane, VehicleCmd
from geometry_msgs.msg import PoseStamped
from shapely.geometry import LineString, Point
from shapely import prepare, distance
from tf.transformations import euler_from_quaternion
from scipy.interpolate import interp1d

class PurePursuitFollower:
    def __init__(self):

        # Parameters
        # Reading in the parameter values
        self.lookahead_distance = rospy.get_param("~lookahead_distance")
        self.wheel_base = rospy.get_param("/vehicle/wheel_base")
        self.path = None
        self.distance_to_velocity_interpolator = None

        # Publishers
        self.vehicle_cmd_pub = rospy.Publisher("/control/vehicle_cmd", VehicleCmd, queue_size=1)

        # Subscribers
        rospy.Subscriber('path', Lane, self.path_callback, queue_size=1)
        rospy.Subscriber('/localization/current_pose', PoseStamped, self.current_pose_callback, queue_size=1)

    def path_callback(self, msg):
        if msg.waypoints == None or len(msg.waypoints) < 2:
            return
        # convert waypoints to shapely linestring
        self.path = LineString([(w.pose.pose.position.x, w.pose.pose.position.y) for w in msg.waypoints])
        # prepare path - creates spatial tree, making the spatial queries more efficient
        prepare(self.path)

        # Create a distance to velocity interpolator for the path
        # collect waypoint x and y coordinates
        waypoints_xy = np.array([(w.pose.pose.position.x, w.pose.pose.position.y) for w in msg.waypoints])
        distances = np.cumsum(np.sqrt(np.sum(np.diff(waypoints_xy, axis=0)**2, axis=1)))
        distances = np.insert(distances, 0, 0) # add 0 distance at the start

        velocities = np.array([w.twist.twist.linear.x for w in msg.waypoints])
        self.distance_to_velocity_interpolator = interp1d(distances, velocities, kind='linear', bounds_error=False, fill_value=0.0)

    def current_pose_callback(self, msg):
        if self.path == None or self.distance_to_velocity_interpolator == None:
            return
        
        current_pose = Point([msg.pose.position.x, msg.pose.position.y])
        d_ego_from_path_start = self.path.project(current_pose)
        lookahead_point = self.path.interpolate(d_ego_from_path_start + self.lookahead_distance)

        # lookahead point heading calculation
        lookahead_heading = np.arctan2(lookahead_point.y - current_pose.y, lookahead_point.x - current_pose.x)

        _, _, heading = euler_from_quaternion([msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w])

        self.lookahead_distance = current_pose.distance(lookahead_point)

        alpha = lookahead_heading - heading
        steering_angle = (2 * self.wheel_base * np.sin(alpha)) / self.lookahead_distance

        velocity = self.distance_to_velocity_interpolator(d_ego_from_path_start)

        vehicle_cmd = VehicleCmd()
        vehicle_cmd.ctrl_cmd.steering_angle = steering_angle
        vehicle_cmd.ctrl_cmd.linear_velocity = velocity
        vehicle_cmd.header.stamp = msg.header.stamp
        vehicle_cmd.header.frame_id = "base_link"
        self.vehicle_cmd_pub.publish(vehicle_cmd)
        

    def run(self):
        rospy.spin()

if __name__ == '__main__':
    rospy.init_node('pure_pursuit_follower')
    node = PurePursuitFollower()
    node.run()