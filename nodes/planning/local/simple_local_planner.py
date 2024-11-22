#!/usr/bin/env python3

import rospy
import math
import threading
from tf2_ros import Buffer, TransformListener, TransformException
import numpy as np
from autoware_msgs.msg import Lane, DetectedObjectArray, Waypoint
from geometry_msgs.msg import PoseStamped, TwistStamped, Vector3, Vector3Stamped
from shapely.geometry import LineString, Point, Polygon
from shapely import prepare
from tf2_geometry_msgs import do_transform_vector3
from scipy.interpolate import interp1d
from numpy.lib.recfunctions import unstructured_to_structured

from lanelet2.io import Origin, load
from lanelet2.projection import UtmProjector
from autoware_msgs.msg import TrafficLightResultArray

class SimpleLocalPlanner:

    def __init__(self):

        # Parameters
        self.output_frame = rospy.get_param("~output_frame")
        self.local_path_length = rospy.get_param("~local_path_length")
        self.transform_timeout = rospy.get_param("~transform_timeout")
        self.braking_safety_distance_obstacle = rospy.get_param("~braking_safety_distance_obstacle")
        self.braking_safety_distance_goal = rospy.get_param("~braking_safety_distance_goal")
        self.braking_safety_distance_stopline = rospy.get_param("~braking_safety_distance_stopline",
                                                                self.braking_safety_distance_obstacle)
        self.braking_safety_distance_goal = rospy.get_param("~braking_safety_distance_goal")
        self.braking_reaction_time = rospy.get_param("braking_reaction_time")
        self.stopping_lateral_distance = rospy.get_param("stopping_lateral_distance")
        self.current_pose_to_car_front = rospy.get_param("current_pose_to_car_front")
        self.default_deceleration = rospy.get_param("default_deceleration")
        self.tfl_maximum_deceleration = rospy.get_param("/planning/simple_local_planner/tfl_maximum_deceleration")
        coordinate_transformer = rospy.get_param("/localization/coordinate_transformer")
        use_custom_origin = rospy.get_param("/localization/use_custom_origin")
        utm_origin_lat = rospy.get_param("/localization/utm_origin_lat")
        utm_origin_lon = rospy.get_param("/localization/utm_origin_lon")
        lanelet2_map_name = rospy.get_param("~lanelet2_map_name")

        # Variables
        self.lock = threading.Lock()
        self.global_path_linestring = None
        self.global_path_distances = None
        self.distance_to_velocity_interpolator = None
        self.current_speed = None
        self.current_position = None
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer)

        if coordinate_transformer == "utm":
            projector = UtmProjector(Origin(utm_origin_lat, utm_origin_lon), use_custom_origin, False)
        else:
            raise RuntimeError('Only "utm" is supported for lanelet2 map loading')

        lanelet2_map = load(lanelet2_map_name, projector)

        # Stop lines
        self.stoplines = self.get_stoplines(lanelet2_map)
        self.red_lines = set()

        # Publishers
        self.local_path_pub = rospy.Publisher('local_path', Lane, queue_size=1, tcp_nodelay=True)

        # Subscribers
        rospy.Subscriber('global_path', Lane, self.path_callback, queue_size=None, tcp_nodelay=True)
        rospy.Subscriber('/localization/current_pose', PoseStamped, self.current_pose_callback, queue_size=1, tcp_nodelay=True)
        rospy.Subscriber('/localization/current_velocity', TwistStamped, self.current_velocity_callback, queue_size=1, tcp_nodelay=True)
        rospy.Subscriber('/detection/final_objects', DetectedObjectArray, self.detected_objects_callback, queue_size=1, buff_size=2**20, tcp_nodelay=True)
        rospy.Subscriber('/detection/traffic_light_status', TrafficLightResultArray, self.traffic_light_status_callback, queue_size=1)

    def path_callback(self, msg):

        if len(msg.waypoints) == 0:
            global_path_linestring = None
            global_path_distances = None
            distance_to_velocity_interpolator = None
            rospy.loginfo("%s - Empty global path received", rospy.get_name())

        else:
            waypoints_xyz = np.array([(w.pose.pose.position.x, w.pose.pose.position.y, w.pose.pose.position.z) for w in msg.waypoints])
            # convert waypoints to shapely linestring
            global_path_linestring = LineString(waypoints_xyz)
            prepare(global_path_linestring)

            # calculate distances between points, use only xy, and insert 0 at start of distances array
            global_path_distances = np.cumsum(np.sqrt(np.sum(np.diff(waypoints_xyz[:,:2], axis=0)**2, axis=1)))
            global_path_distances = np.insert(global_path_distances, 0, 0)

            # extract velocity values at waypoints
            velocities = np.array([w.twist.twist.linear.x for w in msg.waypoints])
            # create interpolator
            distance_to_velocity_interpolator = interp1d(global_path_distances, velocities, kind='linear', bounds_error=False, fill_value=0.0)

            rospy.loginfo("%s - Global path received with %i waypoints", rospy.get_name(), len(msg.waypoints))

        with self.lock:
            self.global_path_linestring = global_path_linestring
            self.global_path_distances = global_path_distances
            self.distance_to_velocity_interpolator = distance_to_velocity_interpolator

    def current_velocity_callback(self, msg):
        # save current velocity
        self.current_speed = msg.twist.linear.x

    def current_pose_callback(self, msg):
        # save current pose
        current_position = Point([msg.pose.position.x, msg.pose.position.y])
        self.current_position = current_position

    def detected_objects_callback(self, msg):
        with self.lock:
            global_path_linestring = self.global_path_linestring
            global_path_distances = self.global_path_distances
            distance_to_velocity_interpolator = self.distance_to_velocity_interpolator
            current_position = self.current_position
            local_path_length = self.local_path_length
            red_lines = self.red_lines
            stoplines = self.stoplines


        if global_path_linestring is None or global_path_distances is None or distance_to_velocity_interpolator is None:
            self.publish_local_path_wp([], msg.header.stamp, self.output_frame, 0.0, 0.0, False, 0.0)
            return
        
        d_ego_from_path_start = global_path_linestring.project(current_position)

        local_path = self.extract_local_path(global_path_linestring, global_path_distances, d_ego_from_path_start, local_path_length)
        if local_path is None:
            rospy.loginfo(f"{rospy.get_name()} Reached goal.")
            self.publish_local_path_wp([], msg.header.stamp, self.output_frame, 0.0, 0.0, False, 0.0)
            return
        
        map_based_velocity = float(distance_to_velocity_interpolator(d_ego_from_path_start))
        local_path_to_wp = self.convert_local_path_to_waypoints(local_path, map_based_velocity)
        local_path_buffer = local_path.buffer(self.stopping_lateral_distance, cap_style="flat")
        prepare(local_path_buffer)

        closest_obj_d = float('inf')
        closest_obj_velocity = 0.0
        local_path_blocked = False
        object_distances = []
        object_velocities = []
        adjust_stopping_distances = []
        target_distances = []
        object_braking_distances = []

        goal_pos = Point(global_path_linestring.coords[-1])
        d_goal_from_path_start = global_path_linestring.project(goal_pos)
        distance_to_goal = d_goal_from_path_start - d_ego_from_path_start

        if distance_to_goal > 0 and distance_to_goal <= local_path_length:
            object_distances.append(distance_to_goal)
            object_velocities.append(0.0)
            adjust_stopping_distances.append(distance_to_goal - self.braking_safety_distance_goal)
            object_braking_distances.append(self.braking_safety_distance_goal)
            target_distances.append(distance_to_goal - (self.current_pose_to_car_front + object_braking_distances[-1]))

        stopping_point_distance = float('inf')
        for line_id in red_lines:
            if line_id in stoplines:
                stopline = stoplines[line_id]
                if stopline.intersects(local_path_buffer):
                    intersect_points = local_path_buffer.intersection(stopline)
                    stopline_distance = min([global_path_linestring.project(Point(intersect_point))
                                             for intersect_point in intersect_points.coords]) - d_ego_from_path_start
                    if stopline_distance > 0:
                        stopping_distance = max(0, stopline_distance - self.braking_safety_distance_stopline)
                        current_speed = self.current_speed
                        if stopping_distance > 0 and (current_speed**2)/(2*stopping_distance) > self.tfl_maximum_deceleration:
                            if (rospy.Time.now() - self.last_warn_time).to_sec() >= 3.0:
                                rospy.loginfo("Ignoring red traffic light")
                                self.last_warn_time = rospy.Time.now()
                            
                        else:
                            rospy.loginfo(f"{rospy.get_name()} - Red stopline detected at distance: {stopline_distance}")
                            object_distances.append(stopline_distance)
                            object_velocities.append(0)
                            adjust_stopping_distances.append(stopping_distance)

                            object_braking_distances.append(self.braking_safety_distance_obstacle) 
                            target_distance = stopline_distance - (self.current_pose_to_car_front + object_braking_distances[-1])
                            target_distances.append(target_distance)

        for obj in msg.objects:
            obj_polygon = Polygon([(p.x, p.y) for p in obj.convex_hull.polygon.points])

            if local_path_buffer.intersects(obj_polygon):
                obj_position = Point(obj.pose.position.x, obj.pose.position.y)
                d_obj_from_path_start = global_path_linestring.project(obj_position)
                d_to_object = d_obj_from_path_start - d_ego_from_path_start

                try:
                    transform = self.tf_buffer.lookup_transform(self.output_frame, msg.header.frame_id,
                                                                rospy.Time(), rospy.Duration(self.transform_timeout))
                except TransformException as e:
                    rospy.logwarn(f"{rospy.get_name()} - Transform lookup failed: {e}")
                    transform = None

                if transform is not None:
                    vector3_stamped = Vector3Stamped(vector=obj.velocity.linear)
                    velocity = do_transform_vector3(vector3_stamped, transform).vector
                else:
                    velocity = Vector3()
                
                obj_velocity = velocity.x

                adjust_stopping_d = d_to_object - self.braking_safety_distance_obstacle

                if adjust_stopping_d > 0:
                    object_distances.append(d_to_object)
                    object_velocities.append(obj_velocity)
                    adjust_stopping_distances.append(adjust_stopping_d)

                    object_braking_distances.append(self.braking_safety_distance_obstacle)
                    reaction_distance = self.braking_reaction_time * abs(obj_velocity)
                    target_distance = d_to_object - (self.current_pose_to_car_front + reaction_distance + object_braking_distances[-1])
                    target_distances.append(target_distance)

        if len(target_distances) > 0:
            target_velocities = np.sqrt(np.maximum(0.0, 2 * self.default_deceleration *
                np.array(target_distances) + np.array(object_velocities)**2))
            min_index = np.argmin(target_velocities)
            closest_obj_d = object_distances[min_index]
            closest_obj_velocity = object_velocities[min_index]
            stopping_point_distance = adjust_stopping_distances[min_index]

            target_velocity = min(target_velocities[min_index], map_based_velocity)
            local_path_blocked = True if (closest_obj_velocity != 0) else False

        else:
            closest_obj_d = 0.0
            closest_obj_velocity = 0.0
            stopping_point_distance = 0.0
            local_path_blocked = False
            target_velocity = map_based_velocity


        local_path_to_wp = self.convert_local_path_to_waypoints(local_path, target_velocity)
        self.publish_local_path_wp(local_path_to_wp, msg.header.stamp, self.output_frame, closest_obj_d,
                                   closest_obj_velocity, local_path_blocked, stopping_point_distance)


    def extract_local_path(self, global_path_linestring, global_path_distances, d_ego_from_path_start, local_path_length):

        # current position is projected at the end of the global path - goal reached
        if math.isclose(d_ego_from_path_start, global_path_linestring.length):
            return None

        d_to_local_path_end = d_ego_from_path_start + local_path_length

        # find index where distances are higher than ego_d_on_global_path
        index_start = np.argmax(global_path_distances >= d_ego_from_path_start)
        index_end = np.argmax(global_path_distances >= d_to_local_path_end)

        # if end point of local_path is past the end of the global path (returns 0) then take index of last point
        if index_end == 0:
            index_end = len(global_path_linestring.coords) - 1

        # create local path from global path add interpolated points at start and end, use sliced point coordinates in between
        start_point = global_path_linestring.interpolate(d_ego_from_path_start)
        end_point = global_path_linestring.interpolate(d_to_local_path_end)
        local_path = LineString([start_point] + list(global_path_linestring.coords[index_start:index_end]) + [end_point])

        return local_path


    def convert_local_path_to_waypoints(self, local_path, target_velocity):
        # convert local path to waypoints
        local_path_waypoints = []
        for point in local_path.coords:
            waypoint = Waypoint()
            waypoint.pose.pose.position.x = point[0]
            waypoint.pose.pose.position.y = point[1]
            waypoint.pose.pose.position.z = point[2]
            waypoint.twist.twist.linear.x = target_velocity
            local_path_waypoints.append(waypoint)
        return local_path_waypoints


    def publish_local_path_wp(self, local_path_waypoints, stamp, output_frame, closest_object_distance=0.0, closest_object_velocity=0.0, local_path_blocked=False, stopping_point_distance=0.0):
        # create lane message
        lane = Lane()
        lane.header.frame_id = output_frame
        lane.header.stamp = stamp
        lane.waypoints = local_path_waypoints
        lane.closest_object_distance = closest_object_distance
        lane.closest_object_velocity = closest_object_velocity
        lane.is_blocked = local_path_blocked
        lane.cost = stopping_point_distance
        self.local_path_pub.publish(lane)

    def get_stoplines(self, lanelet2_map):
        stoplines = {}
        for line in lanelet2_map.lineStringLayer:
            if line.attributes:
                if line.attributes["type"] == "stop_line":
                    stoplines[line.id] = LineString([(p.x, p.y) for p in line])

        return stoplines

    def traffic_light_status_callback(self, msg):
        self.red_lines = set()
        for result in msg.results:
            if result.recognition_result_str == "RED cam":
                self.red_lines.add(result.lane_id)

    def run(self):
        rospy.spin()

if __name__ == '__main__':
    rospy.init_node('simple_local_planner')
    node = SimpleLocalPlanner()
    node.run()