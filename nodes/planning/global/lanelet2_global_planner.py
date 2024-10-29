#!/usr/bin/env python3

import rospy
import lanelet2

from lanelet2.io import Origin, load
from lanelet2.projection import UtmProjector
from lanelet2.core import BasicPoint2d
from lanelet2.geometry import findNearest

from shapely.geometry import Point, LineString

from geometry_msgs.msg import PoseStamped
from autoware_msgs.msg import Lane, Waypoint

from math import sqrt

class Lanelet2GlobalPlanner:
    def __init__(self) -> None:
        self.current_location = None
        self.goal_point = None
        self.reached_goal = False

        self.map_name = rospy.get_param("~lanelet2_map_name")
        self.speed_limit = rospy.get_param("~speed_limit", default=40.0)
        self.coordinate_transformer = rospy.get_param("/localization/coordinate_transformer")
        self.use_custom_origin = rospy.get_param("/localization/use_custom_origin")
        self.utm_origin_lat = rospy.get_param("/localization/utm_origin_lat")
        self.utm_origin_lon = rospy.get_param("/localization/utm_origin_lon")
        self.output_frame = rospy.get_param("/planning/lanelet2_global_planner/output_frame")
        self.distance_to_goal_limit = rospy.get_param("/planning/lanelet2_global_planner/distance_to_goal_limit")

        if self.coordinate_transformer == "utm":
            projector = UtmProjector(Origin(self.utm_origin_lat, self.utm_origin_lon), self.use_custom_origin, False)
        else:
            raise ValueError('Unknown coordinate_transformer for loading the Lanelet2 map ("utm" should be used): ' + self.coordinate_transformer)

        self.lanelet2_map = load(self.map_name, projector)

        # traffic rules
        traffic_rules = lanelet2.traffic_rules.create(lanelet2.traffic_rules.Locations.Germany,
                                                      lanelet2.traffic_rules.Participants.VehicleTaxi)
        # routing graph
        self.graph = lanelet2.routing.RoutingGraph(self.lanelet2_map, traffic_rules)

        self.waypoints_pub = rospy.Publisher('global_path', Lane, queue_size=1, latch=True)

        rospy.Subscriber("/move_base_simple/goal", PoseStamped, self.goal_point_callback)
        rospy.Subscriber("/localization/current_pose", PoseStamped, self.current_pose_callback) 

    def compute_route(self):
        # get start and end lanelets
        start_lanelet = findNearest(self.lanelet2_map.laneletLayer, self.current_location, 1)[0][1]
        goal_lanelet = findNearest(self.lanelet2_map.laneletLayer, self.goal_point, 1)[0][1]
        
        try:
            # find routing graph
            route = self.graph.getRoute(start_lanelet, goal_lanelet, 0, True)
            # find shortest path
            path = route.shortestPath()

            path_no_lane_change = path.getRemainingLane(start_lanelet)
            if not path_no_lane_change:
                print("No path w/o lane change")
                return
            
            projected_goal, goal_distance = self.goal_projection(path_no_lane_change[-1].centerline, self.goal_point)

            waypoints = self.lanelet_2_waypoint(path_no_lane_change, projected_goal, goal_distance)
            self.publish_global(waypoints)

        except:
            rospy.logwarn("No route found")

    def goal_projection(self, lanelet, goal_point):
        coordinates = [(point.x, point.y) for point in lanelet]
        line = LineString(coordinates)

        goal = Point(goal_point.x, goal_point.y)

        projected_dist = line.project(goal)
        projected_point = line.interpolate(projected_dist)
        distance = line.project(projected_point)

        return BasicPoint2d(projected_point.x, projected_point.y), distance 

    def goal_point_callback(self, msg):
        # loginfo message about receiving the goal point
        rospy.loginfo("%s - goal position (%f, %f, %f) orientation (%f, %f, %f, %f) in %s frame", rospy.get_name(),
                       msg.pose.position.x, msg.pose.position.y, msg.pose.position.z,
                       msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z,
                       msg.pose.orientation.w, msg.header.frame_id)
        self.goal_point = BasicPoint2d(msg.pose.position.x, msg.pose.position.y)
        self.reached_goal = False
        if self.current_location is not None:
            self.compute_route()

    def current_pose_callback(self, msg):
        self.current_location = BasicPoint2d(msg.pose.position.x, msg.pose.position.y)

        if self.goal_point is not None:
            distance_to_goal = sqrt(pow(self.current_location.x - self.goal_point.x, 2) +
                                    pow(self.current_location.y - self.goal_point.y, 2))
            
            if distance_to_goal < self.distance_to_goal_limit:
                if not self.reached_goal:
                    rospy.loginfo("Reached goal, stopping and clearing path!")
                    self.clear_path(msg)
                    self.reached_goal = True

                return


    def clear_path(self, msg):
        lane = Lane()
        lane.header.frame_id = self.output_frame
        lane.header.stamp = msg.header.stamp
        lane.waypoints = []

        self.waypoints_pub.publish(lane)

    def lanelet_2_waypoint(self, lanelet_path, projected_goal, goal_distance):
        waypoints = []

        for lanelet in lanelet_path:
            if "speed_ref" in lanelet.attributes:
                speed = float(lanelet.attributes['speed_ref']) * 1000/3600
            else:
                speed = self.speed_limit * 1000/3600

            line = LineString([(point.x, point.y) for point in lanelet.centerline])
            for i, point in enumerate(lanelet.centerline):
                if i == 0 and len(waypoints) > 0:
                    continue
                
                distance_travelled = line.project(Point(point.x, point.y))

                if distance_travelled >= goal_distance:
                    rospy.loginfo("Reached goal point")
                    break

                waypoint = Waypoint()
                waypoint.pose.pose.position.x = point.x
                waypoint.pose.pose.position.y = point.y
                waypoint.pose.pose.position.z = point.z
                waypoint.twist.twist.linear.x = speed

                waypoints.append(waypoint)

        if projected_goal is not None:
            # Append last goal waypoint
            waypoint = Waypoint()
            waypoint.pose.pose.position.x = projected_goal.x
            waypoint.pose.pose.position.y = projected_goal.y
            waypoint.pose.pose.position.z = 0.0

            waypoints.append(waypoint)

        return waypoints       

    def publish_global(self, waypoints):
        lane = Lane()        
        lane.header.frame_id = self.output_frame
        lane.header.stamp = rospy.Time.now()
        lane.waypoints = waypoints
        self.waypoints_pub.publish(lane)

    def run(self):
        rospy.spin()

if __name__ == '__main__':
    rospy.init_node('lanelet2_global_planner')
    node = Lanelet2GlobalPlanner()
    node.run()