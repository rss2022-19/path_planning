#!/usr/bin/env python

from Queue import PriorityQueue
# from black import Priority
import rospy
import numpy as np
from geometry_msgs.msg import PoseStamped, PoseArray
from nav_msgs.msg import Odometry, OccupancyGrid
import rospkg
import time, os
from utils import LineTrajectory
from math import sqrt
import cv2
import matplotlib.pyplot as plt

class PathPlan(object):
    """ Listens for goal pose published by RViz and uses it to plan a path from
    current car pose. 
    """
    def __init__(self):
        self.odom_topic = rospy.get_param("~odom_topic")
        self.map_sub = rospy.Subscriber("/map", OccupancyGrid, self.map_cb)
        self.trajectory = LineTrajectory("/planned_trajectory")
        self.goal_sub = rospy.Subscriber("/move_base_simple/goal", PoseStamped, self.goal_cb, queue_size=10)
        self.traj_pub = rospy.Publisher("/trajectory/current", PoseArray, queue_size=10)
        self.odom_sub = rospy.Subscriber(self.odom_topic, Odometry, self.odom_cb)

        # TODO: maybe make one Map object?
        self.map = None
        self.map_resolution = None
        self.map_pose_of_real_world_origin = None
        self.map_set = False

    def map_cb(self, msg):
        map_nrows, map_ncols = msg.info.height, msg.info.width

        self.map = np.zeros((map_nrows, map_ncols))


        for r in range(map_nrows):
            for c in range(map_ncols):
                self.map[r,c] = msg.data[(map_ncols*r) + c]
            # print("msg.data:", msg.data)
            # row_data = msg.data[map_nrows*r : (map_nrows*r) + map_ncols]
            # print("row_data:", row_data.shape)
            # print("self.map:", self.map.shape)
            # print("self.map[r, :]:", self.map[r, :].shape)
            # self.map[r, :] = np.array(row_data) #(map_ncols, )

        map_occupied = np.where(self.map != 0, 1, 0).astype('uint8') #self.map[(self.map != 0)]
        # plt.imshow(255 - map_occupied*255, cmap='gray') #viz check
        # plt.show()

        dilate_kernel = np.ones((10,10), 'uint8')

        map_occupied_dilated = cv2.dilate(map_occupied, dilate_kernel, iterations=1)
        self.map = map_occupied_dilated #(0 if empty, 1 otherwise)
        
        # plt.imshow(255 - map_occupied_dilated*255, cmap='gray') #viz check
        # plt.show()

        #TODO: possible variables of interest?
        self.map_resolution = msg.info.resolution
        # self.map_pose_of_real_world_origin = msg.info.origin #TODO: Pose object, may get type error
        self.map_set = True

    def odom_cb(self, msg):
        pass ## REMOVE AND FILL IN ##


    def goal_cb(self, msg):
        pass ## REMOVE AND FILL IN ##

    def plan_path(self, start_point, end_point, map):
        ## CODE FOR PATH PLANNING ##

        path = self.A_star(start_point, end_point, map)
        
        # publish trajectory
        self.traj_pub.publish(self.trajectory.toPoseArray())

        # visualize trajectory Markers
        self.trajectory.publish_viz()

    
    # #TODO: helper methods for A* algorithm; may move inside main method or static if only one-use
    def distance(self, first_point, second_point):
        return sqrt((first_point[0]-second_point[0])**2 + (first_point[1]-second_point[1])**2)

    def heuristic(end_point, current_point):
        # return self.distance(end_point, current_point) #using regular distance
        return abs(end_point[0]-current_point[0]) + abs(end_point[1]-current_point[1]) #using Manhanttan distance

    def A_star(self, start_point, end_point, map):
        #code based on: https://www.redblobgames.com/pathfinding/a-star/introduction.html
        frontier = PriorityQueue()
        frontier.put(start_point, 0)
        came_from = dict()
        cost_so_far = dict()
        came_from[start_point] = None
        cost_so_far[start_point] = 0


        while not frontier.empty():
            current = frontier.get()

            if current == end_point:
                break

            for next in map.neighbors(current):
                new_cost = cost_so_far[current] + self.distance(current, next)
                #TODO: maybe + self.cost(current, next)?
                # implement cost method to account for realistic steering?
                # for now, just using abs distance
                if next not in cost_so_far or new_cost < cost_so_far[next]: #unvisited or visited, but found more optimal path
                    cost_so_far[next] = new_cost
                    priority = new_cost + self.heuristic(end_point, next)
                    frontier.put(next, priority)
                    came_from[next] = current

        # retrace path
        current = end_point
        path = []
        while current != start_point:
            path.append(current)
            current = came_from[current]
        path.append(start_point)
        path.reverse()

    #     return path
    

    # def pixel_to_real_world(self, u, v, map_resolution, origin_pose):
    #     pixel_position = np.array([[u, v, 1]]) #(3x1)

    #     q = origin_pose.orientation
    #     rotation = R.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
    #     return np.dot(pixel_position.T * map_resolution, rotation)

    # def real_world(self, x, y, map_resolution, origin_pose):
    #     real_world_position = np.array([[x, y, 1]]) #(3x1)
    #     q = origin_pose.orientation
    #     rotation = R.from_quat([q.x, q.y, q.z, q.w]).inv().as_matrix()
    #     return np.dot(real_world_position.T * map_resolution, rotation)
                


if __name__=="__main__":
    rospy.init_node("path_planning")
    pf = PathPlan()
    rospy.spin()
