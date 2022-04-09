#!/usr/bin/env python

from cv2 import pointPolygonTest
from Queue import PriorityQueue, Queue
from scipy.spatial.transform import Rotation as R
# from black import Priority
import rospy
import numpy as np
from geometry_msgs.msg import PoseStamped, PoseArray, PointStamped
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
        self.odom_topic = "/odom"
        self.odom_sub = rospy.Subscriber(self.odom_topic, Odometry, self.odom_cb)

        self.clicked_point_sub = rospy.Subscriber("/clicked_point", PointStamped, self.cp_cb)

        # TODO: maybe make one Map object?
        self.map = None
        self.map_resolution = None
        self.real_world_origin_position = None
        self.real_world_origin_orientation = None
        self.map_set = False

        self.current_point = None

    def cp_cb(self, msg):
        print("world coordinate:", msg.point.x, msg.point.y)
        pixel_coord = self.real_world_to_pixel(msg.point.x, msg.point.y)
        print("pixel coordinate:", pixel_coord)
        print("occupied:", self.map[pixel_coord[1], pixel_coord[0]])
        return 

    def map_cb(self, msg):
        map_nrows, map_ncols = msg.info.height, msg.info.width

        self.map = np.zeros((map_nrows, map_ncols))

        for r in range(map_nrows):
            for c in range(map_ncols):
                self.map[r,c] = msg.data[(map_ncols*r) + c]


        map_occupied = np.where(self.map != 0, 1, 0).astype('uint8') #self.map[(self.map != 0)]
        # plt.imshow(255 - map_occupied*255, cmap='gray') #viz check
        # plt.show()

        # dilate_kernel = np.ones((5,5), 'uint8')

        # map_occupied_dilated = cv2.dilate(map_occupied, dilate_kernel, iterations=1)
        # self.map = map_occupied_dilated #(0 if empty, 1 otherwise)
        self.map = map_occupied

        # plt.imshow(255 - map_occupied_dilated*255, cmap='gray') #viz check
        # plt.show()

        #TODO: possible variables of interest?
        self.map_resolution = msg.info.resolution
        self.real_world_origin_position = msg.info.origin.position
        self.real_world_origin_orientaion = msg.info.origin.orientation

        # self.map_pose_of_real_world_origin = msg.info.origin #TODO: Pose object, may get type error
        self.map_set = True

    def odom_cb(self, msg):
        #pass ## REMOVE AND FILL IN ##
        # print("ODOM CALLBACK")
        self.current_point = (msg.pose.pose.position.x, msg.pose.pose.position.y)
        
    def goal_cb(self, msg):
        #pass ## REMOVE AND FILL IN ##
        start_point = self.current_point
        end_point = (msg.pose.position.x, msg.pose.position.y)

        if self.map_set:
            self.plan_path(start_point, end_point, self.map)

    def plan_path(self, start_point, end_point, map):
        ## CODE FOR PATH PLANNING ##
        path_points = self.A_star(start_point, end_point, map)
        print("path_points:", len(path_points))
        for x,y in path_points:
            self.trajectory.addPoint(x,y)
        
        # publish trajectory
        self.traj_pub.publish(self.trajectory.toPoseArray())

        # visualize trajectory Markers
        self.trajectory.publish_viz()

    
    # #TODO: helper methods for A* algorithm; may move inside main method or static if only one-use
    def distance(self, first_point, second_point):
        return sqrt((first_point[0]-second_point[0])**2 + (first_point[1]-second_point[1])**2)

    def heuristic(self, end_point, current_point):
        # return self.distance(end_point, current_point) #using regular distance
        return abs(end_point[0]-current_point[0]) + abs(end_point[1]-current_point[1]) #using Manhanttan distance
    def get_neighbors(self, current):
        # print("current occupied?:", self.map[current[0], current[1]])
        neighbors = []
        map_nrows, map_ncols = self.map.shape
        for (dx, dy) in [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0,1), (1,-1), (1,0), (1,1)]:
            next_x, next_y = current[0]+dx, current[1]+dy
            # print("next_x, next_y:", next_x, next_y)
            # print("map_nrows:", map_nrows, "map_ncols:", map_ncols)
            # print("self.map[next_x, next_y]:", self.map[next_x, next_y])
            if (0 <= current[0]+dx < map_nrows and 0 <= current[1]+dy < map_ncols and self.map[current[0]+dx, current[1]+dy] == 0):
                neighbors.append((next_x, next_y))
        return neighbors

    def A_star(self, start_point, end_point, map):
        #code based on: https://www.redblobgames.com/pathfinding/a-star/introduction.html
        print("start_point:", start_point, "end_point:", end_point)
        start_point_y, start_point_x = self.real_world_to_pixel(start_point[0], start_point[1])
        end_point_y, end_point_x = self.real_world_to_pixel(end_point[0], end_point[1])
        
        start_point = (start_point_x, start_point_y)
        end_point = (end_point_x, end_point_y)

        frontier = Queue()
        frontier.put(start_point)
        came_from = dict()
        came_from[start_point] = None

        while not frontier.empty():
            current = frontier.get()

            if current == end_point:
                break

            for next in self.get_neighbors(current):
                if next not in came_from:
                    frontier.put(next)
                    came_from[next] = current


        ### DIJKSTRAAA
        # frontier = PriorityQueue()
        # frontier.put(start_point, 0)
        # came_from = dict()
        # cost_so_far = dict()
        # came_from[start_point] = None
        # cost_so_far[start_point] = 0
        # print("frontier:", frontier)

        # print("pixel start_point:", start_point, "pixel end_point:", end_point)

        # while not frontier.empty():
        #     print("frontier size:", frontier.qsize())
        #     current = frontier.get()
        #     print("current:", current)

        #     if current == end_point:
        #         print("BREAK")
        #         break

        #     # print("neighbors:", self.get_neighbors(current))

        #     for next in self.get_neighbors(current):
        #         new_cost = cost_so_far[current] + self.distance(current, next)
        #         #TODO: maybe + self.cost(current, next)?
        #         # implement cost method to account for realistic steering?
        #         # for now, just using abs distance
        #         # print("next:", next)
        #         if next not in cost_so_far or new_cost < cost_so_far[next]: #unvisited or visited, but found more optimal path
        #             cost_so_far[next] = new_cost
        #             priority = new_cost + self.heuristic(end_point, next)
        #             frontier.put(next, priority)
        #             came_from[next] = current

        # (493, 978), (592, 982)

        # retrace path
        current = end_point
        path = []
        while current != start_point:
            path.append(current)
            current = came_from[current]
        path.append(start_point)
        path.reverse()

        print("path:", path)
        # path = []
        return path
    

    # def pixel_to_real_world(self, u, v, map_resolution, origin_pose):
    #     pixel_position = np.array([[u, v, 1]]) #(3x1)

    #     q = origin_pose.orientation
    #     rotation = R.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
    #     return np.dot(pixel_position.T * map_resolution, rotation)

    def real_world_to_pixel(self, x, y): #, map_resolution, origin_pose):
        real_world_position = np.array([[x, y, 1]]) #(3x1)
        q = [self.real_world_origin_orientaion.x, self.real_world_origin_orientaion.y, self.real_world_origin_orientaion.z, self.real_world_origin_orientaion.w]
        
        pixel_to_world_rot = R.from_quat(q).as_dcm()
        pixel_to_world_rot[0, 2] = self.real_world_origin_position.x
        pixel_to_world_rot[1, 2] = self.real_world_origin_position.y

        #rotation = np.linalg.inv(pixel_to_world_rot)
        rotation = pixel_to_world_rot
        pixel_position = np.dot(np.linalg.inv(rotation), real_world_position.T).reshape((3,))[0:2]
        pixel_position = tuple((pixel_position/self.map_resolution).astype(int))
        print("pixel_position:", pixel_position)

        return pixel_position

if __name__=="__main__":
    rospy.init_node("path_planning")
    pf = PathPlan()
    rospy.spin()
