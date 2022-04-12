#!/usr/bin/env python

from tokenize import String
import rospy
import numpy as np
import time
import utils
import tf

from geometry_msgs.msg import PoseArray, PoseStamped
from visualization_msgs.msg import Marker
from ackermann_msgs.msg import AckermannDriveStamped
from nav_msgs.msg import Odometry
from scipy.spatial.transform import Rotation as R
from visualization_tools import *


class PurePursuit(object):
    """ Implements Pure Pursuit trajectory tracking with a fixed lookahead and speed.
    """
    def __init__(self):
        self.lookahead        = rospy.get_param("lookahead", 0.8)
        self.speed            = rospy.get_param("speed", 1.0)
        self.base_link_offset = rospy.get_param("base_link_offset", 0.0)
        self.wheelbase_length = 0.568
        
        self.trajectory  = utils.LineTrajectory("/followed_trajectory")
        self.traj_sub    = rospy.Subscriber("/trajectory/current", PoseArray, self.trajectory_callback, queue_size=1)
        # self.traj_sub = rospy.Subscriber("/loaded_trajectory/path", PoseArray, self.trajectory_callback, queue_size=1) # for simulation i tink...
        self.robot_pose  = None
        self.odom_topic  = rospy.get_param("~odom_topic")
        self.odom_sub    = rospy.Subscriber(self.odom_topic, Odometry, self.pose_callback, queue_size = 1)
        self.drive_pub   = rospy.Publisher("/drive", AckermannDriveStamped, queue_size=1)

        self.goal_pub = rospy.Publisher("/goalpos", Marker, queue_size=1)
        
        #How many segments to search ahead of the nearest
        #self.segment_lookahead = 3
        self.segments = None
        self.segment_index = None
        self.goal_point = None
        
        self.kp_velocity = rospy.get_param("kp_velocity", 0.8)
        
    def trajectory_callback(self, msg):
        ''' Clears the currently followed trajectory, and loads the new one from the message
        '''
        print("Receiving new trajectory:", len(msg.poses), "points")
        self.trajectory.clear()
        self.trajectory.fromPoseArray(msg)
        self.trajectory.publish_viz(duration=0.0)
        self.segments = np.array(self.trajectory.points)
        
    def find_circle_interesection(self):
        #
        # Find Circle Intersection
        # Start at nearest segment and search next three segments
        #
        #search_end_index = self.segment_index+self.segment_lookahead
        #if search_end_index >  self.segments.shape[0]-1:
            #search_end_index = self.segments.shape[0]-1
        search_end_index = self.segments.shape[0]-1
        
        goal_point = None
        
        #Check nearest segment as well as a couple segments up. Choose the furthest goal point
        for i in range(self.segment_index,search_end_index):
            P1 = self.segments[self.segment_index]
            P2 = self.segments[self.segment_index+1]
            Q = self.robot_pose[:2]
            r = self.lookahead
            V = P2-P1

            a = V.dot(V)
            b = 2.*V.dot(P1-Q)
            c = P1.dot(P1)+Q.dot(Q)-2.*P1.dot(Q)-r**2.

            disc = b**2.-4.*a*c
            
            if disc < 0.:
                #No intersection at all
                pass
            else:
                #Two solutions
                sqrt_disc = np.sqrt(disc)
                t1 = (-b + sqrt_disc) / (2. * a)
                t2 = (-b - sqrt_disc) / (2. * a)
               
                #Choose the solution that is actually on the line segmnet (0 <= t <= 1)
                if ((t1 < 0. or t1 > 1.) and (t2 < 0. or t2 > 1.)):
                    #No intersection on segment
                    pass
                elif ((t1 >= 0. or t1 <= 1.) and (t2 < 0. or t2 > 1.)):
                    goal_point = P1+t1*V
                elif ((t1 < 0. or t1 > 1.) and (t2 >= 0. or t2 <= 1.)):
                    goal_point = P1+t2*V
                #Otherwise choose the one closest to the next point
                else:
                    intersects = [P1+t1*V,P1+t2*V]
                    
                    dist_P2_intersects = np.array([np.linalg.norm(P2-intersects[0]),np.linalg.norm(P2-intersects[1])])
                    goal_point = intersects[np.argmin(dist_P2_intersects)]
        
        #Return intersection. (None if no intersections)
        return goal_point
    
    def find_nearest_segment(self):
        #
        # Find nearest line segment
        #
        num_segments = self.segments.shape[0]
        robot_point = np.tile(self.robot_pose[:2],(num_segments-1,1))
        
        segment_start = self.segments[:-1,:]
        segment_end = self.segments[1:,:]
        
        #Vectorize
        robot_vector = robot_point-segment_start
        segment_vector = segment_end-segment_start
        
        #Normalize and Scale
        segment_length = np.linalg.norm(segment_vector, axis=1)
        segment_length = np.reshape(segment_length, (num_segments-1,1))
        
        unit_segment = np.divide(segment_vector,segment_length)
        scaled_robot_point = np.divide(robot_vector,segment_length)
        
        #Project
        scaled_dot_product = np.sum(unit_segment*scaled_robot_point, axis=1)
        
        #Clip and Rescale
        scaled_dot_product = np.clip(scaled_dot_product,0.0,1.0)
        scaled_dot_product = np.reshape(scaled_dot_product, (num_segments-1,1))
        
        scaled_segment = segment_vector*scaled_dot_product
        
        #Offset
        shifted = np.add(scaled_segment,segment_start)
        
        #Find Distances
        shortest_distance = np.linalg.norm(shifted-robot_point, axis=1)
        
        #Grab relevant segment
        return(np.argmin(shortest_distance))
        
    def pose_callback(self, msg):
        ''' Clears the currently followed trajectory, and loads the new one from the message
        '''
        pose = [msg.pose.pose.position.x+self.base_link_offset, msg.pose.pose.position.y]
        r = R.from_quat([msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z, msg.pose.pose.orientation.w])
        theta = r.as_euler('XYZ')[2]
        pose.append(theta)
        self.robot_pose = np.array(pose)

        if self.segments is not None:
            self.segment_index = self.find_nearest_segment()
            self.goal_point = self.find_circle_interesection()

            if self.goal_point is None:
                self.goal_point = self.segments[self.segment_index + 1]
            VisualizationTools.plot_line([self.goal_point[0], self.goal_point[0] + .01], [self.goal_point[1], self.goal_point[1] + .01], self.goal_pub, frame="/map")
            
            end_point = self.segments[-1]
            dist_error = np.linalg.norm(end_point-self.robot_pose[0:2])
            velocity = self.kp_velocity*dist_error
            velocity_out = np.clip(velocity,-self.speed,self.speed)

        x_c, y_c, theta_c = self.robot_pose
        x_g, y_g = self.goal_point
        T_R_W = np.array([[ np.cos(theta_c), -np.sin(theta_c), x_c ],
                          [ np.sin(theta_c),  np.cos(theta_c), y_c ],
                          [ 0,                0,               1   ]])

        T_G_W = np.array([[ np.cos(0), -np.sin(0), x_g ],
                          [ np.sin(0),  np.cos(0), y_g ],
                          [ 0,          0,         1   ]])

        T_G_R = np.dot(np.linalg.inv(T_R_W), T_G_W)

        x = T_G_R[0][2]
        y = T_G_R[1][2]

        theta = np.arctan(y/x) # bearing
        d = np.sqrt(x**2 + y**2) # euclidean distance
        delta = np.arctan((2*self.wheelbase_length*np.sin(theta))/d) # steering angle 
        
        if self.goal_point is None:
            pass

        # Drive to goal point
        drive_msg = AckermannDriveStamped()
        drive_msg.header.stamp = rospy.get_rostime()
        drive_msg.header.frame_id = "base_link"
        drive_msg.drive.steering_angle = delta
        drive_msg.drive.steering_angle_velocity = 0
        if self.segments is None:
            drive_msg.drive.speed = 0
        else:
            drive_msg.drive.speed = velocity_out
        drive_msg.drive.acceleration = 0
        drive_msg.drive.jerk = 0
        self.drive_pub.publish(drive_msg)
        

if __name__=="__main__":
    rospy.init_node("pure_pursuit")
    pf = PurePursuit()
    rospy.spin()
