import rospy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseWithCovarianceStamped
from geometry_msgs.msg import PoseStamped
from move_base_msgs.msg import MoveBaseActionResult
from cv_bridge import CvBridge, CvBridgeError

from imutils.object_detection import non_max_suppression
from imutils import paths
import numpy as np
import imutils
import cv2



#Object to convert the ROS Image to 
bridge = CvBridge()

hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

#Message to publish to the pursuer when a human is detected
msg_to_track_evader = PoseStamped()

#Puruser node to track the evader. Used to send goal poses
pub = rospy.Publisher('/tb3_0/move_base_simple/goal',PoseStamped, queue_size=10)

class ImageDetectionAndTracking:
    def callback_for_every_pose(self,data):
        # converting the ROS image into CV format
        cv_img = bridge.imgmsg_to_cv2(data, "bgr8")
        image = cv_img
        image = imutils.resize(image, width=min(400, image.shape[1]))
        orig = image.copy()

        # Detecting human in the image
        (rects, weights) = hog.detectMultiScale(image, winStride=(4, 4),
            padding=(8, 8), scale=1.05)

        for (x, y, w, h) in rects:
            cv2.rectangle(orig, (x, y), (x + w, y + h), (0, 0, 255), 2)

        rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
        pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)

        # draw the final bounding boxes
        for (xA, yA, xB, yB) in pick:
            cv2.rectangle(image, (xA, yA), (xB, yB), (0, 255, 0), 2)
        
        if(len(rects) > 0):
            print("Human detected...")
            #Publishing the orientation data to the pursuer

            #Keeping track of last 10 locations where the evader was detected
            if(len(self.humanPositionArrayX) < 10):
                self.humanPositionArrayX.append(self.msg_detected_human_pose.pose.pose.position.x)
                self.humanPositionArrayY.append(self.msg_detected_human_pose.pose.pose.position.y)
                
                # For the first instance alone initiating the pursuer.This is based on assumption that a human will be visible at first.
                # We need to handle the case, If no human detected in the inital phase.
                if(len(self.humanPositionArrayX) == 1):
                    msg_to_track_evader.header = self.msg_detected_human_pose.header
                    msg_to_track_evader.pose= self.msg_detected_human_pose.pose.pose
                    pub.publish(msg_to_track_evader)  
        else:
            print("Unable to track human. Searching....")


    #Callback when the position of the human changes
    def evader_pose_callback(self,data):
        self.msg_detected_human_pose = data

    #pursuer callback
    def purseur_pose_callback(self,data):
        if data.status.status == 3:
            msg_to_track_evader.header.stamp = rospy.Time.now()
            msg_to_track_evader.pose.position.x = self.humanPositionArrayX.pop(0)
            msg_to_track_evader.pose.position.y = self.humanPositionArrayY.pop(0)
            pub.publish(msg_to_track_evader)
            print("Shifting to the next location...", len(self.humanPositionArrayX))
        else:
            print("Still Pursuing the given goal")


    #server to handle client request to draw the initials
    def __init__(self):
        rospy.init_node('human_detection_node')
        rospy.sleep(1)
        self.humanPositionArrayX=[]
        self.humanPositionArrayY=[]
        self.currentGoal=0
        rospy.Subscriber("/tb3_0/camera/rgb/image_raw", Image, self.callback_for_every_pose, queue_size=10)
        rospy.Subscriber("/tb3_1/amcl_pose", PoseWithCovarianceStamped, self.evader_pose_callback, queue_size=10)
        rospy.Subscriber("/tb3_0/move_base/result",MoveBaseActionResult, self.purseur_pose_callback, queue_size=10)
        rospy.spin()
    
if __name__ == '__main__':
    print('Server up and running to receive data...')
    detectionClass =  ImageDetectionAndTracking()
