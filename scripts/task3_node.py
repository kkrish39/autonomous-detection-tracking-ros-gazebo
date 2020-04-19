import rospy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseWithCovarianceStamped
from geometry_msgs.msg import PoseStamped
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

msg_to_track_evader = PoseStamped()
pub = rospy.Publisher('/tb3_0/move_base_simple/goal',PoseStamped, queue_size=10)
# m = Darknet(cfg_file)
def callback(data):
    cv_img = bridge.imgmsg_to_cv2(data, "bgr8")
    image = cv_img
    image = imutils.resize(image, width=min(400, image.shape[1]))
    orig = image.copy()
	# detect people in the image
    (rects, weights) = hog.detectMultiScale(image, winStride=(4, 4),
		padding=(8, 8), scale=1.05)
	# draw the original bounding boxes
    for (x, y, w, h) in rects:
		cv2.rectangle(orig, (x, y), (x + w, y + h), (0, 0, 255), 2)
	# apply non-maxima suppression to the bounding boxes using a
	# fairly large overlap threshold to try to maintain overlapping
	# boxes that are still people
    rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
    pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)
	# draw the final bounding boxes
    for (xA, yA, xB, yB) in pick:
        cv2.rectangle(image, (xA, yA), (xB, yB), (0, 255, 0), 2)
	# show some information on the number of bounding boxes
	# filename = imagePath[imagePath.rfind("/") + 1:]
    print("--",len(rects),len(pick))

#Callback when the position of the human changes
def pose_callback(data):
    print("#######")
    msg_to_track_evader.header = data.header
    msg_to_track_evader.pose= data.pose.pose
    pub.publish(msg_to_track_evader)


#server to handle client request to draw the initials
def draw_initials_server():
    rospy.init_node('human_detection_node')
    rospy.sleep(1)
    rospy.Subscriber("/tb3_0/camera/rgb/image_raw", Image, callback)
    rospy.Subscriber("/tb3_1/amcl_pose",PoseWithCovarianceStamped, pose_callback )
    rospy.spin()
    
if __name__ == '__main__':
    print('Server up and running to receive data...')
    draw_initials_server()
