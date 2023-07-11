import socket
import cv2
import pickle
import struct
import argparse
from detect_on_stream import detect

# create socket
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
host_ip = '172.24.1.55'  # paste your server IP address here
port = 9999
client_socket.connect((host_ip, port))  # a tuple
data = b""
payload_size = struct.calcsize("Q")

# Parse command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--weights', type=str, default='yolov7.pt', help='path to weights file')
parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
parser.add_argument('--view-img', action='store_true', help='display results')
parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
parser.add_argument('--augment', action='store_true', help='augmented inference')
parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
parser.add_argument('--update', action='store_true', help='update all models')
parser.add_argument('--project', default='runs/detect', help='save results to project/name')
parser.add_argument('--name', default='exp', help='save results to project/name')
parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
parser.add_argument('--no-trace', action='store_true', help='don`t trace model')

opt = parser.parse_args()

frames = []  # List to store received frames

while True:
    while len(data) < payload_size:
        packet = client_socket.recv(4 * 1024)  # 4K
        if not packet:
            break
        data += packet
    packed_msg_size = data[:payload_size]
    data = data[payload_size:]
    msg_size = struct.unpack("Q", packed_msg_size)[0]

    while len(data) < msg_size:
        data += client_socket.recv(4 * 1024)
    frame_data = data[:msg_size]
    data = data[msg_size:]
    frame = pickle.loads(frame_data)
    

    # Perform object detection on the frames
    pred_frame = detect(weights=opt.weights, frame=frame)

    # Display frame using OpenCV
    if pred_frame is not None:
        cv2.imshow("img", pred_frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

client_socket.close()
cv2.destroyAllWindows()
