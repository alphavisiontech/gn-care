import argparse
import cv2
import numpy as np
import os 
import time 
import subprocess
import logging
import sys
import threading
import queue
from collections import deque
from rknn.api import RKNN

from utils.cam_utils import open_cam, reset_usb_devices
from utils.rules_utils import iou, calculate_bbox_ratio
from utils.draw_utils import display_fall_detection_result, add_fps_info, add_performance_info, create_side_by_side_display

from ppyoloe.py_utils.coco_utils import COCO_test_helper
from ppyoloe.rknn_utils import setup_model, post_process, draw
from rtmpose.rknn_utils import process_multiple_people, visualize_keypoints, load_rknn_model

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='pose_estimation.log',
    filemode='a',
)

class Config:
    """Configuration constants"""
    PAD_COLOR = (0, 0, 0)
    MODEL_INPUT_SIZE = (192, 256)
    USB_RESET_DELAY = 2
    MODULE_RELOAD_DELAY = 2
    MAX_RETRY_COUNT = 10
    RECONNECT_DELAY = 5
    
    # Threading configuration
    QUEUE_SIZE = 8
    THREAD_TIMEOUT = 0.05
    THREAD_JOIN_TIMEOUT = 2.0

class NPUCoreConfig:
    """NPU Core assignment configuration for RK3588"""
    
    # Core assignments (most computationally intensive first)
    PPYOLOE_CORE = RKNN.NPU_CORE_0      # Object detection (heaviest)
    RTMPOSE_CORE = RKNN.NPU_CORE_1      # Pose estimation (medium)

    @staticmethod
    def get_core_mask(model_type):
        """Get appropriate core mask for each model"""
        core_map = {
            'ppyoloe': NPUCoreConfig.PPYOLOE_CORE,
            'rtmpose': NPUCoreConfig.RTMPOSE_CORE, 
        }
        return core_map.get(model_type, RKNN.NPU_CORE_0)

class ThreadedFrameProcessor:
    """Multi-threaded frame processing for improved performance"""
    
    def __init__(self, det_model, pose_model, co_helper, args):
        self.det_model = det_model
        self.pose_model = pose_model
        self.co_helper = co_helper
        self.args = args
        self.det_img_shape = [int(s) for s in args.image_shape.split(',')]
        self.ratio_threshold = args.ratio_threshold  # Threshold for bbox width/height ratio to indicate lying down
        self.human_det_threshold = args.score_threshold  # Threshold for human detection confidence
        self.keypoint_threshold = args.keypoint_threshold  # Threshold for keypoint confidence
        self.backward_ratio_threshold = args.backward_ratio_threshold  # Threshold for backward fall check
        self.consecutive_falls = 0
        
        # Thread-safe queues
        self.input_queue = queue.Queue(maxsize=Config.QUEUE_SIZE)
        self.detection_queue = queue.Queue(maxsize=Config.QUEUE_SIZE)
        self.pose_queue = queue.Queue(maxsize=Config.QUEUE_SIZE)
        self.result_queue = queue.Queue(maxsize=Config.QUEUE_SIZE)
        
        # Thread control
        self.stop_event = threading.Event() # default value: False 
        self.threads = []
        
        # Start worker threads
        self.start_threads()
    
    def start_threads(self):
        """Start all worker threads"""
        # Detection thread
        detection_thread = threading.Thread(target=self._detection_worker, daemon=True)
        detection_thread.start()
        self.threads.append(detection_thread)
        
        # Pose estimation thread
        pose_thread = threading.Thread(target=self._pose_worker, daemon=True)
        pose_thread.start()
        self.threads.append(pose_thread)

        fall_thread = threading.Thread(target=self._fall_detection_worker, daemon=True)
        fall_thread.start()
        self.threads.append(fall_thread)
    
    def _detection_worker(self):
        """Worker thread for object detection"""
        while not self.stop_event.is_set(): # keep running while stop event is NOT set (while not False)
            try:
                frame_data = self.input_queue.get(timeout=Config.THREAD_TIMEOUT)
                if frame_data is None:
                    break
                
                frame, frame_count, verbose = frame_data
                
                # Preprocess frame
                input_data = self._preprocess_frame(frame)
                
                # Run detection
                outputs = self.det_model.run([input_data])
                boxes, classes, scores = post_process(outputs, self.human_det_threshold)
                '''
                print(f"boxes:{boxes}, type:{type(boxes)}")
                print(f"scores:{scores}, type:{type(scores)}")
                '''

                person_bboxes = []
                person_classes = []
                person_scores = []

                bed_bboxes = []
                bed_scores = []
                
                if boxes is not None and len(boxes) > 0:
                    for box, cls, score in zip(boxes, classes, scores):
                        if cls == 0:  # Person class
                            person_bboxes.append(box)
                            person_classes.append(cls)
                            person_scores.append(score)
                        elif cls == 59:  # Bed class
                            bed_bboxes.append(box)
                            bed_scores.append(score)
                    print(f"Person boxes: {len(person_bboxes)}, Bed boxes: {len(bed_bboxes)}")
                else:
                    print(f"No detections in frame {frame_count}")

                assert len(person_bboxes) == len(person_scores), "Mismatch in person boxes and scores length"
                assert len(bed_bboxes) == len(bed_scores), "Mismatch in bed boxes and scores length"
                
                person_bboxes = np.array(person_bboxes)
                person_classes = np.array(person_classes)
                person_scores = np.array(person_scores)
                bed_bboxes = np.array(bed_bboxes)
                bed_scores = np.array(bed_scores)
                
                '''
                print(f"person_boxes:{person_bboxes}, type:{type(person_bboxes)}")
                print(f"person_scores:{person_scores}, type:{type(person_scores)}")
                print(f"bed_boxes:{bed_bboxes}, type:{type(bed_bboxes)}")
                print(f"bed_scores:{bed_scores}, type:{type(bed_scores)}")
                '''

                # Pass to pose estimation
                self.detection_queue.put((frame, person_bboxes, person_classes, person_scores, bed_bboxes, bed_scores, frame_count, verbose))
                self.input_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Detection worker error: {e}")
                continue
    
    def _pose_worker(self):
        """Worker thread for pose estimation"""
        while not self.stop_event.is_set():
            try:
                detection_data = self.detection_queue.get(timeout=Config.THREAD_TIMEOUT)
                if detection_data is None:
                    break

                frame, person_bboxes, person_classes, person_scores, bed_bboxes, bed_scores, frame_count, verbose = detection_data

                # Run pose estimation
                keypoints_list = []
                kpts_scores_list = []

                if person_bboxes is not None and len(person_bboxes) > 0:
                    real_bboxes = self.co_helper.get_real_box(person_bboxes)
                    keypoints_list, kpts_scores_list = process_multiple_people(
                        frame, self.pose_model, real_bboxes, Config.MODEL_INPUT_SIZE
                    )
                    person_bboxes = real_bboxes
                
                # Pass to fall detection
                self.pose_queue.put((frame, keypoints_list, kpts_scores_list, person_bboxes, person_classes, person_scores, bed_bboxes, bed_scores, frame_count, verbose))
                self.detection_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Pose worker error: {e}")
                continue
    
    def _fall_detection_worker(self):
        """Worker thread for fall detection and visualization"""
        while not self.stop_event.is_set():
            try:
                pose_data = self.pose_queue.get(timeout=Config.THREAD_TIMEOUT)
                if pose_data is None:
                    break

                frame, keypoints_list, kpts_scores_list, person_bboxes, person_classes, person_scores, bed_bboxes, bed_scores, frame_count, _ = pose_data

                # Initialize default fall result
                fall_result = {
                    'predicted_class': "No person detected",
                    'human_det_confidence': 0.0,
                    'is_fall': False,
                    'status': 'no_person/no_detection'
                }
                
                ratio_list, person_scores = calculate_bbox_ratio(
                    person_bboxes, person_scores, keypoints_list, kpts_scores_list, self.backward_ratio_threshold, self.keypoint_threshold)
                
                assert len(ratio_list) == len(person_scores), "Mismatch in ratio and scores length"
                
                in_bed_status_list = [self.get_in_bed_status([p_bbox], bed_bboxes, bed_scores) for p_bbox in person_bboxes]
            
                reason = "ratio"
                for det_idx, (ratio, score) in enumerate(zip(ratio_list, person_scores)):
                    print(f"Detection {det_idx}: BBox ratio = {ratio:.3f}, Score = {score:.3f}")
                    in_bed_status = in_bed_status_list[det_idx] if det_idx < len(in_bed_status_list) else False

                    print(f"Detection {det_idx}: ratio: {ratio}, in bed status = {in_bed_status}")

                    if ratio > self.ratio_threshold and score >= self.human_det_threshold and not in_bed_status:
                        if ratio == 5:
                            reason = "forward"
                        elif ratio == 6:
                            reason = "backward"
                        print("Person detected with bbox ratio indicating fall")
                        print("Ratio: ", ratio)
                        fall_result = {
                            'predicted_class': "Fall",
                            'human_det_confidence': score,
                            'is_fall': True,
                            'status': f'fall_detected_{reason}',
                            'fall_det_index': int(det_idx) 
                        }
                    if in_bed_status and score >= self.human_det_threshold:
                        print("Person detected in bed, no fall")
                        print("Ratio: ", ratio)
                        fall_result = {
                            'predicted_class': "No Fall",
                            'human_det_confidence': score,
                            'is_fall': False,
                            'status': 'person_in_bed',
                            'fall_det_index': None,
                        }
                    else: 
                        fall_result = { 
                            'predicted_class': "No Fall",
                            'confidence': score,
                            'is_fall': False,
                            'status': 'no_fall_detected',
                            'fall_det_index': None,
                        }    

                # Create visualization frames
                left_frame = frame.copy()
                right_frame = np.zeros_like(frame)
                
                # Visualize keypoints on both frames
                if keypoints_list is not None and len(keypoints_list) > 0:
                    left_frame = draw(left_frame, person_bboxes, person_scores, person_classes)
                    left_frame = visualize_keypoints(
                        left_frame, keypoints_list, kpts_scores_list, thr=self.args.keypoint_threshold
                    )
                    right_frame = draw(right_frame, person_bboxes, person_scores, person_classes)
                    right_frame = visualize_keypoints(
                        right_frame, keypoints_list, kpts_scores_list, thr=self.args.keypoint_threshold
                    )
                        
                # Add fall detection overlay
                left_frame, self.consecutive_falls = display_fall_detection_result(
                    left_frame, fall_result,
                    self.consecutive_falls, self.args.fall_consecutive_threshold, frame_count
                )
                
                right_frame, _ = display_fall_detection_result(
                    right_frame, fall_result, 
                    self.consecutive_falls, self.args.fall_consecutive_threshold, frame_count
                )
                
                # Put result in output queue
                self.result_queue.put((left_frame, right_frame, fall_result, frame_count))
                self.pose_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Fall detection worker error: {e}")
                continue
    
    def _preprocess_frame(self, frame):
        """Preprocess frame for detection"""
        img = self.co_helper.letter_box(
            im=frame.copy(),
            new_shape=(self.det_img_shape[0], self.det_img_shape[1]),
            pad_color=Config.PAD_COLOR
        )
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        input_data = img.transpose((2, 0, 1))
        input_data = input_data.reshape(1, *input_data.shape).astype(np.float32)
        return input_data
    
    def process_frame_async(self, frame, frame_count=None, verbose=False):
        """Submit frame for asynchronous processing"""
        try:
            self.input_queue.put((frame, frame_count, verbose), timeout=Config.THREAD_TIMEOUT)
            return True
        except queue.Full:
            print("Warning: Processing queue is full, dropping frame")
            return False
        
    def get_in_bed_status(self, person_bboxes, bed_bboxes, bed_scores, iou_threshold=0.7):
        """Determine if person is in bed based on IoU with bed bounding boxes"""
        if not person_bboxes or not bed_bboxes:
            return False
        
        bed_bboxes = [b_box for b_box, b_conf in zip(bed_bboxes, bed_scores) if b_conf >= self.human_det_threshold]
        
        iou_value = iou(person_bboxes, bed_bboxes)
        return iou_value >= iou_threshold
    
    def get_result(self, timeout=None):
        """Get processed result from output queue"""
        try:
            return self.result_queue.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def stop(self):
        """Stop all worker threads"""
        self.stop_event.set() # stop_event = True -> stopping all the threads 

        # Clear queues
        while not self.input_queue.empty():
            try:
                self.input_queue.get_nowait()
            except queue.Empty:
                break
        
        while not self.detection_queue.empty():
            try:
                self.detection_queue.get_nowait()
            except queue.Empty:
                break
        
        while not self.pose_queue.empty():
            try:
                self.pose_queue.get_nowait()
            except queue.Empty:
                break
        
        # Wait for threads to finish
        for thread in self.threads:
            thread.join(timeout=Config.THREAD_JOIN_TIMEOUT)
            if thread.is_alive():
                print(f"Warning: Thread {thread.name} did not terminate in time")
    
    def get_queue_sizes(self):
        """Get current queue sizes for monitoring"""
        return {
            'input': self.input_queue.qsize(),
            'detection': self.detection_queue.qsize(),
            'pose': self.pose_queue.qsize(),
            'result': self.result_queue.qsize()
        }

def validate_model_files(args):
    """Validate that all required model files exist"""
    models = [
        (args.ppyoloe_model, "PP-YOLOE"),
        (args.rtmpose_model, "RTMPose"),
    ]
    
    for model_path, model_name in models:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"{model_name} model not found: {model_path}")

def validate_input_source(args):
    """Validate input source based on input type"""
    if args.input_type == 'image' and not args.source:
        raise ValueError("--source is required for image input")
    elif args.input_type == 'video' and not args.source:
        raise ValueError("--source is required for video input")
    elif args.input_type == 'webcam':
        args.source = int(args.source) if args.source else 0
    
    # Check if source file exists for image/video
    if args.input_type in ['image', 'video'] and not os.path.exists(args.source):
        raise FileNotFoundError(f"Source file not found: {args.source}")

def initialize_models(args):
    """Initialize all models and return them"""
    # Initialize detection model
    det_model, platform = setup_model(args.ppyoloe_model, core_mask=NPUCoreConfig.get_core_mask('ppyoloe'))
    co_helper = COCO_test_helper(enable_letter_box=True)
    print(f"PP-YOLOE model loaded from {args.ppyoloe_model}")
    
    # Initialize pose estimation model
    pose_model = load_rknn_model(args.rtmpose_model, core_mask=NPUCoreConfig.get_core_mask('rtmpose'))
    print(f"RTMPose model loaded from {args.rtmpose_model}")
    
    return det_model, pose_model, co_helper


def process_video_stream_threaded(args, det_model, pose_model, co_helper, cap, window_name, current_usb_path, is_webcam=False):
    """Process video stream using multi-threading for improved performance"""
    # Create threaded processor
    threaded_processor = ThreadedFrameProcessor(det_model, pose_model, co_helper, args)
    
    fps_time = time.time()
    frame_count = 0
    
    # Performance metrics
    processing_times = deque(maxlen=30)
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                if is_webcam:
                    print("No frame. Resetting USB and reconnecting...")
                    logging.warning("No frame. Resetting USB and reconnecting...")
                    cap.release()
                    reset_usb_devices(current_usb_path)
                    cap, current_video_dev, current_usb_path = open_cam()
                    continue
                else:
                    print("End of video stream")
                    break

            frame_count += 1
            start_time = time.time()
            
            # Submit frame for processing
            if not threaded_processor.process_frame_async(frame, frame_count, verbose=True):
                print(f"Frame {frame_count} dropped due to queue overflow")
                continue
            
            # Try to get a result (non-blocking)
            result = threaded_processor.get_result(timeout=0.001)
            if result is not None:
                left_frame, right_frame, fall_result, result_frame_count = result
                confidence = fall_result.get('human_det_confidence', 0.0)
                
                # Log results
                if fall_result:
                    print(f"Frame {result_frame_count} - Fall detection: {fall_result['predicted_class']} "
                          f"(confidence: {confidence:.3f})")
                    if fall_result.get('transition_detected'):
                        print(f"Frame {result_frame_count} - Transition: {fall_result['transition_detected']}")
                    print(f"Frame {result_frame_count} - Consecutive falls: {threaded_processor.consecutive_falls}")
                print()  # Empty line for readability

                # Calculate and display FPS
                current_time = time.time()
                fps_value = 1.0 / (current_time - fps_time)
                fps_time = current_time
                
                # Track processing time
                processing_time = current_time - start_time
                processing_times.append(processing_time)
                
                # Get queue sizes for monitoring
                queue_sizes = threaded_processor.get_queue_sizes()
                
                # Add performance info to the right frame
                right_frame = add_fps_info(right_frame, fps_value, result_frame_count)
                # right_frame = add_performance_info(right_frame, processing_times, queue_sizes)
                
                # Create side-by-side display
                combined_frame = create_side_by_side_display(left_frame, right_frame)
                
                # Display the combined frame
                cv2.imshow(window_name, combined_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    finally:
        # Clean up
        threaded_processor.stop()
        cap.release()
        cv2.destroyAllWindows()
    
    return cap if is_webcam else None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Multi-person pose estimation with PP-YOLOE + RTMPose + CTRGCN')
    parser.add_argument('--ppyoloe-model', type=str, default='./ppyoloe/models/rknn/ppyoloe_s.rknn',
                        help='Path to PP-YOLOE RKNN model')
    parser.add_argument('--rtmpose-model', type=str, default='./rtmpose/models/rknn/rtmpose-m.rknn',
                        help='Path to RTMPose RKNN model')
    parser.add_argument('--input-type', type=str, choices=['image', 'video', 'webcam'], required=True,
                        help='Input type: image, video, or webcam')
    parser.add_argument('--source', type=str, default=None,
                        help='Input source: image path, video path, or webcam index (0, 1, etc.)')
    parser.add_argument('--enable-threading', action='store_true',
	                    help='Enable threading mode')
    
    # Fall detection parameters
    parser.add_argument('--score-threshold', type=float, default=0.6,	
                        help='Detection confidence threshold (default: 0.6)')
    parser.add_argument('--keypoint-threshold', type=float, default=0.3,
                        help='Pose keypoint confidence threshold (default: 0.3)')
    parser.add_argument('--fall-consecutive-threshold', type=int, default=5,
                        help='Number of detected falls before alert is raised (default: 5)')
    parser.add_argument('--ratio-threshold', type=float, default=1.2,
                        help='Bounding box width/height ratio threshold to indicate lying down (default: 1.2)')
    parser.add_argument('--backward-ratio-threshold', type=float, default=0.91,
                        help='Upper body to lower body ratio threshold for backward fall check (default: 0.91)')

    # Others 
    parser.add_argument('--image-shape', type=str, default='640,640',
                        help='Input image shape for the model (default: 640,640)')
    parser.add_argument('--output-dir', type=str, default='.',
                        help='Output directory for results (default: current directory)')
    

    args = parser.parse_args()
    
    try:
        # Validate arguments and model files
        validate_input_source(args)
        validate_model_files(args)
        
        print(f"Detection image shape: {[int(s) for s in args.image_shape.split(',')]}")
        print(f"Threading enabled: {args.enable_threading}")
        
        # Initialize models
        det_model, pose_model, co_helper = initialize_models(args)
        
        # Process based on input type
        if args.input_type == 'video':
            cap = cv2.VideoCapture(args.source)
            if not cap.isOpened():
                raise ValueError(f"Could not open video file: {args.source}")
            
            window_name = 'PP-YOLOE + RTMPose + CTRGCN Fall Detection - Video (Threaded)'
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            process_video_stream_threaded(args, det_model, pose_model, co_helper, cap, window_name, is_webcam=False)

            
        else:  # webcam
            window_name = 'PP-YOLOE + RTMPose + CTRGCN Fall Detection - Webcam (Threaded)'
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cap, current_video_dev, current_usb_path = open_cam()
            process_video_stream_threaded(args, det_model, pose_model, co_helper, cap, window_name, current_usb_path, is_webcam=True)

    except Exception as e:
        print(f"Error: {e}")
        logging.error(f"Application error: {e}")
        sys.exit(1)

