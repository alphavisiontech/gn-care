import argparse
import cv2
import numpy as np
import torch
import time
from collections import deque
import sys
import os

# Add the current directory to the path to import local modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from rfdetr import RFDETRMedium
from rtmpose.onnx_utils import (
    build_session, visualize_keypoints, process_multiple_people
)
from ctrgcn.detector import FallDetector

class FallDetectionPipeline:
    """Complete fall detection pipeline: RF-DETR -> RTMPose -> CTRGCN"""
    
    def __init__(self, args):
        self.args = args
        
        # Initialize RTMPose
        self.rtmpose_session = build_session(args.rtmpose_model, args.device)
        print("RTMPose model successfully loaded")
        
        # Initialize CTRGCN Fall Detector
        self.fall_detector = FallDetector(
            model_path=args.ctrgcn_model,
            device=args.device,
            confidence_threshold=args.fall_detection_threshold,
            temporal_window=args.temporal_window,
            smoothing_window=args.smoothing_window,
            num_classes=args.num_classes
        )
        print("CTRGCN model successfully loaded")

        # Initialize RF-DETR for person detection
        self.predictor = RFDETRMedium()
        print("RF-DETR model successfully loaded")

        # Fall alert tracking
        self.fall_frame_count = 0
        self.consecutive_fall_threshold = args.fall_frames
        self.fall_alert_active = False
        self.fall_alert_start_time = None
        self.fall_alert_duration = 3.0  # Show alert for 3 seconds
        
        # FPS tracking
        self.fps_counter = deque(maxlen=30)
        self.frame_times = deque(maxlen=30)
        
    def detect_person_bbox(self, frame):
        """
        Simple person detection using background subtraction or full frame
        In a real implementation, this would use RF-DETR
        For now, we'll use the full frame as a single person bbox
        """

        detections = self.predictor.predict(frame, score_threshold=self.args.human_det_threshold, input_shape=self.args.det_img_shape)
        person_bboxes = []
        person_scores = []
        
        for index, (class_id, confidence) in enumerate(zip(detections.class_id, detections.confidence)):
            bbox = detections.xyxy[index]
            
            if class_id == 1:  # person
                person_bboxes.append(bbox)
                person_scores.append(confidence)
        
        if person_bboxes:
            print(f"Found {len(person_bboxes)} person detections")
        
        return person_bboxes, person_scores

    def run_rtmpose(self, frame, bboxes):
        """Run RTMPose inference on detected person bboxes"""
        if not bboxes:
            return [], [], []
        
        h, w = self.rtmpose_session.get_inputs()[0].shape[2:]
        model_input_size = (w, h)
            
        # Process multiple people (though we expect only one from person detection)
        keypoints_list, scores_list = process_multiple_people(
            img=frame,
            sess=self.rtmpose_session,
            bboxes=bboxes,
            input_size=model_input_size,
            use_batch_inference=self.args.use_batch_inference,
        )

        final_kpts_list = []
        for i, (keypoints, scores) in enumerate(zip(keypoints_list, scores_list)):
            if keypoints is not None and scores is not None:
                # Ensure keypoints is numpy array 
                if not isinstance(keypoints, np.ndarray):
                    keypoints = np.array(keypoints)
                if not isinstance(scores, np.ndarray):
                    scores = np.array(scores)
            
                final_kpts = np.concatenate([keypoints, scores.reshape(-1, 1)], axis=1) 
                final_kpts_list.append(final_kpts)
            
            else: 
                # If no valid keypoints, create zero array 
                final_kpts_list.append(np.zeros((17, 3)))
        
        return final_kpts_list, keypoints_list, scores_list
    
    def run_ctrgcn(self, keypoints, person_scores):
        """Run CTRGCN fall detection on pose keypoints"""
        if not keypoints or len(keypoints) == 0:
            return {
                'predicted_class': 0, # Default to standing
                'class_probabilities': [1.0] + [0.0] * (self.args.num_classes - 1),
                'confidence': 0.0,
                'is_fall': False,
                'status': 'no_pose'
            }
        
        # Select person with highest detection confidence
        if person_scores and len(person_scores) > 0:
            # Find index of person with highest confidence
            highest_conf_idx = np.argmax(person_scores)
            person_keypoints = keypoints[highest_conf_idx]
            selected_confidence = person_scores[highest_conf_idx]
            print(f"Selected person with idx {highest_conf_idx} with confidence: {selected_confidence:.3f}")
        else:
            # Fallback to first person if no scores available
            person_keypoints = keypoints[0]
            print("No detection scores available, using first person")
        
        if person_keypoints.shape != (17, 3):
            return {
                'predicted_class': 0,
                'class_probabilities': [1.0] + [0.0] * (self.args.num_classes - 1),
                'confidence': 0.0,
                'is_fall': False,
                'status': 'invalid_pose'
            }
        
        return self.fall_detector.predict(person_keypoints)
    
    def update_fall_alert(self, prediction_result):
        """Update fall alert status based on consecutive fall predictions"""
        current_time = time.time()
        
        if prediction_result['is_fall']:
            self.fall_frame_count += 1
        else:
            self.fall_frame_count = 0
        
        # Activate fall alert if consecutive frames exceed threshold
        if self.fall_frame_count >= self.consecutive_fall_threshold:
            if not self.fall_alert_active:
                self.fall_alert_active = True
                self.fall_alert_start_time = current_time
                print("FALL ALERT ACTIVATED! 🚨")
        
        # Deactivate alert after duration
        if (self.fall_alert_active and self.fall_alert_start_time and 
            current_time - self.fall_alert_start_time > self.fall_alert_duration):
            if self.fall_frame_count < self.consecutive_fall_threshold // 2:
                self.fall_alert_active = False
                self.fall_alert_start_time = None
    
    def draw_info_panel(self, frame, fps, prediction_result, position='top'):
        """Draw information panel on frame"""
        h, w = frame.shape[:2]
        panel_height = 120
        
        # Create semi-transparent overlay
        overlay = frame.copy()
        if position == 'top':
            cv2.rectangle(overlay, (0, 0), (w, panel_height), (0, 0, 0), -1)
        else:
            cv2.rectangle(overlay, (0, h-panel_height), (w, h), (0, 0, 0), -1)
        
        alpha = 0.7
        frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
        
        # Text parameters
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 1
        
        # Y positions for text
        if position == 'top':
            y_start = 25
        else:
            y_start = h - 95
        
        # FPS
        fps_text = f"FPS: {fps:.1f}"
        cv2.putText(frame, fps_text, (10, y_start), font, font_scale, (0, 255, 0), thickness)
        
        # Action prediction
        action_text = prediction_result.get('status', 'Unknown').upper()
        cv2.putText(frame, action_text, (10, y_start + 25), font, font_scale, (255, 255, 255), thickness)
        
        # Confidence
        confidence = prediction_result.get('confidence', 0.0)
        conf_text = f"Confidence: {confidence:.2f}"
        cv2.putText(frame, conf_text, (10, y_start + 50), font, font_scale, (255, 255, 255), thickness)
        
        # Fall Alert
        if prediction_result.get('predicted_class') == 1:
            if self.fall_alert_active:
                alert_text = f"FALL ALERT!"
                cv2.putText(frame, alert_text, (10, y_start + 75), font, font_scale, (0, 0, 255), thickness)
            else:
                status_text = f"Status: Monitoring ({self.fall_frame_count}/{self.consecutive_fall_threshold})"
                cv2.putText(frame, status_text, (10, y_start + 75), font, font_scale, (0, 255, 255), thickness)
        else:
            status_text = f"Status: Normal Activity"
            cv2.putText(frame, status_text, (10, y_start + 75), font, font_scale, (0, 255, 255), thickness)
        return frame
    
    def create_skeleton_display(self, original_frame, keypoints_list, scores_list):
        """Create skeleton display on black background"""
        h, w = original_frame.shape[:2]
        black_frame = np.zeros((h, w, 3), dtype=np.uint8)
        
        if keypoints_list and scores_list:
            black_frame = visualize_keypoints(
                black_frame, keypoints_list, scores_list, thr=0.3
            )
        
        return black_frame
    
    def process_frame(self, frame):
        """Process a single frame through the entire pipeline"""
        frame_start_time = time.time()
        
        # Step 1: Person Detection 
        person_bboxes, person_scores = self.detect_person_bbox(frame)
        
        # Step 2: RTMPose
        ctrgcn_kpts_list, keypoints_list, scores_list = self.run_rtmpose(frame, person_bboxes)
        
        # Step 3: CTRGCN Fall Detection
        prediction_result = self.run_ctrgcn(ctrgcn_kpts_list, person_scores)
        # Debug print
        print(prediction_result)
        print()
        
        # Update fall alert
        self.update_fall_alert(prediction_result)
        
        # Calculate FPS
        frame_time = time.time() - frame_start_time
        self.frame_times.append(frame_time)
        fps = 1.0 / np.mean(list(self.frame_times)) if self.frame_times else 0
        
        # Create visualizations
        # Original frame with skeleton
        original_with_skeleton = frame.copy()
        if keypoints_list and scores_list:
            original_with_skeleton = visualize_keypoints(
                original_with_skeleton, keypoints_list, scores_list, thr=self.args.keypoint_threshold
            )
        
        # Skeleton on black background
        skeleton_display = self.create_skeleton_display(frame, keypoints_list, scores_list)
        
        # Add info panels
        original_with_skeleton = self.draw_info_panel(
            original_with_skeleton, fps, prediction_result, 'top'
        )
        skeleton_display = self.draw_info_panel(
            skeleton_display, fps, prediction_result, 'top'
        )
        
        # Combine displays side by side
        combined_display = np.hstack([original_with_skeleton, skeleton_display])
        
        return combined_display
    
    def run(self):
        """Run the fall detection pipeline"""
        # Initialize video capture
        if self.args.input_video:
            cap = cv2.VideoCapture(self.args.input_video)
            print(f"Processing video: {self.args.input_video}")
        else:
            cap = cv2.VideoCapture(0)  # Webcam
            print("Using webcam input")
        
        if not cap.isOpened():
            print("Error: Could not open video source")
            return
        
        # Set webcam properties if using webcam
        if not self.args.input_video:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_FPS, 30)
        
        print("Pipeline started. Press 'q' to quit, 'r' to reset fall detector")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    if self.args.input_video:
                        print("End of video reached")
                    else:
                        print("Failed to capture frame from webcam")
                    break
                
                # Process frame
                display_frame = self.process_frame(frame)
                
                # Show result
                cv2.imshow('Fall Detection - Original | Skeleton', display_frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('r'):
                    self.fall_detector.reset()
                    self.fall_frame_count = 0
                    self.fall_alert_active = False
                    self.fall_alert_start_time = None
                    print("Fall detector reset")
                elif key == ord('s') and self.args.input_video:
                    # Space to pause/resume video
                    cv2.waitKey(0)
        
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
            
            # Print statistics
            stats = self.fall_detector.get_statistics()
            print("\n=== Detection Statistics ===")
            print(f"Frames processed: {stats['frames_processed']}")
            print(f"Average inference time: {stats['average_inference_time_ms']:.1f} ms")
            print(f"Average FPS: {stats['average_fps']:.1f}")

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Fall Detection Pipeline: RF-DETR -> RTMPose -> CTRGCN')
    
    # Input options
    parser.add_argument('--input-video', type=str, default=None,
                       help='Path to input video file (default: use webcam)')
    
    # Model paths
    parser.add_argument('--rtmpose-model', type=str, 
                       default='/Users/vionna/Desktop/new_fd_train_v2/rtmpose/models/onnx/rtmpose-s.onnx',
                       help='Path to RTMPose ONNX model')
    parser.add_argument('--ctrgcn-model', type=str, 
                        default='/Users/vionna/Desktop/new_fd_train_v2/ctrgcn/runs/3/checkpoints/best_model.pth',
                       help='Path to CTRGCN model weights')
    
    # Device settings
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cpu', 'cuda'],
                       help='Device for inference (default: auto)')
    
    # Fall detection parameters
    parser.add_argument('--det-img-shape', type=int, nargs=2, default=[640, 640],
                       help='Detection image shape (width height), default: 640 640')
    parser.add_argument('--human-det-threshold', type=float, default=0.5,
                       help='Score threshold for person detection (default: 0.5)')
    parser.add_argument('--keypoint-threshold', type=float, default=0.3,
                       help='Threshold for keypoint (default: 0.3)')
    parser.add_argument('--fall-detection-threshold', type=float, default=0.7,
                    help='Confidence threshold for fall detection (default: 0.7)')
    parser.add_argument('--num-classes', type=int, default=6,
                       help='Number of classes for action recognition (default: 6)')
    parser.add_argument('--fall-frames', type=int, default=10,
                       help='Number of consecutive fall frames to trigger alert (default: 10)')
    parser.add_argument('--temporal-window', type=int, default=30,
                       help='Temporal window size for pose sequence (default: 30)')
    parser.add_argument('--smoothing-window', type=int, default=5,
                       help='Window size for prediction smoothing (default: 5)')
    parser.add_argument('--use-batch-inference', action='store_true',
                        help='Use batch inference for pose estimation')
    
    return parser.parse_args()

def main():
    """Main function"""
    args = parse_arguments()
    
    # Validate model files exist (if specified)
    if not os.path.exists(args.rtmpose_model):
        print(f"Error: RTMPose model not found: {args.rtmpose_model}")
        print("Please download the RTMPose ONNX model or provide correct path")
        return
    
    if args.ctrgcn_model and not os.path.exists(args.ctrgcn_model):
        print(f"Warning: CTRGCN model not found: {args.ctrgcn_model}")
        print("Using randomly initialized weights")
    
    # Initialize and run pipeline
    pipeline = FallDetectionPipeline(args)
    pipeline.run()

if __name__ == '__main__':
    main()