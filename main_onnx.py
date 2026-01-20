import argparse
import functools
import cv2
import numpy as np
import os 
import onnxruntime as ort
import time 

from ppyoloe.onnx_utils import PPYOLOEONNXPredictor
from rtmpose.onnx_utils import process_multiple_people, visualize_keypoints
from ctrgcn.detector import CTRGCNFallDetector

# COCO-17 skeleton connections
COCO_SKELETON = [
    [15, 13], [13, 11], [16, 14], [14, 12], [11, 12],
    [5, 11], [6, 12], [5, 6], [5, 7], [6, 8],
    [7, 9], [8, 10], [1, 2], [0, 1], [0, 2],
    [1, 3], [2, 4], [3, 5], [4, 6]
]

SKELETON_COLORS = [
        (255, 0, 0), (255, 85, 0), (255, 170, 0), (255, 255, 0),
        (170, 255, 0), (85, 255, 0), (0, 255, 0), (0, 255, 85),
        (0, 255, 170), (0, 255, 255), (0, 170, 255), (0, 85, 255),
        (0, 0, 255), (85, 0, 255), (170, 0, 255), (255, 0, 255),
        (255, 0, 170), (255, 0, 85)
    ]

def draw_skeleton_on_black(keypoints_list, scores_list, img_shape, thr=0.3):
    """Draw skeleton on black background for all detected persons"""
    # Create black background
    black_img = np.zeros(img_shape, dtype=np.uint8)
    
    if not keypoints_list or len(keypoints_list) == 0:
        return black_img
    
    # Process all detected persons
    for person_idx, keypoints in enumerate(keypoints_list):
        scores = scores_list[person_idx] if scores_list and person_idx < len(scores_list) else None
        
        # Draw skeleton connections
        for i, (start_idx, end_idx) in enumerate(COCO_SKELETON):
            if start_idx < len(keypoints) and end_idx < len(keypoints):
                start_point = keypoints[start_idx]
                end_point = keypoints[end_idx]
                
                # Check confidence if scores available
                if scores is not None:
                    if scores[start_idx] < thr or scores[end_idx] < thr:
                        continue
                
                start_pos = (int(start_point[0]), int(start_point[1]))
                end_pos = (int(end_point[0]), int(end_point[1]))
                
                # Use original colors for skeleton connections
                color = SKELETON_COLORS[i % len(SKELETON_COLORS)]
                cv2.line(black_img, start_pos, end_pos, color, 2)
        
        # Draw keypoints 
        for i, keypoint in enumerate(keypoints):
            if scores is not None and scores[i] < thr:
                continue
            
            pos = (int(keypoint[0]), int(keypoint[1]))
            cv2.circle(black_img, pos, 4, (255, 255, 255), -1)
            
    return black_img

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Fall Detection with PP-YOLOE + CTRGCN + RTMPose')
    parser.add_argument('--ppyoloe-model', type=str, default='./ppyoloe/models/onnx/ppyoloe_plus_s.onnx',
                        help='Path to PP-YOLOE ONNX model')
    parser.add_argument('--rtmpose-model', type=str, default='./rtmpose/models/onnx/rtmpose-s.onnx',
                        help='Path to RTMPose ONNX model')
    parser.add_argument('--ctrgcn-model', type=str, default='./ctrgcn/models/ctrgcn.onnx',
                        help='Path to CTRGCN fall detection model')
    parser.add_argument('--input-type', type=str, choices=['image', 'video', 'webcam'], required=True,
                        help='Input type: image, video, or webcam')
    parser.add_argument('--source', type=str, default=None,
                        help='Input source: image path, video path, or webcam index (0, 1, etc.)')
    parser.add_argument('--score-threshold', type=float, default=0.5,
                        help='Detection confidence threshold (default: 0.5)')
    parser.add_argument('--pose-threshold', type=float, default=0.3,
                        help='Pose keypoint confidence threshold (default: 0.3)')
    parser.add_argument('--fall-threshold', type=float, default=0.9,
                        help='Fall detection confidence threshold (default: 0.9)')
    parser.add_argument('--fall-consecutive-threshold', type=int, default=5,
                        help='Number of detected falls before alert is raised (default: 5)')
    parser.add_argument('--image-shape', type=str, default='640,640',
                        help='Input image shape for the model (default: 640,640)')
    parser.add_argument('--use-batch-inference', action='store_true', default=False,
                        help='Use batch inference for pose estimation (default: False)')
    parser.add_argument('--output-dir', type=str, default='.',
                        help='Output directory for results (default: current directory)')
    parser.add_argument('--use-gpu', action='store_true', default=False,
                        help='Use GPU for inference (default: False)')
    parser.add_argument('--use-tensorrt', action='store_true', default=False,
                        help='Use TensorRT for acceleration (default: False)')
    
    # Enhance detection parameters
    parser.add_argument('--min-frames-for-detection', type=int, default=10,
                    help='Minimum frames to consider before making ANY prediction (default: 10)')
    parser.add_argument('--buffer-fill-strategy', type=str, choices=['gradual', 'duplicate'], default='gradual',
                        help='Buffer fill strategy for fall detection: gradual or duplicate (default: gradual)')
    
    args = parser.parse_args()

    labels_list_path = './dataset/label_list.txt'
    
    # Validate arguments
    if args.input_type == 'image' and not args.source:
        parser.error("--source is required for image input")
    elif args.input_type == 'video' and not args.source:
        parser.error("--source is required for video input")
    elif args.input_type == 'webcam':
        args.source = int(args.source) if args.source else 0

    # Check if model files exist
    if not os.path.exists(args.ppyoloe_model):
        raise FileNotFoundError(f"PP-YOLOE model not found: {args.ppyoloe_model}")
    if not os.path.exists(args.rtmpose_model):
        raise FileNotFoundError(f"RTMPose model not found: {args.rtmpose_model}")
    if not os.path.exists(args.ctrgcn_model):
        raise FileNotFoundError(f"CTRGCN model not found: {args.ctrgcn_model}")

    det_img_shape = [int(s) for s in args.image_shape.split(',')] # 640, 640
    print(f"Detection image shape: {det_img_shape}")

    # Initialize PP-YOLOE model
    predictor = PPYOLOEONNXPredictor(model_path=args.ppyoloe_model,
                                        labels_list_path=labels_list_path,
                                        use_gpu=args.use_gpu,
                                        use_tensorrt=args.use_tensorrt,
                                        height=det_img_shape[0],
                                        width=det_img_shape[1],
                                        threshold=args.score_threshold)
    print(f"PP-YOLOE model loaded from {args.ppyoloe_model}")

    # Initialize RTMPose model
    rtmpose_session = ort.InferenceSession(args.rtmpose_model)
    print(f"RTMPose model loaded from {args.rtmpose_model}")

    # Initialize fall detection pipeline
    fall_pipeline = CTRGCNFallDetector(
        args.ctrgcn_model, 
        use_gpu=args.use_gpu,
        min_frames_for_detection=args.min_frames_for_detection,  # Minimum frames to consider for fall detection
        buffer_fill_strategy=args.buffer_fill_strategy,  # 'gradual' or 'duplicate'
    )
    print(f"CTRGCN fall detection ONNX model loaded from {args.ctrgcn_model}")

    if args.input_type == 'image':
        image = cv2.imdecode(np.fromfile(args.source, dtype=np.uint8), cv2.IMREAD_COLOR)
        bboxes, det_scores = predictor.infer(image)

        if len(bboxes) == 0:
            print("No detections found!")
            fall_result = None
            keypoints_list, scores_list = [], []
        else:
            print(f"Found {len(bboxes)} detections")
            
            # Get RTMPose model input size
            h, w = rtmpose_session.get_inputs()[0].shape[2:]
            model_input_size = (w, h)
            
            # Process poses
            keypoints_list, scores_list = process_multiple_people(
                image, rtmpose_session, bboxes, model_input_size, 
                use_batch_inference=args.use_batch_inference
            )
            
        # Add keypoints to fall detection pipeline
        fall_pipeline.add_keypoints(keypoints_list, scores_list, det_scores)
        
        # For single image, we need to fill the buffer
        for _ in range(fall_pipeline.sequence_length - 1):
            fall_pipeline.add_keypoints(keypoints_list, scores_list, det_scores)
        
        fall_result = fall_pipeline.detect_fall()

        # Create left display (original image with annotations)
        left_img = image.copy()
        
        # Draw bounding boxes
        for bbox in bboxes:
            if bbox is not None:
                x1, y1, x2, y2 = map(int, bbox)
                cv2.rectangle(left_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Draw skeleton on left image
        if keypoints_list and scores_list:
            left_img = visualize_keypoints(left_img, keypoints_list, scores_list, thr=args.pose_threshold)
        
        # Create right display (skeleton only on black background)
        right_img = draw_skeleton_on_black(keypoints_list, scores_list, image.shape, thr=args.pose_threshold)
        
        # Add fall detection result to both images
        if fall_result:
            fall_text = f"Fall: {fall_result['predicted_class']} ({fall_result['confidence']:.2f})"
            color = (0, 255, 255) if fall_result['predicted_class'] == 'fall' else (0, 255, 0)
            
            cv2.putText(left_img, fall_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            cv2.putText(right_img, fall_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            

        # Combine images side by side
        combined_img = np.hstack([left_img, right_img])

        # Save result
        base_name = os.path.splitext(os.path.basename(args.source))[0]
        output_path = os.path.join(args.output_dir, f"fall_detection_{base_name}.jpg")
        cv2.imwrite(output_path, combined_img)
        print(f"Result saved to: {output_path}")
        
    else: # webcam or video 
        cap = cv2.VideoCapture(args.source)
        if not cap.isOpened():
            raise ValueError(f"Could not open webcam/video file: {args.source}")
        
        window_name = 'Fall Detection System - Left: Original | Right: Skeleton Only'
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

        fps_time = time.time()
        frame_count = 0
        consecutive_falls = 0
        fall_alert_threshold = args.fall_consecutive_threshold

        while True: 
            ret, frame = cap.read()
            if not ret:
                print("End of video stream or failed to read frame.")
                break
            
            frame_count += 1
            
            # Object detection
            bboxes, det_scores = predictor.infer(frame)
            
            # Pose estimation
            keypoints_list, scores_list = [], []
            if len(bboxes) > 0:
                # Get RTMPose model input size
                h, w = rtmpose_session.get_inputs()[0].shape[2:]
                model_input_size = (w, h)
                
                # Process poses
                keypoints_list, scores_list = process_multiple_people(
                    frame, rtmpose_session, bboxes, model_input_size, 
                    use_batch_inference=args.use_batch_inference
                )

            # Add keypoints to fall detection pipeline
            fall_pipeline.add_keypoints(keypoints_list, scores_list, det_scores)

            # Fall detection
            fall_result = fall_pipeline.detect_fall()
            
            # Create left display (original frame with annotations)
            left_frame = frame.copy()
            
            # Draw bounding boxes
            for bbox in bboxes:
                if bbox is not None:
                    x1, y1, x2, y2 = map(int, bbox)
                    cv2.rectangle(left_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw skeleton on left frame
            if keypoints_list and scores_list:
                left_frame = visualize_keypoints(left_frame, keypoints_list, scores_list, thr=args.pose_threshold)
            
            # Create right display (skeleton only on black background)
            right_frame = draw_skeleton_on_black(keypoints_list, scores_list, frame.shape, thr=args.pose_threshold)
            
            # Display fall detection result on both frames
            if fall_result:
                fall_text = f"Action: {fall_result['predicted_class']} ({fall_result['confidence']:.2f})"
                color = (0, 255, 255) if fall_result['predicted_class'] == 'fall' else (0, 255, 0)
                
                cv2.putText(left_frame, fall_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                cv2.putText(right_frame, fall_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                
                # Fall alert logic 
                if fall_result['predicted_class'] == 'fall' and fall_result['confidence'] > args.fall_threshold:
                    consecutive_falls += 1
                    if consecutive_falls >= fall_alert_threshold:
                        cv2.putText(left_frame, "FALL ALERT!", (10, 100), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                        cv2.putText(right_frame, "FALL ALERT!", (10, 100), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                        print(f"🚨 FALL DETECTED! Confidence: {fall_result['confidence']:.2f}")

            # Calculate and display FPS
            current_time = time.time()
            fps_value = 1.0 / (current_time - fps_time)
            fps_time = current_time
            cv2.putText(left_frame, f"FPS: {fps_value:.1f}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(right_frame, f"FPS: {fps_value:.1f}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Display detection info
            cv2.putText(left_frame, f"Persons: {len(bboxes)}", (10, left_frame.shape[0] - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(right_frame, f"Persons: {len(bboxes)}", (10, right_frame.shape[0] - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # Combine frames side by side
            combined_frame = np.hstack([left_frame, right_frame])
            cv2.imshow(window_name, combined_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Exiting video stream.")
                break
                
        cap.release()
        cv2.destroyAllWindows()