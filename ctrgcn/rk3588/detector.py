import numpy as np
from collections import deque
import onnxruntime as ort

class ActionTransitionDetector:
    def __init__(self, temporal_window=20):
        self.temporal_window = temporal_window
        self.keypoint_history = deque(maxlen=temporal_window)
        self.velocity_history = deque(maxlen=temporal_window)
        self.height_history = deque(maxlen=temporal_window)
        
    def add_keypoints(self, keypoints):
        """Add keypoints and calculate motion metrics"""
        if keypoints is None:
            return
            
        self.keypoint_history.append(keypoints)
        
        # Calculate center of mass height (approximate standing height)
        valid_joints = keypoints[keypoints[:, 2] > 0.3]
        if len(valid_joints) > 0:
            center_y = np.mean(valid_joints[:, 1])
            self.height_history.append(center_y)
        
        # Calculate velocity if we have previous frame
        if len(self.keypoint_history) >= 2:
            prev_kp = self.keypoint_history[-2]
            curr_kp = self.keypoint_history[-1]
            
            # Calculate average velocity of valid keypoints
            velocities = []
            for i in range(len(curr_kp)):
                if curr_kp[i, 2] > 0.3 and prev_kp[i, 2] > 0.3:
                    vel = np.sqrt((curr_kp[i, 0] - prev_kp[i, 0])**2 + 
                                 (curr_kp[i, 1] - prev_kp[i, 1])**2)
                    velocities.append(vel)
            
            avg_velocity = np.mean(velocities) if velocities else 0.0
            self.velocity_history.append(avg_velocity)
    
    def detect_transition_type(self):
        """Detect type of action transition"""
        if len(self.height_history) < 10 or len(self.velocity_history) < 5:
            return "unknown", 0.0
        
        recent_heights = list(self.height_history)[-10:]
        recent_velocities = list(self.velocity_history)[-5:]
        
        # Detect getting up: increasing height trend + moderate velocity
        height_trend = np.polyfit(range(len(recent_heights)), recent_heights, 1)[0]
        avg_velocity = np.mean(recent_velocities)
        
        if height_trend < -5 and avg_velocity > 0.1:  # Negative trend = going up (y increases downward)
            return "getting_up", 0.8
        elif height_trend > 5 and avg_velocity > 0.2:  # Positive trend = going down
            return "bending_down", 0.7
        elif avg_velocity < 0.05:
            return "stationary", 0.6
        else:
            return "normal_movement", 0.5
        
class CTRGCNFallDetector:
    # This detector only works for a single person in the frame
    def __init__(self, 
                 ctrgcn_model_path, 
                 use_gpu=False,
                 min_frames_for_detection=15,  # Minimum frames to consider for fall detection
                 buffer_fill_strategy='gradual', # 'gradual' or 'duplicate' 
                 enable_transition_detection=True,
                ):
        """Initialize the CTRGCN-based fall detection model with ONNX runtime"""
        
        # Initialize ONNX session for CTRGCN
        providers = ['CPUExecutionProvider']
        if use_gpu:
            if ort.get_device() == 'GPU':
                providers.insert(0, 'CUDAExecutionProvider')
            # For Mac with MPS
            elif hasattr(ort, 'get_available_providers') and 'CoreMLExecutionProvider' in ort.get_available_providers():
                providers.insert(0, 'CoreMLExecutionProvider')
        
        self.ctrgcn_session = ort.InferenceSession(ctrgcn_model_path, providers=providers)
            
        # Get model input details
        self.input_name = self.ctrgcn_session.get_inputs()[0].name
        self.input_shape = self.ctrgcn_session.get_inputs()[0].shape
        
        print(f"CTRGCN ONNX model input: {self.input_name}, shape: {self.input_shape}")
        
        # Extract sequence parameters from model input shape
        # Expected shape: [batch_size, channels, seq_len, num_joints]
        # Example: [1, 3, 30, 17]
        if len(self.input_shape) == 4:
            self.batch_size = self.input_shape[0] if self.input_shape[0] != -1 else 1
            self.num_channels = self.input_shape[1]  # Should be 3 (x, y, confidence)
            self.sequence_length = self.input_shape[2]  # Temporal frames
            self.num_joints = self.input_shape[3]  # Should be 17 for COCO
        else:
            # Fallback to defaults
            self.sequence_length = 30
            self.num_joints = 17
            self.num_channels = 3
            self.batch_size = 1
        
        print(f"Model expects: seq_len={self.sequence_length}, joints={self.num_joints}, channels={self.num_channels}")
        
        # Initialize keypoints buffer
        self.keypoints_buffer = deque(maxlen=self.sequence_length)
        self.frame_counter = 0
        
        # Class labels
        self.class_labels = ['walking',
            'sitting', 
            'standing',
            'falling',
            'bending_down',
            'crouching',
            'waving_hands',
            'sleeping']

        # Frame parameters
        self.min_frames_for_detection = min_frames_for_detection
        self.buffer_fill_strategy = buffer_fill_strategy
        self.valid_detections_count = 0  # Track frames with valid human detection

        # Initialize transition detection
        self.enable_transition_detection = enable_transition_detection
        if self.enable_transition_detection:
            self.transition_detector = ActionTransitionDetector(temporal_window=self.sequence_length)
        
        # Bed detection parameters
        self.bed_bbox = None
        self.bed_overlap_threshold = 0.7 # Default threshold
        self.bed_overlap_history = deque(maxlen=10)  # Track recent bed overlap ratios

    def add_bed_params(self, bed_bbox, bed_overlap_threshold):
        """Set the bed bounding box for overlap checking"""
        if self.bed_bbox is not None:
            self.bed_overlap_history.clear()  # Clear history when bed bbox changes
        self.bed_bbox = bed_bbox  # [x1, y1, x2, y2]
        self.bed_overlap_threshold = bed_overlap_threshold

    def check_keypoints_in_bed(self, keypoints):
        """
        Check if keypoints are inside the bed bounding box
        
        Args:
            keypoints: numpy array of shape (17, 3) with [x, y, confidence]
            
        Returns:
            dict: Contains overlap ratio and whether person is in bed
        """
        if self.bed_bbox is None or keypoints is None:
            return {'in_bed': False, 'overlap_ratio': 0.0, 'keypoints_in_bed': 0, 'total_valid_keypoints': 0}
        
        x1, y1, x2, y2 = self.bed_bbox
        
        # Count valid keypoints (confidence > 0.3)
        valid_keypoints = keypoints[keypoints[:, 2] > 0.3]
        total_valid = len(valid_keypoints)
        
        if total_valid == 0:
            return {'in_bed': False, 'overlap_ratio': 0.0, 'keypoints_in_bed': 0, 'total_valid_keypoints': 0}
        
        # Count keypoints inside bed bbox
        keypoints_in_bed = 0
        for kp in valid_keypoints:
            x, y, conf = kp
            if x1 <= x <= x2 and y1 <= y <= y2:
                keypoints_in_bed += 1
        
        # Calculate overlap ratio
        overlap_ratio = keypoints_in_bed / total_valid
        
        # Determine if person is considered to be in bed
        in_bed = overlap_ratio >= self.bed_overlap_threshold
        
        return {
            'in_bed': in_bed,
            'overlap_ratio': overlap_ratio,
            'keypoints_in_bed': keypoints_in_bed,
            'total_valid_keypoints': total_valid
        }
    
    def update_bed_overlap_history(self, overlap_ratio):
        """Update the history of bed overlap ratios for temporal consistency"""
        self.bed_overlap_history.append(overlap_ratio)
    
    def is_consistently_in_bed(self, min_frames=5):
        """
        Check if person has been consistently in bed for a minimum number of frames
        
        Args:
            min_frames: Minimum frames to consider for consistency
            
        Returns:
            bool: True if consistently in bed
        """
        if len(self.bed_overlap_history) < min_frames:
            return False
        
        recent_overlaps = list(self.bed_overlap_history)[-min_frames:]
        consistent_in_bed = sum(1 for ratio in recent_overlaps if ratio >= self.bed_overlap_threshold)
        
        # Require at least 80% of recent frames to have high bed overlap
        consistency_threshold = max(1, int(min_frames * 0.8))
        return consistent_in_bed >= consistency_threshold
    
    def add_keypoints(self, keypoints_list, scores_list, det_scores):
        """Add keypoints to the buffer for temporal modeling"""
        self.frame_counter += 1
        
        # Create default keypoints if no person detected
        if not keypoints_list or len(keypoints_list) == 0:
            # Use zero keypoints as placeholder
            keypoints = np.zeros((self.num_joints, 3))
            self.valid_detections_count = 0
            # Update bed overlap history with 0
            if self.bed_bbox is not None:
                self.update_bed_overlap_history(0.0)
        else:
            # Select person with highest detection score
            if det_scores is not None and len(det_scores) > 0:
                best_person_idx = np.argmax(det_scores)
            else:
                best_person_idx = 0
            
            # Ensure the index is valid
            best_person_idx = min(best_person_idx, len(keypoints_list) - 1)
            
            # Use best person's keypoints
            keypoints = keypoints_list[best_person_idx]  # Shape: (17, 2) or (17, 3)

            # Ensure keypoints have confidence scores
            if keypoints.shape[1] == 2:
                conf_scores = scores_list[best_person_idx] if scores_list else np.ones(self.num_joints)
                keypoints = np.column_stack([keypoints, conf_scores])
            
            self.valid_detections_count += 1
            
            # Check bed overlap and update history
            if self.bed_bbox is not None:
                bed_check = self.check_keypoints_in_bed(keypoints)
                self.update_bed_overlap_history(bed_check['overlap_ratio'])
        
        # Add to buffer
        self.keypoints_buffer.append(keypoints)

        # Add to transition detector if enabled
        if self.enable_transition_detection and hasattr(self, 'transition_detector'):
            current_keypoints = keypoints if keypoints_list else None
            self.transition_detector.add_keypoints(current_keypoints)
    
    def detect_fall(self):
        """Detect fall from buffered keypoints using ONNX model"""
        if len(self.keypoints_buffer) < self.sequence_length:
            # Not enough frames for prediction
            return None
        
        # Don't make predictions until we have enough valid detections
        if self.valid_detections_count < self.min_frames_for_detection:
            return {
                'predicted_class': 'no_fall',
                'confidence': 0.1,  # Very low confidence
                'original_confidence': 0.1,
                'probabilities': [0.9, 0.1],
                'frame_id': self.frame_counter,
                'reason': 'insufficient_valid_frames'
            }
        
        # Check if person is consistently in bed
        consistently_in_bed = self.is_consistently_in_bed(min_frames=5)
        current_keypoints = list(self.keypoints_buffer)[-1]  # Get most recent keypoints
        
        # Get current bed overlap info
        bed_info = self.check_keypoints_in_bed(current_keypoints) if self.bed_bbox is not None else {
            'in_bed': False, 'overlap_ratio': 0.0, 'keypoints_in_bed': 0, 'total_valid_keypoints': 0
        }
        
        # If person is consistently in bed, suppress fall detection
        if consistently_in_bed and bed_info['in_bed']:
            return {
                'predicted_class': 'no_fall',
                'confidence': 0.95,  # High confidence that it's not a fall
                'original_confidence': 0.95,
                'probabilities': [0.95, 0.05],
                'frame_id': self.frame_counter,
                'reason': 'person_in_bed',
                'bed_info': bed_info,
                'consistently_in_bed': consistently_in_bed
            }
        
        # Prepare input tensor
        # Convert deque to numpy array: (seq_len, num_joints, 3)
        sequence = np.array(list(self.keypoints_buffer))
        
        # Reshape to model input format: [batch_size, channels, seq_len, num_joints]
        # Transpose from (seq_len, num_joints, 3) to (3, seq_len, num_joints)
        sequence = sequence.transpose(2, 0, 1)  # (3, seq_len, num_joints)
        
        # Add batch dimension: (1, 3, seq_len, num_joints)
        input_tensor = sequence[np.newaxis, :, :, :].astype(np.float32)
        
        # Handle dynamic axes - only check non-dynamic dimensions
        expected_shape = []
        for i, dim in enumerate(self.input_shape):
            if isinstance(dim, str) or dim == -1:
                # Dynamic dimension, use actual input size
                expected_shape.append(input_tensor.shape[i])
            else:
                # Fixed dimension, must match
                expected_shape.append(dim)
        
        if input_tensor.shape != tuple(expected_shape):
            print(f"Warning: Input shape mismatch. Expected: {expected_shape}, Got: {input_tensor.shape}")
            return None
        
        try:
            # Run inference
            outputs = self.ctrgcn_session.run(None, {self.input_name: input_tensor})
            
            # Get predictions
            logits = outputs[0][0]  # Remove batch dimension
            
            # Apply softmax to get probabilities
            exp_logits = np.exp(logits - np.max(logits))  # Numerical stability
            probabilities = exp_logits / np.sum(exp_logits)
            
            # Get predicted class
            predicted_class_idx = np.argmax(probabilities)
            predicted_class = self.class_labels[predicted_class_idx]
            confidence = probabilities[predicted_class_idx]
            
            # Apply bed-based confidence adjustment
            final_confidence = float(confidence)
            adjustment_reason = None
            
            # If model predicts fall but person is partially in bed, reduce confidence
            if predicted_class == 'fall' and bed_info['overlap_ratio'] > 0.3:
                # Reduce confidence based on bed overlap
                reduction_factor = min(0.8, bed_info['overlap_ratio'] * 1.2)
                final_confidence *= (1.0 - reduction_factor)
                adjustment_reason = f"reduced_due_to_bed_overlap_{bed_info['overlap_ratio']:.2f}"
            
            # If transition detection is enabled, check for transitions
            transition_detected = None
            if self.enable_transition_detection and hasattr(self, 'transition_detector'):
                transition_type, transition_confidence = self.transition_detector.detect_transition_type()

                # Adjust confidence based on transition type
                if predicted_class == 'fall':
                    if transition_type == "getting_up" and transition_confidence > 0.2:
                        final_confidence *= 0.2  # Significantly reduce confidence
                        transition_detected = "getting_up"
                    elif transition_type == "bending_down" and transition_confidence > 0.2:
                        final_confidence *= 0.3  # Reduce confidence for bending
                        transition_detected = "bending_down"

            result = {
                'predicted_class': predicted_class,
                'confidence': final_confidence,
                'original_confidence': float(confidence),
                'probabilities': probabilities.tolist(),
                'frame_id': self.frame_counter,
                'transition_detected': transition_detected,
                'bed_info': bed_info,
                'consistently_in_bed': consistently_in_bed,
                'adjustment_reason': adjustment_reason
            }
            
            return result
            
        except Exception as e:
            print(f"Error in fall detection inference: {e}")
            return None
    
    def get_bed_overlap_stats(self):
        """Get statistics about bed overlap history"""
        if not self.bed_overlap_history:
            return {'mean': 0.0, 'max': 0.0, 'min': 0.0, 'current': 0.0}
        
        overlaps = list(self.bed_overlap_history)
        return {
            'mean': np.mean(overlaps),
            'max': np.max(overlaps),
            'min': np.min(overlaps),
            'current': overlaps[-1] if overlaps else 0.0
        }
