import torch
import torch.nn.functional as F
import numpy as np
import cv2
from collections import deque
import time
from .ctrgcn import CTRGCN, Graph

class FallDetector:
    """
    Fall Detection system using CTR-GCN for real-time pose-based fall detection
    """
    
    def __init__(self, model_path=None, device='auto', confidence_threshold=0.7, 
                 temporal_window=30, smoothing_window=5, num_classes=6, temperature=2.0):
        """
        Initialize Fall Detector
        
        Args:
            model_path (str): Path to trained CTR-GCN model weights
            device (str): Device to run inference ('cuda', 'cpu', or 'auto')
            confidence_threshold (float): Threshold for fall detection confidence
            temporal_window (int): Number of frames to analyze for temporal features
            smoothing_window (int): Number of predictions to smooth for final decision
        """
        self.device = self._setup_device(device)
        self.confidence_threshold = confidence_threshold
        self.temporal_window = temporal_window
        self.smoothing_window = smoothing_window
        self.num_classes = num_classes
        self.temperature = temperature

        # Class mapping
        self.class_names = ['Standing', 'Falling', 'Sitting', 'Bending', 'Sleeping', 'Walking', "Unknown"]
        
        # Initialize model
        self.model = self._load_model(model_path)
        
        # Temporal storage for pose sequences
        self.pose_buffer = deque(maxlen=temporal_window)

        # Fall state tracking
        self.is_falling = False
        self.fall_start_time = None
        self.last_fall_time = 0
        self.fall_cooldown = 3.0  # Seconds between fall detections
        
        # Statistics
        self.frame_count = 0
        self.inference_times = deque(maxlen=100)
        
    def _setup_device(self, device):
        """Setup computation device"""
        if device == 'auto':
            return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return torch.device(device)
    
    def _load_model(self, model_path):
        """Load and initialize CTR-GCN model"""
        # Create model with COCO skeleton (17 keypoints)
        graph = Graph(layout='coco', strategy='spatial', max_hop=1)
        model = CTRGCN(
            num_class=self.num_classes,  
            num_point=17,  # COCO keypoints
            num_person=1,
            graph=graph,
            in_channels=3,  # x, y, confidence
            drop_out=0.0,  # No dropout during inference
            adaptive=True,
            attention=True
        )
        
        # Load weights if provided
        if model_path:
            try:
                checkpoint = torch.load(model_path, map_location=self.device)
                if 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    model.load_state_dict(checkpoint)
                print(f"Loaded model weights from {model_path}")
            except Exception as e:
                print(f"Warning: Could not load model weights: {e}")
                print("Using randomly initialized weights")
        
        model.to(self.device)
        model.eval()
        return model
    
    def preprocess_pose(self, keypoints):
        """
        Preprocess pose keypoints for model input
        
        Args:
            keypoints: Array of shape (17, 3) 
            
        Returns:
            Normalized keypoints ready for model input
        """
        if keypoints is None or len(keypoints) == 0:
            # Return zero pose if no keypoints detected
            return np.zeros((17, 3))
        
        # Ensure we have 17 keypoints
        if len(keypoints) != 17:
            # Pad or truncate to 17 keypoints
            padded_kpts = np.zeros((17, 3))
            n_kpts = min(len(keypoints), 17)
            if keypoints.shape[1] == 2:
                # Add confidence scores of 1.0 if not provided
                padded_kpts[:n_kpts, :2] = keypoints[:n_kpts]
                padded_kpts[:n_kpts, 2] = 1.0
            else:
                padded_kpts[:n_kpts] = keypoints[:n_kpts]
            keypoints = padded_kpts
        
        # Normalize coordinates 
        valid_mask = keypoints[:, 2] > 0.3  # Same confidence threshold
        valid_points = keypoints[valid_mask, :2]
        
        if len(valid_points) > 4:  # Same minimum points requirement
            x_coords = valid_points[:, 0]
            y_coords = valid_points[:, 1]
            
            x_min, x_max = np.min(x_coords), np.max(x_coords)
            y_min, y_max = np.min(y_coords), np.max(y_coords)
            
            width = max(x_max - x_min, 1.0)  # Same fallback
            height = max(y_max - y_min, 1.0)
            center_x = (x_min + x_max) / 2
            center_y = (y_min + y_max) / 2
            
            normalized_kpts = keypoints.copy()
            normalized_kpts[:, 0] = (keypoints[:, 0] - center_x) / (width / 2)
            normalized_kpts[:, 1] = (keypoints[:, 1] - center_y) / (height / 2)
            
            normalized_kpts[:, :2] = np.clip(normalized_kpts[:, :2], -5, 5)

            # Debug print
            # data_range = [np.min(normalized_kpts[:, :2]), np.max(normalized_kpts[:, :2])]
            # print(f"Normalized range: [{data_range[0]:.2f}, {data_range[1]:.2f}]")
            
            return normalized_kpts.astype(np.float32)
        else:
            return np.zeros((17, 3), dtype=np.float32)
    
    def update_pose_buffer(self, keypoints):
        """Add new pose to temporal buffer"""
        processed_pose = self.preprocess_pose(keypoints)
        self.pose_buffer.append(processed_pose)
    
    def extract_sequence(self):
        """Extract pose sequence for model input"""
        if len(self.pose_buffer) < self.temporal_window:
            # Pad sequence if not enough frames
            sequence = list(self.pose_buffer)
            while len(sequence) < self.temporal_window:
                if len(sequence) > 0:
                    sequence.insert(0, sequence[0])  # Repeat first frame
                else:
                    sequence.append(np.zeros((17, 3)))  # Add zero pose
        else:
            sequence = list(self.pose_buffer)
        
        # Convert to tensor format (T, V, C) -> (1, C, T, V)
        sequence = np.array(sequence)  # (T, V, C)
        sequence = sequence.transpose(2, 0, 1)  # (C, T, V)
        sequence = sequence[np.newaxis, ...]  # (1, C, T, V)
        sequence = torch.FloatTensor(sequence).to(self.device)
        
        return sequence
    
    def _apply_temperature_scaling(self, logits):
        """Apply temperature scaling to calibrate confidence"""
        scaled_logits = logits / self.temperature
        return F.softmax(scaled_logits, dim=1)
    
    def predict(self, keypoints):
        """
        Predict fall probability from pose keypoints
        
        Args:
            keypoints: Array of shape (17, 3) or (17, 2) containing pose keypoints
            
        Returns:
            dict: Prediction results with probability, confidence, and decision
        """
        start_time = time.time()
        
        # Update pose buffer
        self.update_pose_buffer(keypoints)
        self.frame_count += 1
        
        # Need minimum frames for prediction
        if len(self.pose_buffer) < min(10, self.temporal_window):
            return {
                'predicted_class': 0,  # Default to standing
                'class_probabilities': [1.0] + [0.0] * (self.num_classes),
                'confidence': 0.0,
                'is_fall': False,
                'status': 'insufficient_data'
            }
        
        # Extract sequence and predict
        with torch.no_grad():
            sequence = self.extract_sequence()
            logits = self.model(sequence)
            # print(f"Logits: {logits}")
            probabilities = self._apply_temperature_scaling(logits)
            class_probs = probabilities[0].cpu().numpy()
            predicted_class = np.argmax(class_probs)
        
        max_prob = np.max(class_probs)
        if max_prob < 0.5:
            predicted_class = self.num_classes  # Assign to "Unknown" class
        
        # Track inference time
        inference_time = time.time() - start_time
        self.inference_times.append(inference_time)
        
        # Make fall decision with temporal consistency
        current_time = time.time()
        is_fall = (predicted_class == 1 and class_probs[1] > self.confidence_threshold)
        
        if is_fall:
            self.is_falling = True
            self.fall_start_time = current_time
            self.last_fall_time = current_time
        else:
            self.is_falling = False
        
        return {
            'predicted_class': predicted_class,
            'class_probabilities': class_probs.tolist(),
            'confidence': max_prob,
            'is_fall': is_fall,
            'inference_time_ms': inference_time * 1000,
            'frame_count': self.frame_count,
            'status': self.class_names[predicted_class].lower()
        }
    
    def get_statistics(self):
        """Get detector performance statistics"""
        avg_inference_time = np.mean(list(self.inference_times)) if self.inference_times else 0
        
        return {
            'frames_processed': self.frame_count,
            'average_inference_time_ms': avg_inference_time * 1000,
            'average_fps': 1.0 / max(avg_inference_time, 0.001) if avg_inference_time > 0 else 0,
            'pose_buffer_size': len(self.pose_buffer),
        }
    
    def reset(self):
        """Reset detector state"""
        self.pose_buffer.clear()
        self.is_falling = False
        self.fall_start_time = None
        self.frame_count = 0
