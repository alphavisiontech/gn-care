import torch
import json
import numpy as np
import os
from collections import defaultdict
import glob
from torch.utils.data import Dataset
import re

class COCOSkeletonDataset(Dataset):
    """Dataset class for COCO Skeleton data with new folder structure"""
    
    def __init__(self, data_dir, sequence_length=30, num_classes=6, 
                 class_mapping=None, transform=None):
        """
        Args:
            data_dir: Base directory containing numbered folders
            sequence_length: Length of input sequences
            num_classes: Number of classes to use
            class_mapping: Dict mapping class names to indices
            transform: Optional transform to apply to data
        """
        self.data_dir = data_dir
        self.sequence_length = sequence_length
        self.num_classes = num_classes
        self.transform = transform
        
        # Define action patterns for fuzzy matching
        self.standing_patterns = [
            r'\bstand\b', r'\bstanding\b', r'\bstand\s*up\b', r'\bstandup\b',
            r'\bstanding\s*up\b', r'\bupright\b', r'\berect\b', r'\bnormal\b',
            r'\bidle\b', r'\bstill\b', r'\bstationary\b'
        ]
        
        self.falling_patterns = [
            r'\bfall\b', r'\bfalling\b', r'\bfall\s*down\b', r'\bfalldown\b',
            r'\bfalling\s*down\b', r'\bfell\b', r'\bfell\s*down\b', r'\btumble\b',
            r'\btumbling\b', r'\btrip\b', r'\btripping\b', r'\bslip\b', r'\bslipping\b'
        ]
        
        self.sitting_patterns = [
            r'\bsit\b', r'\bsitting\b', r'\bsit\s*down\b', r'\bsitdown\b',
            r'\bsitting\s*down\b', r'\bseated\b', r'\bchair\b', r'\bbench\b'
        ]
        
        self.bending_patterns = [
            r'\bbend\b', r'\bbending\b', r'\bbent\b', r'\bbow\b', r'\bbowing\b',
            r'\bbending\s*down\b', r'\bbend\s*down\b', r'\bstoop\b', r'\bstooping\b'
        ]

        self.sleeping_patterns = [
            r'\bsleep\b', r'\bsleeping\b', r'\brest\b', r'\bresting\b', r'\bbed\b',
            r'\bnap\b', r'\bnapping\b', r'\bprone\b', r'\bsupine\b'
        ]
        
        self.walking_patterns = [
            r'\bwalk\b', r'\bwalking\b', r'\bstroll\b', r'\bstrolling\b',
            r'\bpace\b', r'\bpacing\b', r'\bmarch\b', r'\bmarching\b'
        ]
        
        # Set up class mapping
        if class_mapping is None:
            if num_classes == 2:
                self.class_mapping = {'standing': 0, 'falling': 1}
            elif num_classes == 6:
                self.class_mapping = {
                    'standing': 0, 
                    'falling': 1, 
                    'sitting': 2, 
                    'bending': 3, 
                    'sleeping': 4, 
                    'walking': 5
                }
            else:
                # For other multi-class configurations
                self.class_mapping = {}  
        else:
            self.class_mapping = class_mapping
        
        self.samples = self._prepare_samples()
        
    def _normalize_action_name(self, action_name):
        """Remove extra spaces and convert to lowercase"""
        if not action_name:
            return ""
        return re.sub(r'\s+', ' ', str(action_name).lower().strip())
    
    def _match_action_pattern(self, action_name, patterns):
        """Check if action name matches the given patterns"""
        normalized_name = self._normalize_action_name(action_name)
        for pattern in patterns:
            if re.search(pattern, normalized_name, re.IGNORECASE):
                return True
        return False
    
    def _classify_action(self, action_name, category_id=None):
        """Classify action into standing (0) or falling (1) based on name in COCO"""
        # Fallback using category_id if action_name is missing
        if not action_name and category_id is not None:
            # Default mapping for category IDs
            return min(category_id, self.num_classes - 1)
        
        # Check patterns in order of specificity
        if self._match_action_pattern(action_name, self.falling_patterns):
            return self.class_mapping.get('falling', 1)
        
        if self._match_action_pattern(action_name, self.sitting_patterns):
            return self.class_mapping.get('sitting', 2)
            
        if self._match_action_pattern(action_name, self.bending_patterns):
            return self.class_mapping.get('bending', 3)
            
        if self._match_action_pattern(action_name, self.sleeping_patterns):
            return self.class_mapping.get('sleeping', 4)
            
        if self._match_action_pattern(action_name, self.walking_patterns):
            return self.class_mapping.get('walking', 5)
            
        if self._match_action_pattern(action_name, self.standing_patterns):
            return self.class_mapping.get('standing', 0)

        return -1  # Unknown action
    
    def _get_all_sample_folders(self):
        """Get all numbered sample folders from data directory"""
        sample_folders = []
        if not os.path.exists(self.data_dir):
            print(f"Warning: Data directory {self.data_dir} does not exist")
            return sample_folders
            
        for item in os.listdir(self.data_dir):
            item_path = os.path.join(self.data_dir, item)
            if os.path.isdir(item_path):
                # Check if annotations folder exists
                annotations_path = os.path.join(item_path, 'annotations')
                if os.path.exists(annotations_path):
                    sample_folders.append(item)
        
        return sorted(sample_folders)

    def _get_category_info(self, data):
        """Get category information from COCO data"""
        categories = {}
        
        # Build category mapping from categories section
        for category in data.get('categories', []):
            cat_id = category.get('id')
            cat_name = category.get('name', '')
            if cat_id is not None:
                categories[cat_id] = cat_name
        
        return categories
        
    def _prepare_samples(self):
        """Prepare samples from COCO annotations in numbered folders"""
        samples = []
        sample_folders = self._get_all_sample_folders()
        
        for sample_folder in sample_folders:
            sample_path = os.path.join(self.data_dir, sample_folder)
            annotations_dir = os.path.join(sample_path, 'annotations')
            
            if not os.path.exists(annotations_dir):
                print(f"Warning: Annotations directory not found for sample {sample_folder}")
                continue
            
            # Find all JSON files in annotations directory
            json_files = glob.glob(os.path.join(annotations_dir, '*.json'))
            
            if not json_files:
                print(f"Warning: No JSON files found in {annotations_dir}")
                continue
            
            # Process each JSON file (assuming one per sample)
            for json_file in json_files:
                try:
                    with open(json_file, 'r') as f:
                        data = json.load(f)
                except (json.JSONDecodeError, FileNotFoundError) as e:
                    print(f"Error reading {json_file}: {e}")
                    continue
                
                # Get category information
                categories = self._get_category_info(data)
                # print(f"Processing {json_file} with categories: {categories}")
                
                # Group annotations by sequence/frame
                sequences = defaultdict(list)
                
                for annotation in data.get('annotations', []):
                    if 'keypoints' in annotation:
                        # Get label info for validation
                        category_id = annotation.get('category_id', 0)
                        category_name = categories.get(category_id, '')
                        # print(f"Category Name: {category_name} -> Classified Action: {self._classify_action(category_name, category_id)}")

                        if self._classify_action(category_name, category_id) == -1:
                            continue  # Skip unknown actions

                        # Use image_id or create sequence based on order
                        seq_id = annotation.get('image_id', 0)
                        sequences[seq_id].append(annotation)
                
                # Process each sequence
                for seq_id, seq_annotations in sequences.items():
                    # Sort by frame order if available
                    seq_annotations.sort(key=lambda x: x.get('image_id', 0))
                    
                    keypoints_sequence = []
                    labels = []
                    
                    for annotation in seq_annotations:
                        keypoints_data = annotation.get('keypoints', [])
                        if not keypoints_data:
                            continue
                            
                        kpts = np.array(keypoints_data).reshape(-1, 3)
                        kpts_xy = kpts[:, :2]  # Shape: (17, 2)
                        confidence_scores = kpts[:, 2]  # Confidence scores
                        
                        if len(kpts_xy) != 17:
                            print(f"Warning: Expected 17 keypoints, got {len(kpts_xy)} in {json_file}")
                            continue
                        
                        # Filter valid keypoints based on confidence
                        valid_mask = confidence_scores > 0.3
                        valid_points = kpts_xy[valid_mask]
                        
                        if len(valid_points) > 4:  # Need enough points for meaningful bbox
                            # Calculate bounding box of visible keypoints
                            x_coords = valid_points[:, 0]
                            y_coords = valid_points[:, 1]
                            
                            x_min, x_max = np.min(x_coords), np.max(x_coords)
                            y_min, y_max = np.min(y_coords), np.max(y_coords)
                            
                            # Calculate bounding box dimensions
                            width = max(x_max - x_min, 1.0)  # Avoid division by zero
                            height = max(y_max - y_min, 1.0)
                            center_x = (x_min + x_max) / 2
                            center_y = (y_min + y_max) / 2
                            
                            # Normalize to [-1, 1] range (resolution-independent)
                            normalized_kpts = kpts_xy.copy()
                            normalized_kpts[:, 0] = (kpts_xy[:, 0] - center_x) / (width / 2)
                            normalized_kpts[:, 1] = (kpts_xy[:, 1] - center_y) / (height / 2)
                            
                            # Clip extreme values to prevent outliers
                            normalized_kpts = np.clip(normalized_kpts, -5, 5)
                            
                            kpts_xy = normalized_kpts
                            
                            # Debug print
                            # if len(samples) % 100 == 0:
                            #     print(f"Sample {len(samples)}: bbox=({width:.1f}x{height:.1f}), "
                            #         f"range=[{np.min(kpts_xy):.2f}, {np.max(kpts_xy):.2f}]")
                        else:
                            # If not enough valid points, use zero coordinates
                            kpts_xy = np.zeros_like(kpts_xy)
                            print(f"Warning: Not enough valid keypoints in {json_file}, using zeros")
                        
                        keypoints_sequence.append(kpts_xy)
                        
                        # Get label based on category name and ID
                        category_id = annotation.get('category_id', 0)
                        category_name = categories.get(category_id, '')
                        
                        # Classify based on action name pattern matching
                        action_label = self._classify_action(category_name, category_id)
                        labels.append(action_label)

                    # Create sequences of fixed length
                    if len(keypoints_sequence) >= self.sequence_length:
                        for i in range(len(keypoints_sequence) - self.sequence_length + 1):
                            seq_data = np.array(keypoints_sequence[i:i+self.sequence_length])
                            seq_label = labels[i + self.sequence_length - 1]  # Use last frame label
                            
                            # Validate label
                            if 0 <= seq_label < self.num_classes:
                                samples.append((seq_data, seq_label))
                                print(f"Added sequence with label {seq_label} ({self.get_class_name(seq_label)})")
                                
                    elif len(keypoints_sequence) > 0:
                        # Pad or truncate sequence to required length
                        seq_data = np.array(keypoints_sequence)
                        if len(seq_data) < self.sequence_length:
                            # Pad with last frame
                            last_frame = seq_data[-1:] if len(seq_data) > 0 else np.zeros((1, 17, 2))
                            padding_needed = self.sequence_length - len(seq_data)
                            padding = np.repeat(last_frame, padding_needed, axis=0)
                            seq_data = np.concatenate([seq_data, padding], axis=0)
                        else:
                            # Truncate to sequence length
                            seq_data = seq_data[:self.sequence_length]
                        
                        seq_label = labels[-1] if labels else 0
                        
                        # Validate label
                        if 0 <= seq_label < self.num_classes:
                            samples.append((seq_data, seq_label))
        
        print(f"Processed {len(samples)} sequences from {len(sample_folders)} samples")
        # Count samples per class
        class_counts = {i: 0 for i in range(self.num_classes)}
        for _, label in samples:
            class_counts[label] += 1
        
        # Print class distribution
        if self.num_classes == 6:
            class_names = ['Standing', 'Falling', 'Sitting', 'Bending', 'Sleeping', 'Walking']
            print("Class distribution:")
            for i, name in enumerate(class_names):
                print(f"  {name}: {class_counts[i]}")
        else:
            print(f"Class distribution: {class_counts}")
        
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        data, label = self.samples[idx]
        
        # Convert to tensor format (C, T, V)
        # data shape: (T, V, C) -> (C, T, V)
        data = torch.FloatTensor(data).permute(2, 0, 1)  # (2, T, 17)
        
        # Add coordinate dimension (x, y, z)
        if data.shape[0] == 2:
            zeros = torch.zeros(1, data.shape[1], data.shape[2])
            data = torch.cat([data, zeros], dim=0)  # (3, T, V)
        
        label = torch.LongTensor([label])
        
        if self.transform:
            data = self.transform(data)
        
        return data, label.squeeze()
    
    def get_class_mapping(self):
        """Return the class mapping dictionary"""
        return self.class_mapping

    def print_class_info(self):
        """Print information about discovered classes"""
        print("Class mapping:")
        for class_name, class_id in self.class_mapping.items():
            print(f"  {class_id}: {class_name}")
    