import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pickle
from collections import defaultdict
import gc
import re

class COCOFallDetectionDataset(Dataset):
    """Dataset for COCO format fall detection with multi-class temporal labels"""
    
    def __init__(self, dataset_path, seq_len=30, stride=1, cache_data=True, 
                 label_strategy='transition'):
        self.dataset_path = dataset_path
        self.seq_len = seq_len
        self.stride = stride
        self.cache_data = cache_data
        self.label_strategy = label_strategy  # 'majority', 'transition', 'weighted'
        
        # COCO keypoint connections
        self.coco_skeleton = [
            [16, 14], [14, 12], [17, 15], [15, 13], [12, 13],  # legs
            [6, 12], [7, 13], [6, 7],                          # torso
            [6, 8], [7, 9], [8, 10], [9, 11],                 # arms
            [2, 3], [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]  # head
        ]
        
        # Class mapping - matches your JSON categories
        self.class_to_idx = {
            'walking': 0,
            'sitting': 1, 
            'standing': 2,
            'falling': 3,
            'bending_down': 4,
            'crouching': 5,
            'waving_hands': 6,
            'sleeping': 7
        }
        
        # Comprehensive class name variations mapping
        self.class_name_variations = {
            # Walking variations
            'walking': 'walking',
            'walk': 'walking',
            'walks': 'walking',
            'exercising': 'walking',
            
            # Sitting variations
            'sitting': 'sitting',
            'sit': 'sitting',
            'sits': 'sitting',
            'seated': 'sitting',
            'sit down': 'sitting',
            'drinking': 'sitting',
            'eating': 'sitting',
            'reading': 'sitting',
            'writing': 'sitting',
            
            # Standing variations
            'standing': 'standing',
            'stand': 'standing',
            'stands': 'standing',
            'upright': 'standing',
            
            # Falling variations
            'falling': 'falling',
            'fall': 'falling',
            'falls': 'falling',
            'fell': 'falling',
            'fallen': 'falling',
            'fall_down': 'falling',
            'fall down': 'falling',
            'falling down': 'falling',
            'falling_down': 'falling',
            'slipping': 'falling',
            'falldown': 'falling',
            
            # Bending down variations
            'bending_down': 'bending_down',
            'bending down': 'bending_down',
            'bend_down': 'bending_down',
            'bend down': 'bending_down',
            'bending': 'bending_down',
            'bend': 'bending_down',
            'stooping': 'bending_down',
            'bends': 'bending_down',
            'bend over': 'bending_down',
            
            # Crouching variations
            'crouching': 'crouching',
            'crouch': 'crouching',
            'crouches': 'crouching',
            'squatting': 'crouching',
            'squat': 'crouching',
            'tie shoe laces': 'crouching',
            'tie_shoe_laces': 'crouching',
            'tie_shoelaces': 'crouching',
            'wearing shoes': 'crouching',
            'wearing_shoes': 'crouching',
            'wear shoes': 'crouching',
            'Wearing/fixing shoes': 'crouching',
            'wearing/fixing_shoes': 'crouching',
            
            # Waving hands variations
            'waving_hands': 'waving_hands',
            'waving hands': 'waving_hands',
            'wave_hands': 'waving_hands',
            'wave hands': 'waving_hands',
            'waving': 'waving_hands',
            'wave': 'waving_hands',
            'hand_waving': 'waving_hands',
            'hand waving': 'waving_hands',
            'wave ones hand': 'waving_hands',  
            'wave ones hands': 'waving_hands',
            'wave_ones_hand': 'waving_hands',   
            'wave_ones_hands': 'waving_hands',

            # Sleeping variations
            'sleeping': 'sleeping',
            'sleep': 'sleeping',
            'sleeps': 'sleeping',
            'lying_down': 'sleeping',
            'lying down': 'sleeping',
            'lying': 'sleeping',
            'lie': 'sleeping',
            'lies': 'sleeping'
        }
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}
        self.num_classes = len(self.class_to_idx)
        
        # Category ID to class mapping (will be populated from JSON)
        self.category_id_to_class = {}

        self.validation_stats = {
            'total_videos_processed': 0,
            'videos_with_valid_sequences': 0,
            'sequences_created': 0,
            'sequences_with_valid_labels': 0,
            'invalid_label_count': 0,
            'empty_sequence_count': 0
        }
        
        # Load and process dataset
        self.sequences = []
        self.labels = []
        self.sequence_info = []  # Store metadata for analysis
        self._load_dataset()

        self._validate_dataset_integrity()
        
        print(f"📊 Loaded {len(self.sequences)} sequences")
        print(f"🏷️  Classes: {list(self.class_to_idx.keys())}")
        print(f"📈 Label strategy: {self.label_strategy}")
        
    def _load_dataset(self):
        """Load all video data and create temporal sequences"""
        video_folders = [f for f in os.listdir(self.dataset_path) 
                        if os.path.isdir(os.path.join(self.dataset_path, f))]
        
        for video_idx, video_folder in enumerate(video_folders):
            self.validation_stats['total_videos_processed'] += 1
            try:
                video_path = os.path.join(self.dataset_path, video_folder)
                annotations_path = os.path.join(video_path, 'annotations')
                
                if not os.path.exists(annotations_path):
                    print(f"⚠️  No annotations folder in {video_folder}")
                    continue
                    
                # Find annotation file
                json_files = [f for f in os.listdir(annotations_path) if f.endswith('.json')]
                if not json_files:
                    print(f"⚠️  No JSON annotations in {video_folder}")
                    continue
                    
                annotation_file = os.path.join(annotations_path, json_files[0])
                sequences, labels, seq_info = self._process_video(annotation_file, video_folder)
                
                if sequences and labels and len(sequences) == len(labels):
                    self.sequences.extend(sequences)
                    self.labels.extend(labels)
                    self.sequence_info.extend(seq_info)
                    self.validation_stats['videos_with_valid_sequences'] += 1
                    self.validation_stats['sequences_created'] += len(sequences)
                else:
                    print(f"⚠️  Video {video_folder} produced invalid sequences: "
                          f"sequences={len(sequences) if sequences else 0}, "
                          f"labels={len(labels) if labels else 0}")
                
                # Progress update
                if (video_idx + 1) % 10 == 0:
                    print(f"📈 Processed {video_idx + 1}/{len(video_folders)} videos, "
                          f"Created {len(self.sequences)} sequences so far")
                
            except Exception as e:
                print(f"❌ Error processing video {video_folder}: {e}")
                import traceback
                traceback.print_exc()
                continue
                
            # Memory cleanup
            if video_idx % 20 == 0:
                gc.collect()
    
    def _process_video(self, annotation_file, video_name):
        """Process single video and extract temporal sequences"""
        with open(annotation_file, 'r') as f:
            coco_data = json.load(f)
        
        # Validate required fields
        if 'categories' not in coco_data:
            print(f"⚠️  No categories in {video_name}")
            return [], [], []
        
        if 'annotations' not in coco_data:
            print(f"⚠️  No annotations in {video_name}")
            return [], [], []
        
        if 'images' not in coco_data:
            print(f"⚠️  No images in {video_name}")
            return [], [], []
        
        # Build category mapping from JSON with normalization
        categories = coco_data.get('categories', [])
        valid_category_count = 0
        
        for cat in categories:
            original_name = cat['name']
            normalized_name = self._normalize_class_name(original_name)
            
            if normalized_name in self.class_to_idx:
                self.category_id_to_class[cat['id']] = self.class_to_idx[normalized_name]
                valid_category_count += 1
            else:
                print(f"⚠️  Unknown class in {video_name}: '{original_name}' -> '{normalized_name}' (category_id: {cat['id']})")
        
        if valid_category_count == 0:
            print(f"❌ No valid categories found in {video_name}")
            return [], [], []
        
        # Extract keypoints and labels per frame
        frame_keypoints = defaultdict(list)
        frame_labels = {}
        
        # Create image_id to frame_num mapping
        image_id_to_frame = {}
        for img in coco_data.get('images', []):
            frame_num = self._extract_frame_number(img['file_name'])
            image_id_to_frame[img['id']] = frame_num

        # Process annotations with validation
        valid_annotations = 0
        for ann in coco_data.get('annotations', []):
            image_id = ann['image_id']
            
            if image_id not in image_id_to_frame:
                continue
                
            frame_num = image_id_to_frame[image_id]
            
            # Extract keypoints (COCO format: [x1, y1, v1, x2, y2, v2, ...])
            keypoints = ann.get('keypoints', [])
            if len(keypoints) == 51:  # 17 keypoints * 3 (x, y, visibility)
                kpts = np.array(keypoints).reshape(17, 3)
                frame_keypoints[frame_num].append(kpts)
                valid_annotations += 1
            
            # Extract category-based label for this frame
            category_id = ann.get('category_id')
            if category_id in self.category_id_to_class:
                frame_labels[frame_num] = self.category_id_to_class[category_id]
        
        if valid_annotations == 0:
            print(f"❌ No valid annotations with keypoints in {video_name}")
            return [], [], []
        
        print(f"✅ {video_name}: {valid_annotations} valid annotations, "
              f"{len(frame_keypoints)} frames with keypoints, "
              f"{len(frame_labels)} frames with labels")
        
        return self._create_sequences(frame_keypoints, frame_labels, video_name)
    
    def _create_sequences(self, frame_keypoints, frame_labels, video_name):
        """Create sequences from frame data"""
        sequences = []
        labels = []
        seq_info = []
        
        sorted_frames = sorted(frame_keypoints.keys())
        if len(sorted_frames) < self.seq_len:
            return sequences, labels, seq_info
        
        for i in range(0, len(sorted_frames) - self.seq_len + 1, self.stride):
            frame_range = sorted_frames[i:i + self.seq_len]
            
            # Build sequence
            sequence = []
            sequence_labels = []
            valid_frames_count = 0
            
            for frame_num in frame_range:
                if frame_num in frame_keypoints and len(frame_keypoints[frame_num]) > 0:
                    kpts = frame_keypoints[frame_num][0]
                    sequence.append(kpts)
                    valid_frames_count += 1
                    
                    if frame_num in frame_labels:
                        sequence_labels.append(frame_labels[frame_num])
                    else:
                        sequence_labels.append(1)  # Default to 'sitting'
                else:
                    sequence.append(np.zeros((17, 3)))
                    sequence_labels.append(1)  # Default to 'sitting'
            
            # Validation: Require minimum valid frames
            min_valid_frames = max(5, self.seq_len // 3)  # At least 1/3 of frames must be valid
            if valid_frames_count < min_valid_frames:
                self.validation_stats['empty_sequence_count'] += 1
                continue

            if len(sequence) == self.seq_len:
                sequence = np.array(sequence)
                
                try: 
                    # Apply labeling strategy
                    sequence_label, info = self._apply_sequence_labeling(sequence_labels, frame_range)
                    
                    # Validate label is in correct range
                    if not isinstance(sequence_label, int) or sequence_label < 0 or sequence_label >= self.num_classes:
                        print(f"❌ Invalid label {sequence_label} for {self.num_classes} classes in {video_name}, skipping sequence")
                        continue
                    
                    sequences.append(sequence)
                    labels.append(sequence_label)
                    seq_info.append({
                        'video_name': video_name,
                        'frame_range': frame_range,
                        'frame_labels': sequence_labels.copy(),
                        'sequence_label': sequence_label,
                        'valid_frames_count': valid_frames_count,
                        **info
                    })
                    self.validation_stats['sequences_with_valid_labels'] += 1
                except Exception as e:
                    print(f"❌ Error in sequence labelling for {video_name} frames {frame_range}: {e}")
                    continue
        
        return sequences, labels, seq_info
    def _validate_dataset_integrity(self):
        """Validate dataset integrity after loading"""
        print(f"\n🔍 Dataset Integrity Validation:")
        print(f"   Total videos processed: {self.validation_stats['total_videos_processed']}")
        print(f"   Videos with valid sequences: {self.validation_stats['videos_with_valid_sequences']}")
        print(f"   Total sequences created: {self.validation_stats['sequences_created']}")
        print(f"   Sequences with valid labels: {self.validation_stats['sequences_with_valid_labels']}")
        print(f"   Invalid labels filtered: {self.validation_stats['invalid_label_count']}")
        print(f"   Empty sequences filtered: {self.validation_stats['empty_sequence_count']}")
        
        # Critical checks
        if len(self.sequences) == 0:
            raise ValueError("❌ No valid sequences created! Check your dataset and annotations.")
        
        if len(self.sequences) != len(self.labels):
            raise ValueError(f"❌ Sequence-label mismatch: {len(self.sequences)} sequences vs {len(self.labels)} labels")
        
        # Validate all labels are in correct range
        invalid_labels = [label for label in self.labels if label < 0 or label >= self.num_classes]
        if invalid_labels:
            raise ValueError(f"❌ Found {len(invalid_labels)} invalid labels: {set(invalid_labels)}")
        
        # Check for minimum dataset size
        if len(self.sequences) < 10:
            print(f"⚠️  Very small dataset: only {len(self.sequences)} sequences")
        
        print(f"✅ Dataset integrity validated: {len(self.sequences)} valid sequences")
    
    def _apply_sequence_labeling(self, sequence_labels, frame_range):
        """Apply different strategies to assign one label to a sequence of frame labels"""
        info = {}
        
        # Analyze the sequence
        unique_labels = list(set(sequence_labels))
        info['unique_labels'] = unique_labels
        info['has_transition'] = len(unique_labels) > 1
        
        # Detect transitions
        transitions = []
        for i in range(1, len(sequence_labels)):
            if sequence_labels[i] != sequence_labels[i-1]:
                transitions.append((sequence_labels[i-1], sequence_labels[i], i))
        
        info['transitions'] = transitions
        info['num_transitions'] = len(transitions)
        
        if self.label_strategy == 'majority':
            label = self._majority_vote(sequence_labels)
            return label, info  # Return tuple
            
        elif self.label_strategy == 'transition':
            label = self._transition_aware_labeling(sequence_labels, transitions, info)
            return label, info  # Return tuple
            
        elif self.label_strategy == 'weighted':
            label = self._weighted_labeling(sequence_labels, info)
            return label, info  # Return tuple
            
        else:
            label = self._majority_vote(sequence_labels)
            return label, info  # Return tuple

    def _majority_vote(self, sequence_labels):
        """Simple majority vote for sequence label"""
        if not sequence_labels:
            return 0
        return max(set(sequence_labels), key=sequence_labels.count)

    def _transition_aware_labeling(self, sequence_labels, transitions, info):
        """Updated transition labeling for 8 classes"""
        
        # Critical transitions for fall detection
        critical_transitions = [
            (2, 3),  # standing -> falling
            (0, 3),  # walking -> falling
            (1, 3),  # sitting -> falling
            (3, 7),  # falling -> sleeping (lying down)
            (4, 3),  # bending_down -> falling
            (5, 3),  # crouching -> falling
        ]
        
        # Check for critical transitions
        for transition in transitions:
            trans_type = (transition[0], transition[1])
            if trans_type in critical_transitions:
                info['has_critical_transition'] = True
                info['critical_transition_type'] = trans_type
                
                if transition[1] == 3:  # Transition TO falling
                    return 3
                elif transition[0] == 3:  # Transition FROM falling
                    fall_count = sequence_labels.count(3)
                    if fall_count >= self.seq_len * 0.2:
                        return 3
        
        # Prioritize falling if present
        if 3 in sequence_labels:
            fall_ratio = sequence_labels.count(3) / len(sequence_labels)
            if fall_ratio >= 0.15:
                return 3
        
        return self._weighted_labeling(sequence_labels, info)
    
    def _weighted_labeling(self, sequence_labels, info):
        """Updated weighted labeling for 8 classes"""
        
        label_counts = {}
        for label in sequence_labels:
            label_counts[label] = label_counts.get(label, 0) + 1
        
        # Activity importance weights
        importance_weights = {
            0: 1.5,   # walking - dynamic activity
            1: 1.0,   # sitting - baseline
            2: 1.2,   # standing - neutral
            3: 4.0,   # falling - most critical!
            4: 1.8,   # bending_down - transition state
            5: 1.6,   # crouching - unstable position
            6: 1.3,   # waving_hands - minor activity
            7: 2.5,   # sleeping - important outcome state
        }
        
        # Calculate weighted scores
        weighted_scores = {}
        for label, count in label_counts.items():
            weight = importance_weights.get(label, 1.0)
            weighted_scores[label] = count * weight
        
        best_label = max(weighted_scores.items(), key=lambda x: x[1])[0]
        info['weighted_scores'] = weighted_scores
        
        return best_label
    
    def _normalize_class_name(self, class_name):
        """Enhanced normalize class name to handle variations including apostrophes"""
        # Convert to lowercase and strip whitespace
        normalized = class_name.lower().strip()
        
        # Handle punctuation and separators
        normalized = normalized.replace("'", "")  # Remove apostrophes
        normalized = normalized.replace('"', "")  # Remove quotes
        normalized = normalized.replace(' ', '_').replace('-', '_')
        normalized = normalized.replace('__', '_')  # Remove double underscores
        
        # Handle plural/singular variations
        plural_mappings = {
            'hands': 'hand',
            'waves': 'wave',
            'falls': 'fall',
            'sits': 'sit',
            'stands': 'stand',
            'walks': 'walk',
            'bends': 'bend',
            'crouches': 'crouch',
            'sleeps': 'sleep',
            'lies': 'lie'
        }
        
        # Apply plural-to-singular conversion
        words = normalized.split('_')
        normalized_words = []
        for word in words:
            if word in plural_mappings:
                normalized_words.append(plural_mappings[word])
            else:
                normalized_words.append(word)
        normalized = '_'.join(normalized_words)
        
        # Direct mapping lookup
        if normalized in self.class_name_variations:
            return self.class_name_variations[normalized]
        
        # Fuzzy matching for complex cases
        return self._fuzzy_class_match(normalized)

    def _fuzzy_class_match(self, normalized):
        """Fuzzy matching for complex class name variations"""
        # Keywords for each class
        class_keywords = {
            'waving_hands': ['wave', 'hand', 'waving'],
            'falling': ['fall', 'falling', 'slip', 'fell'],
            'sitting': ['sit', 'sitting', 'seated'],
            'standing': ['stand', 'standing', 'upright'],
            'walking': ['walk', 'walking'],
            'bending_down': ['bend', 'bending', 'stoop'],
            'crouching': ['crouch', 'crouching', 'squat'],
            'sleeping': ['sleep', 'sleeping', 'lying', 'lie']
        }
        
        normalized_words = set(normalized.split('_'))
        
        # Find best match based on keyword overlap
        best_match = None
        max_overlap = 0
        
        for class_name, keywords in class_keywords.items():
            overlap = len(normalized_words.intersection(set(keywords)))
            if overlap > max_overlap and overlap > 0:
                max_overlap = overlap
                best_match = class_name
        
        return best_match if best_match else normalized
    
    def _extract_frame_number(self, filename):
        """Extract frame number from filename"""
        if not filename:
            return 0
        
        # Try specific patterns in order of preference
        patterns = [
            r'frame_(\d+)',      # frame_123
            r'_frame_(\d+)',     # _frame_123  
            r'frame(\d+)',       # frame123
            r'img_(\d+)',        # img_123
            r'image_(\d+)',      # image_123
            r'_(\d+)\.',         # _123.jpg (number before extension)
            r'(\d+)\.',          # 123.jpg (last resort - number before extension)
        ]
        
        for pattern in patterns:
            match = re.search(pattern, str(filename))
            if match:
                return int(match.group(1))
        
        # Fallback: extract all numbers and take the last one
        numbers = re.findall(r'\d+', str(filename))
        if numbers:
            return int(numbers[-1])  # Take the last number found
        
        return 0  # Ultimate fallback
    
    def get_class_distribution(self):
        """Get distribution of classes in dataset"""
        from collections import Counter
        return Counter(self.labels)
    
    def get_transition_analysis(self):
        """Analyze transitions in the dataset"""
        transition_stats = {
            'total_sequences': len(self.sequence_info),
            'sequences_with_transitions': 0,
            'sequences_with_falls': 0,
            'transition_types': {},
            'critical_transitions': 0
        }
        
        for info in self.sequence_info:
            if info.get('has_transition', False):
                transition_stats['sequences_with_transitions'] += 1
                
                for transition in info.get('transitions', []):
                    trans_type = (transition[0], transition[1])
                    if trans_type not in transition_stats['transition_types']:
                        transition_stats['transition_types'][trans_type] = 0
                    transition_stats['transition_types'][trans_type] += 1
            
            if info.get('has_critical_transition', False):
                transition_stats['critical_transitions'] += 1
            
            if 3 in info.get('frame_labels', []):
                transition_stats['sequences_with_falls'] += 1
        
        return transition_stats
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        if idx >= len(self.sequences) or idx >= len(self.labels):
            raise IndexError(f"Index {idx} out of range for dataset size {len(self.sequences)}")
        
        sequence = self.sequences[idx]  # Shape: (T, V, C)
        label = self.labels[idx]

        if not isinstance(label, int) or label < 0 or label >= self.num_classes:
            print(f"❌ Invalid label {label} at index {idx}, using fallback")
            label = 1  # Fallback to sitting
        
        # Convert to tensor and rearrange to (C, T, V, M)
        # M=1 for single person, C=3 for (x, y, confidence)
        sequence = torch.FloatTensor(sequence)
        sequence = sequence.permute(2, 0, 1) # (C, T, V, M)
        
        # Ensure correct dimensions
        assert sequence.shape[0] == 3, f"Expected 3 channels, got {sequence.shape[0]}"
        assert sequence.shape[2] == 17, f"Expected 17 keypoints, got {sequence.shape[2]}"

        # Normalize coordinates
        sequence = self._normalize_keypoints(sequence)
        
        return sequence, label
    
    def _normalize_keypoints(self, keypoints):
        """Normalize keypoints for RK3588 model - optimized version"""
        # Extract x, y coordinates (ignore confidence for normalization)
        x_coords = keypoints[0]  # (T, V)
        y_coords = keypoints[1]  # (T, V)
        confidence = keypoints[2]  # (T, V)
        
        eps = 1e-8  # Small epsilon for numerical stability
        
        # Normalize each frame independently
        for t in range(x_coords.shape[0]):
            # Get valid keypoints (confidence > 0.3 for RK3588 - more lenient)
            valid_mask = confidence[t] > 0.3
            
            if valid_mask.any():
                # Normalize x coordinates
                x_valid = x_coords[t][valid_mask]
                if len(x_valid) > 1:  # Need at least 2 points for range
                    x_min, x_max = x_valid.min(), x_valid.max()
                    x_range = x_max - x_min
                    if x_range > eps:
                        # Normalize to [0, 1] range for better RK3588 performance
                        x_coords[t] = (x_coords[t] - x_min) / (x_range + eps)
                    else:
                        x_coords[t] = 0.5  # Center if no range
                
                # Normalize y coordinates
                y_valid = y_coords[t][valid_mask]
                if len(y_valid) > 1:
                    y_min, y_max = y_valid.min(), y_valid.max()
                    y_range = y_max - y_min
                    if y_range > eps:
                        # Normalize to [0, 1] range for better RK3588 performance
                        y_coords[t] = (y_coords[t] - y_min) / (y_range + eps)
                    else:
                        y_coords[t] = 0.5  # Center if no range
            else:
                # No valid keypoints - set to center
                x_coords[t] = 0.5
                y_coords[t] = 0.5
        
        # Clip confidence values to ensure stability
        keypoints[2] = torch.clamp(confidence, 0.0, 1.0)
        
        return keypoints

def create_data_loaders(dataset_path, batch_size=16, seq_len=30, stride=1, 
                       train_split=0.8, num_workers=2, label_strategy='transition'):
    """Create train and validation data loaders"""
    
    # Load full dataset
    full_dataset = COCOFallDetectionDataset(
        dataset_path=dataset_path,
        seq_len=seq_len,
        stride=stride,
        label_strategy=label_strategy
    )
    
    # Print dataset statistics
    print("\n📊 Dataset Statistics:")
    print(f"Class distribution: {full_dataset.get_class_distribution()}")
    print(f"Transition analysis: {full_dataset.get_transition_analysis()}")
    
    # Stratified split to maintain class distribution
    try:
        from sklearn.model_selection import train_test_split
        
        indices = list(range(len(full_dataset)))
        labels = full_dataset.labels
        
        # Check if we have enough samples for stratification
        unique_labels = set(labels)
        min_class_size = min([labels.count(label) for label in unique_labels])
        
        if min_class_size >= 2:
            train_indices, val_indices = train_test_split(
                indices, 
                test_size=1-train_split, 
                stratify=labels, 
                random_state=42
            )
        else:
            print("⚠️  Not enough samples per class for stratification, using random split")
            train_size = int(train_split * len(full_dataset))
            train_indices = indices[:train_size]
            val_indices = indices[train_size:]
        
        # Create subset datasets
        train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
        val_dataset = torch.utils.data.Subset(full_dataset, val_indices)

    except ImportError:
        # Fallback to simple split if sklearn not available
        dataset_size = len(full_dataset)
        train_size = int(train_split * dataset_size)
        
        train_dataset, val_dataset = torch.utils.data.random_split(
            full_dataset, [train_size, dataset_size - train_size]
        )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False
    )
    
    return train_loader, val_loader, full_dataset.num_classes

# Memory optimization utilities
def cleanup_memory():
    """Clean up memory"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def optimize_dataloader_memory(loader):
    """Optimize dataloader for memory efficiency"""
    if hasattr(loader.dataset, 'dataset'):  # For Subset
        if hasattr(loader.dataset.dataset, 'cache_data'):
            loader.dataset.dataset.cache_data = False
    elif hasattr(loader.dataset, 'cache_data'):
        loader.dataset.cache_data = False
    return loader

def create_balanced_sampler(dataset, strategy='weighted'):
    """Create sampler for handling class imbalance"""
    from torch.utils.data import WeightedRandomSampler
    
    # Get labels (handle Subset case)
    if hasattr(dataset, 'indices'):  # Subset dataset
        labels = [dataset.dataset.labels[i] for i in dataset.indices]
    else:
        labels = dataset.labels
    
    # Calculate class weights
    class_counts = torch.bincount(torch.tensor(labels))
    class_weights = 1.0 / class_counts.float()
    
    # Boost critical classes
    if strategy == 'fall_aware':
        class_weights[3] *= 3.0  # Extra boost for falling class
        class_weights[4] *= 1.5  # Slight boost for lying
    
    # Create sample weights
    sample_weights = [class_weights[label] for label in labels]
    
    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )