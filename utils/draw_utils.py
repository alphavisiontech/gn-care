import cv2
import numpy as np

def display_fall_detection_result(frame, fall_result, consecutive_falls, fall_alert_threshold, frame_count=None):
    """Helper function to display fall detection results on frame with consecutive fall tracking"""
    if not fall_result:
        return frame, consecutive_falls
    
    # Print fall result for debugging
    print(fall_result)
    
    # Determine action text and color
    confidence = fall_result.get('human_det_confidence', 0.0)
    fall_text = f"Action: {fall_result['predicted_class']} ({confidence:.2f})"
    color = (0, 255, 255) if fall_result['predicted_class'] == 'fall' else (0, 255, 0)
    
    # Display action text
    cv2.putText(frame, fall_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    
    # Fall alert logic
    if fall_result['is_fall'] == True:
        consecutive_falls += 1
        if consecutive_falls >= fall_alert_threshold:
            cv2.putText(frame, "FALL ALERT!", (10, 100), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
            print(f"🚨 FALL DETECTED! Human Detection Confidence: {confidence:.2f}")

            # Flash border effect for video streams
            if frame_count is not None and frame_count % 10 < 5:
                cv2.rectangle(frame, (0, 0), (frame.shape[1]-1, frame.shape[0]-1), 
                             (0, 0, 255), 10)
    else:
        # Reset consecutive falls if not a fall or confidence too low
        consecutive_falls = 0
    
    return frame, consecutive_falls

def add_fps_info(frame, fps_value, frame_count):
    """Add FPS and frame count information to frame"""
    cv2.putText(frame, f"FPS: {fps_value:.1f}", (frame.shape[1] - 150, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    # cv2.putText(frame, f"Frame: {frame_count}", (frame.shape[1] - 150, 60), 
               #cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return frame

def add_performance_info(frame, processing_times, queue_sizes, config_queue_size):
    """Add performance monitoring information to frame"""
    if processing_times:
        avg_time = sum(processing_times) / len(processing_times)
        cv2.putText(frame, f"Avg Process: {avg_time*1000:.1f}ms", 
                   (10, frame.shape[0] - 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Display queue sizes
    queue_text = f"Queues - I:{queue_sizes['input']} D:{queue_sizes['detection']} P:{queue_sizes['pose']} R:{queue_sizes['result']}"
    cv2.putText(frame, queue_text, (10, frame.shape[0] - 80), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Calculate total queue load
    total_queue_items = sum(queue_sizes.values())
    max_queue_items = config_queue_size * 4  # 4 queues
    load_percentage = (total_queue_items / max_queue_items) * 100
    
    color = (0, 255, 0) if load_percentage < 50 else (0, 255, 255) if load_percentage < 80 else (0, 0, 255)
    cv2.putText(frame, f"Queue Load: {load_percentage:.1f}%", 
               (10, frame.shape[0] - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    return frame

def create_side_by_side_display(left_frame, right_frame):
    """Create a side-by-side display of two frames"""
    # Add background rectangles for better label visibility
    cv2.rectangle(left_frame, (5, 5), (300, 35), (0, 0, 0), -1)
    cv2.rectangle(right_frame, (5, 5), (200, 35), (0, 0, 0), -1)
    
    # Add labels to distinguish the frames
    cv2.putText(left_frame, "Original + Skeleton", (10, 25), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(right_frame, "Skeleton Only", (10, 25), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Add a vertical separator line
    separator = np.zeros((left_frame.shape[0], 3, 3), dtype=np.uint8)
    separator[:, :, :] = [255, 255, 255]  # White separator
    
    # Combine frames horizontally with separator
    combined_frame = np.hstack((left_frame, separator, right_frame))
    return combined_frame
