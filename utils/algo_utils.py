import numpy as np
    
def iou(person_bboxes, bed_bboxes): 
    """Calculate IoU between person and bed bboxes"""
    if not person_bboxes or not bed_bboxes:
        return 0.0
    
    def bbox_area(bbox):
        x1, y1, x2, y2 = bbox
        return max(0, x2 - x1) * max(0, y2 - y1)
    
    max_iou = 0.0
    for p_bbox in person_bboxes:
        for b_bbox in bed_bboxes:
            xA = max(p_bbox[0], b_bbox[0])
            yA = max(p_bbox[1], b_bbox[1])
            xB = min(p_bbox[2], b_bbox[2])
            yB = min(p_bbox[3], b_bbox[3])
            
            interArea = max(0, xB - xA) * max(0, yB - yA)
            if interArea == 0:
                continue
            
            boxAArea = bbox_area(p_bbox)
            boxBArea = bbox_area(b_bbox)
            
            iou = interArea / float(boxAArea + boxBArea - interArea)
            max_iou = max(max_iou, iou)
    
    return max_iou

def fall_forward_check(kpts, upper_half_limit):
    """Check if person is falling forward based on keypoints
    Args:
        kpts: Keypoints of the person
        upper_half_limit: Y-coordinate limit for upper half of bbox
    """
    status = False
    nose_kpt_y = kpts[0][1] # Nose keypoint
    left_ankle_kpt_y = kpts[15][1]  # Left ankle keypoint
    right_ankle_kpt_y = kpts[16][1]  # Right ankle keypoint

    print(f"Nose keypoints vs nose_kpt_y > left_ankle_kpt_y: {nose_kpt_y > left_ankle_kpt_y}")
    print(f"Nose keypoints vs nose_kpt_y > right_ankle_kpt_y: {nose_kpt_y > right_ankle_kpt_y}")
    print(f"left_ankle_kpt_y < upper_half_limit: {left_ankle_kpt_y < upper_half_limit}")
    print(f"right_ankle_kpt_y < upper_half_limit: {right_ankle_kpt_y < upper_half_limit}")

    if nose_kpt_y > left_ankle_kpt_y and nose_kpt_y > right_ankle_kpt_y and left_ankle_kpt_y < upper_half_limit and right_ankle_kpt_y < upper_half_limit:
        status = True  # person is upside down 

    return status

def fall_backward_check(kpts, backward_ratio_threshold=0.91):
    """Check if person is falling backward based on keypoints
    Args:
        kpts: Keypoints of the person 
    """
    status = False
    lShoulder_y = kpts[5][1]  # Left shoulder keypoint
    rShoulder_y = kpts[6][1]  # Right shoulder keypoint
    lHip_y = kpts[7][1]  # Left hip keypoint
    rHip_y = kpts[8][1]  # Right hip keypoint

    upper_body_ratio = abs(((lShoulder_y + rShoulder_y) / 2 ) / ((lHip_y + rHip_y) / 2 + 1e-5))
    print(f"Fall Backward Check - Upper body ratio: {upper_body_ratio:.3f}")

    if upper_body_ratio > backward_ratio_threshold:
        status = True  # person is falling backward

    return status

def calculate_bbox_ratio(person_bboxes, person_scores, keypoints_list, kpts_scores_list, 
                            backward_ratio_threshold=0.91, keypoint_threshold=0.3): 
    """Calculate ratio of person bbox (w:h)"""
    if not person_bboxes:
        return [], []
    
    person_area_ratio = []

    for bbox, keypoints, kpts_scores in zip(person_bboxes, keypoints_list, kpts_scores_list):
        x1, y1, x2, y2 = bbox
        width = x2 - x1 # x-axis
        height = y2 - y1 # y-axis
        half_height = height / 2
        upper_half_limit = y1 + half_height
        ratio = width / height if height > 0 else 0

        kpts_status = [True if score > keypoint_threshold else False for score in kpts_scores]
        print(f"Number of valid keypoints: {np.array(kpts_status).sum()}")
        if np.array(kpts_status).sum() < 14:
            print("Invalid keypoints detected, skipping fall checks")
            ratio = -1 # < 0 to indicate invalid keypoints

        elif fall_forward_check(keypoints, upper_half_limit) and ratio > 0.5:
            ratio = 5 # blow up ratio to indicate fall forward

        elif fall_backward_check(keypoints, backward_ratio_threshold) and ratio > 0.5:
            ratio = 6 # blow up ratio to indicate fall backward

        person_area_ratio.append(ratio) 

    return person_area_ratio, person_scores