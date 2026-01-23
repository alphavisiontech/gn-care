import argparse
import time
from typing import List, Tuple

import cv2
import numpy as np
import onnxruntime as ort

def preprocess_rtmpose(
    img: np.ndarray, input_size: Tuple[int, int] = (192, 256), bboxes: List[np.ndarray] = None
) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    """Do preprocessing for RTMPose model inference for multiple people in a single image.
    
    This function takes a single image and crops/preprocesses regions for each person
    based on the provided bounding boxes.

    Args:
        img (np.ndarray): Input image containing multiple people.
        input_size (tuple): Input image size in shape (w, h).
        bboxes (List[np.ndarray], optional): List of bounding boxes in format (x1, y1, x2, y2).
                                           Each bbox defines a person's location in the image.
                                           If None, processes entire image as single person.

    Returns:
        tuple:
        - resized_imgs (List[np.ndarray]): List of preprocessed image crops for each person.
        - centers (List[np.ndarray]): List of centers for each person's bbox.
        - scales (List[np.ndarray]): List of scales for each person's bbox.
    """
    resized_imgs = []
    centers = []
    scales = []
    
    # If no bboxes provided, use entire image as single person (original behavior)
    if bboxes is None or len(bboxes) == 0:
        img_shape = img.shape[:2]
        bboxes = [np.array([0, 0, img_shape[1], img_shape[0]])]
    
    # Process each bounding box
    for bbox in bboxes:
        # get center and scale
        center, scale = bbox_xyxy2cs(bbox, padding=1.25)

        # do affine transformation
        resized_img, scale = top_down_affine(input_size, scale, center, img)
        print(f"Resized image shape: {resized_img.shape}, Center: {center}, Scale: {scale}")

        # normalize image
        mean = np.array([123.675, 116.28, 103.53])
        std = np.array([58.395, 57.12, 57.375])
        resized_img = (resized_img - mean) / std
        print(f"Normalized image stats - max: {resized_img.max():.3f}, min: {resized_img.min():.3f}")

        resized_imgs.append(resized_img)
        centers.append(center)
        scales.append(scale)

    return resized_imgs, centers, scales


def build_session(onnx_file: str, device: str = 'cpu') -> ort.InferenceSession:
    """Build onnxruntime session.

    Args:
        onnx_file (str): ONNX file path.
        device (str): Device type for inference.

    Returns:
        sess (ort.InferenceSession): ONNXRuntime session.
    """
    providers = ['CPUExecutionProvider'
                 ] if device == 'cpu' else ['CUDAExecutionProvider']
    sess = ort.InferenceSession(path_or_bytes=onnx_file, providers=providers)

    return sess


def inference_multiple(sess: ort.InferenceSession, imgs: List[np.ndarray]) -> List[List[np.ndarray]]:
    """Inference RTMPose model for multiple people from a single image.
    
    This function processes multiple cropped regions (bounding boxes) from the same image,
    where each cropped region contains one person.

    Args:
        sess (ort.InferenceSession): ONNXRuntime session.
        imgs (List[np.ndarray]): List of preprocessed image crops for each person from the same image.

    Returns:
        List[List[np.ndarray]]: List of outputs for each person.
    """
    all_outputs = []
    
    # Process each person's cropped image region
    for img_crop in imgs:
        # build input - transpose to CHW format for ONNX model
        input_data = [img_crop.transpose(2, 0, 1)]

        # build output
        sess_input = {sess.get_inputs()[0].name: input_data}
        sess_output = []
        for out in sess.get_outputs():
            sess_output.append(out.name)

        # run model inference for this person
        outputs = sess.run(sess_output, sess_input)
        all_outputs.append(outputs)

    return all_outputs


def inference_batch(sess: ort.InferenceSession, imgs: List[np.ndarray]) -> List[List[np.ndarray]]:
    """Batch inference RTMPose model for multiple people from a single image.
    
    This function processes multiple cropped regions in a single batch for better efficiency.
    Note: This requires all input crops to have the same dimensions.

    Args:
        sess (ort.InferenceSession): ONNXRuntime session.
        imgs (List[np.ndarray]): List of preprocessed image crops for each person.

    Returns:
        List[List[np.ndarray]]: List of outputs for each person.
    """
    if not imgs:
        return []
    
    # Stack all images into a batch
    batch_input = np.stack([img.transpose(2, 0, 1) for img in imgs], axis=0)
    batch_input = batch_input.astype(np.float32)  # Ensure float32 for ONNX input
    
    # build output
    sess_input = {sess.get_inputs()[0].name: batch_input}
    sess_output = []
    for out in sess.get_outputs():
        sess_output.append(out.name)

    # run model inference for all people at once
    batch_outputs = sess.run(sess_output, sess_input)
    
    # Split batch outputs back into individual person outputs
    all_outputs = []
    num_people = len(imgs)
    
    for i in range(num_people):
        person_outputs = []
        for output in batch_outputs:
            person_outputs.append(output[i:i+1])  # Keep batch dimension of 1
        all_outputs.append(person_outputs)
    
    return all_outputs


def postprocess_rtmpose_multiple(all_outputs: List[List[np.ndarray]],
                               model_input_size: Tuple[int, int],
                               centers: List[np.ndarray],
                               scales: List[np.ndarray],
                               simcc_split_ratio: float = 2.0
                               ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Postprocess for RTMPose model output for multiple people.

    Args:
        all_outputs (List[List[np.ndarray]]): List of outputs for each person.
        model_input_size (tuple): RTMPose model Input image size.
        centers (List[np.ndarray]): List of centers for each person.
        scales (List[np.ndarray]): List of scales for each person.
        simcc_split_ratio (float): Split ratio of simcc.

    Returns:
        tuple:
        - all_keypoints (List[np.ndarray]): List of rescaled keypoints for each person.
        - all_scores (List[np.ndarray]): List of model predict scores for each person.
    """
    all_keypoints = []
    all_scores = []
    
    for outputs, center, scale in zip(all_outputs, centers, scales):
        # use simcc to decode
        simcc_x, simcc_y = outputs
        print(f"SimCC outputs - X shape: {simcc_x.shape}, Y shape: {simcc_y.shape}")
        print(f"SimCC X stats - max: {simcc_x.max()}, min: {simcc_x.min()}")
        print(f"SimCC Y stats - max: {simcc_y.max()}, min: {simcc_y.min()}")
        print(f"Center: {center}, Scale: {scale}")
        print(f"Model input size: {model_input_size}")

        keypoints, scores = decode(simcc_x, simcc_y, simcc_split_ratio)
        
        print(f"Decoded keypoints shape: {keypoints.shape}, Scores shape: {scores.shape}")
        print(f"Keypoints stats - max: {keypoints.max()}, min: {keypoints.min()}")
        print(f"Scores stats - max: {scores.max()}, min: {scores.min()}")

        # Ensure correct shapes for coordinate transformation
        # keypoints: shape (N, K, 2) where N=1 (single person), K=num_keypoints
        # model_input_size: (w, h) -> shape (2,)
        # center: shape (2,) -> (x, y)
        # scale: shape (2,) -> (w, h)
        
        # Convert model_input_size to numpy array if it's not already
        model_input_size_array = np.array(model_input_size)
        
        # Rescale keypoints from model coordinates to image coordinates
        # keypoints is (1, K, 2), we need to broadcast correctly
        keypoints = keypoints / model_input_size_array * scale + center - scale / 2
        print(f"Rescaled keypoints stats - max: {keypoints.max()}, min: {keypoints.min()}")
        
        # Remove batch dimension for compatibility with visualization
        keypoints = keypoints.squeeze(0)  # Shape: (K, 2)
        scores = scores.squeeze(0)  # Shape: (K,)
        
        all_keypoints.append(keypoints)
        all_scores.append(scores)

    return all_keypoints, all_scores


def process_multiple_people(img: np.ndarray, 
                          sess: ort.InferenceSession,
                          bboxes: List[np.ndarray] = None,
                          input_size: Tuple[int, int] = (192, 256),
                          simcc_split_ratio: float = 2.0,
                          use_batch_inference: bool = False) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Complete pipeline to process multiple people in a single image.

    Args:
        img (np.ndarray): Input image containing multiple people.
        sess (ort.InferenceSession): ONNXRuntime session.
        bboxes (List[np.ndarray], optional): List of bounding boxes for each person in format (x1, y1, x2, y2).
                                           If None, processes entire image as single person.
        input_size (tuple): Input image size in shape (w, h).
        simcc_split_ratio (float): Split ratio of simcc.
        use_batch_inference (bool): Whether to use batch inference for better efficiency.
                                   Only works if all crops have same dimensions.

    Returns:
        tuple:
        - all_keypoints (List[np.ndarray]): List of keypoints for each person in the image.
        - all_scores (List[np.ndarray]): List of scores for each person in the image.
    """
    # Preprocessing - crop and preprocess regions for each person
    resized_imgs, centers, scales = preprocess_rtmpose(img, input_size, bboxes)
    
    # Inference - process all person crops
    if use_batch_inference and len(resized_imgs) > 1:
        try:
            all_outputs = inference_batch(sess, resized_imgs)
        except Exception as e:
            print(f"Batch inference failed, falling back to sequential: {e}")
            all_outputs = inference_multiple(sess, resized_imgs)
    else:
        all_outputs = inference_multiple(sess, resized_imgs)
    
    # Postprocessing - convert model outputs to keypoints for each person
    all_keypoints, all_scores = postprocess_rtmpose_multiple(
        all_outputs, input_size, centers, scales, simcc_split_ratio
    )
    
    return all_keypoints, all_scores


def visualize_keypoints(img: np.ndarray,
              keypoints: List[np.ndarray],
              scores: List[np.ndarray],
              thr=0.3) -> np.ndarray:
    """Visualize the keypoints and skeleton on image for multiple people.

    Args:
        img (np.ndarray): Input image in shape.
        keypoints (List[np.ndarray]): List of keypoints for each person in image.
        scores (List[np.ndarray]): List of model predict scores for each person.
        thr (float): Threshold for visualize.

    Returns:
        img (np.ndarray): Visualized image.
    """
    # default color
    skeleton = [(15, 13), (13, 11), (16, 14), (14, 12), (11, 12), (5, 11),
                (6, 12), (5, 6), (5, 7), (6, 8), (7, 9), (8, 10), (1, 2),
                (0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6), (15, 17),
                (15, 18), (15, 19), (16, 20), (16, 21), (16, 22), (91, 92),
                (92, 93), (93, 94), (94, 95), (91, 96), (96, 97), (97, 98),
                (98, 99), (91, 100), (100, 101), (101, 102), (102, 103),
                (91, 104), (104, 105), (105, 106), (106, 107), (91, 108),
                (108, 109), (109, 110), (110, 111), (112, 113), (113, 114),
                (114, 115), (115, 116), (112, 117), (117, 118), (118, 119),
                (119, 120), (112, 121), (121, 122), (122, 123), (123, 124),
                (112, 125), (125, 126), (126, 127), (127, 128), (112, 129),
                (129, 130), (130, 131), (131, 132)]
    palette = [[51, 153, 255], [0, 255, 0], [255, 128, 0], [255, 255, 255],
               [255, 153, 255], [102, 178, 255], [255, 51, 51]]
    link_color = [
        1, 1, 2, 2, 0, 0, 0, 0, 1, 2, 1, 2, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 2, 2,
        2, 2, 2, 2, 2, 4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 1, 1, 1, 1, 2, 2, 2,
        2, 4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 1, 1, 1, 1
    ]
    point_color = [
        0, 0, 0, 0, 0, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 2, 2, 2, 2, 2, 2, 3,
        3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
        3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
        3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2, 2, 2, 2,
        4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 1, 1, 1, 1, 3, 2, 2, 2, 2, 4, 4, 4,
        4, 5, 5, 5, 5, 6, 6, 6, 6, 1, 1, 1, 1
    ]

    # draw keypoints and skeleton for each person
    for person_kpts, person_scores in zip(keypoints, scores):
        keypoints_num = len(person_scores)
        # Draw keypoints for this person
        for kpt, color in zip(person_kpts, point_color):
            if color < len(palette):  # Safety check
                cv2.circle(img, tuple(kpt.astype(np.int32)), 1, palette[color], 1,
                           cv2.LINE_AA)
        # Draw skeleton for this person
        for (u, v), color in zip(skeleton, link_color):
            if u < keypoints_num and v < keypoints_num \
                        and person_scores[u] > thr and person_scores[v] > thr:
                if color < len(palette):  # Safety check
                    cv2.line(img, tuple(person_kpts[u].astype(np.int32)),
                             tuple(person_kpts[v].astype(np.int32)), palette[color], 2,
                             cv2.LINE_AA)

    return img


def bbox_xyxy2cs(bbox,
                 padding: float = 1.) -> Tuple[np.ndarray, np.ndarray]:
    """Transform the bbox format from (x,y,w,h) into (center, scale)

    Args:
        bbox (ndarray): Bounding box(es) in shape (4,) or (n, 4), formatted
            as (left, top, right, bottom)
        padding (float): BBox padding factor that will be multilied to scale.
            Default: 1.0

    Returns:
        tuple: A tuple containing center and scale.
        - np.ndarray[float32]: Center (x, y) of the bbox in shape (2,) or
            (n, 2)
        - np.ndarray[float32]: Scale (w, h) of the bbox in shape (2,) or
            (n, 2)
    """
    # Convert to numpy array if it's a list
    if isinstance(bbox, list):
        bbox = np.array(bbox)

    # convert single bbox from (4, ) to (1, 4)
    dim = bbox.ndim
    if dim == 1:
        bbox = bbox[None, :]

    # get bbox center and scale
    x1, y1, x2, y2 = np.hsplit(bbox, [1, 2, 3])
    center = np.hstack([x1 + x2, y1 + y2]) * 0.5
    scale = np.hstack([x2 - x1, y2 - y1]) * padding

    if dim == 1:
        center = center[0]
        scale = scale[0]

    return center, scale


def _fix_aspect_ratio(bbox_scale: np.ndarray,
                      aspect_ratio: float) -> np.ndarray:
    """Extend the scale to match the given aspect ratio.

    Args:
        scale (np.ndarray): The image scale (w, h) in shape (2, )
        aspect_ratio (float): The ratio of ``w/h``

    Returns:
        np.ndarray: The reshaped image scale in (2, )
    """
    w, h = np.hsplit(bbox_scale, [1])
    bbox_scale = np.where(w > h * aspect_ratio,
                          np.hstack([w, w / aspect_ratio]),
                          np.hstack([h * aspect_ratio, h]))
    return bbox_scale


def _rotate_point(pt: np.ndarray, angle_rad: float) -> np.ndarray:
    """Rotate a point by an angle.

    Args:
        pt (np.ndarray): 2D point coordinates (x, y) in shape (2, )
        angle_rad (float): rotation angle in radian

    Returns:
        np.ndarray: Rotated point in shape (2, )
    """
    sn, cs = np.sin(angle_rad), np.cos(angle_rad)
    rot_mat = np.array([[cs, -sn], [sn, cs]])
    return rot_mat @ pt


def _get_3rd_point(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """To calculate the affine matrix, three pairs of points are required. This
    function is used to get the 3rd point, given 2D points a & b.

    The 3rd point is defined by rotating vector `a - b` by 90 degrees
    anticlockwise, using b as the rotation center.

    Args:
        a (np.ndarray): The 1st point (x,y) in shape (2, )
        b (np.ndarray): The 2nd point (x,y) in shape (2, )

    Returns:
        np.ndarray: The 3rd point.
    """
    direction = a - b
    c = b + np.r_[-direction[1], direction[0]]
    return c


def get_warp_matrix(center: np.ndarray,
                    scale: np.ndarray,
                    rot: float,
                    output_size: Tuple[int, int],
                    shift: Tuple[float, float] = (0., 0.),
                    inv: bool = False) -> np.ndarray:
    """Calculate the affine transformation matrix that can warp the bbox area
    in the input image to the output size.

    Args:
        center (np.ndarray[2, ]): Center of the bounding box (x, y).
        scale (np.ndarray[2, ]): Scale of the bounding box
            wrt [width, height].
        rot (float): Rotation angle (degree).
        output_size (np.ndarray[2, ] | list(2,)): Size of the
            destination heatmaps.
        shift (0-100%): Shift translation ratio wrt the width/height.
            Default (0., 0.).
        inv (bool): Option to inverse the affine transform direction.
            (inv=False: src->dst or inv=True: dst->src)

    Returns:
        np.ndarray: A 2x3 transformation matrix
    """
    shift = np.array(shift)
    src_w = scale[0]
    dst_w = output_size[0]
    dst_h = output_size[1]

    # compute transformation matrix
    rot_rad = np.deg2rad(rot)
    src_dir = _rotate_point(np.array([0., src_w * -0.5]), rot_rad)
    dst_dir = np.array([0., dst_w * -0.5])

    # get four corners of the src rectangle in the original image
    src = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale * shift
    src[1, :] = center + src_dir + scale * shift
    src[2, :] = _get_3rd_point(src[0, :], src[1, :])

    # get four corners of the dst rectangle in the input image
    dst = np.zeros((3, 2), dtype=np.float32)
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5]) + dst_dir
    dst[2, :] = _get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        warp_mat = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        warp_mat = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return warp_mat


def top_down_affine(input_size: dict, bbox_scale: dict, bbox_center: dict,
                    img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Get the bbox image as the model input by affine transform.

    Args:
        input_size (dict): The input size of the model.
        bbox_scale (dict): The bbox scale of the img.
        bbox_center (dict): The bbox center of the img.
        img (np.ndarray): The original image.

    Returns:
        tuple: A tuple containing center and scale.
        - np.ndarray[float32]: img after affine transform.
        - np.ndarray[float32]: bbox scale after affine transform.
    """
    w, h = input_size
    warp_size = (int(w), int(h))

    # reshape bbox to fixed aspect ratio
    bbox_scale = _fix_aspect_ratio(bbox_scale, aspect_ratio=w / h)

    # get the affine matrix
    center = bbox_center
    scale = bbox_scale
    rot = 0
    warp_mat = get_warp_matrix(center, scale, rot, output_size=(w, h))

    # do affine transform
    img = cv2.warpAffine(img, warp_mat, warp_size, flags=cv2.INTER_LINEAR)

    return img, bbox_scale


def get_simcc_maximum(simcc_x: np.ndarray,
                      simcc_y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Get maximum response location and value from simcc representations.

    Note:
        instance number: N
        num_keypoints: K
        heatmap height: H
        heatmap width: W

    Args:
        simcc_x (np.ndarray): x-axis SimCC in shape (K, Wx) or (N, K, Wx)
        simcc_y (np.ndarray): y-axis SimCC in shape (K, Wy) or (N, K, Wy)

    Returns:
        tuple:
        - locs (np.ndarray): locations of maximum heatmap responses in shape
            (K, 2) or (N, K, 2)
        - vals (np.ndarray): values of maximum heatmap responses in shape
            (K,) or (N, K)
    """
    N, K, Wx = simcc_x.shape
    simcc_x = simcc_x.reshape(N * K, -1)
    simcc_y = simcc_y.reshape(N * K, -1)

    # get maximum value locations
    x_locs = np.argmax(simcc_x, axis=1)
    y_locs = np.argmax(simcc_y, axis=1)
    locs = np.stack((x_locs, y_locs), axis=-1).astype(np.float32)
    max_val_x = np.amax(simcc_x, axis=1)
    max_val_y = np.amax(simcc_y, axis=1)

    # get maximum value across x and y axis
    mask = max_val_x > max_val_y
    max_val_x[mask] = max_val_y[mask]
    vals = max_val_x
    locs[vals <= 0.] = -1

    # reshape
    locs = locs.reshape(N, K, 2)
    vals = vals.reshape(N, K)

    return locs, vals


def decode(simcc_x: np.ndarray, simcc_y: np.ndarray,
           simcc_split_ratio) -> Tuple[np.ndarray, np.ndarray]:
    """Modulate simcc distribution with Gaussian.

    Args:
        simcc_x (np.ndarray[K, Wx]): model predicted simcc in x.
        simcc_y (np.ndarray[K, Wy]): model predicted simcc in y.
        simcc_split_ratio (int): The split ratio of simcc.

    Returns:
        tuple: A tuple containing center and scale.
        - np.ndarray[float32]: keypoints in shape (K, 2) or (n, K, 2)
        - np.ndarray[float32]: scores in shape (K,) or (n, K)
    """
    keypoints, scores = get_simcc_maximum(simcc_x, simcc_y)
    keypoints /= simcc_split_ratio

    return keypoints, scores
