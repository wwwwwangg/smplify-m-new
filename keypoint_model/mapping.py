"""Keypoint format mapping utilities"""
import numpy as np
import json


def create_openpose_format_json(keypoints, output_path):
    """Create OpenPose-format JSON from keypoints (N, 3) with [x, y, conf]"""
    if keypoints.shape[1] == 2:
        conf = np.ones((keypoints.shape[0], 1), dtype=np.float32)
        keypoints = np.concatenate([keypoints, conf], axis=1)
    
    data = {
        "version": 1.2,
        "people": [{
            "pose_keypoints_2d": keypoints.flatten().tolist(),
            "face_keypoints_2d": [],
            "hand_left_keypoints_2d": [],
            "hand_right_keypoints_2d": []
        }]
    }
    
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    return output_path


def validate_keypoints(keypoints, image_shape):
    """Validate keypoints within image bounds"""
    h, w = image_shape[:2]
    keypoints[:, 0] = np.clip(keypoints[:, 0], 0, w - 1)
    keypoints[:, 1] = np.clip(keypoints[:, 1], 0, h - 1)
    return keypoints


def mobilenet_to_smplx(keypoints, model_type='smplx'):
    """Convert MobileNet keypoints to SMPL-X format"""
    import sys
    sys.path.insert(0, 'E:/Lxf_test/smplify-x-master/smplifyx')
    from smplifyx.utils import smpl_to_openpose
    
    openpose_indices = smpl_to_openpose(model_type=model_type, use_hands=False, use_face=False)
    num_smplx_joints = 127 if model_type == 'smplx' else 23
    
    smplx_kp = np.zeros((num_smplx_joints, 3), dtype=np.float32)
    for smplx_idx, op_idx in enumerate(openpose_indices[:25]):
        if op_idx < len(keypoints):
            smplx_kp[smplx_idx] = keypoints[op_idx]
    
    return smplx_kp, openpose_indices
