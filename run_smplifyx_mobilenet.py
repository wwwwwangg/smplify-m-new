#!/usr/bin/env python
"""
Run SMPLify-X with MobileNetV3 initialization
"""
import sys
import os
import time
import yaml
import torch
import smplx
import json

# Set paths
PROJECT_ROOT = '/mnt/e/Lxf_test/smplify-x-master'
# Do NOT add PROJECT_ROOT - has conflicting smplx folder

from smplifyx.utils import JointMapper
from smplifyx.data_parser import create_dataset
from smplifyx.fit_single_frame import fit_single_frame
from smplifyx.camera import create_camera
from smplifyx.prior import create_prior

torch.backends.cudnn.enabled = False

def main():
    # Load config
    config_path = os.path.join(PROJECT_ROOT, 'cfg_files', 'fit_smplx.yaml')
    with open(config_path, 'r') as f:
        args = yaml.safe_load(f)

    # Override with MobileNet settings
    args['data_folder'] = PROJECT_ROOT + '/data'
    args['output_folder'] = PROJECT_ROOT + '/smplx_mobilenet_output'
    args['model_folder'] = PROJECT_ROOT + '/models'
    args['vposer_ckpt'] = PROJECT_ROOT + '/vposer'
    args['use_mobilenet_init'] = True
    args['mobilenet_ckpt'] = PROJECT_ROOT + '/mobilenetv3-master/smplify_pth/checkpoint-best.pth'
    args['max_persons'] = 1  # Only process 1 person for testing

    print("=" * 80)
    print("SMPLify-X with MobileNetV3 Initialization")
    print("=" * 80)
    print(f"Data folder: {args['data_folder']}")
    print(f"Output folder: {args['output_folder']}")
    print(f"MobileNetV3: {args['use_mobilenet_init']}")
    print(f"MobileNetV3 checkpoint: {args['mobilenet_ckpt']}")
    print("=" * 80)

    output_folder = args.pop('output_folder')
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    result_folder = args.pop('result_folder', 'results')
    result_folder = os.path.join(output_folder, result_folder)
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)

    mesh_folder = args.pop('mesh_folder', 'meshes')
    mesh_folder = os.path.join(output_folder, mesh_folder)
    if not os.path.exists(mesh_folder):
        os.makedirs(mesh_folder)

    out_img_folder = os.path.join(output_folder, 'images')
    if not os.path.exists(out_img_folder):
        os.makedirs(out_img_folder)

    use_cuda = args.get('use_cuda', True)
    if use_cuda and not torch.cuda.is_available():
        print('CUDA is not available, using CPU')
        use_cuda = False

    img_folder = args.pop('img_folder', 'images')
    dataset_obj = create_dataset(img_folder=img_folder, **args)

    print(f"Found {len(dataset_obj)} images to process")

    dtype = torch.float32
    device = torch.device('cuda') if use_cuda else torch.device('cpu')

    joint_mapper = JointMapper(dataset_obj.get_model2data())

    model_params = dict(model_path=args.get('model_folder'),
                        joint_mapper=joint_mapper,
                        create_global_orient=True,
                        create_body_pose=not args.get('use_vposer'),
                        create_betas=True,
                        create_left_hand_pose=True,
                        create_right_hand_pose=True,
                        create_expression=True,
                        create_jaw_pose=True,
                        create_leye_pose=True,
                        create_reye_pose=True,
                        create_transl=False,
                        dtype=dtype,
                        **args)

    male_model = smplx.create(gender='male', **model_params)
    neutral_model = smplx.create(gender='neutral', **model_params)
    female_model = smplx.create(gender='female', **model_params)

    focal_length = args.get('focal_length')
    camera = create_camera(focal_length_x=focal_length,
                           focal_length_y=focal_length,
                           dtype=dtype, **args)

    if use_cuda:
        camera = camera.to(device=device)
        female_model = female_model.to(device=device)
        male_model = male_model.to(device=device)
        neutral_model = neutral_model.to(device=device)

    use_hands = args.get('use_hands', True)
    use_face = args.get('use_face', True)

    body_pose_prior = create_prior(
        prior_type=args.get('body_prior_type'), dtype=dtype, **args)

    jaw_prior = create_prior(
        prior_type=args.get('jaw_prior_type'), dtype=dtype, **args) if use_face else None

    expr_prior = create_prior(
        prior_type=args.get('expr_prior_type', 'l2'), dtype=dtype, **args) if use_face else None

    left_hand_prior = create_prior(
        prior_type=args.get('left_hand_prior_type'), dtype=dtype,
        use_left_hand=True, num_gaussians=args.get('num_pca_comps'), **args) if use_hands else None

    right_hand_prior = create_prior(
        prior_type=args.get('right_hand_prior_type'), dtype=dtype,
        use_right_hand=True, num_gaussians=args.get('num_pca_comps'), **args) if use_hands else None

    shape_prior = create_prior(
        prior_type=args.get('shape_prior_type', 'l2'), dtype=dtype, **args)

    angle_prior = create_prior(prior_type='angle', dtype=dtype)

    if use_cuda:
        body_pose_prior = body_pose_prior.to(device=device)
        angle_prior = angle_prior.to(device=device)
        shape_prior = shape_prior.to(device=device)
        if use_face:
            expr_prior = expr_prior.to(device=device)
            jaw_prior = jaw_prior.to(device=device)
        if use_hands:
            left_hand_prior = left_hand_prior.to(device=device)
            right_hand_prior = right_hand_prior.to(device=device)

    joint_weights = dataset_obj.get_joint_weights().to(device=device, dtype=dtype)
    joint_weights.unsqueeze_(dim=0)

    input_gender = args.pop('gender', 'neutral')
    gender_lbl_type = args.pop('gender_lbl_type', 'none')
    max_persons = args.pop('max_persons', -1)

    start_time = time.time()
    processed_count = 0

    for idx, data in enumerate(dataset_obj):
        img = data['img']
        fn = data['fn']
        keypoints = data['keypoints']
        print(f'\nProcessing image {idx+1}/{len(dataset_obj)}: {data["img_path"]}')

        curr_result_folder = os.path.join(result_folder, fn)
        if not os.path.exists(curr_result_folder):
            os.makedirs(curr_result_folder)
        curr_mesh_folder = os.path.join(mesh_folder, fn)
        if not os.path.exists(curr_mesh_folder):
            os.makedirs(curr_mesh_folder)

        for person_id in range(keypoints.shape[0]):
            if person_id >= max_persons and max_persons > 0:
                continue

            curr_result_fn = os.path.join(curr_result_folder, '{:03d}.pkl'.format(person_id))
            curr_mesh_fn = os.path.join(curr_mesh_folder, '{:03d}.obj'.format(person_id))

            curr_img_folder = os.path.join(output_folder, 'images', fn, '{:03d}'.format(person_id))
            if not os.path.exists(curr_img_folder):
                os.makedirs(curr_img_folder)

            gender = input_gender

            if gender == 'neutral':
                body_model = neutral_model
            elif gender == 'female':
                body_model = female_model
            elif gender == 'male':
                body_model = male_model

            out_img_fn = os.path.join(curr_img_folder, 'output.png')

            print(f"  Running SMPLify-X fit with MobileNetV3 initialization...")

            fit_single_frame(img, keypoints[[person_id]],
                             body_model=body_model,
                             camera=camera,
                             joint_weights=joint_weights,
                             dtype=dtype,
                             output_folder=output_folder,
                             result_folder=curr_result_folder,
                             out_img_fn=out_img_fn,
                             result_fn=curr_result_fn,
                             mesh_fn=curr_mesh_fn,
                             shape_prior=shape_prior,
                             expr_prior=expr_prior,
                             body_pose_prior=body_pose_prior,
                             left_hand_prior=left_hand_prior,
                             right_hand_prior=right_hand_prior,
                             jaw_prior=jaw_prior,
                             angle_prior=angle_prior,
                             **args)

            processed_count += 1
            print(f"  [OK] Processed person {person_id} in image {fn}")

    elapsed = time.time() - start_time
    time_msg = time.strftime('%H hours, %M minutes, %S seconds', time.gmtime(elapsed))
    print(f'\nProcessing the data took: {time_msg}')
    print(f'Total processed: {processed_count}')

    # Save summary
    summary = {
        'processed_count': processed_count,
        'total_time': elapsed,
        'time_message': time_msg,
        'use_mobilenet_init': True,
        'mobilenet_ckpt': args.get('mobilenet_ckpt', ''),
        'output_folder': output_folder
    }

    summary_fn = os.path.join(output_folder, 'summary.json')
    import json
    with open(summary_fn, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'\nSummary saved to: {summary_fn}')

if __name__ == "__main__":
    main()
