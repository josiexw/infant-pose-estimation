from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import pprint
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import _init_paths
from config import cfg
from config import update_config
from core.loss import JointsMSELoss
from utils.utils import create_logger
import dataset
import models
import numpy as np
import cv2
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description='Apply HRNet-FIDIP to video')
    parser.add_argument('--cfg', help='experiment configure file name', required=True, type=str)
    parser.add_argument('--videoPath', help='path to input video', required=True, type=str)
    parser.add_argument('--outputPath', help='path to output video', required=True, type=str)
    parser.add_argument('opts', help="Modify config options using the command-line", default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()
    return args

def process_frame(model, frame, transform):
    frame_tensor = transform(frame).unsqueeze(0).cuda()
    with torch.no_grad():
        output = model(frame_tensor)
    # Process the output as needed
    # This is a placeholder - modify according to your model's output
    processed_frame = output[0].cpu().numpy().transpose(1, 2, 0)
    processed_frame = (processed_frame * 255).astype(np.uint8)
    return processed_frame

def process_video(model, video_path, output_path, transform):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    for _ in tqdm(range(total_frames)):
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        processed_frame = process_frame(model, frame_rgb, transform)
        processed_frame_bgr = cv2.cvtColor(processed_frame, cv2.COLOR_RGB2BGR)
        
        out.write(processed_frame_bgr)
    
    cap.release()
    out.release()

def main():
    args = parse_args()
    update_config(cfg, args)

    logger, final_output_dir, tb_log_dir = create_logger(cfg, args.cfg, 'valid')
    logger.info(pprint.pformat(args))
    logger.info(cfg)

    # cudnn related setting
    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

    model = eval('models.'+cfg.MODEL.NAME+'.get_adaptive_pose_net')(cfg, is_train=False)

    if cfg.TEST.MODEL_FILE:
        logger.info('=> loading model from {}'.format(cfg.TEST.MODEL_FILE))
        model.load_state_dict(torch.load(cfg.TEST.MODEL_FILE), strict=False)
    else:
        model_state_file = os.path.join(final_output_dir, 'final_state.pth')
        logger.info('=> loading model from {}'.format(model_state_file))
        model.load_state_dict(torch.load(model_state_file))

    model = torch.nn.DataParallel(model, device_ids=cfg.GPUS).cuda()

    # Data transform
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

    # Process video
    process_video(model, args.videoPath, args.outputPath, transform)

if __name__ == '__main__':
    main()
    