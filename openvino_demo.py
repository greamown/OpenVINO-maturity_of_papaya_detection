#!/usr/bin/python3
# -*- coding: utf-8 -*-
import cv2, sys, time, logging
import numpy as np
from common.utils import read_json, image_prepare
from common.draw import Draw, get_mask, cls_other, u2net_postprocessing
from common.logger import config_logger 
import argparse

def main(args):
    # Instantiation
    draw = Draw()
    # Get all cfg parameter
    custom_cfg = read_json(args.config)
    model_cfg = {key:read_json(custom_cfg[key]) for key in custom_cfg.keys() if "model" in key}
    # Building initial model
    from common.segmentation import Segmentation
    from common.classification import Classification
    seg = Segmentation(model_cfg["model_1"])
    cls = Classification(model_cfg["model_2"])
    seg_palette = seg.load_model()
    cls_palette = cls.load_model()
    # Check input is camera or image and initial frame id/show id
    from common.images_capture import open_images_capture
    cap_r = open_images_capture(custom_cfg['input_data_r'], model_cfg["model_1"]['loop'])
    cap_l = open_images_capture(custom_cfg['input_data_l'], model_cfg["model_2"]['loop'])
    # Image processing
    while True:
        try:
            start_time = time.time()
            frame = [image_prepare(x) for x in [cap_r.read(), cap_l.read()]]
            frame = np.concatenate([frame[0], frame[1]],axis=1)
            # Inference
            seg_info = seg.inference(frame)
            if seg_info is not None:
                removed_frame = u2net_postprocessing(frame, seg_info['detections'][0]['objects'])
                mask_info = get_mask(removed_frame, seg_info['detections'][0]['objects'])
                try:
                    if type(mask_info['merge_image']) != str:
                        cls_info = cls.inference(mask_info['merge_image'].astype('uint8'))
                    else:
                        cls_info = cls_other()
                    if cls_info is not None:
                        frame = draw.draw_detections(frame, cls_info, cls_palette)
                    else:
                        continue
                except Exception as e:
                    logging.error(e)
                    continue
            else:
                continue
            end_time = time.time()
            fps = 1/(end_time - start_time)
            frame = draw.draw_fps(frame, fps)
            cv2.namedWindow('frame', cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
            cv2.imshow('frame', frame)
            time.sleep(0.01)
            key = cv2.waitKey(1)
            ESC_KEY = 27
            # Quit.
            if key in {ord('q'), ord('Q'), ESC_KEY}:
                break
        except Exception as e:
            logging.error(e)
            continue

if __name__ == '__main__':
    config_logger('./openvino.log', 'w', "info")
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', help = "The path of application config")
    args = parser.parse_args()

    sys.exit(main(args) or 0)