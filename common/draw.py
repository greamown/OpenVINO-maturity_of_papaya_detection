
import cv2
import numpy as np
import logging
from math import floor

class Draw():

    def __init__(self) -> None:
        pass

    def draw_detections(self, frame, info, palette):
        output_transform = info['output_transform']
        if info["detections"] is not None and info["detections"] != []:
            if output_transform != None:
                frame = output_transform.resize(frame)
            for detection in info["detections"]:
                # class_id = int(detection['id'])
                det_label = detection['det_label']
                xmin = max(int(detection['xmin']), 0)
                ymin = max(int(detection['ymin']), 0)
                cv2.putText(frame, '{} {:.1%}'.format(det_label, detection['score']),
                            (xmin, ymin - 7), cv2.FONT_HERSHEY_COMPLEX, 0.6, palette[0], 1)
                
        return frame
        
    def draw_fps(self, frame, fps):
        logging.info("FPS    |    {}".format(round(fps, 1)))
        cv2.putText(frame, '{} {}'.format("FPS:", round(fps, 1)),
                    (frame.shape[1]-100, 35), cv2.FONT_HERSHEY_COMPLEX, 0.6, [0, 0, 255], 1)
        return frame

def u2net_postprocessing(image, resized_result):
    bg_removed_result = image.copy()
    bg_removed_result[resized_result == 0] = 255
    return bg_removed_result

def get_mask(img, mask):
    # unet 預測結果以信賴度 0.5 為界，轉為 0 與 255
    mask = np.where(mask >= 0.5, 255, 0).astype('uint8')
    # 找出邊界
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # 必須有兩個以上的邊界否則判定為其他
    if(len(contours)>=2):
        # 若有三個或三個以上的區域，僅切割前兩個面積最大的，
        max_index = list(map(lambda x:len(x), contours))
        max_index_ = max_index.copy()
        max_index_.sort(reverse=True)
        max_index = [max_index.index(x) for x in max_index_[:2]]
        # 分割成兩張圖片，且高 >= 寬，否則轉 90 度
        crop_img = list(map(lambda x: detect_papaya(x, contours, img), max_index))
        # resize & padding to 244*224
        resize_img = [img_pad(x) for x in crop_img]
        # rotate image 180 degree (共 4 張，正+反，0度+180度)
        rotate_img = [np.rot90(x, k = 2) for x in resize_img]
        rotate_img = resize_img + rotate_img 
        # merge four image
        merge_img = merge_channel_img(rotate_img)
    else:
        merge_img = ''
        rotate_img = ''
    
    return {'remove_image':img, 'mask_image':mask, 'merge_image':merge_img, 'rotate_img':rotate_img}

def detect_papaya(max_index_, contours_, result_):
    x_min = min(contours_[max_index_][:,0][:,0])
    x_max = max(contours_[max_index_][:,0][:,0])
    y_min = min(contours_[max_index_][:,0][:,1])
    y_mix = max(contours_[max_index_][:,0][:,1])
    result_ = result_[y_min:y_mix, x_min:x_max]
    result_ = result_[:,:,:3]    
    if result_.shape[0]<result_.shape[1]:
        return np.rot90(result_, k = 1)
    else:
        return result_

def img_pad(im_, image_h = 224):
    # 縮放
    if im_.shape[0]==max(im_.shape):
        im_ = cv2.resize(im_, (floor(image_h*im_.shape[1]/im_.shape[0]), image_h), interpolation=cv2.INTER_AREA)
    else:
        im_ = cv2.resize(im_, (image_h, floor(image_h*im_.shape[0]/im_.shape[1])), interpolation=cv2.INTER_AREA)
    # padding
    delta_w = image_h - im_.shape[1]
    delta_h = image_h - im_.shape[0]
    top, bottom = delta_h//2, delta_h-(delta_h//2)
    left, right = delta_w//2, delta_w-(delta_w//2)
    img_pad = cv2.copyMakeBorder(im_, top, bottom, left, right, cv2.BORDER_CONSTANT, value = [255, 255, 255])
    return img_pad

def make_mask(img_):
    mask = (img_ == 255).all(axis=2).astype(int)
    mask = mask^(mask&1==mask)
    return mask

def merge_channel_img(img_, color_ = 255):
    mask = [make_mask(x) for x in img_]
    img_new = [img_[x]*np.stack([mask[x], mask[x], mask[x]], axis=2) for x in range(len(img_))]
    img_new = np.sum(img_new, axis=0)
    mask = np.sum(mask, axis=0)
    mask[mask==0] = 4
    img_new = img_new/np.stack([mask,mask,mask], axis=2)
    mask = (img_new == 0).all(axis=2).astype(int)*color_
    mask = np.stack([mask, mask, mask], axis=2)
    img_new = img_new+mask
    return img_new

def cls_other():
    info = {"detections":[{  'xmin':15, 
                            'ymin':40, 
                            'xmax':30, 
                            'ymax':30, 
                            'det_label':"other", 
                            'score': 1.0,
                        }],
            "output_transform":None}

    return info