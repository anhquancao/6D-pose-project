import logging
import numpy as np
import torch 

def setup_logger(logger_name, log_file, level=logging.INFO):
    l = logging.getLogger(logger_name)
    formatter = logging.Formatter('%(asctime)s : %(message)s')
    fileHandler = logging.FileHandler(log_file, mode='w')
    fileHandler.setFormatter(formatter)

    l.setLevel(level)
    l.addHandler(fileHandler)

#     streamHandler = logging.StreamHandler()
#     streamHandler.setFormatter(formatter)
#     l.addHandler(streamHandler)
    return l

def im_convert(tensor):
    image = tensor.to("cpu").clone().detach()
    image = image.numpy().squeeze()
    image = image.transpose(1,2,0)
    image = image * np.array((0.229, 0.224, 0.225)) + np.array([0.485, 0.456, 0.406])
    image = image.clip(0, 1)
 
    return image

def depth_to_img(tensor):
    max_d = torch.max(tensor)
    min_d = torch.min(tensor)
    depth_norm = (tensor - min_d) * 255 / (max_d - min_d)
    return depth_norm[0]