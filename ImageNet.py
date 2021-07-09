CLASS_TO_CALCULATE = 1000 # reduce it for quick test

PATH_LABEL_TO_WORDNET = '/home/projects/RobutnessScore/imagenet_label_to_wordnet_synset.txt'
PATH_LABEL_TO_WORDNET = '/home/projects/RobutnessScore/imagenet_label_to_wordnet_synset.txt'

PATH_IMAGENET_CLASS = '/home/datasets/imagenet_2012/val/{}/'
PATH_IMAGENET_BBOX = '/home/datasets/imagenet_2012/val/xml/'

PATH_OUT = './'

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import xml.etree.ElementTree as ET
import os
import gc
import sys
import json
import random
import warnings
from PIL import Image
from torchvision import models
from sklearn.metrics import accuracy_score
from cam import CAM, GradCAMpp, SmoothGradCAMpp, ScoreCAM
from efficientnet_pytorch import EfficientNet

np.random.seed(0)
torch.manual_seed(0)
random.seed(0)
os.environ["CUDA_VISIBLE_DEVICES"]=str(0); 
device = torch.device("cuda:0")

###

def class_id_to_name(class_id):
    with open(PATH_LABEL_TO_WORDNET) as f:
        json_dict = json.load(f)
    return json_dict[str(class_id)]['label'].replace(" ", "_").replace(",", "__")

def class_id_to_code(class_id):
    with open(PATH_LABEL_TO_WORDNET) as f:
        json_dict = json.load(f)
    return "n{}".format(json_dict[str(class_id)]['id'].split("-")[0])

def openXML(path):
    file = open(path)
    root = ET.fromstring(file.read())
    file.close()
    bbox = []
    for box in root.findall('object'):
        xmin = int (box.find('bndbox').find('xmin').text)
        ymin = int (box.find('bndbox').find('ymin').text)
        xmax = int (box.find('bndbox').find('xmax').text)
        ymax = int (box.find('bndbox').find('ymax').text)
        bbox.append((xmin, ymin, xmax, ymax))
    return bbox

def getData(class_code):
    image_class_path = PATH_IMAGENET_CLASS.format(class_code) 
    bbox_class_path = PATH_IMAGENET_BBOX.format(class_code) 
    
    results = []
    for name in os.listdir(image_class_path):
        jpg_file = os.path.join(image_class_path, name)
        xml_file = os.path.join(bbox_class_path, name).replace("JPEG", "xml")

        if not (os.path.isfile(jpg_file) and os.path.isfile(xml_file)):
            continue
        
        img = Image.open(jpg_file).convert('RGB')
        bbox = openXML(xml_file)
        
        results.append((img, bbox))   
    return results

def tansform_bbox(bbox, img, image_size):
    x1, y1, x2, y2 = bbox
    
    width, height = img.width, img.height
    if height > width:
        new_sizes = [image_size, image_size * height / width]
    else:
        new_sizes = [image_size * width/ height, image_size]
    
    new_sizes[0] = int(new_sizes[0])
    new_sizes[1] = int(new_sizes[1])
    x1 = int(x1 * new_sizes[0]/width)
    x2 = int(x2 * new_sizes[0]/width)
    y1 = int(y1 * new_sizes[1]/height)
    y2 = int(y2 * new_sizes[1]/height)
    
    bbox = (x1, y1, x2, y2)
    
    if new_sizes[0] > image_size:   
        x1 -= (new_sizes[0] - image_size)//2
        x2 -= (new_sizes[0] - image_size)//2     
    
    if new_sizes[1] > image_size:
        y1 -= (new_sizes[1] - image_size)//2
        y2 -= (new_sizes[1] - image_size)//2  
        
    x1 = max(0, x1)
    x2 = max(0, x2)    
    y1 = max(0, y1)    
    y2 = max(0, y2)    
    bbox = (x1, y1, x2, y2)
    
    return bbox

def get_transforms():
    return transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),     
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

def calculate_rs(img, bbox, id_class):
    orginal_mask, _= saliency_maps(transform_net(img).to(device).unsqueeze(0), id_class)
    orginal_mask = torch.nn.functional.upsample(orginal_mask, size=(image_size, image_size), mode='bilinear', align_corners=False)
    orginal_mask = orginal_mask.to(device)
    
    mask = torch.ones(orginal_mask.shape).to(device)
    for box in bbox:
        x1, y1, x2, y2 = tansform_bbox(box, img, image_size)
        sub_mask_x = torch.ones((image_size)).to(device)
        sub_mask_x[int(x1):int(x2)] = 0

        sub_mask_y = torch.zeros((image_size))
        sub_mask_y[int(y1):int(y2)] = 1
        mask[0][0][sub_mask_y.nonzero()] *= sub_mask_x
        
    mask_in = orginal_mask.detach().clone().mul(1-mask).clamp(max = 1)
    mask_out = orginal_mask.detach().clone().mul(mask).clamp(max = 1)
    orginal_mask = orginal_mask.clamp(max = 1)
    
    rs = mask_in.sum().item() / orginal_mask.sum().item() if orginal_mask.sum().item() > 0 else 0
    
    return rs, mask_in, mask_out

def get_top_5_classes(img):
    out = net(transform_net(img).unsqueeze(0).to(device))
    return out.argsort().detach().cpu().numpy()[0][::-1][:5]

def calclulate_class(data, id_class):
    result = []
    result_rs = []
    for index, (img, bbox) in enumerate(data): 
        is_in_top_5 = id_class in get_top_5_classes(img)
        rs, _, _ = calculate_rs(img, bbox, id_class)
        
        result.append(is_in_top_5)
        result_rs.append(rs)
        
    ground_true = [True]*len(result) 
    
    acc = accuracy_score(result, ground_true)
    rs_mean = np.array(result_rs).sum()/len(result_rs)

    print("id:{}    acc:{:.4f}    rs:{:.4f}    name:{}".format(id_class, acc, rs_mean, class_id_to_name(id_class)))
    return (id_class, acc, rs_mean, class_id_to_name(id_class))

def run(name):
    path = os.path.join(PATH_OUT, name)
    if os.path.isfile(path):
        return
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        with open(path, 'w') as csv:
            csv.write("id;acc;crs;name\r\n")
            for i in range(CLASS_TO_CALCULATE):
                data = getData(class_id_to_code(i)) 
                results = calclulate_class(data, i)  
                csv.write("{};{:.4f};{:.4f};{}\r\n".format(results[0], results[1], results[2], results[3]))
                
###            
            
if __name__ == "__main__":

    net = models.resnet152(pretrained=True)
    net = net.to(device)
    net.eval()
    image_size = 224
    
    # ImageNet + ResNet-152 + GradCAM++
    print(">>> ImageNet + ResNet-152 + GradCAM++")    
    saliency_maps = GradCAMpp(net, net.layer4[2].conv3)
    transform_net = get_transforms()
    run("ImageNet__ResNet_152__GradCAMpp.csv")
    
    # ImageNet + ResNet-152 + SmoothGrad-Cam++
    print(">>> ImageNet + ResNet-152 + SmoothGrad-Cam++")
    saliency_maps = SmoothGradCAMpp(net, net.layer4[2].conv3)
    transform_net = get_transforms()
    run("ImageNet__ResNet_152__SmoothGradCAMpp.csv")  
    
    # ImageNet + ResNet-152 + ScoreCAM
    print(">>> ImageNet + ResNet-152 + ScoreCAM")
    saliency_maps = ScoreCAM(net, net.layer4[2].conv3)
    transform_net = get_transforms()
    run("ImageNet__ResNet_152__ScoreCAM.csv")

    ###
    
    # ImageNet + AlexNet + GradCAM++
    net = models.alexnet(pretrained=True)
    net = net.to(device)
    net.eval()
    image_size = 224
    
    print(">>> ImageNet + AlexNet + GradCAM++")
    saliency_maps = GradCAMpp(net, net._modules['avgpool'])
    transform_net = get_transforms()
    run("ImageNet__AlexNet__GradCAMpp.csv")
    
    ###

    # ImageNet + EfficientNet-B0 + GradCAM++
    net = EfficientNet.from_pretrained('efficientnet-b0')
    net = net.to(device)
    net.eval()
    image_size = 224
    
    print(">>> ImageNet + EfficientNet-B0 + GradCAM++")
    saliency_maps = GradCAMpp(net, net._modules['_conv_head'])
    transform_net = get_transforms()
    run("ImageNet__EfficientNet-B0__GradCAMpp.csv")
    
    # ImageNet + EfficientNet-B3 + GradCAM++
    net = EfficientNet.from_pretrained('efficientnet-b3')
    net = net.to(device)
    net.eval()
    image_size = 300
    
    print(">>> ImageNet + EfficientNet-B3 + GradCAM++")
    saliency_maps = GradCAMpp(net, net._modules['_conv_head'])
    transform_net = get_transforms()
    run("ImageNet__EfficientNet-B3__GradCAMpp.csv")      
    
    # ImageNet + EfficientNet-B7 + GradCAM++
    net = EfficientNet.from_pretrained('efficientnet-b7')
    net = net.to(device)
    net.eval()
    image_size = 600
    
    print(">>> ImageNet + EfficientNet-B7 + GradCAM++")
    saliency_maps = GradCAMpp(net, net._modules['_conv_head'])
    transform_net = get_transforms()
    run("ImageNet__EfficientNet-B7__GradCAMpp.csv")    