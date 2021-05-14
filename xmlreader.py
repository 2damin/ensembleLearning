import xml.etree.ElementTree as ET
import ast
import os
import numpy as np
import io
import natsort

from wbf import weighted_boxes_fusion

from detectron2.data import MetadataCatalog
from detectron2.utils.file_io import PathManager
from detectron2.evaluation.coco_evaluation import _evaluate_predictions_on_coco
from pycocotools.coco import COCO

import torchvision.datasets as dset

r50c4_result_path = os.getcwd() + "/detectron2/detection_result/r50c4/"
r50dc5_result_path = os.getcwd() + "/detectron2/detection_result/r50dc5/"
r50fpn_result_path = os.getcwd() + "/detectron2/detection_result/r50fpn/"

xmlfile_c4 = os.listdir(r50c4_result_path)
xmlfile_dc5 = os.listdir(r50dc5_result_path)
xmlfile_fpn = os.listdir(r50fpn_result_path)

def read_xmlfile(path):
    results = []
    tree = ET.parse(path)
    root = tree.getroot()
    image_id = int(root.find('image_id').text)
    num_boxes = int(root.find('num_boxes').text)
    width = int(root.find('width').text)
    height = int(root.find('height').text)
    box_infos = root.find('image_id').findall("box_info")
    for box in box_infos:
        bbox = ast.literal_eval(box.find('bbox').text)
        score = float(box.find('score').text)
        category_id = int(box.find('category_id').text)
        result = {
            "image_id":image_id,
            "category_id" : category_id,
            "bbox" : bbox,
            "score":score
        }
        results.append(result)
    return image_id, width, height, results

#change format and normalize. [x,y,w,h] -> [xmin, ymin, xmax, ymax]
def normalize_box(box, width, height):
    xmin = box[0] / width
    ymin = box[1] / height
    xmax = (box[0] + box[2]) / width
    ymax = (box[1] + box[3]) / height
    return [xmin,ymin,xmax,ymax]

#change format and denormalize.  [x_min, y_min, x_max, y_max] -> [x,y,w,h]
def denormalize_box(box, width, height):
    xLT= box[0] * width
    yLT = box[1] * height
    w = (box[2] -box[0]) * width
    h = (box[3] -box[1]) * height
    return [xLT,yLT,w,h]

#convert format to cocodata type
def convert_to_cocotype(boxes, scores, labels, image_id, width, height):
    result = []
    for box, score, label in zip(boxes, scores, labels):
        tmp = {
            "image_id":image_id,
            "category_id" : label,
            "bbox" : denormalize_box(box,width,height),
            "score":score
        }
        result.append(tmp)
    return result

if __name__ == "__main__":

    #read coco_2017_val 
    dataset_name = "coco_2017_val"
    metadata = MetadataCatalog.get(dataset_name)
    json_file = PathManager.get_local_path("./detectron2/"+metadata.json_file)
    coco_api = COCO(json_file)
    dataloader = dset.CocoDetection(root = "./detectron2/datasets/coco/val2017",
                                    annFile = "./detectron2/datasets/coco/annotations/instances_val2017.json")

    print("c4 model file :", len(xmlfile_c4))
    print("dc5 model file :", len(xmlfile_dc5))
    print("fpn model file :", len(xmlfile_fpn))

    #sort xmlfiles in ascending order (image_id) 
    xmlfile_c4_sorted = natsort.natsorted(xmlfile_c4)
    xmlfile_dc5_sorted = natsort.natsorted(xmlfile_dc5)
    xmlfile_fpn_sorted = natsort.natsorted(xmlfile_fpn)

    #coco category_id setting
    dataset_id_to_contiguous_id = metadata.thing_dataset_id_to_contiguous_id
    all_contiguous_ids = list(dataset_id_to_contiguous_id.values())
    num_classes = len(all_contiguous_ids)
    assert min(all_contiguous_ids) == 0 and max(all_contiguous_ids) == num_classes - 1
    reverse_id_mapping = {v: k for k, v in dataset_id_to_contiguous_id.items()}

    result_all = []
    for c4, dc5, fpn, anno in zip(xmlfile_c4_sorted, xmlfile_dc5_sorted, xmlfile_fpn_sorted, dataloader):
        
        #read xmlfile
        img_c4, width, height, result_c4 = read_xmlfile(r50c4_result_path+c4)
        img_dc5, width, height, result_dc5 = read_xmlfile(r50dc5_result_path+dc5)
        img_fpn, width, height, result_fpn = read_xmlfile(r50fpn_result_path+fpn)

        boxes_c4 = []
        boxes_dc5 = []
        boxes_fpn = []
        score_c4 = []
        score_dc5 = []
        score_fpn = []
        cate_c4 = []
        cate_dc5 = []
        cate_fpn = []
        for box_c4, box_dc5, box_fpn in zip(result_c4, result_dc5, result_fpn):
            #for bc, bdc, dfpn in zip(box_c4, box_dc5, box_fpn)
            box_c4["category_id"] = reverse_id_mapping[box_c4["category_id"]]
            box_dc5["category_id"] = reverse_id_mapping[box_dc5["category_id"]] 
            box_fpn["category_id"] = reverse_id_mapping[box_fpn["category_id"]] 
            boxes_c4.append(normalize_box(box_c4["bbox"], width, height))
            boxes_dc5.append(normalize_box(box_dc5["bbox"], width, height))
            boxes_fpn.append(normalize_box(box_fpn["bbox"], width, height))
            score_c4.append(box_c4["score"])
            score_dc5.append(box_dc5["score"])
            score_fpn.append(box_fpn["score"])
            cate_c4.append(box_c4["category_id"])
            cate_dc5.append(box_dc5["category_id"])
            cate_fpn.append(box_fpn["category_id"])
        
        allbox = [boxes_c4, boxes_dc5, boxes_fpn]
        allscore = [score_c4, score_dc5, score_fpn]
        allcategory = [cate_c4, cate_dc5, cate_fpn]
        weights = [1, 1, 2]

        boxes, scores, labels = weighted_boxes_fusion(allbox, allscore, allcategory, weights=weights, iou_thr=0.55)
        
        chaned_data = convert_to_cocotype(boxes, scores, labels, img_c4, width, height)
        result_all += chaned_data

    _evaluate_predictions_on_coco(
        coco_api,
        result_all,
        iou_type = "bbox",
        img_ids=None,
    )
            