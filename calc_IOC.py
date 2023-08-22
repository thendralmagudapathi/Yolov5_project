import numpy as np

"""loading the ground truth annotations and model predictions into dictionaries"""

def load_groundtruthannnotation(filepath):
    ground_truth_annotations = {}
    with open(filepath, 'r') as f:
        lines = f.readlines()
        for line in lines[1:]:  # Skip the first line with headers
            class_label, x_center, y_center, width, height = map(float, line.strip().split())
            ground_truth_annotations[class_label] = (x_center, y_center, width, height)
    return ground_truth_annotations

def load_modelpredictions(filepath):
    model_predictions = {}
    with open(filepath, 'r') as f:
        lines = f.readlines()
        for line in lines[1:]:  # Skip the first line with headers
            class_label, x_center, y_center, width, height = map(float, line.strip().split())
            model_predictions[class_label] = (x_center, y_center, width, height)
    return model_predictions

def calculate_iou(box1, box2):
    # Calculate intersection coordinates
    x1 = max(box1[0] - box1[2] / 2, box2[0] - box2[2] / 2)
    y1 = max(box1[1] - box1[3] / 2, box2[1] - box2[3] / 2)
    x2 = min(box1[0] + box1[2] / 2, box2[0] + box2[2] / 2)
    y2 = min(box1[1] + box1[3] / 2, box2[1] + box2[3] / 2)
    
    # Calculate intersection area
    intersection_area = max(0, x2 - x1) * max(0, y2 - y1)
    
    # Calculate areas of the two boxes
    box1_area = box1[2] * box1[3]
    box2_area = box2[2] * box2[3]

    print(box1_area, "GROUND TRUTH BOX AREA")
    print(box2_area, "MODEL PREDICTION AREA")
    
    # Calculate IOU
    iou = intersection_area / (box1_area + box2_area - intersection_area)
    
    print(iou, "CALCULATED IOU")
    return iou

def identify_false_positives_and_negatives(gt_annotations, pred_annotations):
    false_positives = []
    false_negatives = []
    
    for class_label, gt_box in gt_annotations.items():
        if class_label in pred_annotations:
            pred_box = pred_annotations[class_label]
            iou = calculate_iou(gt_box[:4], pred_box[:4])
            
            if iou < 0.5 or (len(pred_box) > 4 and pred_box[4] < 0.4):
                false_positives.append((class_label, pred_box))
        else:
            false_negatives.append((class_label, gt_box))

    # print(false_positives, "FALSE POSITIVE")
    # print(false_negatives, "FALSE NEGATIVE")
    
    return false_positives, false_negatives

# Load ground truth annotations and model predictions (as shown in previous steps)

validation_images = ["C:/Users/thend/Desktop/56secure/annotations/7c2ffbec93afbd9e034b2cc70e7365d2--future-car-the-future.txt",
                     "C:/Users/thend/Desktop/56secure/annotations/berlin-june-14-2015-full-size-car-pontiac-bonneville-convertible-1963-EYJ9XC.txt",
                     "C:/Users/thend/Desktop/56secure/annotations/close-up-red-emergency-triangle-road-front-damaged-car-unrecognizable-people-accident-concept-copy-space-121124292.txt",
                     "C:/Users/thend/Desktop/56secure/annotations/download.txt",
                     "C:/Users/thend/Desktop/56secure/annotations/ford_vtti_research_04_hr_1280x720.txt",
                     "C:/Users/thend/Desktop/56secure/annotations/lots-of-people-crossing-the-street-on-a-crossroad-walking-between-H5HJ4C.txt",
                     "C:/Users/thend/Desktop/56secure/annotations/pexels-jacek-herbut-8318771.txt",
                     "C:/Users/thend/Desktop/56secure/annotations/Screenshot 2023-08-22 010652.txt",
                     "C:/Users/thend/Desktop/56secure/annotations/Screenshot 2023-08-22 010748.txt"]

initial = 0
   
# Iterate through images
for image_path in validation_images:
    print("---------------- ", "FILE ", initial+1, " ----------------")
    gt_annotations = load_groundtruthannnotation(image_path)
    pred_annotations = load_modelpredictions(image_path)
    
    # Identify false positives and false negatives
    false_positives, false_negatives = identify_false_positives_and_negatives(gt_annotations, pred_annotations)
    initial = initial+1