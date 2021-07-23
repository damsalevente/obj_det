import selectivesearch
import numpy as np

def extract_candidates(img):
    image_label, regions = selectivesearch.selective_search(img, scale = 200, min_size = 100)
    img_area = np.prod(img.shape[:2])
    candidates = []
    for r in regions:
        if r['rect'] in candidates:
            continue
        if r['size'] < 0.05*img_area:
            continue
        if r['size'] > img_area:
            continue
        candidates.append(list(r['rect']))
    return candidates

# iou: metszetet elosztom az unioval 
# from the book
def extract_iou(boxA, boxB, epsilon=1e-5):
    x1 = max(boxA[0], boxB[0])
    y1 = max(boxA[1], boxB[1])
    x2 = min(boxA[2], boxB[2])
    y2 = min(boxA[3], boxB[3])
    width = (x2 - x1)
    height = (y2 - y1)
    if (width<0) or (height <0):
        return 0.0
    area_overlap = width * height
    area_a = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    area_b = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    area_combined = area_a + area_b - area_overlap
    iou = area_overlap / (area_combined+epsilon)
    return iou
