import numpy as np
import cython
import time


def iou(box, boxes, isMin=False):  # [x1,y1,x2,y2,c]
    # 计算面积
    box_area = (box[2] - box[0]) * (box[3] - box[1])
    boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    # 找交集
    xx1 = np.maximum(box[0], boxes[:, 0])
    yy1 = np.maximum(box[1], boxes[:, 1])
    xx2 = np.minimum(box[2], boxes[:, 2])
    yy2 = np.minimum(box[3], boxes[:, 3])

    # 判断是否有交集
    w = np.maximum(0, xx2 - xx1)
    h = np.maximum(0, yy2 - yy1)
    # 交集面积
    inter = w * h

    if isMin:
        over = np.true_divide(inter, np.minimum(box_area, boxes_area))
    else:
        over = np.true_divide(inter, (box_area + boxes_area - inter))

    return over


def nms(boxes, threshold=0.3, isMin=False):
    tt = time.process_time()
    # 根据置信度进行排序
    _boxes = boxes[(-boxes[:, 4]).argsort()]
    # 保留剩余的框
    r_boxes = []
    while _boxes.shape[0] > 1:
        # 取出第一个框
        a_box = _boxes[0]
        # 取出剩余的框
        b_boxes = _boxes[1:]
        # 保留第一个框
        r_boxes.append(a_box)
        # 比较iou后保留阈值小的值
        index = np.where(iou(a_box, b_boxes, isMin) < threshold)
        _boxes = b_boxes[index]
    if _boxes.shape[0] > 0:
        r_boxes.append(_boxes[0])
    ee = time.process_time()
    print('nmsTime:', ee - tt)
    # 将array组装成矩阵
    try:
        return np.stack(r_boxes)
    except Exception as e:
        print(str(e))


def convert_to_square(bbox):
    square_bbox = bbox.copy()
    if bbox.shape[0] == 0:
        return np.array([])
    h = bbox[:, 3] - bbox[:, 1]
    w = bbox[:, 2] - bbox[:, 0]
    max_side = np.maximum(h, w)
    square_bbox[:, 0] = bbox[:, 0] + w * 0.5 - max_side * 0.5
    square_bbox[:, 1] = bbox[:, 1] + h * 0.5 - max_side * 0.5
    square_bbox[:, 2] = square_bbox[:, 0] + max_side
    square_bbox[:, 3] = square_bbox[:, 1] + max_side

    return square_bbox


def prewhiten(x):
    mean = np.mean(x)
    std = np.std(x)
    std_adj = np.maximum(std, 1.0 / np.sqrt(x.size))
    y = np.multiply(np.subtract(x, mean), 1 / std_adj)
    return y


# if __name__ == '__main__':
#     bs = np.array([[2,2,30,30,40],[3,3,23,23,60],[18,18,27,27,15]])
#     print(nms(bs))


def __box(self, start_index, offset, cls, scale, stride=2, side_len=12):
    _x1 = (start_index[1] * stride).item() / scale
    _y1 = (start_index[0] * stride).item() / scale
    _x2 = (start_index[1] * stride + side_len).item() / scale
    _y2 = (start_index[0] * stride + side_len).item() / scale

    ow = _x2 - _x1
    oh = _y2 - _y1

    _offset = offset[start_index[0], start_index[1]]
    x1 = _x1 + ow * _offset[0]
    y1 = _y1 + oh * _offset[1]
    x2 = _x2 + ow * _offset[2]
    y2 = _y2 + oh * _offset[3]
    return [x1, y1, x2, y2, cls]
