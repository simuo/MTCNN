import numpy as np
import os
from wider import WIDER
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from tools import utils

wider = WIDER('D:\dataset\wider_face\wider_face_split\wider_face_split',
              'D:\dataset\wider_face\WIDER_train\WIDER_train\images',
              'wider_face_train.mat')

# if not os.path.exists('label.txt'):
#     with open('label.txt', 'w') as f:
#         for i, data in enumerate(wider.next()):
#             boxes = []
#             name = data.image_name
#             f.write(str(name) + " ")
#             for bbox in data.bboxes:
#                 x1 = bbox[0]
#                 x2 = bbox[1]
#                 w = bbox[2]
#                 h = bbox[3]
#
#                 f.write(str(x1) + " " + str(x2) + " " + str(w) + " " + str(h) + " ")
#             f.write("\n")

save_path = r'F:\widerface'

with open('label.txt', 'r') as f:
    lines = f.readlines()
    for face_size in [48, 24, 12]:
        positive_img_dir = os.path.join(save_path, str(face_size), 'positive')
        part_img_dir = os.path.join(save_path, str(face_size), 'part')
        negative_img_dir = os.path.join(save_path, str(face_size), 'negative')

        for dir in [positive_img_dir, part_img_dir, negative_img_dir]:
            if not os.path.exists(dir):
                os.makedirs(dir)

        # 样本的参数保存路径
        positive_anno_dir = os.path.join(save_path, str(face_size), 'positive.txt')
        part_anno_dir = os.path.join(save_path, str(face_size), 'part.txt')
        negative_anno_dir = os.path.join(save_path, str(face_size), 'negative.txt')

        positive_anno_dir = open(positive_anno_dir, 'w')
        part_anno_dir = open(part_anno_dir, 'w')
        part_count = 0
        positive_count = 0
        negative_count = 0
        negative_anno_dir = open(negative_anno_dir, 'w')

        # 计数，方便为生成的样本数据标名字

        for line in lines:
            strs = line.split()
            img_name = strs[0].strip()
            boxes = np.array(strs[1:], dtype=np.int32)
            boxes_ = np.split(boxes, len(boxes) / 4)
            for box_ in boxes_:
                x1 = box_[0]
                y1 = box_[1]
                x2 = box_[2]
                y2 = box_[3]
                w = x2 - x1
                h = y2 - y1

                if w < 40 or h < 40:
                    break

                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2

                box = np.array([x1, y1, x2, y2])
                with Image.open(img_name) as img:
                    for i in range(5):
                        mw = np.random.randint(-w * 0.2, w * 0.2)
                        mh = np.random.randint(-w * 0.2, w * 0.2)
                        cx_ = cx + mw
                        cy_ = cy + mh
                        img_crop = img.crop(box)
                        # img_crop.show()
                        img_crop = img_crop.resize((face_size, face_size))
                        img_crop.save('{}/{}/positive/{}.{}.{}.{}.jpg'.format(save_path, face_size, x1, y1, x2, y2))

# for i, data in enumerate(wider.next()):
#     boxes = []
#     name = data.image_name
#
#     for bbox in data.bboxes:
#         # print(bbox)
#         img = Image.open(name)
#         img_crop = img.crop(bbox)
#         img_crop.save("F:\widerface\img/{}.{}.{}.{}.{}.jpg".format(bbox[0], bbox[1], bbox[2], bbox[3], i))
#         # boxes.append(bbox)
#
#     print(boxes)

# press ctrl-C to stop the process
# for data in wider.next():
#
#     im = cv2.imread(data.image_name)
#
#     im = im[:, :, (2, 1, 0)]
#     fig, ax = plt.subplots(figsize=(12, 12))
#     ax.imshow(im, aspect='equal')
#
#     for bbox in data.bboxes:
#
#         ax.add_patch(
#             plt.Rectangle((bbox[0], bbox[1]),
#                           bbox[2] - bbox[0],
#                           bbox[3] - bbox[1], fill=False,
#                           edgecolor='red', linewidth=3.5)
#             )
#
#     plt.axis('off')
#     plt.tight_layout()
#     plt.draw()
#     plt.show()
