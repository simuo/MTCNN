import os
from PIL import Image, ImageDraw
import numpy as np
from tools import utils
import traceback

imgs_path = r'G:\dataset\celeba\img_celeba\img_celeba'
anno_path = r'G:\dataset\celeba\Anno\list_bbox_celeba.txt'

save_path = r'F:\celeba3'

for face_size in [48, 24, 12]:
    print("Generating {} image ... ".format(face_size))

    # 造出的样本保存的路径
    positive_img_dir = os.path.join(save_path, str(face_size), 'positive')
    part_img_dir = os.path.join(save_path, str(face_size), 'part')
    negative_img_dir = os.path.join(save_path, str(face_size), 'negative')

    # for dir in [negative_img_dir]:
    for dir in [positive_img_dir, part_img_dir, negative_img_dir]:
        if not os.path.exists(dir):
            os.makedirs(dir)
    # 样本的参数保存路径
    positive_anno_dir = os.path.join(save_path, str(face_size), 'positive.txt')
    part_anno_dir = os.path.join(save_path, str(face_size), 'part.txt')
    negative_anno_dir = os.path.join(save_path, str(face_size), 'negative.txt')

    # positive_anno_dir = open(positive_anno_dir, 'w')
    part_anno_dir = open(part_anno_dir, 'w')
    # negative_anno_dir = open(negative_anno_dir, 'w')

    # 计数，方便为生成的样本数据标名字
    # positive_count = 0
    part_count = 0
    # negative_count = 0
    try:
        p = 0
        for i, line in enumerate(open(anno_path)):  # 打开标签文件，一行行读取
            if i < 2:
                continue
            strs = line.split()
            img_path = os.path.join(imgs_path, strs[0])  # 获取每张图片的文件名，并生成图片的绝对路径

            with Image.open(img_path) as img:  # 打开图片
                img_w, img_h = img.size  # 图片的宽高

                x1 = float(strs[1].strip())  # 原图上的框的左上角x坐标
                y1 = float(strs[2].strip())  # 原图上的框的左上角y坐标
                w = float(strs[3].strip())  # 框的宽
                h = float(strs[4].strip())  # 框的高
                x2 = x1 + w  # 右下角x坐标
                y2 = y1 + h  # 右下角y坐标

                # 过滤去掉（图片最小边小于40）或（左上角x,y坐标都小于0）或（宽高小于0）的图片
                if max(w, h) < 40 or x1 < 0 or y1 < 0 or w < 0 or h < 0:
                    continue

                # 原图人脸的框
                boxes = np.array([[x1, y1, x2, y2]])

                # imgdraw = ImageDraw.Draw(img)
                # imgdraw.rectangle((x1, y1, x2, y2), outline='red')
                # img.show()

                # 找到框的中心点
                cx = x1 + w / 2
                cy = y1 + h / 2

                for i in range(10):
                    # 让框的中心点移一移
                    mw = np.random.randint(-w * 0.3, w * 0.3)
                    mh = np.random.randint(-h * 0.3, h * 0.3)

                    # 新的中心点
                    cx_ = cx + mw
                    cy_ = cy + mh

                    # imgD = ImageDraw.Draw(img)
                    # imgD.rectangle((x1_, y1_, x2_, y2_), outline='blue')
                    # img.show()

                    # 将人脸框变为正方形
                    side_len = np.random.randint(int(min(w, h) * 0.3), np.ceil(max(w, h) * 0.5))
                    x1_ = cx_ - side_len / 2
                    y1_ = cy_ - side_len / 2

                    x2_ = cx_ + side_len / 2
                    y2_ = cy_ + side_len / 2

                    # 计算每个框的偏移值
                    offset_x1 = (x1 - x1_) / side_len
                    offset_y1 = (y1 - y1_) / side_len
                    offset_x2 = (x2 - x2_) / side_len
                    offset_y2 = (y2 - y2_) / side_len

                    crop_box = np.array([x1_, y1_, x2_, y2_])

                    # 剪切图片
                    img_crop = img.crop(crop_box)
                    img_resize = img_crop.resize((face_size, face_size))

                    iou = utils.iou(crop_box, boxes)

                    # if iou > 0.65:  # 正样本
                    #     positive_anno_dir.write(
                    #         'positive/{0}.jpg {1} {2} {3} {4} {5} {6}\n'.format(positive_count, 1, iou[0], offset_x1,
                    #                                                             offset_y1,
                    #                                                             offset_x2, offset_y2))
                    #     positive_anno_dir.flush()
                    #     img_resize.save(os.path.join(positive_img_dir, '{}.jpg'.format(positive_count)))
                    #     positive_count += 1

                    if iou < 0.5 and iou > 0.3:  # 部分样本
                        part_anno_dir.write(
                            'part/{0}.jpg {1} {2} {3} {4} {5} {6}\n'.format(part_count, 2, iou[0], offset_x1, offset_y1,
                                                                            offset_x2, offset_y2))
                        part_anno_dir.flush()
                        img_resize.save(os.path.join(part_img_dir, '{}.jpg'.format(part_count)))
                        part_count += 1
                    # if iou < 0.3:  # 负样本
                    #     negative_anno_dir.write('part/{0}.jpg {1} {2} {3} {4} {5}\n'.format(negative_count, 0, offset_x1, offset_y1, offset_x2,
                    #                                                     offset_y2))
                    #     negative_anno_dir.flush()
                    #     img_resize.save(os.path.join(negative_img_dir, '{}.jpg'.format(negative_count)))
                    #     negative_count += 1
                    #     print('-{}-'.format(p+1))
                    #     p+=1

                # for i in range(10):
                #     side_len = np.random.randint(face_size, min(img_w, img_h))
                #     x_ = np.random.randint(0, img_w - side_len)
                #     y_ = np.random.randint(0, img_h - side_len)
                #     crop_box = np.array([x_, y_, x_ + side_len, y_ + side_len])
                #     iou = utils.iou(crop_box, boxes)
                #     if iou < 0.03:
                #         face_crop = img.crop(crop_box)
                #         face_resize = face_crop.resize((face_size, face_size), Image.ANTIALIAS)
                #
                #         negative_anno_dir.write(
                #             "negative/{0}.jpg 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0\n".format(negative_count))
                #         negative_anno_dir.flush()
                #         face_resize.save(os.path.join(negative_img_dir, "{0}.jpg".format(negative_count)))
                #         negative_count += 1
                #         print('-{}-'.format(p + 1))
                #         p+=1
    except Exception as e:
        print("-------------------", str(e))
        print('traceback.format_exc():\n%s' % traceback.format_exc())
