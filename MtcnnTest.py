import torch
import torchvision
import numpy as np
import time
from tools import utils
import time
import os
import cv2
from PIL import Image, ImageDraw, ImageFont
import traceback
from numba import jit, autojit


class mtcnnTest():
    def __init__(self, pnetpath, rnetpath, onetpath, scale):
        self.pnetpath = pnetpath
        self.rnetpath = rnetpath
        self.onetpath = onetpath
        self.pnet = torch.load(self.pnetpath)
        self.rnet = torch.load(self.rnetpath)
        self.onet = torch.load(self.onetpath)
        self.pnet.eval()
        self.rnet.eval()
        self.onet.eval()

        self.scale = scale
        self.transform = torchvision.transforms.ToTensor()

    def detect_pic(self, image_path):
        img_path = image_path
        img_dirs = os.listdir(img_path)
        for m in img_dirs:
            img_names = os.path.join(img_path, m)
            _im = cv2.imread(img_names, 1)
            im = np.array(cv2.cvtColor(_im, cv2.COLOR_BGR2RGB))
            print('TESTING--{}'.format(img_names))
            """??????????????????????????????????????P网络测试??????????????????????????????????????"""
            try:
                pnet_boxes = self.__pnet_detect(im)
                rnet_boxes = self.__rnet_detect(im, pnet_boxes)
                onet_boxes = self.__onet_detect(im, rnet_boxes)
                o_end = time.process_time()
                if len(pnet_boxes) == 0:  # 若没有返回框，则返回np.array([[0, 0, 1, 1, 0]])
                    pnet_boxes = np.array([[0, 0, 0, 0, 0]])
                elif len(rnet_boxes) == 0:
                    rnet_boxes = np.array([[0, 0, 0, 0, 0]])
                elif len(onet_boxes) == 0:
                    onet_boxes = np.array([[0, 0, 0, 0, 0]])
                self.draw(_im, pnet_boxes)
                self.draw(_im, rnet_boxes)
                self.draw(_im, onet_boxes)
            except Exception as e:
                print(str(e))
                print('traceback.format_exc():\n%s' % traceback.format_exc())

    def __pnet_detect(self, image):
        boxes = torch.Tensor([])

        img = image
        w, h = img.shape[1], img.shape[0]  # opencv2的宽、高
        min_side_len = min(w, h)
        scale = 1
        tt = time.process_time()
        while min_side_len > 12:
            img_data = self.transform(img) - 0.5
            img_data = img_data.cuda().unsqueeze_(0)
            img_data = img_data.float()

            _cls, _offset = self.pnet(img_data)
            cls, offset = _cls[0].cpu().data.permute(1, 2, 0), _offset[0].cpu().data.permute(1, 2, 0)

            idxs = torch.nonzero(torch.gt(cls, 0.9))
            _x1 = (idxs[:, 1] * 2).float() / scale
            _y1 = (idxs[:, 0] * 2).float() / scale
            _x2 = (idxs[:, 1] * 2 + 12).float() / scale
            _y2 = (idxs[:, 0] * 2 + 12).float() / scale
            bw = _x2 - _x1
            bh = _y2 - _y1
            boxoffset = offset[idxs[:, 0], idxs[:, 1]]
            x1 = _x1.float() + bw.float() * boxoffset[:, 0]
            y1 = _y1.float() + bh.float() * boxoffset[:, 1]
            x2 = _x2.float() + bw.float() * boxoffset[:, 2]
            y2 = _y2.float() + bh.float() * boxoffset[:, 3]
            conf = cls[idxs[:, 0], idxs[:, 1], idxs[:, -1]]
            box = torch.stack([x1, y1, x2, y2, conf], dim=1)
            scale *= self.scale
            _w = int(w * scale)
            _h = int(h * scale)

            img = cv2.resize(image, (_w, _h), interpolation=cv2.INTER_AREA)
            min_side_len = min(_w, _h)
            boxes = torch.cat((boxes, box), dim=0)
        ee = time.process_time()
        print('Pnet time:', ee - tt)
        return utils.nms(np.array(boxes), 0.3)

    def __rnet_detect(self, image, pnet_boxes):
        _img_dataset = []
        try:
            _pnet_boxes = utils.convert_to_square(pnet_boxes)
            tt = time.process_time()
            for _box in _pnet_boxes:
                _x1 = int(_box[0])
                _y1 = int(_box[1])
                _x2 = int(_box[2])
                _y2 = int(_box[3])

                if _x1 <= 0 or _y1 <= 0 or _x2 <= 0 or _y2 <= 0 or _x1 > _x2 or _y1 > _y2:
                    continue
                img = image[_y1:_y2, _x1:_x2]  # 抠图
                if img.shape[0] <= 0 or img.shape[1] <= 0:
                    continue
                img = cv2.resize(img, (24, 24))
                img_data = self.transform(img) - 0.5
                _img_dataset.append(img_data)
            ee = time.process_time()
            print('Rf  time:', ee - tt)
            img_dataset = torch.stack(_img_dataset, dim=0).cuda()

            aa = time.process_time()
            _cls, _offset = self.rnet(img_dataset)
            bb = time.process_time()
            print('Rnet  time:', bb - aa)
            cls = _cls.cpu().data.numpy()
            offset = _offset.cpu().data.numpy()

            idxs, _ = np.where(cls > 0.9)
            _box = _pnet_boxes[idxs]
            _x1 = _pnet_boxes[idxs][:, 0]
            _y1 = _pnet_boxes[idxs][:, 1]
            _x2 = _pnet_boxes[idxs][:, 2]
            _y2 = _pnet_boxes[idxs][:, 3]
            bw = _x2 - _x1
            bh = _y2 - _y1
            x1 = _x1 + bw * offset[idxs][:, 0]
            y1 = _y1 + bh * offset[idxs][:, 1]
            x2 = _x2 + bw * offset[idxs][:, 2]
            y2 = _y2 + bh * offset[idxs][:, 3]
            conf = cls[idxs, 0]
            boxes = np.stack([x1, y1, x2, y2, conf], axis=1)
            return utils.nms(np.array(boxes), 0.3, isMin=True)
        except Exception as e:
            print(traceback.format_exc())

    def __onet_detect(self, image, rnet_boxes):
        _img_dataset = []
        if rnet_boxes is None:
            return
        _rnet_boxes = utils.convert_to_square(rnet_boxes)
        tt = time.process_time()
        for _box in _rnet_boxes:
            _x1 = int(_box[0])
            _y1 = int(_box[1])
            _x2 = int(_box[2])
            _y2 = int(_box[3])

            if _x1 <= 0 or _y1 <= 0 or _x2 <= 0 or _y2 <= 0 or _x1 > _x2 or _y1 > _y2:
                continue
            img = image[_y1:_y2, _x1:_x2]
            if img.shape[0] <= 0 or img.shape[1] <= 0:
                continue
            img = cv2.resize(img, (48, 48))
            img_data = self.transform(img) - 0.5
            _img_dataset.append(img_data)
        ee = time.process_time()
        print('Of time:', ee - tt)
        if _img_dataset is None:
            return
        try:
            img_dataset = torch.stack(_img_dataset, dim=0)
            img_dataset = img_dataset.cuda()
            aa = time.process_time()
            _cls, _offset = self.onet(img_dataset)
            bb = time.process_time()
            print('Onet time:', bb - aa)

            cls = _cls.cpu().data.numpy()
            offset = _offset.cpu().data.numpy()

            idxs, _ = np.where(cls > 0.999)

            _box = _rnet_boxes[idxs]
            _x1 = _rnet_boxes[idxs][:, 0]
            _y1 = _rnet_boxes[idxs][:, 1]
            _x2 = _rnet_boxes[idxs][:, 2]
            _y2 = _rnet_boxes[idxs][:, 3]
            bw = _x2 - _x1
            bh = _y2 - _y1
            x1 = _x1 + bw * offset[idxs][:, 0]
            y1 = _y1 + bh * offset[idxs][:, 1]
            x2 = _x2 + bw * offset[idxs][:, 2]
            y2 = _y2 + bh * offset[idxs][:, 3]
            conf = cls[idxs, 0]
            boxes = np.stack([x1, y1, x2, y2, conf], axis=1)

            return utils.nms(np.array(boxes), 0.3, isMin=True)
        except Exception as e:
            print(traceback.format_exc())

    def draw(self, image, boxes):
        img = image.copy()
        for i, box in enumerate(boxes):
            x1 = int(box[0])
            y1 = int(box[1])
            x2 = int(box[2])
            y2 = int(box[3])
            font = cv2.FONT_HERSHEY_COMPLEX
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), thickness=1)
            cv2.putText(img, str(box[4]), (x1, y1 - 8), font, 0.3, (255, 0, 0), 1)
        return img

    def detect_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        while (cap.isOpened()):
            # Capture frame-by-frame
            ret, frame = cap.read()
            pnet_boxes = self.__pnet_detect(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            rnet_boxes = self.__rnet_detect(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), pnet_boxes)
            onet_boxes = self.__onet_detect(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), rnet_boxes)
            if onet_boxes is None:
                cv2.imshow('img', frame)
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    break
                continue
            Detected_image=self.draw(frame, onet_boxes)
            cv2.imshow('Video', Detected_image)
            if cv2.waitKey(0) & 0xFF == ord('q'):
                break
        # When everything done, release the capture
        cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    detector = mtcnnTest(r'models/pnet.pth',r'models/rnet.pth',r'models/onet.pth',scale=0.707)
    detector.detect_video(r"G:\CloudMusic\MV\testvi.mp4")
