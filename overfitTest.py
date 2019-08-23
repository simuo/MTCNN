import os
import cv2
import torch
import src.originNet as Net
import PIL.Image as Image
import torchvision.transforms as tf


def verify(model:str, img:str):

    img = Image.open(img)
    side = img.size[0]
    data = tf.ToTensor()(img).unsqueeze(dim=0) - 0.5

    if os.path.exists(model):
        net = torch.load(model)
    else:
        if side == 12:
            net = Net.PNet().cuda()
        elif side == 24:
            net = Net.RNet().cuda()
        elif side== 48:
            net = Net.ONet().cuda()
        else:
            raise RuntimeError

    with torch.no_grad():
        confi, offset = net(data.cuda())

    if offset.ndimension() > 2:
        offset = offset.permute(0, 3, 2, 1).reshape(shape=(-1))
    else:
        offset = offset.reshape(-1)
    print(offset)
    x1_offset, y1_offset, x2_offset, y2_offset = offset*side
    x1, y1, x2, y2 = x1_offset, y1_offset, x2_offset+side, y2_offset+side
    return [int(x.item()) for x in [x1, y1, x2, y2]]


def draw(coordinate:list, img:str):
    img = cv2.imread(img)
    x1, y1, x2, y2 = coordinate
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 3)
    cv2.imshow("Test-Overfit", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    model = "F:/MTCNN/weight/onet.pth"
    img = "F:/celeba3/48/positive/115.jpg"
    draw(verify(model, img), img)


