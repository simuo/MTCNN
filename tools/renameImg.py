import os
from PIL import Image

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
imgpath = os.path.join(rootPath, 'IMG')

imgs = os.listdir(imgpath)
for i, img in enumerate(imgs):
    image = os.path.join(imgpath, img)
    newimage=os.path.join(imgpath,'{}.jpg'.format(i+1))
    os.rename(image,newimage)

