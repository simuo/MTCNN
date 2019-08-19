import sys
import os
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

from src.MtcnnTrain import trainer
from src.MtcnnNet import ONet
import cfg

if __name__ == '__main__':
    # trainer = trainer(ONet(), '../models/onet.pth', r'F:\MTCNN\celeba3\48', '../log/Olog.txt')
    trainer = trainer(ONet(), '../models/onet.pth', r'F:\celeba3\48', '../log/Olog.txt')

    # trainer = trainer(ONet(), cfg.Omodelpath, cfg.Odata, cfg.Olog)
    trainer.train()
