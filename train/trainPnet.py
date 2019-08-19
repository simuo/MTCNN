import sys
import os
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

from src.MtcnnTrain import trainer
from src.MtcnnNet import PNet
import cfg

if __name__ == '__main__':
    # trainer = trainer(PNet(), '../models/pnet.pth', r'F:\MTCNN\celeba3\12', '../log/Plog.txt')
    trainer = trainer(PNet(), '../models/pnet.pth', r'F:\celeba3\12', '../log/Plog.txt')
    # trainer = trainer(PNet(), cfg.Pmodelpath, cfg.Pdata, cfg.Plog)
    trainer.train()
