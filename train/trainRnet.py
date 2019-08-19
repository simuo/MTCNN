import sys
import os

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

from src.MtcnnTrain import trainer
from src.MtcnnNet import RNet
import cfg

if __name__ == '__main__':
    # trainer = trainer(RNet(), '../models/rnet.pth', r'F:\MTCNN\celeba3\24', '../log/Rlog.txt')
    trainer = trainer(RNet(), '../models/rnet.pth', r'F:\celeba3\24', '../log/Rlog.txt')
    # trainer = trainer(RNet(), cfg.Rmodelpath, cfg.Rdata, cfg.Olog)
    trainer.train()
