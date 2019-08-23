import sys
import os
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

from src.MtcnnTrain import trainer
from src.myNet import P12Net

if __name__ == '__main__':
    trainer = trainer(P12Net(), '../models/pnet.pth', r'F:\celeba3\12', '../log/Plog.txt')
    trainer.train()
