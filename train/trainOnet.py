import sys
import os
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

from src.MtcnnTrain import trainer
from src.myNet import O48Net

if __name__ == '__main__':
    trainer = trainer(O48Net(), '../models/onet.pth', r'F:\celeba3\48', '../log/Olog.txt')
    trainer.train()
