import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils import data
import os
from src.loaddata import loaddata
import matplotlib.pyplot as plt
import time
from datetime import datetime
from lookahead import Lookahead

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


class trainer():
    def __init__(self, net, netsavepath, datasetpath, filename):
        self.net = net.to(device)
        self.netsavepath = netsavepath
        if os.path.exists(self.netsavepath):
            self.net = torch.load(self.netsavepath).to(device)
        print(net.__class__.__name__)
        from torch.utils.tensorboard import SummaryWriter
        self.writer = SummaryWriter(log_dir='./runs/{}_loss'.format(str(net.__class__.__name__)))
        self.datasetpath = datasetpath
        self.lossconf = nn.BCELoss()
        self.lossoffset = nn.MSELoss()
        self.filename = filename

        # self.optimizer = optim.Adam(self.net.parameters())
        self.optimizer = torch.optim.Adam(
            self.net.parameters(),
            lr=1e-3,
            betas=(
                0.9,
                0.999))  # Any optimizer
        self.lookahead = Lookahead(
            self.optimizer, k=5, alpha=0.5)  # Initialize Lookahead

    def train(self):

        epoch = 0
        losses = []
        dataset = loaddata(self.datasetpath)
        datas = data.DataLoader(
            dataset=dataset,
            # batch_size=3,
            batch_size=1024,
            shuffle=True,
            # num_workers=0,
            num_workers=4,
            drop_last=True)
        # j=0
        while True:
            # j=j+1
            starttime = time.time()
            for i, (imgdata, conf, offset) in enumerate(datas):
                imgdata = imgdata.to(device)
                conf = conf.to(device)
                offset = offset.to(device)

                outconf, outoffset = self.net(imgdata)
                conf = conf.view(-1, 1)
                outconf = outconf.view(-1, 1)

                _conf = conf[conf <= 1]
                outconf = outconf[conf <= 1]
                confloss = self.lossconf(outconf, _conf)

                # 召回率
                recall = torch.sum(outconf > 0.9).float() / torch.sum(conf == 1)
                # 精确度
                precision = torch.sum(_conf[torch.nonzero(outconf > 0.9)[:, 0]] == 1).float() \
                            / torch.sum(outconf > 0.9)

                outoffset = outoffset.view(-1, 4)
                # confidence = torch.nonzero(conf > 0)[:, 0]
                confidence = conf[:, 0] > 0
                outoffset = outoffset[confidence]
                offset = offset[confidence]
                offsetloss = self.lossoffset(outoffset, offset)

                loss = confloss + offsetloss
                conf = conf[conf < 2]
                confloss = self.lossconf(outconf, conf)

                outoffset = outoffset.view(-1, 4)
                outoffset = outoffset[offset > 0]
                offset = offset[offset > 0]
                offsetloss = self.lossoffset(outoffset, offset)

                loss = confloss + offsetloss
                self.writer.add_scalar(self.net.__class__.__name__+'_loss', loss.data, global_step=epoch)
                losses.append(loss)

                self.lookahead.zero_grad()
                loss.backward()  # Self-defined loss function
                self.lookahead.step()

                # if j % 10==0:
                with open(self.filename, 'a') as file:
                    file.write(
                        'epoch : {}    {} / {} , loss:{:.6f} , confloss:{:.5f} ,  offsetloss:{:.5f} \n'.format(
                            epoch,
                            i + 1,
                            len(datas),
                            loss.cpu().data.numpy(),
                            confloss.cpu().data.numpy(),
                            offsetloss.cpu().data.numpy()))
                    print((
                        'epoch : {}    {} / {} , loss:{:.6f} , confloss:{:.5f} ,  offsetloss:{:.5f}, recall:{}% , precision:{}%'.format(
                            epoch,
                            i + 1,
                            len(datas),
                            loss.cpu().data.numpy(),
                            confloss.cpu().data.numpy(),
                            offsetloss.cpu().data.numpy(),
                            recall.data * 100,
                            precision.data * 100)))
            print("saving.......")
            torch.save(self.net, self.netsavepath)
            print("save successfully!")
            epoch += 1
            endtime = time.time()
            with open(self.filename, 'a') as file:
                file.write(
                    '训练一轮所需时间：{0}分{1}秒\n{2}\n\n\n'.format(
                        (endtime - starttime) // 60, (endtime - starttime) %
                        60, str(
                            datetime.now())))
