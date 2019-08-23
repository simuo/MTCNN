import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils import data
from datetime import datetime
import matplotlib.pyplot as plt
from lookahead import Lookahead
from src.loaddata import loaddata
from torch.utils.tensorboard import SummaryWriter

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class trainer():
    def __init__(self, net, netsavepath, datasetpath, filename):
        self.net = net.to(device)
        self.netsavepath = netsavepath
        if os.path.exists(self.netsavepath):
            self.net = torch.load(self.netsavepath).to(device)
        print(net.__class__.__name__)
        self.writer = SummaryWriter(log_dir='./runs/{}loss'.format(str(net.__class__.__name__[0])))
        self.datasetpath = datasetpath
        self.lossconf = nn.BCELoss()
        self.lossoffset = nn.MSELoss()
        self.iouloss = nn.MSELoss()
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
        print(device)
        epoch = 0
        losses = []
        dataset = loaddata(self.datasetpath)
        datas = data.DataLoader(
            dataset=dataset,
            # batch_size=3,
            batch_size=64,
            shuffle=True,
            # num_workers=0,
            num_workers=4,
            drop_last=True)
        # j=0
        while True:
            # j=j+1
            starttime = time.time()
            for i, (imgdata, conf, iou, offset) in enumerate(datas):
                imgdata = imgdata.to(device)
                conf = conf.to(device)
                iou = iou.to(device)
                offset = offset.to(device)

                outconf, outoffset, outiou = self.net(imgdata)  # 网络输出置信度、偏移量、iou
                conf = conf.view(-1, 1)
                outconf = outconf.view(-1, 1)
                # 置信度损失
                _conf = conf[conf <= 1]
                outconf = outconf[conf <= 1]
                confloss = self.lossconf(outconf, _conf)

                # 召回率
                recall = torch.sum(outconf > 0.9).float() / torch.sum(conf == 1)
                # 精确度
                precision = torch.sum(_conf[torch.nonzero(outconf > 0.9)[:, 0]] == 1).float() \
                            / torch.sum(outconf > 0.9)

                # 偏移量损失
                outoffset = outoffset.view(-1, 4)
                # confidence = torch.nonzero(conf > 0)[:, 0]
                confidence = conf[:, 0] > 0  # part、positive的索引
                outoffset = outoffset[confidence]  # 输出结果为part、与positive的
                offset = offset[confidence]  # part、positive
                offsetloss = self.lossoffset(outoffset, offset)
                # iou损失
                outiou = outiou[confidence]  # 输出iou为part、positive
                iou = iou[confidence]  # part、positive的iou
                iouloss = self.iouloss(outiou.view(-1, 1), iou)

                loss = confloss + offsetloss + iouloss
                self.writer.add_scalars(self.net.__class__.__name__[0] + 'loss', {
                    'loss': loss.data,
                    'confloss': confloss.data,
                    'offsetloss': offsetloss.data,
                    'iouloss': iouloss.data
                }, global_step=i)

                losses.append(loss)

                self.lookahead.zero_grad()
                loss.backward()  # Self-defined loss function
                self.lookahead.step()

                # if j % 10==0:
                with open(self.filename, 'a') as file:
                    file.write(
                        'epoch : {} | {} / {} , loss:{:.6f} , confloss:{:.5f} ,  offsetloss:{:.5f} \n'.format(
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
                file.write('训练一轮所需时间：{0}分{1}秒\n{2}\n\n\n'.format(
                    (endtime - starttime) // 60, (endtime - starttime) % 60, str(datetime.now())))
