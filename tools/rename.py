import os

SRC_DIR = 'IMG'
for i, filename in enumerate(os.listdir(SRC_DIR)):
    os.rename(os.path.join(SRC_DIR, filename), os.path.join(SRC_DIR, '{}.jpg'.format(i + 1)))

import os

root = r'F:\celeba312\12'
with open(os.path.join(root, 'negative.txt')) as f:
    with open('tem.txt', 'w') as fp:
        for i, line in enumerate(f.readlines()):
            strs = line.split()
            name = strs[0].split('/')
            number = name[1].split('.')[0]
            # print(strs[0],strs[1],strs[2],strs[3],strs[4],strs[5])
            fp.write(
                'negative/{0}.jpg 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0\n'.format((i + 286155)))
# imgs=os.listdir(root)
# for i,img in enumerate(imgs):
#     # name=img.split('.')[0]
#     os.rename(os.path.join(root,img),os.path.join(root,'{}.jpg'.format(i+260721)))

