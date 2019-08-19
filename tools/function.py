import urllib.request
import urllib.parse
import re
import os
from PIL import Image
import random
import sys
sys.path[0]=r''

def handle_data(source_path,size=(100, 100), channels='RGB', save_path='D:\picture'):
    listdir = os.listdir(source_path)
    try:
        if os.path.exists(save_path):
            print("找到了{}".format(save_path))
        else:
            os.mkdir(save_path)
    except FileNotFoundError:
        print("找不到{}".format(save_path))
    for i, imgname in enumerate(listdir):
        imgpath = os.path.join(source_path,imgname)
        img = Image.open(imgpath)
        img = img.resize(size).convert(channels)    #重置图片大小并转换通道
        img.save('{}\{}.jpg'.format(save_path,i))

def paste_logo(logo_path,source_path):
    listdir = os.listdir(source_path)
    for i, imgname in enumerate(listdir):
        logo = Image.open(logo_path).convert('RGB')
        logo_size = random.randint(30, 50)
        logo = logo.resize((logo_size, logo_size))
        logo_xpos = random.randint(0, 150 - logo_size)
        logo_ypos = random.randint(0, 150 - logo_size)
        imgpath = os.path.join(source_path, imgname)
        img = Image.open(imgpath)
        img.paste(logo, (logo_xpos, logo_ypos))
        img.save("../imgdata\{}.{}.{}.{}.jpg".format(logo_xpos, logo_ypos, logo_xpos+logo_size, logo_ypos+logo_size))

def download_data():
    # 添加header，其中Referer是必须的,否则会返回403错误，User-Agent是必须的，这样才可以伪装成浏览器进行访问
    header = \
        {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/56.0.2924.87 Safari/537.36',
            "referer": "https://image.baidu.com"
        }
    url = "https://image.baidu.com/search/acjson?tn=resultjson_com&ipn=rj&ct=201326592&is=&fp=result&queryWord={word}&cl=2&lm=-1&ie=utf-8&oe=utf-8&adpicid=&st=-1&z=&ic=0&word={word}&s=&se=&tab=&width=&height=&face=0&istype=2&qc=&nc=1&fr=&cg=girl&pn={pageNum}&rn=30&gsm=1e00000000001e&1490169411926="
    keyword = input("请输入搜索关键字：")
    # 转码
    keyword = urllib.parse.quote(keyword, 'utf-8')

    n = 0
    j = 0

    while (n < 3000):
        error = 0
        n += 30
        # url
        url1 = url.format(word=keyword, pageNum=str(n))
        # 获取请求
        rep = urllib.request.Request(url1, headers=header)
        # 打开网页
        rep = urllib.request.urlopen(rep)
        # 获取网页内容
        try:
            html = rep.read().decode('utf-8')
            # print(html)
        except:
            print("出错了！")
            error = 1
            print("出错页数：" + str(n))
        if error == 1:
            continue
        # 正则匹配
        p = re.compile("thumbURL.*?\.jpg")
        # 获取正则匹配到的结果，返回list
        s = p.findall(html)
        if os.path.isdir("D://pic") != True:
            os.makedirs("D://pic")
        with open("testpic.txt", "a") as f:
            # 获取图片
            for i in s:
                print(i)
                i = i.replace('thumbURL":"', '')
                print(i)
                f.write(i)
                f.write("\n")
                # 保存图片
                urllib.request.urlretrieve(i, r"D:\celeba3\negative\pic{num}.jpg".format(num=j))
                j += 1
            f.close()
    print("总共爬取图片数为：" + str(j))

# handle_data("D:\pic",(150,150))
# paste_logo('9.jpg','D:\picture')

# download_data()