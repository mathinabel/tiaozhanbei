# 导入flask以及相关子模块(安装方式：pip install flask)
import os

from flask import Flask, render_template, request
import requests
import json
import cv2
import base64


def cv2_to_base64(image):
    data = cv2.imencode('.jpg', image)[1]
    return base64.b64encode(data.tostring()).decode('utf8')


class net_api:

    def send_http_maskDetect(img):
        # 发送HTTP请求
        data = {'images': [cv2_to_base64(cv2.imread(img))]}
        headers = {"Content-type": "application/json"}
        url = "http://120.79.178.226:8867/predict/pyramidbox_lite_server_mask"
        r = requests.post(url=url, headers=headers, data=json.dumps(data))

        # 打印预测结果
        print(r.json()["results"])
        a = r.json()["results"]
        return json.dumps(a)

    def send_http_ocr(img):
        # 发送HTTP请求
        data = {'images': [cv2_to_base64(cv2.imread(img))]}
        headers = {"Content-type": "application/json"}
        url = "http://120.79.178.226:8866/predict/chinese_ocr_db_crnn_server"
        r = requests.post(url=url, headers=headers, data=json.dumps(data))

        # 打印预测结果
        print(r.json()["results"])
        a = r.json()["results"]
        return json.dumps(a)

    def send_http_genLoveWords(lw):
        data = {'texts': [lw],
                'use_gpu': False, 'beam_width': 1}
        headers = {"Content-type": "application/json"}
        url = "http://122.112.246.244:8866/predict/ernie_gen_lover_words"
        r = requests.post(url=url, headers=headers, data=json.dumps(data))

        # 返回预测结果
        print(r.json()["results"])
        a = r.json()["results"]
        return json.dumps(a)

    def send_http_genAcrostic(acr):
        data = {'texts': [acr],
                'use_gpu': False, 'beam_width': 1}
        headers = {"Content-type": "application/json"}
        url = "http://127.0.0.1:8866/predict/ernie_gen_acrostic_poetry"
        r = requests.post(url=url, headers=headers, data=json.dumps(data))

        # 打印预测结果
        print(r.json()["results"])
        a = r.json()["results"]
        return json.dumps(a)

    def send_http_genCouplet(cou):
        # 发送HTTP请求
        data = {'texts': [cou],
                'use_gpu': False, 'beam_width': 1}
        headers = {"Content-type": "application/json"}
        url = "http://127.0.0.1:8867/predict/ernie_gen_couplet"
        r = requests.post(url=url, headers=headers, data=json.dumps(data))

        # 打印预测结果
        print(r.json()["results"])
        a = r.json()["results"]
        return json.dumps(a)


# 获取项目当前绝对路径
# 比如我的项目在桌面上存放则得到——"C:\Users\asus\Desktop\shou"
basedir = os.path.abspath(os.path.dirname(__file__))

# 实例
app = Flask(__name__)


# 在根路由下返回上面的表单页面
@app.route('/', methods=['GET'])
def index():
    return render_template('test.html')


# 表单提交路径，需要指定接受方式
@app.route('/maskDetect', methods=['GET', 'POST'])
def getImg():
    # 通过表单中name值获取图片
    imgData = request.files["maskDetect"]

    # 设置图片要保存到的路径
    path = basedir

    # 获取图片名称及后缀名
    imgName = imgData.filename

    # 图片path和名称组成图片的保存路径
    file_path = path + '\\' + imgName
    print(file_path)
    # 保存图片
    imgData.save(imgName)

    # url是图片的路径
    url = imgName
    print(url)

    return net_api.send_http_maskDetect(url)


@app.route('/ocr', methods=['GET', 'POST'])
def getImgAndOCR():
    # 通过表单中name值获取图片
    imgData = request.files["ocr"]

    # 设置图片要保存到的路径
    path = basedir

    # 获取图片名称及后缀名
    imgName = imgData.filename

    # 图片path和名称组成图片的保存路径
    file_path = path + '\\' + imgName
    print(file_path)
    # 保存图片
    imgData.save(imgName)

    # url是图片的路径
    url = imgName
    print(url)

    return net_api.send_http_ocr(url)


@app.route('/lovePoem', methods=['GET', 'POST'])
def genLoveWords():
    lw = request.args.get("lovePoem")
    return net_api.send_http_genLoveWords(lw)


@app.route('/acrostic', methods=['GET', 'POST'])
def genAcrostic():
    ac = request.args.get("acrostic")
    return net_api.send_http_genAcrostic(ac)


@app.route('/couplet', methods=['GET', 'POST'])
def genCouplet():
    co = request.args.get("couplet")
    return net_api.send_http_genCouplet(co)


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)
