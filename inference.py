#-*-coding:utf-8-*-
__author__ = 'taobiaoli'

import onnxruntime as rt
import numpy as np
import cv2
import onnxruntime.backend as backend
from onnx import load
import onnx
from pkl_reader import DataGenerator
import timeit


#backend onnxruntime with sess, we need choose python3.5.2


def top5_acc(pred,k=5):
    Inf = 0.
    results =[]
    for i in range(k):
       results.append(pred.index(max(pred)))
       pred[pred.index(max(pred))] = Inf
    return results

def inference(model_path,data_path):
    sess = rt.InferenceSession(model_path)
    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name
    acc_top1 = 0
    acc_top5 = 0
    img = cv2.imread('ILSVRC2012_val_00049517.JPEG')
    img = cv2.resize(img,(224,224))
    img = np.transpose(img,(2,0,1))
    print(img.shape)
    img = img.astype('float32')/255
    img = img.reshape(1,224,224,3)
    print(img.shape)
    print(img.dtype)
    starttime = timeit.default_timer()
    res = sess.run([output_name],{input_name:img})
    endtime = timeit.default_timer()
    print('cost time: ',endtime-starttime)
    print('result:',np.argmax(res))
'''    
    dg = DataGenerator(data_path,model = 'mobilenet', dtype='float32')

    for im, label in dg.generator():
         res = sess.run([output_name],{input_name:im})
         if(np.argmax(res) == label):
             acc_top1 = acc_top1 + 1
         if label in top5_acc(res):
             acc_top5 = acc_top5 + 1
    print('top1 accuracy: {}'.format(acc_top1/50000))
    print('top5 accuracy: {}'.format(acc_top5/50000))
    
'''
'''
input_name = sess.get_inputs()[0].name
print('input name',input_name)
#input_shape = sess.get_inputs()[0].shape
#print('input shape',input_shape)

input_type = sess.get_inputs()[0].type
print('input type',input_type)

output_name = sess.get_outputs()[0].name
print('output name',output_name)


#backend foronnxruntime for backend ,we need choose python3.5.2
#model = onnx.load('alex_cat_dog.onnx')
#rep = backend.prepare(model,'CPU')

#prepare for model input image
img = cv2.imread('0050.jpg')
img = cv2.resize(img,(224,224))
img = np.transpose(img,(2,0,1))
print(img.shape)
img = img.astype('float32')/255
img = img.reshape(1,224,224,3)
print(img.shape)
print(img.dtype)

#backend onnxruntime with sess
res = sess.run([output_name],{input_name:img})
print(res)
print(np.argmax(res))
'''
if __name__ == '__main__':
    inference('./mobilenet.onnx','./data/val224_compressed.pkl')

