# # -*- coding:utf-8 -*-
# import torch
# from torch import nn
# from torch.autograd import Variable
# from torch.autograd.gradcheck import zero_gradients
# import torchvision.transforms as T
# from torchvision.models.inception import inception_v3
# from PIL import Image
# import matplotlib.pyplot as plt
# import numpy as np 

# classes = eval(open('classes.txt').read())
# trans = T.Compose([T.ToTensor(),T.Lambda(lambda t:t.unsqueeze(0))])
# reverse_trans = lambda x:np.asarray(T.ToPILImage()(x))
# eps = 2*8/255
# steps = 40
# norm = float('inf')
# step_alpha = 0.0001
# model = inception_v3(pretrained=True,transforms_input=True).cpu()
# loss = nn.CrossEntropyLoss()
# model.eval()

# # visualization
# def loat_image(img_path):
# 	img = trans(Image.open(img_path).convert('RGB'))
# 	return img
# def get_class(img):
# 	x = Variable(img,volatile=True).cpu()
# 	cls = model(x).data.max(1)[1].cpu().numpy()[0]
# 	return classes[cls]
# def draw_result(img,noise,adv_img):
# 	fig,ax = plt.subplots(1,3,figsize=(15,10))
# 	orig_class,attack_class = get_class(img),get_class(adv_img)
# 	ax[0].imshow(reverse_trans(img[0]))
# 	ax[0].set_title('Original image:{}'.format(orig_class.split(','[0]))
# 	ax[1].imshow(noise[0].cpu().numpy().transpose(1,2,0))
# 	ax[1].set_title('Attacking noise')
# 	ax[2].imshow(reverse_trans(adv_img[0]))
# 	ax[2].set_title('Adversarial example:{}'.format(attack_class))
# 	for i in range(3):
# 		ax[i].set_axis_off()
# 		plt.tight_layout()
# 		plt.show()

# def non_targetd_attack(img):
# 	img = img.cpu()
# 	label = troch.zeros(1,1).cpu()
# 	x,y = Variable(img,requires_grad=True),Variable(label)
# 	for step in range(steps):
# 		zeros_gradients(x)
# 		out = model(x)
# 		y.data = out.data.max(1)[1]
# 		_loss = loss(out,y)
# 		_loss.backward()
# 		normed_grad = step_alpha*torch.sign(x.grad.data)
# 		step_adv = x.data + normed_grad
# 		adv = step_adv - img
# 		adv = torch.clamp(adv,-eps,eps)
# 		result = img + adv 
# 		result = torch.clamp(result,0.0,1.0)
# 		x.data = result
# 		return result.cpu(),adv.cpu()
# img = load_image('input.png')
# adv_img.noise = non_targetd_attack(img)
# draw_result(img,noise,adv_img)
import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.slim.nets as nets

tf.logging.set_verbosity(tf.logging.ERROR)
sess = tf.InteractiveSession()
image = tf.Variable(tf.zeros((299, 299, 3)))


def inception(image, reuse):
    preprocessed = tf.multiply(tf.subtract(tf.expand_dims(image, 0), 0.5), 2.0)
    arg_scope = nets.inception.inception_v3_arg_scope(weight_decay=0.0)
    with slim.arg_scope(arg_scope):
        logits, _ = nets.inception.inception_v3(
            preprocessed, 1001, is_training=False, reuse=reuse)
        logits = logits[:,1:] # ignore background class
        probs = tf.nn.softmax(logits) # probabilities
    return logits, probs

logits, probs = inception(image, reuse=False)
import tempfile
from urllib.request import urlretrieve
import tarfile
import os
data_dir = tempfile.mkdtemp()
inception_tarball, _ = urlretrieve(
    'http://download.tensorflow.org/models/inception_v3_2016_08_28.tar.gz')
tarfile.open(inception_tarball, 'r:gz').extractall(data_dir)
# tarfile.open(/home/zq/)
# tarfile.open('/home/zq/桌面', 'r:gz').extractall(data_dir)
restore_vars = [
    var for var in tf.global_variables()
    if var.name.startswith('InceptionV3/')
]
saver = tf.train.Saver(restore_vars)
saver.restore(sess, os.path.join(data_dir, 'inception_v3.ckpt'))

import json
import matplotlib.pyplot as plt
imagenet_json, _ = urlretrieve(
    'http://www.anishathalye.com/media/2017/07/25/imagenet.json')
with open(imagenet_json) as f:
    imagenet_labels = json.load(f)
def classify(img, correct_class=None, target_class=None):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 8))
    fig.sca(ax1)
    p = sess.run(probs, feed_dict={image: img})[0]
    ax1.imshow(img)
    fig.sca(ax1)

    topk = list(p.argsort()[-10:][::-1])
    topprobs = p[topk]
    barlist = ax2.bar(range(10), topprobs)
    if target_class in topk:
        barlist[topk.index(target_class)].set_color('r')
    if correct_class in topk:
        barlist[topk.index(correct_class)].set_color('g')
    plt.sca(ax2)
    plt.ylim([0, 1.1])
    plt.xticks(range(10),
               [imagenet_labels[i][:15] for i in topk],
               rotation='vertical')
    fig.subplots_adjust(bottom=0.2)
    plt.show()

import PIL
import numpy as np
img_path, _ = urlretrieve('http://www.anishathalye.com/media/2017/07/25/cat.jpg')
img_class = 281
img = PIL.Image.open(img_path)
big_dim = max(img.width, img.height)
wide = img.width > img.height
new_w = 299 if not wide else int(img.width * 299 / img.height)
new_h = 299 if wide else int(img.height * 299 / img.width)
img = img.resize((new_w, new_h)).crop((0, 0, 299, 299))
img = (np.asarray(img) / 255.0).astype(np.float32)
classify(img, correct_class=img_class)
