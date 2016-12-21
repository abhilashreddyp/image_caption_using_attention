import six.moves.cPickle as pkl
import numpy
import skimage.transform
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image

dict = {}
with open('results/attn_loss_acc.pkl', 'rb') as f:
	dict = pkl.load(f)

acc = dict['acc']
sum = 0
acc_avg = []
for i, a in enumerate(acc):
    sum = sum + a
    k = i + 1
    if i >= 1000:
        sum = sum - acc[i - 1000]
        k = 1000
    avg = sum/(k)
    acc_avg.append(avg)

numepochs = numpy.linspace(0, 20, num=6000)
plt.plot(numepochs, acc, alpha=0.5)
plt.plot(numepochs, acc_avg)
plt.xlabel("Number of Epochs")
plt.ylabel("Traning Accuracy")
plt.show()

loss = dict['loss']
sum = 0
loss_avg = []
for i, l in enumerate(loss):
    sum = sum + l
    k = i + 1
    if i >= 100:
        sum = sum - loss[i - 100]
        k = 100
    avg = sum/(k)
    loss_avg.append(avg)

plt.plot(numepochs, loss, alpha=0.3)
plt.plot(numepochs, loss_avg)
axes = plt.gca()
axes.set_ylim([0, 10])
plt.xlabel("Number of Epochs")
plt.ylabel("Traning Loss")
plt.show()
