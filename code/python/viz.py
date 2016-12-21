import six.moves.cPickle as pkl
import numpy
import skimage.transform
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image

# keep aspect ratio, and center crop
def LoadImage(file_name, resize=256, crop=224):
  image = Image.open(file_name)
  width, height = image.size

  if width > height:
    width = (width * resize) // height
    height = resize
  else:
    height = (height * resize) // width
    width = resize
  left = (width  - crop) // 2
  top  = (height - crop) // 2
  image_resized = image.resize((width, height), Image.BICUBIC).crop((left, top, left + crop, top + crop))
  data = numpy.array(image_resized.convert('RGB').getdata()).reshape(crop, crop, 3)
  data = data.astype('float32') / 255
  return data

img = LoadImage('results/dog.png')  
caption = "A dog leaps into the air in a grassy field surrounded by trees"
words = caption.split()
print(words)

alphas = None
with open('results/alphab.pkl', 'rb') as f:
    alphas = pkl.load(f)

print(alphas.shape)
alpha = alphas[605, :, :].reshape(44,1, 196)
print(alpha.shape)

# display the visualization
n_words = alpha.shape[0] + 1
w = numpy.round(numpy.sqrt(n_words))
h = numpy.ceil(numpy.float32(n_words) / w)
#plt.figure(figsize=(512, 512))    
plt.subplot(w, h, 1)
plt.imshow(img)
plt.axis('off')

smooth = True

for ii in range(len(words)):
    plt.subplot(w, h, ii+2)
    lab = words[ii]
    #if options['selector']:
    #    lab += '(%0.2f)'%sels[ii]
    plt.text(0, 1, lab, backgroundcolor='white', fontsize=13)
    plt.text(0, 1, lab, color='black', fontsize=13)
    plt.imshow(img)
    if smooth:
        alpha_img = skimage.transform.pyramid_expand(alpha[ii,0,:].reshape(14,14), upscale=16, sigma=20)
    else:
         alpha_img = skimage.transform.resize(alpha[ii,0,:].reshape(14,14), [img.shape[0], img.shape[1]])
    plt.imshow(alpha_img, alpha=0.8)
    plt.set_cmap(cm.Greys_r)
    plt.axis('off')
plt.show()
