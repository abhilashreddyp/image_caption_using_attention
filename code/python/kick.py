from keras.models import Sequential
from keras.layers import Embedding, LSTM, Merge
from keras.layers.core import Flatten, Dense, Dropout, RepeatVector
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers.wrappers import TimeDistributed
from keras.optimizers import SGD
import cv2, numpy as np
import os
from collections import OrderedDict, defaultdict
import six.moves.cPickle as pkl
import h5py
import time

vocab_size=1000
embedding_vector_length=256
max_caption_len=16
output_dim=1000

image_dir="images/"
captions_dir="captions/"
vocab_dir="vocab/"
weights_dir="weights/"

def language_model():
    model = Sequential()
    print('Adding Embedding')
    model.add(Embedding(vocab_size, embedding_vector_length, input_length=max_caption_len))
    print('Adding LSTM')
    model.add(LSTM(output_dim, return_sequences=True))
    print('Adding TimeDistributed Dense')
    model.add(TimeDistributed(Dense(output_dim)))
    #model.add(Flatten())

    #print(model.summary())
    return model

def pop(model):
    '''Removes a layer instance on top of the layer stack.
    This code is thanks to @joelthchao https://github.com/fchollet/keras/issues/2371#issuecomment-211734276
    '''
    if not model.outputs:
        raise Exception('Sequential model cannot be popped: model is empty.')
    else:
        model.layers.pop()
        if not model.layers:
            model.outputs = []
            model.inbound_nodes = []
            model.outbound_nodes = []
        else:
            model.layers[-1].outbound_nodes = []
            model.outputs = [model.layers[-1].output]
        model.built = False

    return model

def VGG_16(weights_file=None):
    model = Sequential()
    model.add(ZeroPadding2D((1,1),input_shape=(3,224,224)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1000, activation='softmax'))
    
    #print(model.summary())
    print('Loading weights')
    if weights_file:
        model = load_weights(model, weights_file)
    print('Loaded weights')
    #model = pop(model)
    #model = pop(model)
    #model.layers.pop()
    #model.layers.pop()
    #model.layers.pop()

    #print(model.summary())
    return model

def build_model(weights_path):
    image_model = VGG_16(weights_path)
    image_model.add(RepeatVector(max_caption_len))
    print('Built Image Model')
    print('Building Language Model')
    lang_model = language_model()
    model = Sequential()
    model.add(Merge([image_model, lang_model], mode='concat',  concat_axis=-1))
    model.add(LSTM(embedding_vector_length, return_sequences=False))
    #print(vocab_size)
    model.add(Dense(vocab_size, activation='softmax'))

    #print(model.summary())
    return model


def train(images, partial_captions, next_words, v_size):
    global vocab_size
    vocab_size = v_size
    model=build_model('vgg16_weights.h5')
    print('Built model.')
    print('Compiling Now')
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    print('Fitting Now')
    #print(images.shape)
    #print(partial_captions.shape)
    #print(next_words.shape)
    model.fit([images, partial_captions], next_words, batch_size=3, nb_epoch=100)
    return model

def load_weights(model, weights_file):
    f = h5py.File(os.path.join(weights_dir, weights_file))
    for k in range(f.attrs['nb_layers']):
        if k >= len(model.layers):
            # we don't look at the last (fully-connected) layers in the savefile
            break
        g = f['layer_{}'.format(k)]
        weights = [g['param_{}'.format(p)] for p in range(g.attrs['nb_params'])]
        model.layers[k].set_weights(weights)
    f.close()
    return model

def predict(model, d_images, index_to_word):
    for image in d_images.values():
        caption = np.zeros(max_caption_len).reshape(1, 16)
        print(caption.shape)
        caption[0] = -1
        count=0
        sentence = []
        while True:
            out = model.predict([image, caption])
            index = out.argmax(-1)
            print(index)
            index = index[0]
            word = index_to_word[index]
            sentence.append(word)
            count+= 1
            if count >= max_caption_len or index == 0: #max caption length reach of '<eos>' encountered
                break
            caption[0,count] = index
        sent_str = " ".join(sentence)
        print("The Oracle says : %s" %sent_str)
'''
def main():
    images=load_images()
    print('Loaded images')
    captions=load_captions()
    print('Loaded captions')
    vocab = load_vocabulary()
    print('Loaded vocabulary')
    global vocab_size
    vocab_size = len(vocab)
    images, partial_captions, next_words = gen_image_partial_captions(images, captions, vocab)
    print('Loaded images, partial_captions, next_words')
    print('Training now')
    #import kick
    model = train(images, partial_captions, next_words)
    timestr = time.strftime("%Y%m%d-%H%M%S")
    file_name = 'weights_'+timestr+'.hf5'
    model.save_weights(file_name)
    print('Trained on %s images, saved weights to %s'%(len(images), file_name))
    for image in images.values():
        caption = np.zeros(max_caption_len)
        caption[0] = -1
        out = model.predict([image, caption])
        print(out.shape)
        print(out)

def load_images():
    images = OrderedDict()
    for image_file in os.listdir(image_dir):
        image = load_image(image_file)
        images[image_file] = image
    return images

def load_image(image_path):
    im = cv2.resize(cv2.imread(os.path.join(image_dir,image_path)), (224, 224)).astype(np.float32)
    im[:,:,0] -= 103.939
    im[:,:,1] -= 116.779
    im[:,:,2] -= 123.68
    im = im.transpose((2,0,1))
    im = np.expand_dims(im, axis=0)
    return im

def load_captions():
    captions = OrderedDict()
    lines = []
    with open(os.path.join(captions_dir, "ref.txt")) as f:
        lines = f.readlines()
    for line in lines:
        space_index = line.index(" ")
        image_name = line[0:space_index].strip()
        caption = line[space_index+1:].strip()
        captions[image_name] = caption
    return captions    

def load_vocabulary():
    with open(os.path.join(vocab_dir,'dictionary.pkl'), 'rb') as f:
        worddict = pkl.load(f)
    vocab = defaultdict(lambda : 1) # return 1, the index for 'UNK' by default
    for word, index in worddict.items():
        vocab[word] = index
    vocab['<eos>'] = 0
    vocab['UNK'] = 1
    #print(vocab)
    #print(len(vocab))
    #vocab_size = len(vocab.keys())
    return vocab

def gen_image_partial_captions(images, captions, vocab):
    a_images = []
    a_captions = []
    next_words = []
    #vocab_size = len(vocab)
    for image_name in images.keys():
        caption = captions[image_name]
        words = [""]
        words.extend(caption.split(" "))
        words.append('<eos>')
        #print(words)
        partial_caption_ar = np.zeros(max_caption_len, dtype=np.int)
        #No need to process <eos> tag
        for i in range(len(words) - 1):
            pc_copy = partial_caption_ar.copy()
            word = words[i]
            #print(word)
            index = -1 if i == 0 else vocab[word]
            pc_copy[i] = index
            #Generate input image and partial caption vectors
            a_images.append(images[image_name])
            a_captions.append(pc_copy)
            #Generate next word output vector
            next_word = words[i + 1]
            next_word_index = vocab[next_word]
            #print(next_word_index)
            next_word_ar = np.zeros(vocab_size, dtype=np.int)
            next_word_ar[next_word_index] = 1
            next_words.append(next_word_ar)
            #print(next_word_ar.shape)
    #print(next_words)
    v_i = np.vstack(a_images)
    v_c = np.vstack(a_captions)
    v_nw = np.vstack(next_words)
    return v_i, v_c, v_nw 

if __name__ == "__main__":
    main()'''