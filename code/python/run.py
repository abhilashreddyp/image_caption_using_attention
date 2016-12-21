import cv2, numpy as np
import time
import os
from collections import OrderedDict, defaultdict
import six.moves.cPickle as pkl

vocab_size=1000
embedding_vector_length=256
max_caption_len=16
output_dim=4096

image_dir="images/"
captions_dir="captions/"
vocab_dir="vocab/"
weights_dir="weights/"


def main():
    d_images=load_images()
    print('Loaded images')
    d_captions=load_captions()
    print('Loaded captions')
    vocab, vocab2 = load_vocabulary()
    print('Loaded vocabulary')
    global vocab_size
    vocab_size = len(vocab)
    images, partial_captions, next_words = gen_image_partial_captions(d_images, d_captions, vocab)
    print('Loaded images, partial_captions, next_words')
    print('Training now')
    import kick
    model = kick.train(images, partial_captions, next_words, vocab_size)
    timestr = time.strftime("%Y%m%d-%H%M%S")
    file_name = 'weights_'+timestr+'.hf5'
    model.save_weights(file_name)
    print('Trained on %s images, saved weights to %s'%(len(images), file_name))
    print("Predicting now")
    kick.predict(model, d_images, vocab2)

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
    vocab2 = defaultdict(lambda : "")
    for word, index in worddict.items():
        vocab[word] = index
        vocab2[index] = word
    vocab['<eos>'] = 0
    vocab['UNK'] = 1
    vocab2[0] = '<eos>'
    vocab2[1] = 'UNK'
    #print(vocab)
    #print(len(vocab))
    #vocab_size = len(vocab.keys())
    return vocab, vocab2

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
    print(a_images)
    v_i = np.vstack(a_images)
    v_c = np.vstack(a_captions)
    v_nw = np.vstack(next_words)
    print(v_i.shape)
    return v_i, v_c, v_nw 

if __name__ == "__main__":
    main()