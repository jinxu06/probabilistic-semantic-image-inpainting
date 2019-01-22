import os
import sys
import numpy as np
from PIL import Image

def transform():
    source = "/data/ziz/not-backed-up/datasets-ziz-all/raw_data/lsun/church_outdoor/train"
    dest = "/data/ziz/not-backed-up/jxu/church_outdoor/train"
    destv = "/data/ziz/not-backed-up/jxu/church_outdoor/valid"

    filenames = []
    for entry in os.scandir(source):
        if not entry.name.startswith('.') and entry.is_file():
            filenames.append(entry.name)
    filenames = sorted(filenames)
    filenames_train = filenames[:110000]
    filenames_valid = filenames[110000:]

    for i, filename in enumerate(filenames_valid):
        if i% 1000==0:
            print(i)
        img = Image.open(os.path.join(source, filename))
        w, h = img.size
        m = min(h, w)
        img = img.crop(((w-m)//2, (h-m)//2, (w-m)//2+m, (h-m)//2+m))
        img = img.resize((32,32))
        img = img.convert("RGB")
        img.save(os.path.join(destv, str(i+1).zfill(6)+'.jpg'), 'jpeg')

    for i, filename in enumerate(filenames_train):
        if i% 1000==0:
            print(i)
        img = Image.open(os.path.join(source, filename))
        w, h = img.size
        m = min(h, w)
        img = img.crop(((w-m)//2, (h-m)//2, (w-m)//2+m, (h-m)//2+m))
        img = img.resize((32,32))
        img = img.convert("RGB")
        img.save(os.path.join(dest, str(i+1).zfill(6)+'.jpg'), 'jpeg')

def transform_test():
    source = "/data/ziz/not-backed-up/datasets-ziz-all/raw_data/lsun/church_outdoor/val"
    dest = "/data/ziz/not-backed-up/jxu/church_outdoor/test"

    filenames = []
    for entry in os.scandir(source):
        if not entry.name.startswith('.') and entry.is_file():
            filenames.append(entry.name)
    filenames = sorted(filenames)
    for i, filename in enumerate(filenames):
        if i% 1000==0:
            print(i)
        img = Image.open(os.path.join(source, filename))
        w, h = img.size
        m = min(h, w)
        img = img.crop(((w-m)//2, (h-m)//2, (w-m)//2+m, (h-m)//2+m))
        img = img.resize((32,32))
        img = img.convert("RGB")
        img.save(os.path.join(dest, str(i+1).zfill(6)+'.jpg'), 'jpeg')


def read_imgs(dir, limit=-1):
    filenames = []
    for entry in os.scandir(dir):
        if limit==0:
            break
        if not entry.name.startswith('.') and entry.is_file():
            limit -= 1
            filenames.append(entry.name)
    filenames = sorted(filenames)
    imgs = []
    for i, filename in enumerate(filenames):
        img = Image.open(os.path.join(dir, filename))
        img = np.array(img)
        imgs.append(img)
    imgs = np.array(imgs).astype(np.uint8)
    return imgs

def load(data_dir, subset='train', size=64, limit=-1):
    if subset in ['train', 'valid', 'test']:
        trainx = read_imgs(os.path.join(data_dir, "{0}".format(subset)), limit=limit)
        trainy = np.ones((trainx.shape[0], ))
        return trainx, trainy
    else:
        raise NotImplementedError('subset should be either train, valid or test')

class DataLoader(object):

    def __init__(self, data_dir, subset, batch_size, rng=None, shuffle=False, return_labels=False, size=64, limit=-1):
        """
        - data_dir is location where to store files
        - subset is train|test
        - batch_size is int, of #examples to load at once
        - rng is np.random.RandomState object for reproducibility
        """

        self.data_dir = data_dir
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.return_labels = return_labels

        # create temporary storage for the data, if not yet created
        if not os.path.exists(data_dir):
            print('creating folder', data_dir)
            os.makedirs(data_dir)

        # load CIFAR-10 training data to RAM
        self.data, self.labels = load(self.data_dir, subset=subset, size=size, limit=limit)
        # self.data = np.transpose(self.data, (0,2,3,1)) # (N,3,32,32) -> (N,32,32,3)

        self.p = 0 # pointer to where we are in iteration
        self.rng = np.random.RandomState(1) if rng is None else rng

    def get_observation_size(self):
        return self.data.shape[1:]

    def get_num_labels(self):
        return np.amax(self.labels) + 1

    def reset(self):
        self.p = 0

    def __iter__(self):
        return self

    def __next__(self, n=None):
        """ n is the number of examples to fetch """
        if n is None: n = self.batch_size

        # on first iteration lazily permute all data
        if self.p == 0 and self.shuffle:
            inds = self.rng.permutation(self.data.shape[0])
            self.data = self.data[inds]
            self.labels = self.labels[inds]

        # on last iteration reset the counter and raise StopIteration
        if self.p + n > self.data.shape[0]:
            self.reset() # reset for next time we get called
            raise StopIteration

        # on intermediate iterations fetch the next batch
        x = self.data[self.p : self.p + n]
        y = self.labels[self.p : self.p + n]
        self.p += self.batch_size

        if self.return_labels:
            return x,y
        else:
            return x

    next = __next__  # Python 2 compatibility (https://stackoverflow.com/questions/29578469/how-to-make-an-object-both-a-python2-and-python3-iterator)
