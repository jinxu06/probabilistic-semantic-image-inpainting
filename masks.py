import numpy as np

np.random.seed(123)

class MaskGenerator(object):

    def __init__(self, height, width, name="mask"):
        self.height = height
        self.width = width
        self.name = name

    def gen(self, n):
        self.masks = np.ones((n, self.height, self.width))
        return self.masks

class CenterMaskGenerator(MaskGenerator):

    def __init__(self, height, width, ratio=0.5, name='mask'):
        super().__init__(height, width, name)
        self.ratio = ratio

    def gen(self, n):
        self.masks = np.ones((n, self.height, self.width))
        c_height = int(self.height * self.ratio)
        c_width = int(self.width * self.ratio)
        height_offset = (self.height - c_height) // 2
        width_offset = (self.width - c_width) // 2
        self.masks[:, height_offset:height_offset+c_height, width_offset:width_offset+c_width] = 0
        return self.masks

class RectangleMaskGenerator(MaskGenerator):

    def __init__(self, height, width, rec=None, name='mask'):
        super().__init__(height, width, name)
        if rec is None:
            rec = int(0.25*self.height), int(0.75*self.width), int(0.75*self.height), int(0.25*self.width)
        self.rec = rec

    def gen(self, n):
        top, right, bottom, left = self.rec
        self.masks = np.ones((n, self.height, self.width))
        self.masks[:, top:bottom, left:right] = 0
        return self.masks

class MultiRectangleMaskGenerator(MaskGenerator):

    def __init__(self, height, width,  recs=None, name='mask'):
        super().__init__(height, width, name)
        if recs is None:
            rec = int(0.25*self.height), int(0.75*self.width), int(0.75*self.height), int(0.25*self.width)
            recs = [rec]
        self.recs = recs

    def gen(self, n):
        self.masks = np.ones((n, self.height, self.width))
        for rec in self.recs:
            top, right, bottom, left = rec
            self.masks[:, top:bottom, left:right] = 0
        return self.masks


class RandomRectangleMaskGenerator(MaskGenerator):

    def __init__(self, height, width, min_ratio=0.25, max_ratio=0.75, margin_ratio=0., batch_same=False, name='mask'):
        super().__init__(height, width, name)
        self.min_ratio = min_ratio
        self.max_ratio = max_ratio
        self.margin_ratio = margin_ratio
        self.batch_same = batch_same

    def gen(self, n):
        self.masks = np.ones((n, self.height, self.width))
        for i in range(self.masks.shape[0]):
            min_height = int(self.height * self.min_ratio)
            min_width = int(self.width * self.min_ratio)
            max_height = int(self.height * self.max_ratio)
            max_width = int(self.width * self.max_ratio)
            margin_height = int(self.height * self.margin_ratio)
            margin_width = int(self.width * self.margin_ratio)

            # rng = np.random.RandomState(None)
            c_height = np.random.randint(low=min_height, high=max_height)
            c_width = np.random.randint(low=min_width, high=max_width)
            height_offset = np.random.randint(low=margin_height, high=self.height-margin_height-c_height)
            width_offset = np.random.randint(low=margin_width, high=self.width-margin_width-c_width)
            # rng = np.random.RandomState(None)
            # c_height = rng.randint(low=min_height, high=max_height)
            # c_width = rng.randint(low=min_width, high=max_width)
            # height_offset = rng.randint(low=margin_height, high=self.height-margin_height-c_height)
            # width_offset = rng.randint(low=margin_width, high=self.width-margin_width-c_width)

            self.masks[i, height_offset:height_offset+c_height, width_offset:width_offset+c_width] = 0
        if self.batch_same:
            self.masks = np.stack([self.masks[i].copy() for i in range(n)], axis=0)
        return self.masks





def get_generator(name, size):
    if name=='full':
        return CenterMaskGenerator(size, size, ratio=1.0, name=name)
    elif name=='transparent':
        return CenterMaskGenerator(size, size, ratio=0.0, name=name)
    elif name=='center':
        return CenterMaskGenerator(size, size, ratio=0.5, name=name)
    elif name=='eye':
        return RectangleMaskGenerator(size, size, rec=[9, 27, 21, 5], name=name)
    elif name=='mouth':
        return RectangleMaskGenerator(size, size, rec=[22, 28, 32, 4], name=name)
        # return RectangleMaskGenerator(size, size, rec=[22, 32, 32, 0])
    elif name=='nose':
        return RectangleMaskGenerator(size, size, rec=[18, 21, 27, 11], name=name)
    elif name=='hair':
        return RectangleMaskGenerator(size, size, rec=[0,32,12,0], name=name)
        # return MultiRectangleMaskGenerator(size, size, recs=[[0,32,10,0], [0,9,32,0], [0,23,32,32]])
    elif name=='face':
        return RectangleMaskGenerator(size, size, rec=[8, 24, 32, 8], name=name)
    elif name=='top half':
        return RectangleMaskGenerator(size, size, rec=[0, 32, 16, 0], name=name)
    elif name=='bottom half':
        return RectangleMaskGenerator(size, size, rec=[16, 32, 32, 0], name=name)
    elif name=='bottom quarter':
        return RectangleMaskGenerator(size, size, rec=[24, 32, 32, 0], name=name)
    elif name=='right half':
        return RectangleMaskGenerator(size, size, rec=[0, 32, 32, 16], name=name)
    elif name=='random rec':
        return RandomRectangleMaskGenerator(size, size, min_ratio=1./8, max_ratio=(1.-1./8), name=name)
    elif name=='random blobs':
        return RandomBlobGenerator(size, size, max_num_blobs=4, iter_min=2, iter_max=7, name=name)
    elif name=='pepper':
        return PepperMaskGenerator(size, size, prob=None, name=name)
    elif name=='pepper50%':
        return PepperMaskGenerator(size, size, prob=0.5, name=name)
    elif name=='pepper80%':
        return PepperMaskGenerator(size, size, prob=0.8, name=name)
    elif name=='center small':
        return CenterMaskGenerator(size, size, ratio=1.0/16., name=name)
    elif name=='mnist top 18':
        return RectangleMaskGenerator(size, size, rec=[0, 28, 18, 0], name=name)
    elif name=='mnist bottom 18':
        return RectangleMaskGenerator(size, size, rec=[10, 28, 28, 0], name=name)



class RandomBlobGenerator(MaskGenerator):

    def __init__(self, width, height, max_num_blobs, iter_min, iter_max, regenerate_samples=None, store="/data/ziz/not-backed-up/jxu/random_blobs_masks.npz", name='mask'):
        super().__init__(height, width, name)
        self.max_num_blobs = max_num_blobs
        self.iter_min = iter_min
        self.iter_max = iter_max
        self.store = store[:-4] + "_{}.npz".format(width)
        self.cached_masks = None
        if regenerate_samples is not None:
            self.__regenerate(regenerate_samples, save=True)

    def _neighbors(self, pos):
        neighbors = []
        x0, y0 = pos
        for i in range(-1, 2):
            for j in range(-1, 2):
                if i==0 and j==0:
                    continue
                x = x0 + i
                y = y0 + j
                if (x < 0 or x >=self.width) or (y < 0 or y >=self.height):
                    continue
                neighbors.append((x, y))
        return neighbors

    def __regenerate(self, n, save=False):
        masks = []
        for idx in range(n):
            if idx % 100 == 0:
                print(idx)
            mask = np.zeros((self.height, self.width))
            num_blobs = np.random.randint(low=1, high=self.max_num_blobs+1)
            for i in range(num_blobs):
                num_iters = np.random.randint(low=self.iter_min, high=self.iter_max+1)
                x0 = np.random.randint(low=0, high=self.width)
                y0 = np.random.randint(low=0, high=self.height)
                mask[x0, y0] = 1
                start_positions = [(x0, y0)]
                for j in range(num_iters):
                    next_start_positions = []
                    for pos in start_positions:
                        for x, y in self._neighbors(pos):
                            p = np.random.uniform(low=0, high=1)
                            if p > 0.5:
                                mask[x, y] = 1
                                next_start_positions.append((x, y))
                    start_positions = next_start_positions
            masks.append(mask)
        masks = np.array(masks)
        masks = 1. - masks
        if not save:
            return masks
        np.savez(self.store, masks=masks)

    def __gen_from_cached(self, n):
        if self.cached_masks is None:
            self.cached_masks = np.load(self.store)['masks']
        idx = np.random.choice(self.cached_masks.shape[0], n)
        return self.cached_masks[idx]


    def gen(self, n, from_cache=True):
        if from_cache:
            return self.__gen_from_cached(n)
        else:
            return self.__regenerate(n)


class PepperMaskGenerator(MaskGenerator):

    def __init__(self, width, height, prob, name='pepper'):
        super().__init__(height, width, name)
        self.prob = prob

    def gen(self, n):
        if self.prob is not None:
            self.masks = np.random.binomial(1, 1-self.prob, size=(n, self.height, self.width))
        else:
            self.masks = []
            for i in range(n):
                p = np.random.uniform(low=0.1, high=0.9)
                mask = np.random.binomial(1, 1-p, size=(self.height, self.width))
                self.masks.append(mask)
            self.masks = np.stack(self.masks, axis=0)
        return self.masks
