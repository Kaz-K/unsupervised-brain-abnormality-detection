import torch
import random
import scipy.misc
import numpy as np
from PIL import Image
from PIL import ImageOps
from torchvision.transforms import ColorJitter


class Normalize(object):
    """Normalizes image with range of 0-255 to 0-1.
    """

    def __init__(self, min_val=0, max_val=255):
        self.min_val = min_val
        self.max_val = max_val

    def __call__(self, sample: dict):
        image = sample['image']
        image -= self.min_val
        image /= (self.max_val - self.min_val)
        image = torch.clamp(image, 0, 1)

        sample.update({
            'image': image,
        })

        return sample


class ZScoreNormalize(object):

    def __call__(self, sample):
        image = sample['image']
        mean = image.mean()
        std = image.std()
        image = image.float()
        image = (image - mean) / std

        sample.update({
            'image': image,
        })

        return sample


class ToImage(object):

    def __call__(self, sample):
        # assert 'label' not in sample.keys()
        image = sample['image']

        sample.update({
            'image': Image.fromarray(image),
        })

        return sample


class ToTensor(object):

    def __call__(self, sample: dict):
        image = sample['image']

        if type(image) == Image.Image:
            image = np.asarray(image)

        if image.ndim == 2:
            image = image[np.newaxis, ...]

        image = torch.from_numpy(image).float()
        sample.update({
            'image': image,
        })

        if 'label' in sample.keys():
            label = sample['label']

            if label.ndim == 2:
                label = label[np.newaxis, ...]

            label = torch.from_numpy(label).int()
            sample.update({
                'label': label,
            })

        return sample


class RandomHorizontalFlip(object):

    def __call__(self, sample: dict):
        assert 'label' not in sample.keys()
        image = sample['image']

        if random.random() < 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)

        sample.update({
            'image': image,
        })

        return sample


class RandomVerticalFlip(object):

    def __call__(self, sample: dict):
        assert 'label' not in sample.keys()
        image = sample['image']

        if random.random() < 0.5:
            image = image.transpose(Image.FLIP_TOP_BOTTOM)

        sample.update({
            'image': image,
        })

        return sample


class RandomRotate(object):

    def __init__(self, degree=20):
        self.degree = degree

    def __call__(self, sample: dict):
        assert 'label' not in sample.keys()
        image = sample['image']

        rotate_degree = random.uniform(-1 * self.degree, self.degree)
        image = image.rotate(rotate_degree, Image.BILINEAR)

        sample.update({
            'image': image,
        })

        return sample


class RandomScale(object):

    def __init__(self, mean=1.0, var=0.05, image_fill=0):
        self.mean = mean
        self.var = var
        self.image_fill = image_fill

    def __call__(self, sample: dict):
        assert 'label' not in sample.keys()
        image = sample['image']
        base_size = image.size

        scale_factor = random.normalvariate(self.mean, self.var)

        size = (
            int(base_size[0] * scale_factor),
            int(base_size[1] * scale_factor),
        )

        image = image.resize(size, Image.BILINEAR)

        if scale_factor < 1.0:
            pad_h = base_size[0] - image.size[0]
            pad_w = base_size[1] - image.size[1]
            ori_h = random.randint(0, pad_h)
            ori_w = random.randint(0, pad_w)

            image = ImageOps.expand(
                image,
                border=(ori_h, ori_w, pad_h - ori_h, pad_w - ori_w),
                fill=self.image_fill
            )

        else:
            ori_h = random.randint(0, image.size[0] - base_size[0])
            ori_w = random.randint(0, image.size[1] - base_size[1])
            image = image.crop((
                ori_h, ori_w,
                ori_h + base_size[0], ori_w + base_size[1]
            ))

        sample.update({
            'image': image,
        })

        return sample


class RandomColorJitter(object):
    def __init__(self, brightness=0.3, contrast=0.3, saturation=0.3):
        self.filter = ColorJitter(brightness, contrast, saturation)

    def __call__(self, sample: dict):
        image = sample['image']

        image = image.convert('RGB')
        image = self.filter(image)
        image = image.convert('L')

        sample.update({
            'image': image,
        })

        return sample


class RandomSliceSelect(object):
    def __init__(self, threshold=1, max_iter=10):
        self.threshold = threshold
        self.max_iter = max_iter

    def __call__(self, sample: dict):
        image = sample['image']

        z_max = image.shape[2]
        mean = 0.0
        n_iter = 0

        while n_iter < self.max_iter:
            selected_z = random.randint(0, z_max - 1)
            selected = image[..., selected_z]
            mean = np.mean(selected)

            if mean > self.threshold:
                break

            n_iter += 1

        sample.update({
            'image': selected,
        })

        return sample
