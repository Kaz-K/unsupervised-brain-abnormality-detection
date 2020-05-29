from torch.utils import data
from torchvision import transforms

from .dataset import CKBrainMetDataset
from .transformation import Normalize
from .transformation import ZScoreNormalize
from .transformation import ToImage
from .transformation import ToTensor
from .transformation import RandomHorizontalFlip
from .transformation import RandomVerticalFlip
from .transformation import RandomRotate
from .transformation import RandomScale
from .transformation import RandomColorJitter
from .transformation import RandomSliceSelect


def get_data_loader(mode, dataset_name, patient_ids, root_dir_path,
                    use_augmentation, batch_size, num_workers, image_size,
                    omit_transform):

    if mode == 'test' or mode == 'test_normal':
        assert not use_augmentation

    if dataset_name == 'CKBrainMetDataset':
        if use_augmentation:
            transform = transforms.Compose([
                ToImage(),
                RandomHorizontalFlip(),
                RandomRotate(degree=20),
                RandomScale(mean=1.0, var=0.05, image_fill=0),
                # RandomColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
                ToTensor(),
                Normalize(min_val=0, max_val=255),
            ])
        else:
            transform = transforms.Compose([
                ToImage(),
                ToTensor(),
                Normalize(min_val=0, max_val=255),
            ])

        if omit_transform:
            transform = None

        dataset = CKBrainMetDataset(
            mode, root_dir_path, patient_ids, transform, image_size
        )

    else:
        raise NotImplementedError

    if mode == 'train':
        shuffle = True
    elif mode == 'test' or mode == 'test_normal':
        shuffle = False

    return data.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
    )
