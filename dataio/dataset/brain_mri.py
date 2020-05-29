import os
import numpy as np
import scipy.misc
from operator import itemgetter
from torch.utils import data


class CKBrainMetDataset(data.Dataset):

    def __init__(self, mode, root_dir_path, patient_ids, transform, image_size):
        super().__init__()
        assert mode in ['train', 'test', 'test_normal']
        """
        if mode == train       -> output only normal images without label
        if mode == test        -> output both normal and abnormal images with label
        if mode == test_normal -> output only normal images with label
        """
        self.mode = mode
        self.root_dir_path = root_dir_path
        self.patient_ids = patient_ids
        self.transform = transform
        self.image_size = image_size
        self.files = self.build_file_paths()

    def get_label_path(self, image_path):
        dirname, basename = os.path.split(image_path)
        basename = basename.split('_')
        basename[-1] = 'lbl.npy'
        basename = '_'.join(basename)
        return os.path.join(dirname, basename)

    def count_volumes(self):
        num = 0
        for patient_id in self.patient_ids:
            image_dir_path = os.path.join(
                self.root_dir_path, patient_id, 'image'
            )
            num += len(os.listdir(image_dir_path))
        return num 

    def build_file_paths(self):
        files = []

        if self.mode == 'train':
            selection = 'normal'
        elif self.mode == 'test':
            selection = 'all'
        elif self.mode == 'test_normal':
            selection = 'normal'
        else:
            raise NotImplementedError

        for patient_id in self.patient_ids:
            image_dir_path = os.path.join(
                self.root_dir_path, patient_id, 'planes'
            )

            for file_name in sorted(os.listdir(image_dir_path)):
                if not '_img.npy' in file_name:
                    continue

                if 'abnormal' in file_name:
                    class_name = 'abnormal'
                else:
                    assert 'normal' in file_name
                    class_name = 'normal'

                if selection == 'normal':
                    if class_name == 'normal':
                        image_path = os.path.join(image_dir_path, file_name)
                    else:
                        continue

                elif selection == 'abnormal':
                    if class_name == 'abnormal':
                        image_path = os.path.join(image_dir_path, file_name)
                    else:
                        continue

                elif selection == 'all':
                    image_path = os.path.join(image_dir_path, file_name)

                study_name = self.get_study_name(file_name)
                slice_num = self.get_slice_num(file_name)
                total_slice_nums = self.get_total_slice_nums(image_dir_path, study_name)

                if self.mode == 'train':
                    files.append({
                        'image': image_path,
                        'patient_id': patient_id,
                        'class_name': class_name,
                        'study_name': study_name,
                        'slice_num': slice_num,
                        'total_slice_nums': total_slice_nums,
                    })

                elif self.mode == 'test' or self.mode == 'test_normal':
                    label_path = self.get_label_path(image_path)

                    files.append({
                        'image': image_path,
                        'label': label_path,
                        'patient_id': patient_id,
                        'class_name': class_name,
                        'study_name': study_name,
                        'slice_num': slice_num,
                        'total_slice_nums': total_slice_nums,
                    })

        return files

    def __len__(self):
        return len(self.files)

    def get_slice_num(self, file_name):
        file_name = file_name.split('_')[-3]
        return int(file_name)

    def get_total_slice_nums(self, image_dir_path, study_name):
        slice_nums = len([f for f in os.listdir(image_dir_path) if study_name in f])
        return slice_nums // 2

    def get_study_name(self, file_name):
        file_name = file_name.split('_')[:-3]
        study_name = '_'.join(file_name)
        return study_name

    def __getitem__(self, index):
        image = np.load(self.files[index]['image'])
        image = np.flipud(np.transpose(image))

        if self.image_size:
            if image.shape != (self.image_size, self.image_size):
                image = scipy.misc.imresize(image,
                                            (self.image_size, self.image_size),
                                            interp='lanczos')

        sample = {
            'image': image.astype(np.float32),
            'patient_id': self.files[index]['patient_id'],
            'class_name': self.files[index]['class_name'],
            'study_name': self.files[index]['study_name'],
            'slice_num': self.files[index]['slice_num'],
            'total_slice_nums': self.files[index]['total_slice_nums'],
        }

        if self.mode == 'test':
            if os.path.exists(self.files[index]['label']):
                label = np.load(self.files[index]['label'])
                label = np.flipud(np.transpose(label))
            else:
                label = np.zeros_like(image)

            if self.image_size:
                if label.shape != (self.image_size, self.image_size):
                    label = scipy.misc.imresize(label,
                                                (self.image_size, self.image_size),
                                                interp='nearest')

            sample.update({
                'label': label.astype(np.int32),
            })

        if self.transform:
            sample = self.transform(sample)

        return sample
