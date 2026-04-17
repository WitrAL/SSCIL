import logging
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from utils.data import iCIFAR224, iImageNetR, iImageNetA, CUB, omnibenchmark, vtab, cars, core50, cddb, domainnet
import torchvision

class DataManager(object):
    def __init__(self, dataset_name, shuffle, seed, init_cls, increment,use_input_norm=False,label_percent=0.1):
        self.dataset_name = dataset_name
        lp=label_percent
        self._setup_data(dataset_name, shuffle, seed,use_input_norm,lp)
        assert init_cls <= len(self._class_order), "No enough classes."
        self._increments = [init_cls]
        while sum(self._increments) + increment < len(self._class_order):
            self._increments.append(increment)
        offset = len(self._class_order) - sum(self._increments)
        if offset > 0:
            self._increments.append(offset)


    @property
    def nb_tasks(self):
        return len(self._increments)

    def get_task_size(self, task):
        return self._increments[task]

    def get_total_classnum(self):
        return len(self._class_order)

    def get_dataset(self, indices, source, mode, appendent=None, ret_data=False):
        '''
        indices: list of class indices
        '''
        is_ulb = False
        trsf_strong = transforms.Compose([*self._train_trsf_strong, *self._common_trsf])
        if source == "train_lb":
            #x, y = self._train_data, self._train_targets
            x, y = self._train_data_lb, self._train_targets_lb

        elif source == "train_ulb":
            x, y = self._train_data_ulb, self._train_targets_ulb
            is_ulb = True
        elif source == "test":
            x, y = self._test_data, self._test_targets
        else:
            raise ValueError("Unknown data source {}.".format(source))

        if mode == "train_lb":
            trsf = transforms.Compose([*self._train_trsf, *self._common_trsf])
        elif mode == "train_ulb":
            #trsf = transforms.Compose([*self._train_trsf_strong,*self._common_trsf])
            trsf = transforms.Compose([*self._train_trsf, *self._common_trsf])
            #trsf_strong = transforms.Compose([*self._train_trsf_strong, *self._common_trsf])
        
        elif mode == "flip":
            trsf = transforms.Compose(
                [
                    *self._test_trsf,
                    transforms.RandomHorizontalFlip(p=1.0),
                    *self._common_trsf,
                ]
            )
        elif mode == "test":
            trsf = transforms.Compose([*self._test_trsf, *self._common_trsf])
        else:
            raise ValueError("Unknown mode {}.".format(mode))

        data, targets = [], []
        for idx in indices:
            class_data, class_targets = self._select(
                    x, y, low_range=idx, high_range=idx + 1
                )
            data.append(class_data)
            targets.append(class_targets)

        if appendent is not None and len(appendent) != 0:
            appendent_data, appendent_targets = appendent
            data.append(appendent_data)
            targets.append(appendent_targets)

        data, targets = np.concatenate(data), np.concatenate(targets)
    #def __init__(self, images, labels, trsf, trsf_strong, is_ulb=False, use_path=False):
        if ret_data:
            return data, targets, MyDataset(data, targets, trsf,trsf_strong, is_ulb,self.use_path)
        else:
            return MyDataset(data, targets, trsf,trsf_strong,is_ulb,self.use_path)

    def _setup_data(self, dataset_name, shuffle, seed,use_input_norm,label_percent=0.1):
        idata = _get_idata(dataset_name,use_input_norm,label_percent)
        idata.download_data()

        # Data
        #self._train_data, self._train_targets = idata.train_data, idata.train_targets
        #self._test_data, self._test_targets = idata.test_data, idata.test_targets
        self.use_path = idata.use_path
        
        self._train_data_lb,self._train_targets_lb = idata.train_data_lb, idata.train_targets_lb
        self._train_data_ulb,self._train_targets_ulb = idata.train_data_ulb, idata.train_targets_ulb
        self._test_data, self._test_targets = idata.test_data, idata.test_targets


        # Transforms
        self._train_trsf = idata.train_trsf
        self._test_trsf = idata.test_trsf
        self._common_trsf = idata.common_trsf
        self._train_trsf_strong = idata.strong_trsf

        if 'domainnet' not in dataset_name :
            # Order
            #order = [i for i in range(len(np.unique(self._train_targets)))]
            order = [i for i in range(len(np.unique(self._train_targets_lb)))]
            if shuffle:
                np.random.seed(seed)
                order = np.random.permutation(len(order)).tolist()
            else:
                order = idata.class_order
            self._class_order = order
            logging.info("Class Order: ["+",".join([str(x) for x in self._class_order])+"]")

            # Map indices
            self._train_targets_lb = _map_new_class_index(
                self._train_targets_lb, self._class_order
            )
            self._train_targets_ulb = _map_new_class_index(
                self._train_targets_ulb, self._class_order
            )
            self._test_targets = _map_new_class_index(self._test_targets, self._class_order)
        else:
            np.random.seed(seed)
            self._class_order =np.arange(345).tolist()
            logging.info("Class Order: ["+",".join([str(x) for x in self._class_order])+"]")

    def _select(self, x, y, low_range, high_range):
        assert isinstance(y, np.ndarray), "Labels must be a numpy array."
        idxes = np.where(np.logical_and(y >= low_range, y < high_range))[0]
        return x[idxes], y[idxes]

    # def getlen(self, index):
    #     y = self._train_targets
    #     return np.sum(np.where(y == index))

# 这是原来的Dataset类，并不兼容半监督学习
class DummyDataset(Dataset):
    def __init__(self, images, labels, trsf, use_path=False):
        assert len(images) == len(labels), "Data size error!"
        self.images = images
        self.labels = labels
        self.trsf = trsf
        self.use_path = use_path

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if self.use_path:
            image = self.trsf(pil_loader(self.images[idx]))
        else:
            image = self.trsf(Image.fromarray(self.images[idx]))
        label = self.labels[idx]

        return idx, image, label

# 参考USB的Dataset类，支持半监督学习
class MyDataset(Dataset):
    def __init__(self, images, labels, trsf, trsf_strong, is_ulb=False, use_path=False):
        #assert len(images) == len(labels), "Data size error!"
        '''
        由is_ulb决定是否是无标签样本
        '''
        self.images = images
        self.labels = labels
        self.trsf = trsf
        self.use_path = use_path

        self.is_ulb = is_ulb
        self.trsf_strong = trsf_strong

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if self.use_path:
            image = self.trsf(pil_loader(self.images[idx]))
        else:
            image = self.trsf(Image.fromarray(self.images[idx]))
        if self.is_ulb:
            if self.use_path:
                image_strong = self.trsf_strong(pil_loader(self.images[idx]))
            else:
                image_strong = self.trsf_strong(Image.fromarray(self.images[idx]))
            #return {'idx_ulb': idx, 'x_ulb_w': image, 'x_ulb_s': image_strong}
            return idx,image,image_strong
        else:
            label = self.labels[idx]
            #return {'idx_lb': idx, 'x_lb': image, 'y_lb': label}
            return idx,image,label
        

        # label = self.labels[idx]

        # return idx, image, label

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=#
# https://github.com/microsoft/Semi-supervised-learning/blob/6cf697e6968f9f77040a9e0162306436a91a00dc/semilearn/datasets/cv_datasets/datasetbase.py

class BasicDataset(Dataset):
    """
    BasicDataset returns a pair of image and labels (targets).
    If targets are not given, BasicDataset returns None as the label.
    This class supports strong augmentation for FixMatch,
    and return both weakly and strongly augmented images.
    """

    def __init__(self,
                 alg,
                 data,
                 targets=None,
                 num_classes=None,
                 transform=None,
                 is_ulb=False,
                 medium_transform=None,
                 strong_transform=None,
                 onehot=False,
                 *args, 
                 **kwargs):
        """
        Args
            alg: algorithm name 我们不需要这个参数
            data: x_data
            targets: y_data (if not exist, None)
            num_classes: number of label classes
            transform: basic transformation of data
            use_strong_transform: If True, this dataset returns both weakly and strongly augmented images.
            strong_transform: list of transformation functions for strong augmentation
            onehot: If True, label is converted into onehot vector.
        """
        super(BasicDataset, self).__init__()
        self.alg = alg
        self.data = data
        self.targets = targets

        self.num_classes = num_classes
        self.is_ulb = is_ulb
        self.onehot = onehot

        self.transform = transform
        self.strong_transform = strong_transform
        self.medium_transform = medium_transform
        if self.strong_transform is None:
            if self.is_ulb:
                assert self.alg not in ['fullysupervised', 'supervised', 'pseudolabel', 'vat', 'pimodel', 'meanteacher', 'mixmatch', 'refixmatch'], f"alg {self.alg} requires strong augmentation"
    
        if self.medium_transform is None:
            if self.is_ulb:
                assert self.alg not in ['sequencematch'], f"alg {self.alg} requires medium augmentation"
    
    def __sample__(self, idx):
        """ dataset specific sample function """
        # set idx-th target
        if self.targets is None:
            target = None
        else:
            target_ = self.targets[idx]
            target = target_ if not self.onehot else get_onehot(self.num_classes, target_)

        # set augmented images
        img = self.data[idx]
        return img, target

    def __getitem__(self, idx):
        """
        If strong augmentation is not used,
            return weak_augment_image, target
        else:
            return weak_augment_image, strong_augment_image, target
        """
        img, target = self.__sample__(idx)

        if self.transform is None:
            return  {'x_lb':  transforms.ToTensor()(img), 'y_lb': target}
        else:
            if isinstance(img, np.ndarray):
                img = Image.fromarray(img)
            img_w = self.transform(img)
            if not self.is_ulb:
                return {'idx_lb': idx, 'x_lb': img_w, 'y_lb': target} 
            else:
                if self.alg == 'fullysupervised' or self.alg == 'supervised':
                    return {'idx_ulb': idx}
                elif self.alg == 'pseudolabel' or self.alg == 'vat':
                    return {'idx_ulb': idx, 'x_ulb_w':img_w} 
                elif self.alg == 'pimodel' or self.alg == 'meanteacher' or self.alg == 'mixmatch':
                    # NOTE x_ulb_s here is weak augmentation
                    return {'idx_ulb': idx, 'x_ulb_w': img_w, 'x_ulb_s': self.transform(img)}
                # elif self.alg == 'sequencematch' or self.alg == 'somematch':
                elif self.alg == 'sequencematch':
                    return {'idx_ulb': idx, 'x_ulb_w': img_w, 'x_ulb_m': self.medium_transform(img), 'x_ulb_s': self.strong_transform(img)} 
                elif self.alg == 'remixmatch':
                    rotate_v_list = [0, 90, 180, 270]
                    rotate_v1 = np.random.choice(rotate_v_list, 1).item()
                    img_s1 = self.strong_transform(img)
                    img_s1_rot = torchvision.transforms.functional.rotate(img_s1, rotate_v1)
                    img_s2 = self.strong_transform(img)
                    return {'idx_ulb': idx, 'x_ulb_w': img_w, 'x_ulb_s_0': img_s1, 'x_ulb_s_1':img_s2, 'x_ulb_s_0_rot':img_s1_rot, 'rot_v':rotate_v_list.index(rotate_v1)}
                elif self.alg == 'comatch':
                    return {'idx_ulb': idx, 'x_ulb_w': img_w, 'x_ulb_s_0': self.strong_transform(img), 'x_ulb_s_1':self.strong_transform(img)} 
                else:
                    return {'idx_ulb': idx, 'x_ulb_w': img_w, 'x_ulb_s': self.strong_transform(img)} 


    def __len__(self):
        return len(self.data)


def get_onehot(num_classes, idx):
    onehot = np.zeros([num_classes], dtype=np.float32)
    onehot[idx] += 1.0
    return onehot

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=#

def _map_new_class_index(y, order):
    return np.array(list(map(lambda x: order.index(x), y)))


def _get_idata(dataset_name,use_input_norm,label_percent):
    name = dataset_name.lower()
    if name== "cifar224":
        return iCIFAR224(use_input_norm,label_percent)
    elif name== "imagenetr":
        return iImageNetR(use_input_norm)
    elif name=="imageneta":
        return iImageNetA(use_input_norm)
    elif name=="cub":
        return CUB(use_input_norm)
    elif name=="omnibenchmark":
        return omnibenchmark(use_input_norm)
    elif name=="vtab":
        return vtab(use_input_norm)
    elif name=="cars":
        return cars(use_input_norm)
    elif "core50" in name:
        logging.info('Starting next DIL task: '+name)
        return core50(name[7::],use_input_norm)
    elif "cddb" in name:
        logging.info('Starting next DIL task: '+name)
        return cddb(name[5::],use_input_norm)
    elif "domainnet" in name:
        logging.info('Starting next DIL task: '+name)
        return domainnet(name[10::],use_input_norm)

    else:
        raise NotImplementedError("Unknown dataset {}.".format(dataset_name))


def pil_loader(path):
    """
    Ref:
    https://pytorch.org/docs/stable/_modules/torchvision/datasets/folder.html#ImageFolder
    """
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")
