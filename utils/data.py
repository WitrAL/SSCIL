import numpy as np
import os
from torchvision import datasets, transforms
from utils.toolkit import split_images_labels
# CUB, ImageNet-R, ImageNet-A, OmnibenchMark and VTAB are the versions defined at https://github.com/zhoudw-zdw/RevisitingCIL from here: 
#   @article{zhou2023revisiting,
#        author = {Zhou, Da-Wei and Ye, Han-Jia and Zhan, De-Chuan and Liu, Ziwei},
#        title = {Revisiting Class-Incremental Learning with Pre-Trained Models: Generalizability and Adaptivity are All You Need},
#        journal = {arXiv preprint arXiv:2303.07338},
#        year = {2023}
#    }

class iData(object):
    train_trsf = []
    test_trsf = []
    strong_trsf = []
    common_trsf = []
    class_order = None

def build_transform(is_train, args,isCifar=False, strong=False):
    input_size = 224
    resize_im = input_size > 32
    if is_train:
        scale = (0.05, 1.0)
        ratio = (3. / 4., 4. / 3.)
        if strong is False:
            transform = [
                transforms.RandomResizedCrop(input_size, scale=scale, ratio=ratio),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
            ]
        else:
            transform = [
                transforms.RandomResizedCrop(input_size, scale=scale, ratio=ratio),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(
                    brightness=0.4,
                    contrast=0.4,
                    saturation=0.4,
                    hue=0.1,
                ),
                transforms.RandomGrayscale(p=0.2),
                transforms.ToTensor(),
            ]
        return transform

    t = []
    if resize_im:
        if isCifar:
            size = input_size
        else:
            size = int((256 / 224) * input_size)
        t.append(
            transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC),  # to maintain same ratio w.r.t. 224 images
        )
        t.append(transforms.CenterCrop(input_size))
    t.append(transforms.ToTensor())
    
    # return transforms.Compose(t)
    return t

class iCIFAR224(iData):
    use_path = False

    train_trsf=build_transform(True, None,True)
    test_trsf=build_transform(False, None,True)
    common_trsf = [
        # transforms.ToTensor(),
    ]
    strong_trsf = build_transform(True, None,True,True)
    class_order = np.arange(100).tolist()

    def __init__(self,use_input_norm,label_percent):
        if use_input_norm:
            self.common_trsf = [transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
        self.label_percent=label_percent
        self.label_percent_str = "{:.0%}".format(self.label_percent)
    def save_indices(self, labeled_indices, unlabeled_indices, output_dir):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        np.save(os.path.join(output_dir, f'{self.label_percent_str}_labeled_indices.npy'), labeled_indices)
        np.save(os.path.join(output_dir, f'{self.label_percent_str}_unlabeled_indices.npy'), unlabeled_indices)

    def load_indices(self, input_dir):
        labeled_indices = np.load(os.path.join(input_dir, f'{self.label_percent_str}_labeled_indices.npy'))
        unlabeled_indices = np.load(os.path.join(input_dir, f'{self.label_percent_str}_unlabeled_indices.npy'))
        return labeled_indices, unlabeled_indices

    def download_data(self):
        do_download=True
        data_dir = './data'
        indices_dir = os.path.join(data_dir,'indices','cifar-100')
        log_file = os.path.join(indices_dir, f'{self.label_percent_str}class_distribution_log.txt')
        if os.path.isfile(os.path.join(data_dir, 'cifar-100-python', 'train')):
            do_download=False
        train_dataset = datasets.cifar.CIFAR100("./data/", train=True, download=do_download)
        test_dataset = datasets.cifar.CIFAR100("./data", train=False, download=False)
        # 数据和标签
        data = train_dataset.data
        targets = np.array(train_dataset.targets)


        if os.path.exists(indices_dir) and os.path.isfile(os.path.join(indices_dir, f'{self.label_percent_str}_labeled_indices.npy')):
            print("Loading indices...")
            labeled_indices, unlabeled_indices = self.load_indices(indices_dir)

        else:
            print("Generating and saving indices...")
            #indices = np.arange(len(targets))
            #np.random.shuffle(indices)

            labeled_indices = []
            unlabeled_indices = []

            num_classes = 100
            labels_per_class = int(self.label_percent * 500)  # x% of 500

            for i in range(num_classes):
                class_indices = np.where(targets == i)[0]
                np.random.shuffle(class_indices)  # 随机打乱索引
                labeled_indices.extend(class_indices[:labels_per_class])
                unlabeled_indices.extend(class_indices[labels_per_class:])

            labeled_indices = np.array(labeled_indices)
            unlabeled_indices = np.array(unlabeled_indices)
            self.save_indices(labeled_indices, unlabeled_indices, indices_dir)
        
        # 使用加载的索引分割数据
        self.train_data_lb = data[labeled_indices]
        self.train_targets_lb = targets[labeled_indices]
        self.train_data_ulb = data[unlabeled_indices]
        self.train_targets_ulb = targets[unlabeled_indices]

        self.test_data = np.array(test_dataset.data)  # 确保类型一致
        self.test_targets = np.array(test_dataset.targets)
        
        self.log_class_distribution(labeled_indices, unlabeled_indices, targets, log_file)

    def log_class_distribution(self, labeled_indices, unlabeled_indices, targets, log_file):
        num_classes = 100
        class_labeled_counts = np.zeros(num_classes, dtype=int)
        class_unlabeled_counts = np.zeros(num_classes, dtype=int)

        for i in range(num_classes):
            class_labeled_counts[i] = np.sum(targets[labeled_indices] == i)
            class_unlabeled_counts[i] = np.sum(targets[unlabeled_indices] == i)

        with open(log_file, 'w') as f:
            for i in range(num_classes):
                f.write(f"Class {i}: Labeled samples: {class_labeled_counts[i]}, Unlabeled samples: {class_unlabeled_counts[i]}\n")
        print(f"Class distribution written to {log_file}")



        # # 每个类别选择10%作为有标签数据
        # num_classes = 100
        # labels_per_class = self.label_percent*500  # x% of 500

        # labeled_data = []
        # labeled_targets = []
        # unlabeled_data = []
        # unlabeled_targets = []

        # for i in range(num_classes):
        #     indices = np.where(targets == i)[0]
        #     np.random.shuffle(indices)
        #     labeled_indices = indices[:labels_per_class]
        #     unlabeled_indices = indices[labels_per_class:]

        #     labeled_data.extend(data[labeled_indices])
        #     labeled_targets.extend(targets[labeled_indices])
        #     unlabeled_data.extend(data[unlabeled_indices])
        #     unlabeled_targets.extend(targets[unlabeled_indices])
        
        # # 转换为numpy数组
        # labeled_data = np.array(labeled_data)
        # labeled_targets = np.array(labeled_targets)
        # unlabeled_data = np.array(unlabeled_data)
        # unlabeled_targets = np.array(unlabeled_targets)

        # self.train_data_lb, self.train_targets_lb = labeled_data, labeled_targets
        # self.train_data_ulb, self.train_targets_ulb = unlabeled_data, unlabeled_targets

        # self.test_data, self.test_targets = test_dataset.data, np.array(
        #     test_dataset.targets
        # )

# class iImageNetR(iData):
#     use_path = True
    
#     train_trsf=build_transform(True, None)
#     test_trsf=build_transform(False, None)
    

#     class_order = np.arange(200).tolist()

#     def __init__(self,use_input_norm):
#         if use_input_norm:
#             self.common_trsf = [transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]

#     def download_data(self):
#         # as per Zhou et al (2023), download from https://drive.google.com/file/d/1SG4TbiL8_DooekztyCVK8mPmfhMo8fkR/view?usp=sharing) or Onedrive: [link](https://entuedu-my.sharepoint.com/:u:/g/personal/n2207876b_e_ntu_edu_sg/EU4jyLL29CtBsZkB6y-JSbgBzWF5YHhBAUz1Qw8qM2954A?e=hlWpNW
#         train_dir = "./data/imagenet-r/train/"
#         test_dir = "./data/imagenet-r/test/"

#         train_dset = datasets.ImageFolder(train_dir)
#         test_dset = datasets.ImageFolder(test_dir)

#         self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
#         self.test_data, self.test_targets = split_images_labels(test_dset.imgs)


# class iImageNetA(iData):
#     use_path = True
    
#     train_trsf=build_transform(True, None)
#     test_trsf=build_transform(False, None)
#     common_trsf = [    ]

#     class_order = np.arange(200).tolist()

#     def __init__(self,use_input_norm):
#         if use_input_norm:
#             self.common_trsf = [transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]

#     def download_data(self):
#         # as per Zhou et al (2023), download from  https://drive.google.com/file/d/19l52ua_vvTtttgVRziCZJjal0TPE9f2p/view?usp=sharing) or Onedrive: [link](https://entuedu-my.sharepoint.com/:u:/g/personal/n2207876b_e_ntu_edu_sg/ERYi36eg9b1KkfEplgFTW3gBg1otwWwkQPSml0igWBC46A?e=NiTUkL
#         train_dir = "./data/imagenet-a/train/"
#         test_dir = "./data/imagenet-a/test/"

#         train_dset = datasets.ImageFolder(train_dir)
#         test_dset = datasets.ImageFolder(test_dir)

#         self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
#         self.test_data, self.test_targets = split_images_labels(test_dset.imgs)
class iImageNetR(iData):
    use_path = True

    train_trsf = build_transform(True, None)
    test_trsf = build_transform(False, None)
    strong_trsf = build_transform(is_train=True, args=None, strong=True)
    common_trsf = []

    class_order = np.arange(200).tolist()

    def __init__(self, use_input_norm, label_percent):
        self.label_percent = label_percent
        self.label_percent_str = "{:.0%}".format(self.label_percent)
        import logging
        logging.info(f"ImageNet-R class initialized with label_percent: {self.label_percent_str}")

        if use_input_norm:
            self.common_trsf = [transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]

    def save_indices(self, labeled_indices, unlabeled_indices, output_dir):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        np.save(os.path.join(output_dir, f'{self.label_percent_str}_labeled_indices.npy'), labeled_indices)
        np.save(os.path.join(output_dir, f'{self.label_percent_str}_unlabeled_indices.npy'), unlabeled_indices)

    def load_indices(self, input_dir):
        labeled_indices = np.load(os.path.join(input_dir, f'{self.label_percent_str}_labeled_indices.npy'))
        unlabeled_indices = np.load(os.path.join(input_dir, f'{self.label_percent_str}_unlabeled_indices.npy'))
        return labeled_indices, unlabeled_indices

    def download_data(self):
        # as per Zhou et al (2023), download from ...
        data_dir = './data'
        indices_dir = os.path.join(data_dir, 'indices', 'imagenet-r')
        log_file = os.path.join(indices_dir, f'{self.label_percent_str}_class_distribution_log.txt')

        train_dir = os.path.join(data_dir, "imagenet-r", "train")
        test_dir = os.path.join(data_dir, "imagenet-r", "test")

        if not os.path.exists(train_dir) or not os.path.exists(test_dir):
            raise FileNotFoundError("ImageNet-R data directories not found...")

        train_dset = datasets.ImageFolder(train_dir, transform=None)
        test_dset = datasets.ImageFolder(test_dir, transform=None)

        # Use YOUR split_images_labels function here:
        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)

        targets = self.train_targets

        if os.path.exists(indices_dir) and os.path.isfile(os.path.join(indices_dir, f'{self.label_percent_str}_labeled_indices.npy')):
            print("Loading existing indices...")
            labeled_indices, unlabeled_indices = self.load_indices(indices_dir)
        else:
            print("Generating and saving new indices...")
            labeled_indices = []
            unlabeled_indices = []

            num_classes = 200
            for i in range(num_classes):
                class_indices = np.where(targets == i)[0]
                np.random.shuffle(class_indices)
                num_labeled = max(1, int(self.label_percent * len(class_indices)))  # Ensure at least one labeled sample
                labeled_indices.extend(class_indices[:num_labeled])
                unlabeled_indices.extend(class_indices[num_labeled:])

            labeled_indices = np.array(labeled_indices)
            unlabeled_indices = np.array(unlabeled_indices)
            self.save_indices(labeled_indices, unlabeled_indices, indices_dir)

        self.train_data_lb = self.train_data[labeled_indices]
        self.train_targets_lb = self.train_targets[labeled_indices]
        self.train_data_ulb = self.train_data[unlabeled_indices]
        self.train_targets_ulb = self.train_targets[unlabeled_indices]

        self.test_data = self.test_data
        self.test_targets = self.test_targets

        self.log_class_distribution(labeled_indices, unlabeled_indices, targets, log_file)

    def log_class_distribution(self, labeled_indices, unlabeled_indices, targets, log_file):
        num_classes = 200
        class_labeled_counts = np.zeros(num_classes, dtype=int)
        class_unlabeled_counts = np.zeros(num_classes, dtype=int)

        for i in range(num_classes):
            class_labeled_counts[i] = np.sum(targets[labeled_indices] == i)
            class_unlabeled_counts[i] = np.sum(targets[unlabeled_indices] == i)

        with open(log_file, 'w') as f:
            for i in range(num_classes):
                f.write(f"Class {i}: Labeled samples: {class_labeled_counts[i]}, Unlabeled samples: {class_unlabeled_counts[i]}\n")
        print(f"Class distribution written to {log_file}")


class iImageNetA(iData):
    use_path = True

    train_trsf = build_transform(True, None)
    test_trsf = build_transform(False, None)
    strong_trsf = build_transform(is_train=True, args=None, strong=True)
    common_trsf = []

    class_order = np.arange(200).tolist()

    def __init__(self, use_input_norm, label_percent):
        self.label_percent = label_percent
        self.label_percent_str = "{:.0%}".format(self.label_percent)
        import logging
        logging.info(f"ImageNet-A class initialized with label_percent: {self.label_percent_str}")
        if use_input_norm:
            self.common_trsf = [transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]

    def save_indices(self, labeled_indices, unlabeled_indices, output_dir):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        np.save(os.path.join(output_dir, f'{self.label_percent_str}_labeled_indices.npy'), labeled_indices)
        np.save(os.path.join(output_dir, f'{self.label_percent_str}_unlabeled_indices.npy'), unlabeled_indices)

    def load_indices(self, input_dir):
        labeled_indices = np.load(os.path.join(input_dir, f'{self.label_percent_str}_labeled_indices.npy'))
        unlabeled_indices = np.load(os.path.join(input_dir, f'{self.label_percent_str}_unlabeled_indices.npy'))
        return labeled_indices, unlabeled_indices
    def download_data(self):
        # as per Zhou et al (2023), download from ...
        data_dir = './data'
        indices_dir = os.path.join(data_dir, 'indices', 'imagenet-a')
        log_file = os.path.join(indices_dir, f'{self.label_percent_str}_class_distribution_log.txt')

        train_dir = os.path.join(data_dir, "imagenet-a", "train")
        test_dir = os.path.join(data_dir, "imagenet-a", "test")

        if not os.path.exists(train_dir) or not os.path.exists(test_dir):
            raise FileNotFoundError("ImageNet-A data directories not found...")

        train_dset = datasets.ImageFolder(train_dir, transform=None)
        test_dset = datasets.ImageFolder(test_dir, transform=None)

        # Use YOUR split_images_labels function:
        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)

        targets = self.train_targets

        if os.path.exists(indices_dir) and os.path.isfile(os.path.join(indices_dir, f'{self.label_percent_str}_labeled_indices.npy')):
            print("Loading existing indices...")
            labeled_indices, unlabeled_indices = self.load_indices(indices_dir)
        else:
            print("Generating and saving new indices...")
            labeled_indices = []
            unlabeled_indices = []

            num_classes = 200
            for i in range(num_classes):
                class_indices = np.where(targets == i)[0]
                np.random.shuffle(class_indices)
                num_labeled = max(1, int(self.label_percent * len(class_indices)))  # Ensure at least one labeled sample
                labeled_indices.extend(class_indices[:num_labeled])
                unlabeled_indices.extend(class_indices[num_labeled:])

            labeled_indices = np.array(labeled_indices)
            unlabeled_indices = np.array(unlabeled_indices)
            self.save_indices(labeled_indices, unlabeled_indices, indices_dir)
        self.train_data_lb = self.train_data[labeled_indices]
        self.train_targets_lb = self.train_targets[labeled_indices]
        self.train_data_ulb = self.train_data[unlabeled_indices]
        self.train_targets_ulb = self.train_targets[unlabeled_indices]

        self.test_data = self.test_data
        self.test_targets = self.test_targets
        self.log_class_distribution(labeled_indices, unlabeled_indices, targets, log_file)

    def log_class_distribution(self, labeled_indices, unlabeled_indices, targets, log_file):
        num_classes = 200
        class_labeled_counts = np.zeros(num_classes, dtype=int)
        class_unlabeled_counts = np.zeros(num_classes, dtype=int)

        for i in range(num_classes):
            class_labeled_counts[i] = np.sum(targets[labeled_indices] == i)
            class_unlabeled_counts[i] = np.sum(targets[unlabeled_indices] == i)

        with open(log_file, 'w') as f:
            for i in range(num_classes):
                f.write(f"Class {i}: Labeled samples: {class_labeled_counts[i]}, Unlabeled samples: {class_unlabeled_counts[i]}\n")
        print(f"Class distribution written to {log_file}")
        
class CUB(iData):
    use_path = True

    # 使用 build_transform 函数生成转换列表
    train_trsf = build_transform(is_train=True, args=None, isCifar=False)
    test_trsf = build_transform(is_train=False, args=None, isCifar=False)
    strong_trsf = build_transform(is_train=True, args=None, isCifar=False, strong=True)
    common_trsf = []

    class_order = np.arange(200).tolist()  # CUB 有200个类别

    def __init__(self, use_input_norm,label_percent):
        """
        初始化 CUB 数据集用于半监督学习。

        Args:
            use_input_norm (bool): 是否应用输入归一化。
            label_percent (float): 每个类别标记数据的百分比。
        """
        self.label_percent = label_percent
        self.label_percent_str = "{:.0%}".format(self.label_percent)
        import logging
        logging.info(f"CUB class initialized with label_percent: {self.label_percent_str}")
        if use_input_norm:
            self.common_trsf = [transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                    std=[0.229, 0.224, 0.225])]

    def save_indices(self, labeled_indices, unlabeled_indices, output_dir):
        """
        将标记和未标记的索引保存到指定目录。

        Args:
            labeled_indices (np.array): 标记样本的索引。
            unlabeled_indices (np.array): 未标记样本的索引。
            output_dir (str): 保存索引的目录。
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        np.save(os.path.join(output_dir, f'{self.label_percent_str}_labeled_indices.npy'), labeled_indices)
        np.save(os.path.join(output_dir, f'{self.label_percent_str}_unlabeled_indices.npy'), unlabeled_indices)

    def load_indices(self, input_dir):
        """
        从指定目录加载标记和未标记的索引。

        Args:
            input_dir (str): 加载索引的目录。

        Returns:
            tuple: (labeled_indices, unlabeled_indices)
        """
        labeled_indices = np.load(os.path.join(input_dir, f'{self.label_percent_str}_labeled_indices.npy'))
        unlabeled_indices = np.load(os.path.join(input_dir, f'{self.label_percent_str}_unlabeled_indices.npy'))
        return labeled_indices, unlabeled_indices

    def download_data(self):
        """
        下载并处理 CUB 数据集，将其划分为标记和未标记的子集。
        """
        data_dir = './data'
        indices_dir = os.path.join(data_dir, 'indices', 'cub')
        log_file = os.path.join(indices_dir, f'{self.label_percent_str}_class_distribution_log.txt')

        train_dir = os.path.join(data_dir, 'cub', 'train')
        test_dir = os.path.join(data_dir, 'cub', 'test')

        # 检查训练和测试目录是否存在
        if not os.path.exists(train_dir) or not os.path.exists(test_dir):
            raise FileNotFoundError("CUB 数据集的训练或测试目录未找到。请确保数据集已下载并按以下结构组织："
                                    "./data/cub/train/<class_id>/image.jpg 和 ./data/cub/test/<class_id>/image.jpg")

        # 使用 ImageFolder 加载数据
        train_dset = datasets.ImageFolder(train_dir, transform=None)  # 转换将在使用时应用
        test_dset = datasets.ImageFolder(test_dir, transform=None)

        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)

        # 转换为 NumPy 数组
        self.train_data = np.array(self.train_data)
        self.train_targets = np.array(self.train_targets)
        self.test_data = np.array(self.test_data)
        self.test_targets = np.array(self.test_targets)

        # 转换目标为 numpy 数组以便于索引
        targets = self.train_targets

        if os.path.exists(indices_dir) and os.path.isfile(os.path.join(indices_dir, f'{self.label_percent_str}_labeled_indices.npy')):
            print("加载现有的索引...")
            labeled_indices, unlabeled_indices = self.load_indices(indices_dir)
        else:
            print("生成并保存新的索引...")
            labeled_indices = []
            unlabeled_indices = []

            num_classes = 200
            # CUB 数据集中每个类别的样本数量可能不同，按实际样本数量进行划分
            for i in range(num_classes):
                class_indices = np.where(targets == i)[0]
                np.random.shuffle(class_indices)  # 随机打乱索引
                num_labeled = max(1, int(self.label_percent * len(class_indices)))  # 至少一个标记样本
                labeled_indices.extend(class_indices[:num_labeled])
                unlabeled_indices.extend(class_indices[num_labeled:])

            labeled_indices = np.array(labeled_indices)
            unlabeled_indices = np.array(unlabeled_indices)
            self.save_indices(labeled_indices, unlabeled_indices, indices_dir)

        # 分割数据为标记和未标记
        self.train_data_lb = self.train_data[labeled_indices]
        self.train_targets_lb = self.train_targets[labeled_indices]
        self.train_data_ulb = self.train_data[unlabeled_indices]
        self.train_targets_ulb = self.train_targets[unlabeled_indices]

        # 测试数据保持不变
        self.test_data = self.test_data  # 已经加载
        self.test_targets = self.test_targets

        # 记录类别分布
        self.log_class_distribution(labeled_indices, unlabeled_indices, targets, log_file)

    def log_class_distribution(self, labeled_indices, unlabeled_indices, targets, log_file):
        """
        记录每个类别的标记和未标记样本数量。

        Args:
            labeled_indices (np.array): 标记样本的索引。
            unlabeled_indices (np.array): 未标记样本的索引。
            targets (np.array): 所有样本的目标标签。
            log_file (str): 日志文件的路径。
        """
        num_classes = 200
        class_labeled_counts = np.zeros(num_classes, dtype=int)
        class_unlabeled_counts = np.zeros(num_classes, dtype=int)

        for i in range(num_classes):
            class_labeled_counts[i] = np.sum(targets[labeled_indices] == i)
            class_unlabeled_counts[i] = np.sum(targets[unlabeled_indices] == i)

        with open(log_file, 'w') as f:
            for i in range(num_classes):
                f.write(f"Class {i}: Labeled samples: {class_labeled_counts[i]}, Unlabeled samples: {class_unlabeled_counts[i]}\n")
        print(f"类别分布已记录到 {log_file}")

        
class omnibenchmark(iData):
    use_path = True
    
    train_trsf=build_transform(True, None)
    test_trsf=build_transform(False, None)
    common_trsf = [    ]

    class_order = np.arange(300).tolist()

    def __init__(self,use_input_norm):
        if use_input_norm:
            self.common_trsf = [transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]

    def download_data(self):
        # as per Zhou et al (2023), download from https://drive.google.com/file/d/1AbCP3zBMtv_TDXJypOCnOgX8hJmvJm3u/view?usp=sharing) or Onedrive: [link](https://entuedu-my.sharepoint.com/:u:/g/personal/n2207876b_e_ntu_edu_sg/EcoUATKl24JFo3jBMnTV2WcBwkuyBH0TmCAy6Lml1gOHJA?e=eCNcoA
        train_dir = "./data/omnibenchmark/train/"
        test_dir = "./data/omnibenchmark/test/"

        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)

        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)

class vtab(iData):
    use_path = True
    
    train_trsf=build_transform(True, None)
    test_trsf=build_transform(False, None)
    common_trsf = [    ]

    class_order = np.arange(50).tolist()

    def __init__(self,use_input_norm):
        if use_input_norm:
            self.common_trsf = [transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]

    def download_data(self):
        # as per Zhou et al (2013), download from https://drive.google.com/file/d/1xUiwlnx4k0oDhYi26KL5KwrCAya-mvJ_/view?usp=sharing) or Onedrive: [link](https://entuedu-my.sharepoint.com/:u:/g/personal/n2207876b_e_ntu_edu_sg/EQyTP1nOIH5PrfhXtpPgKQ8BlEFW2Erda1t7Kdi3Al-ePw?e=Yt4RnV
        train_dir = "./data/vtab/train/"
        test_dir = "./data/vtab/test/"

        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)

        #print(train_dset.class_to_idx)
        #print(test_dset.class_to_idx)

        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)

class cars(iData):
    use_path = True
    
    train_trsf=build_transform(True, None)
    test_trsf=build_transform(False, None)
    common_trsf = [    ]

    class_order = np.arange(196).tolist()

    def __init__(self,use_input_norm):
        if use_input_norm:
            self.common_trsf = [transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]

    def download_data(self):
        train_dir = "./data/cars/train/"
        test_dir = "./data/cars/test/"

        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)

        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)
        
class core50(iData):
    use_path = True
    
    train_trsf=build_transform(True, None)
    test_trsf=build_transform(False, None)
    common_trsf = [    ]

    class_order = np.arange(50).tolist()

    def __init__(self,inc,use_input_norm):
        self.inc=inc
        if use_input_norm:
            self.common_trsf = [transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]

    def download_data(self):
        #download from here: http://bias.csr.unibo.it/maltoni/download/core50/core50_imgs.npz
        train_dir = "./data/core50_imgs/"+self.inc+"/"
        #print(train_dir)
        test_dir = "./data/core50_imgs/test_3_7_10/"

        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)

        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)

class cddb(iData):
    use_path = True
    
    train_trsf=build_transform(True, None)
    test_trsf=build_transform(False, None)
    common_trsf = [    ]

    class_order = np.arange(2).tolist()

    def __init__(self,inc,use_input_norm):
        self.inc=inc
        if use_input_norm:
            self.common_trsf = [transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]

    def download_data(self):
        #download from here: https://coral79.github.io/CDDB_web/
        train_dir = "./data/CDDB/CDDB/"+self.inc+"/train/"
        #print(train_dir)
        test_dir = "./data/CDDB/CDDB-hard_val/"

        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)

        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)

class domainnet(iData):
    use_path = True
    
    train_trsf=build_transform(True, None)
    test_trsf=build_transform(False, None)
    common_trsf = [    ]

    class_order = np.arange(345).tolist()

    def __init__(self,inc,use_input_norm):
        self.inc=inc
        if use_input_norm:
            self.common_trsf = [transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]

    def download_data(self):
        #download from http://ai.bu.edu/M3SDA/#dataset (use "cleaned version")
        aa=np.loadtxt('./data/DomainNet/'+self.inc+'_train.txt',dtype='str')
        self.train_data=np.array(['./data/DomainNet/'+x for x in aa[:,0]])
        self.train_targets=np.array([int(x) for x in aa[:,1]])

        dil_tasks=['real','quickdraw','painting','sketch','infograph','clipart']
        files=[]
        labels=[]
        for task in dil_tasks:
            aa=np.loadtxt('./data/DomainNet/'+task+'_test.txt',dtype='str')
            files+=list(aa[:,0])
            labels+=list(aa[:,1])
        self.test_data=np.array(['./data/DomainNet/'+x for x in files])
        self.test_targets=np.array([int(x) for x in labels])



