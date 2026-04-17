import numpy as np
import torch
import os

def count_parameters(model, trainable=False):
    if trainable:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())

def tensor2numpy(x):
    return x.cpu().data.numpy() if x.is_cuda else x.data.numpy()

def target2onehot(targets, n_classes):
    onehot = torch.zeros(targets.shape[0], n_classes).to(targets.device)
    onehot.scatter_(dim=1, index=targets.long().view(-1, 1), value=1.0)
    return onehot

def accuracy(y_pred, y_true, nb_old, class_increments):
    assert len(y_pred) == len(y_true), "Data length error."
    all_acc = {}
    acc_total = np.around(
        (y_pred == y_true).sum() * 100 / len(y_true), decimals=2
    )

    # Grouped accuracy
    for classes in class_increments:
        idxes = np.where(
            np.logical_and(y_true >= classes[0], y_true <= classes[1])
        )[0]
        label = "{}-{}".format(
            str(classes[0]).rjust(2, "0"), str(classes[1]).rjust(2, "0")
        )
        all_acc[label] = np.around(
            (y_pred[idxes] == y_true[idxes]).sum() * 100 / len(idxes), decimals=2
        )

    return acc_total,all_acc

def split_images_labels(imgs):
    # split trainset.imgs in ImageFolder
    images = []
    labels = []
    for item in imgs:
        images.append(item[0])
        labels.append(item[1])

    return np.array(images), np.array(labels)


#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=#

#来自https://github.com/microsoft/Semi-supervised-learning ，不过它的功能有点太全了，简化一下

#hard code，但我觉得没问题
base_dir = os.path.dirname(os.path.dirname(__file__))


def split_ssl_data(args, data, targets, num_classes,
                   lb_num_labels, ulb_num_labels=None,
                   lb_index=None, ulb_index=None, include_lb_to_ulb=True, load_exist=True):
    """
    data & target is splitted into labeled and unlabeled data.

    Args
        data: data to be split to labeled and unlabeled 
        targets: targets to be split to labeled and unlabeled 
        num_classes: number of total classes
        lb_num_labels: total number of labeled samples
        ulb_num_labels: total number of unlabeled samples, default to None, meaning use all remaining data except for labeled data as unlabeled set
        lb_index: If np.array of index is given, select the data[index], target[index] as labeled samples.
        ulb_index: If np.array of index is given, select the data[index], target[index] as labeled samples.
        include_lb_to_ulb: If True, labeled data is also included in unlabeled data
    """
    data, targets = np.array(data), np.array(targets)
    lb_idx, ulb_idx = sample_labeled_unlabeled_data(args, data, targets, num_classes, 
                                                    lb_num_labels, ulb_num_labels, load_exist=False)
    
    # Manually set lb_idx and ulb_idx if provided, typically for debugging
    if lb_index is not None:
        lb_idx = lb_index
    if ulb_index is not None:
        ulb_idx = ulb_index

    # Include labeled data in unlabeled dataset if specified
    if include_lb_to_ulb:
        ulb_idx = np.concatenate([lb_idx, ulb_idx], axis=0)
    
    return data[lb_idx], targets[lb_idx], data[ulb_idx], targets[ulb_idx]



def sample_labeled_unlabeled_data(args, data, target, num_classes,
                                  lb_num_labels, ulb_num_labels=None,
                                  load_exist=True):
    '''
    samples for labeled data with balanced ratio over classes
    用到args.num_labels, args.seed, args.dataset
    args.num_labels: total number of labeled samples 具体到每个类别的有标样本数为args.num_labels/num_classes
    args.ulb_num_labels: 同上，但是是无标样本,如果这个参数被设置为 None，则默认使用除了有标签样本之外的所有剩余数据作为无标签数据。
    '''
    dump_dir = os.path.join(base_dir, 'data','idx', args.dataset, 'labeled_idx')
    os.makedirs(dump_dir, exist_ok=True)
    lb_dump_path = os.path.join(dump_dir, f'lb_labels{args.num_labels}_seed{args.seed}_idx.npy')
    ulb_dump_path = os.path.join(dump_dir, f'ulb_labels{args.num_labels}_seed{args.seed}_idx.npy')

    if os.path.exists(lb_dump_path) and os.path.exists(ulb_dump_path) and load_exist:
        lb_idx = np.load(lb_dump_path)
        ulb_idx = np.load(ulb_dump_path)
        return lb_idx, ulb_idx 
    
    # Ensure lb_num_labels is dividable by num_classes in balanced setting
    assert lb_num_labels % num_classes == 0, "lb_num_labels must be dividable by num_classes in balanced setting"
    lb_samples_per_class = [int(lb_num_labels / num_classes)] * num_classes
    
    # Set up unlabeled samples per class based on remaining data or specified unlabeled labels
    if ulb_num_labels is None or ulb_num_labels == 'None':
        # Assume all remaining samples are used as unlabeled
        pass
        #ulb_samples_per_class = [int((len(data) - lb_num_labels) / num_classes)] * num_classes
    else:
        # Ensure ulb_num_labels is dividable by num_classes
        assert ulb_num_labels % num_classes == 0, "ulb_num_labels must be dividable by num_classes in balanced setting"
        ulb_samples_per_class = [int(ulb_num_labels / num_classes)] * num_classes

    lb_idx = []
    ulb_idx = []
    
    for c in range(num_classes):
        idx = np.where(target == c)[0]
        np.random.shuffle(idx)
        lb_idx.extend(idx[:lb_samples_per_class[c]])
        ulb_idx.extend(idx[lb_samples_per_class[c]:lb_samples_per_class[c] + ulb_samples_per_class[c]])

    # Convert list to numpy array for storage
    lb_idx = np.asarray(lb_idx)
    ulb_idx = np.asarray(ulb_idx)

    # Save the indices to disk
    np.save(lb_dump_path, lb_idx)
    np.save(ulb_dump_path, ulb_idx)
    
    return lb_idx, ulb_idx









