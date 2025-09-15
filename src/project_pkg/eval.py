"""Evaluation and inference helpers extracted from the original notebook."""

from . import utils
from . import models, data

# ---- cell 1 ----
# torch.manual_seed(seed)
# if cuda:
#     torch.cuda.manual_seed(seed)

sss = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state = 3444)
for train_idx, valt_idx in sss.split(labels, labels):
    train_labels = labels_cat[train_idx]
    valt_labels = labels_cat[valt_idx]
    print("TRAIN indexes:", train_idx, "number of objects:", train_idx.shape)
    print(np.sum(train_labels, axis=0))
    print("Valtest indexes:", valt_idx, "number of objects:", valt_idx.shape)
    print(np.sum(valt_labels, axis=0))

train_labels = labels[train_idx]
valt_labels = labels[valt_idx]
train_images = images[train_idx]
valt_images = images[valt_idx]

# ---- cell 2 ----
ssss = StratifiedShuffleSplit(n_splits=1, test_size = 0.5, random_state = 3444)
for val_idx, test_idx in ssss.split(valt_labels, valt_labels):
    val_labels = valtest_labels_cat[val_idx]
    test_labels = valtest_labels_cat[test_idx]
    print("Validation indexes:", val_idx, "number of objects:", val_idx.shape)
    print(np.sum(val_labels, axis=0))
    print("TEST indexes:", test_idx, "number of objects:", test_idx.shape)
    print(np.sum(test_labels, axis=0))

val_labels = valt_labels[val_idx]
test_labels = valt_labels[test_idx]
val_images = valt_images[val_idx]
test_images = valt_images[test_idx]

# ---- cell 3 ----
main = os.getcwd()

# 1.Create directory tree and save images
dataset_dirname = 'Galaxy10'
try:
    os.mkdir(dataset_dirname)
except OSError:
    print('OSError: Creating or already exists the directory')
# Main dataset folder
os.chdir(dataset_dirname)

# 2.Train partition
os.mkdir('train')
os.chdir('train')
for cls in class_names:   # for each class 'cls'
    os.mkdir(cls)
    # train/<class> folder save images
    cls_int = class_names.index(cls)
    print('Train - Class: ', cls)
    for i in range(len(train_labels)):   # traverse all train lavels
        if train_labels[i] == cls_int:   # save instance 'i' belong to class 'cls'
            img_path = os.path.join(os.getcwd(), cls)
            #print('Save image {} in {}'.format(i, img_path))
            img = Image.fromarray(train_images[i])
            img.save(os.path.join(img_path, 'galaxy10_img{}.jpg'.format(i)))

# 3.Validation partition
os.chdir(os.path.join(main, dataset_dirname))
os.mkdir('val')
os.chdir('val')
for cls in class_names:
    os.mkdir(cls)
    # val/<class> folder save images
    cls_int = class_names.index(cls)
    print('Validation - Class: ', cls)
    for i in range(len(val_labels)):   # traverse all train lavels
        if val_labels[i] == cls_int:   # save instance 'i' belong to class 'cls'
            img_path = os.path.join(os.getcwd(), cls)
            #print('Save image {} in {}'.format(i, img_path))
            img = Image.fromarray(val_images[i])
            img.save(os.path.join(img_path, 'galaxy10_img{}.jpg'.format(i)))

# 3.test partition
os.chdir(os.path.join(main, dataset_dirname))
os.mkdir('test')
os.chdir('test')
for cls in class_names:
    os.mkdir(cls)
    # val/<class> folder save images
    cls_int = class_names.index(cls)
    print('Test - Class: ', cls)
    for i in range(len(test_labels)):   # traverse all train lavels
        if test_labels[i] == cls_int:   # save instance 'i' belong to class 'cls'
            img_path = os.path.join(os.getcwd(), cls)
            #print('Save image {} in {}'.format(i, img_path))
            img = Image.fromarray(test_images[i])
            img.save(os.path.join(img_path, 'galaxy10_img{}.jpg'.format(i)))

os.chdir(main)

