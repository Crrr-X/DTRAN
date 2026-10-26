import random
import os

random.seed(19) 
scale = 6

base_dir = "/public/home/Data"

hr_dir = os.path.join(base_dir, "HR")
lr_dir = os.path.join(base_dir, "LR")
mliA_dir = os.path.join(base_dir, "Feats", "mliA")
mliD_dir = os.path.join(base_dir, "Feats", "mliD")
ccA_dir = os.path.join(base_dir, "Feats", "ccA")
ccD_dir = os.path.join(base_dir, "Feats", "ccD")
slope_dir = os.path.join(base_dir, "Feats", "slope")
aspect_dir = os.path.join(base_dir, "Feats", "aspect")

name_lists_hr = sorted([name for name in os.listdir(hr_dir) if name.endswith(".tif") ])
name_lists_lr = sorted([name for name in os.listdir(lr_dir) if name.endswith(".tif") ])
name_lists_mliA = sorted([name for name in os.listdir(mliA_dir) if name.endswith(".tif") ])
name_lists_mliD = sorted([name for name in os.listdir(mliD_dir) if name.endswith(".tif") ])
name_lists_ccA = sorted([name for name in os.listdir(ccA_dir) if name.endswith(".tif") ])
name_lists_ccD = sorted([name for name in os.listdir(ccD_dir) if name.endswith(".tif") ])
name_lists_slope = sorted([name for name in os.listdir(slope_dir) if name.endswith(".tif") ])
name_lists_aspect = sorted([name for name in os.listdir(aspect_dir) if name.endswith(".tif") ])

dataset_list = [
    os.path.join(hr_dir, i) + " " + \
    os.path.join(lr_dir, j) + " " + \
    os.path.join(mliA_dir, k) + " " + \
    os.path.join(mliD_dir, l) + " " + \
    os.path.join(ccA_dir, m) + " " + \
    os.path.join(ccD_dir, n) + " " + \
    os.path.join(slope_dir, o) + " " + \
    os.path.join(aspect_dir, p) + "\n" 
    for i, j, k, l, m, n, o, p in zip(name_lists_hr, name_lists_lr, name_lists_mliA, name_lists_mliD, name_lists_ccA, name_lists_ccD, name_lists_slope, name_lists_aspect)
]
train_list = random.sample(dataset_list, 1590) 
val_list =  [i for i in dataset_list if i not in train_list]
val_list = random.sample(val_list, 399) 

with open(base_dir + "/train_" + str(scale) + "x.txt", "w") as f:
    for name in train_list:
        f.write(name)

with open(base_dir +  "/val_" + str(scale) + "x.txt", "w") as f:
    for name in val_list:
         f.write(name)
