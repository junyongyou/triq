# Databases for training TRIQ

This work uses two publicly available databases: KonIQ-10k [KonIQ-10k: An ecologically valid database for deep learning of blind image quality assessment](https://ieeexplore.ieee.org/document/8968750) by V. Hosu, H. Lin, T. Sziranyi, and D. Saupe;
 and LIVE-wild [Massive online crowdsourced study of subjective and objective picture quality](https://ieeexplore.ieee.org/document/7327186) by D. Ghadiyaram, and A.C. Bovik

The train_images_koniq(live) and test_images_koniq(live) list the images in the training and testing sets, which were randomly chosen from the two databases in terms of SI (image complexity) and MOS.

It is also no problem to run multiple experiments with randomly split train and test images each time, which can be done by running databases\random_split_imageset.py.