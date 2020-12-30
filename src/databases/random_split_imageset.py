import os
from sklearn.model_selection import train_test_split
import shutil
from PIL import Image


def do_image_resize(image_files, original_image_folder, target_image_folder):
    """
    Halves the images in KonIQ-10k database
    :param image_files:
    :param original_image_folder:
    :param target_image_folder:
    :return:
    """
    for image_file in image_files:
        image = Image.open(os.path.join(original_image_folder, image_file)).resize((512, 384))
        image.save(os.path.join(target_image_folder, image_file))


def do_split(original_image_folder, target_image_folder, mos_file, database='live'):
    all_files = []

    train_image_folder = os.path.join(target_image_folder, 'train', database)
    val_image_folder = os.path.join(target_image_folder, 'val', database)
    if not os.path.exists(train_image_folder):
        os.makedirs(train_image_folder)
    if not os.path.exists(val_image_folder):
        os.makedirs(val_image_folder)

    with open(mos_file) as mf:
        lines = mf.readlines()
        for line in lines:
            content = line.split(',')
            all_files.append(content[0])

    train_images, val_images = train_test_split(all_files, test_size=0.2, random_state=None)

    if database == 'live' or database == 'koniq_normal':
        for train_image in train_images:
            shutil.copy(os.path.join(original_image_folder, train_image),
                        os.path.join(train_image_folder, train_image))
        for val_image in val_images:
            shutil.copy(os.path.join(original_image_folder, val_image),
                        os.path.join(val_image_folder, val_image))
    else:
        do_image_resize(train_images, original_image_folder, train_image_folder)
        do_image_resize(val_images, original_image_folder, val_image_folder)


def random_split():
    # Specify the image folders for KonIQ-10 database and LIVE-wild database, suppose they are stored separately.
    koniq_image_folder = r''
    live_image_folder = r''

    # Specify the MOS files for KonIQ-10 database and LIVE-wild database, respectively.
    # Now the image files will be written to the current database folder, then can be used in model training
    live_mos = r'.\live_mos.csv'
    live_koniq = r'.\koniq10k_images_scores.csv'

    target_image_folder = r'.\\'

    do_split(live_image_folder, target_image_folder, live_mos)
    do_split(koniq_image_folder, target_image_folder, live_koniq, database='koniq_normal')
    do_split(koniq_image_folder, target_image_folder, live_koniq, database='koniq_small')


if __name__ == '__main__':
    random_split()