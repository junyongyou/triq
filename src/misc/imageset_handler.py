"""
This script contains several methods to process images and image groups
"""
import os
from sklearn.model_selection import train_test_split
import numpy as np
from PIL import Image
from scipy.ndimage import sobel
import shutil
import glob
import scipy.stats
import matplotlib.pyplot as plt


def si_image(image):
    """
    SI of image based on the ITU-R Recommendation
    :param image: image array
    :return: SI
    """
    # return np.mean(sobel(image))
    return np.std(sobel(image))


def get_scores(folders, image_scores):
    """
    Get the image scores in folders
    :param folders: data folders
    :param image_scores: a dictionary of images and their MOS scores
    :return: score list in the data folders
    """
    scores = []
    for folder in folders:
        files = os.listdir(folder)
        for file in files:
            file_name = file.lower()
            if file_name.endswith(('.jpg', '.bmp')):
                score = image_scores[file_name]
                scores.append(score)
    return scores


def get_image_means(train_folders):
    """
    Get R,G,B means of images in the train folders
    :param train_folders:
    :return: R,G,B means
    """
    # train_folders = [r'.\image_quality_koniq10k\train\koniq_normal',
    #                  r'.\image_quality_koniq10k\train\koniq_small',
    #                  r'.\image_quality_koniq10k\train\live']
    mean_R = 0
    mean_G = 0
    mean_B = 0
    std_R = 0
    std_G = 0
    std_B = 0
    num = 0
    for folder in train_folders:
        files = os.listdir(folder)
        for file in files:
            file_name = file.lower()
            if file_name.endswith(('.jpg', '.bmp')):
                image_file = os.path.join(folder, file)
                image = np.asarray(Image.open(image_file), dtype=np.float32)
                mean_R += np.mean(image[:, :, 0])
                mean_G += np.mean(image[:, :, 1])
                mean_B += np.mean(image[:, :, 2])
                std_R += np.std(image[:, :, 0])
                std_G += np.std(image[:, :, 1])
                std_B += np.std(image[:, :, 2])
                num += 1
    mean_R /= num
    mean_G /= num
    mean_B /= num
    std_R /= num
    std_G /= num
    std_B /= num
    print('Mean-R: {}, mean-G: {}, mean-B:{}'.format(mean_R, mean_G, mean_B))
    print('Std-R: {}, Std-G: {}, Std-B:{}'.format(std_R, std_G, std_B))


def get_si(folders):
    """
    Get SI values in data folders
    :param folders: data folders
    :return: SI list
    """
    si = []
    for folder in folders:
        files = os.listdir(folder)
        for file in files:
            file_name = file.lower()
            if file_name.endswith(('.jpg', '.bmp')):
                image_file = os.path.join(folder, file)
                image = np.asarray(Image.open(image_file), dtype=np.float32)
                si.append(si_image(image))
            print('{} done'.format(file))
    return si


def draw_train_val_si_hist():
    """
    Draw the histogram of SI of train and validation sets
    :return:
    """
    # train_folders = [r'.\image_quality_koniq10k\train\koniq_normal',
    #                  r'.\image_quality_koniq10k\train\live']
    # val_folders = [r'.\image_quality_koniq10k\val\koniq_normal',
    #                r'.\image_quality_koniq10k\val\live']
    train_folders = [r'.\database\train\koniq_normal',
                     r'.\database\train\live']
    val_folders = [r'.\database\val\koniq_normal',
                   r'.\database\val\live']

    train_si = get_si(train_folders)
    val_si = get_si(val_folders)
    np.save(r'.\database\train_si.npy', train_si)
    np.save(r'.\database\val_si.npy', val_si)
    max_si = np.max(train_si)
    min_si = np.min(train_si)

    plt.figure()
    bins = np.linspace(min_si, max_si, 100)
    # bins = 100
    plt.hist(train_si, bins=bins, alpha=0.5, rwidth=0.95, color='skyblue', label='Train set')
    plt.xlim(min_si, max_si)
    plt.hist(val_si, bins=bins, alpha=1., rwidth=0.95, label='Validation set')
    plt.legend(loc='upper right')
    # plt.ylabel('Density')
    plt.xlabel('SI', fontsize=14)
    # plt.show()

    # plt.subplot(211)
    # plt.hist(train_si, density=True, bins=100)
    # plt.ylabel('Density')
    # plt.xlabel('Train SI')
    #
    # plt.subplot(212)
    # plt.hist(val_si, density=True, bins=100)
    # plt.ylabel('Density')
    # plt.xlabel('Val SI')
    # plt.show()


def draw_train_val_mos_hist():
    """
    Draw the histogram of MOS in the train and val sets
    :return:
    """
    train_folders = [r'.\database\train\koniq_normal',
                     # r'.\database\train\koniq_small',
                     r'.\database\train\live']
    val_folders = [r'.\database\val\koniq_normal',
                   # r'.\database\val\koniq_small',
                   r'.\database\val\live']

    koniq_mos_file = r'.\database\koniq10k_images_scores.csv'
    live_mos_file = r'.\database\live_wild\live_mos.csv'
    image_scores = get_image_scores(koniq_mos_file, live_mos_file)
    train_scores = get_scores(train_folders, image_scores)
    val_scores = get_scores(val_folders, image_scores)

    plt.figure()
    plt.subplot(211)
    bins = np.linspace(1, 5, 100)
    plt.hist(train_scores, bins=bins, alpha=0.5, rwidth=0.95, color='skyblue', label='Training set')
    plt.xlim(1, 5)
    plt.hist(val_scores, bins=bins, alpha=1., rwidth=0.95, label='Testing set')
    plt.legend(loc='upper left')
    # plt.ylabel('Density')
    plt.xlabel('MOS', fontsize=14)

    train_si = np.load(r'.\database\train_si.npy')
    val_si = np.load(r'.\database\val_si.npy')
    max_si = np.max(train_si)
    min_si = np.min(train_si)

    plt.subplot(212)
    bins = np.linspace(min_si, max_si, 100)
    # bins = 100
    plt.hist(train_si, bins=bins, alpha=0.5, rwidth=0.95, color='skyblue', label='Training set')
    plt.xlim(min_si, max_si)
    plt.hist(val_si, bins=bins, alpha=1., rwidth=0.95, label='Testing set')
    plt.legend(loc='upper right')
    # plt.ylabel('Density')
    plt.xlabel('SI', fontsize=14)

    plt.show()


def get_image_scores_from_two_file_formats(mos_file, file_format, mos_format, using_single_mos=True):
    """
    Get single MOS or distribution of scores from mos files with two format: koniq and live
    :param mos_file: mos file containing image path, distribution or std, and MOS
    :param file_format: koniq or live
    :param mos_format: MOS or Z-score
    :param using_single_mos: single MOS or distribution
    :return: dict {image_path: MOS or distribution}
    """
    mos_scale = [1, 2, 3, 4, 5]
    image_files = {}
    with open(mos_file, 'r+') as f:
        lines = f.readlines()
        for line in lines:
            content = line.split(',')
            image_file = content[0].replace('"', '').lower()

            if using_single_mos:
                score = float(content[-1]) if mos_format == 'mos' else float(content[1]) / 25. + 1
            else:
                if file_format == 'koniq':
                    scores_softmax = np.array([float(score) for score in content[1 : 6]])
                    score = [score_softmax / scores_softmax.sum() for score_softmax in scores_softmax]
                else:
                    std = float(content[-2]) if mos_format == 'mos' else float(content[-2]) / 25.
                    mean = float(content[-1]) if mos_format == 'mos' else float(content[-1]) / 25. + 1
                    score = get_distribution(mos_scale, mean, std)

            image_files[image_file] = score
    return image_files


def get_image_scores(koniq_mos_file, live_mos_file, using_single_mos=True):
    image_scores_koniq = get_image_scores_from_two_file_formats(koniq_mos_file, 'koniq', 'mos', using_single_mos)
    image_scores_live = get_image_scores_from_two_file_formats(live_mos_file, 'live', 'z-score', using_single_mos)
    return {**image_scores_koniq, **image_scores_live}


def get_image_score_from_groups(folders, image_scores):
    """
    Get group lists of image files and scores
    :param folders: image folders
    :param image_scores: a dictionary of images and their MOS scores
    :return: two lists
                image_file_groups: a list containing image file groups, each group containing image files
                score_groups: a list containing score groups, each group containing image scores
    """
    image_file_groups = []
    score_groups = []
    for folder in folders:
        files = os.listdir(folder)
        image_file_group = []
        score_group = []
        for file in files:
            file_name = file.lower()
            if file_name in image_scores:
                score = image_scores[file_name]
                score_group.append(score)
                image_file_group.append(os.path.join(folder, file))

        image_file_groups.append(image_file_group)
        score_groups.append(score_group)
    return image_file_groups, score_groups


def get_distribution(score_scale, mean, std, distribution_type='standard'):
    """
    Calculate the distribution of scores from MOS and standard distribution, two types of distribution are supported:
        standard Gaussian and Truncated Gaussian
    :param score_scale: MOS scale, e.g., [1, 2, 3, 4, 5]
    :param mean: MOS
    :param std: standard deviation
    :param distribution_type: distribution type (standard or truncated)
    :return: Distribution of scores
    """
    if distribution_type == 'standard':
        distribution = scipy.stats.norm(loc=mean, scale=std)
    else:
        distribution = scipy.stats.truncnorm((score_scale[0] - mean) / std, (score_scale[-1] - mean) / std, loc=mean, scale=std)
    score_distribution = []
    for s in score_scale:
        score_distribution.append(distribution.pdf(s))

    return score_distribution


def get_live_images():
    image_folder = r'.\database\live_wild\Images'
    image_mos_file = r'.\database\live_wild\live_mos.csv'
    # image_si = []
    # scores = []
    image_files = {}
    with open(image_mos_file, 'r+') as f:
        lines = f.readlines()
        for line in lines:
            content = line.split(',')
            image_file = os.path.join(image_folder, content[0])
            # image_files.append(image_file)
            # image = np.asarray(image_file, dtype=np.float32)
            score = float(content[-1])
            mos = (score / 25.) + 1
            image_files[image_file] = mos
            # scores.append(mos)
            # image_si.append(si_image(image))
    return image_files


def split_train_val(ratio=0.5):
    """
    Randomly split images in a database to training and testing sets in terms of SI and MOS
    :param ratio: splitting ratio
    :return:
    """
    target_train_folder = r'.\database\train\live'
    target_val_folder = r'.\database\val\live'
    # image_files = self.get_si_scores()
    image_files = get_live_images()
    mos_scale = [1, 2, 3, 4, 5]
    image_groups = []
    # train_image_files = []
    # val_image_files = []
    for s in range(len(mos_scale)-1):
        images = []
        for k, v in image_files.items():
            if mos_scale[s] <= v < mos_scale[s + 1]:
                images.append(k)
        image_groups.append(images)

    for image_group in image_groups:
        image_si = {}
        for image_file in image_group:
            image = np.asarray(Image.open(image_file), dtype=np.float32)
            si = si_image(image)
            image_si[image_file] = si

        sorted_si = {k : v for k, v in sorted(image_si.items(), key=lambda item: item[1])}
        val_num = int(1.0 / (1 - ratio)) + 1
        sorted_image_files = sorted_si.keys()
        for i, sorted_image_file in enumerate(sorted_image_files):
            basename = os.path.basename(sorted_image_file)
            image = Image.open(sorted_image_file)
            resized_image = image.resize(512, 512)
            if i % val_num == 0:
                resized_image.save(os.path.join(target_val_folder, basename))
                # shutil.copy(sorted_image_file, os.path.join(target_val_folder, basename))
                # val_image_files.append(sorted_image_file)
            else:
               resized_image.save(os.path.join(target_train_folder, basename))
                # shutil.copy(sorted_image_file, os.path.join(target_train_folder, basename))
                # train_image_files.append(sorted_image_file)


def resize_koniq_images(image_folder):
    """
    Halve size of images in the KonIQ-10k database
    :param image_folder: image folder of KonIQ-10 database
    :return:
    """
    target_folder = r'.\database\train\koniq_small'
    image_files = glob.glob(os.path.join(image_folder, '*.jpg'))
    for image_file in image_files:
        image = Image.open(image_file)
        resized_image = image.resize((512, 384))
        basename = os.path.basename(image_file)
        resized_image.save(os.path.join(target_folder, basename))


class GroupProvider:
    def __init__(self, image_folder, image_mos_file):
        self.image_folder = image_folder
        self.image_mos_file = image_mos_file

    def get_si_scores(self):
        # image_si = []
        # scores = []
        image_files = {}
        with open(self.image_mos_file, 'r+') as f:
            lines = f.readlines()
            for line in lines:
                content = line.split(',')
                image_file = os.path.join(self.image_folder, content[0].replace('"', ''))
                # image = np.asarray(Image.open(image_file), dtype=np.float32)
                # si = si_image(image)
                score = float(content[-3])
                image_files[image_file] = score
                # scores.append(score)
                # image_si.append(si)
        return image_files

    def generate_images(self):
        image_files, scores, image_files = self.get_si_scores()
        train_image_files, test_image_files, train_scores, test_scores = train_test_split(image_files, scores,
                                                                                          test_size=0.1,
                                                                                          random_state=42)
        return train_image_files, test_image_files, train_scores, test_scores

    def get_live_images(self):
        image_folder = r'.\database\live_wild\Images'
        image_mos_file = r'.\database\live_wild\live_mos.csv'
        # image_si = []
        # scores = []
        image_files = {}
        with open(image_mos_file, 'r+') as f:
            lines = f.readlines()
            for line in lines:
                content = line.split(',')
                image_file = os.path.join(image_folder, content[0])
                # image_files.append(image_file)
                # image = np.asarray(image_file, dtype=np.float32)
                score = float(content[1])
                mos = (score / 25.) + 1
                image_files[image_file] = mos
                # scores.append(mos)
                # image_si.append(si_image(image))
        return image_files

    def split_train_val(self, ratio=0.5):
        target_train_folder = r'.\database\train\live'
        target_val_folder = r'.\database\val\live'
        # image_files = self.get_si_scores()
        image_files = self.get_live_images()
        mos_scale = [1, 2, 3, 4, 5]
        image_groups = []
        # train_image_files = []
        # val_image_files = []
        for s in range(len(mos_scale)-1):
            images = []
            for k, v in image_files.items():
                if mos_scale[s] <= v < mos_scale[s + 1]:
                    images.append(k)
            image_groups.append(images)

        for image_group in image_groups:
            image_si = {}
            for image_file in image_group:
                image = np.asarray(Image.open(image_file), dtype=np.float32)
                si = si_image(image)
                image_si[image_file] = si

            sorted_si = {k : v for k, v in sorted(image_si.items(), key=lambda item: item[1])}
            val_num = int(1.0 / (1 - ratio)) + 1
            sorted_image_files = sorted_si.keys()
            for i, sorted_image_file in enumerate(sorted_image_files):
                basename = os.path.basename(sorted_image_file)
                image = Image.open(sorted_image_file)
                resized_image = image.resize(512, 512)
                if i % val_num == 0:
                    resized_image.save(os.path.join(target_val_folder, basename))
                    # shutil.copy(sorted_image_file, os.path.join(target_val_folder, basename))
                    # val_image_files.append(sorted_image_file)
                else:
                   resized_image.save(os.path.join(target_train_folder, basename))
                    # shutil.copy(sorted_image_file, os.path.join(target_train_folder, basename))
                    # train_image_files.append(sorted_image_file)

    def resize_koniq_images(self, image_folder):
        target_folder = r'.\database\train\koniq_small'
        image_files = glob.glob(os.path.join(image_folder, '*.jpg'))
        for image_file in image_files:
            image = Image.open(image_file)
            resized_image = image.resize((512, 384))
            basename = os.path.basename(image_file)
            resized_image.save(os.path.join(target_folder, basename))


if __name__ == '__main__':
    # image_folder = r'.\database\1024x768'
    # image_mos_file = r'.\database\koniq10k_images_scores.csv'

    # provider = GroupProvider(image_folder, image_mos_file)
    # provider.split_train_val()
    # print(1e-4/2)

    # draw_train_val_si_hist()
    draw_train_val_mos_hist()
    # get_distribution()

    # get_image_means()

    # v = [4,5]
    # s = np.std(v)
    #
    # mean = 68.9221 / 25. + 1
    # std = 21.2405 / 25.
    # mos_scale = [1, 2, 3, 4, 5]
    # score = get_distribution(mos_scale, mean, std)


