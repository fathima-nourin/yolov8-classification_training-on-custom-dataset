import os
import cv2
import numpy as np
from pathlib import Path
import random


class DataAugmentation:
    """
    Handles with various augmentations for dataset.
    """

    def __init__(self):
        pass

    def fill(self, img, h, w):
        img = cv2.resize(img, (h, w), cv2.INTER_CUBIC)
        return img

    def horizontal_shift(self, img, ratio=0.0):
        if ratio > 1 or ratio < 0:
            print('Value should be less than 1 and greater than 0')
            return img
        ratio = random.uniform(-ratio, ratio)
        h, w = img.shape[:2]
        to_shift = w * ratio
        if ratio > 0:
            img = img[:, :int(w - to_shift), :]
        if ratio < 0:
            img = img[:, int(-1 * to_shift):, :]
        img = self.fill(img, h, w)
        return img

    def vertical_shift(self, img, ratio=0.0):
        if ratio > 1 or ratio < 0:
            print('Value should be less than 1 and greater than 0')
            return img
        ratio = random.uniform(-ratio, ratio)
        h, w = img.shape[:2]
        to_shift = h * ratio
        if ratio > 0:
            img = img[:int(h - to_shift), :, :]
        if ratio < 0:
            img = img[int(-1 * to_shift):, :, :]
        img = self.fill(img, h, w)
        return img

    def brightness(self, img, low, high):
        value = random.uniform(low, high)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hsv = np.array(hsv, dtype=np.float64)
        hsv[:, :, 1] = hsv[:, :, 1] * value
        hsv[:, :, 1][hsv[:, :, 1] > 255] = 255
        hsv[:, :, 2] = hsv[:, :, 2] * value
        hsv[:, :, 2][hsv[:, :, 2] > 255] = 255
        hsv = np.array(hsv, dtype=np.uint8)
        img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        return img

    def channel_shift(self, img, value):
        value = int(random.uniform(-value, value))
        img = img + value
        img[:, :, :][img[:, :, :] > 255] = 255
        img[:, :, :][img[:, :, :] < 0] = 0
        img = img.astype(np.uint8)
        return img

    def zoom(self, img, value):
        if value > 1 or value < 0:
            print('Value for zoom should be less than 1 and greater than 0')
            return img
        value = random.uniform(value, 1)
        h, w = img.shape[:2]
        h_taken = int(value * h)
        w_taken = int(value * w)
        h_start = random.randint(0, h - h_taken)
        w_start = random.randint(0, w - w_taken)
        img = img[h_start:h_start + h_taken, w_start:w_start + w_taken, :]
        img = self.fill(img, h, w)
        return img

    def horizontal_flip(self, img, flag):
        if flag:
            return cv2.flip(img, 1)
        else:
            return img

    def vertical_flip(self, img, flag):
        if flag:
            return cv2.flip(img, 0)
        else:
            return img

    def rotation(self, img, angle):
        angle = int(random.uniform(-angle, angle))
        h, w = img.shape[:2]
        M = cv2.getRotationMatrix2D((int(w / 2), int(h / 2)), angle, 1)
        img = cv2.warpAffine(img, M, (w, h))
        return img

    def process(self, dataset_directory, augmentation_dataset_directory, cl):
        if not os.path.exists(os.path.join(augmentation_dataset_directory, cl)):
            os.mkdir(os.path.join(augmentation_dataset_directory, cl))

        for each_file in os.listdir(os.path.join(dataset_directory, cl)):
            filename, file_extension = os.path.splitext(each_file)

            if file_extension in ['.jpg', '.jpeg', '.png']:
                image = cv2.imread(os.path.join(dataset_directory, cl, each_file))
                multi_images = (
                    self.horizontal_shift(image), self.vertical_shift(image), self.brightness(image, 0.5, 3),
                    self.zoom(image, 0.5), self.channel_shift(image, 60), self.horizontal_flip(image, True),
                    self.vertical_flip(image, True), self.rotation(image, 60))

                _file_name = 0
                for each_element in multi_images:
                    image = each_element
                    cv2.imwrite(
                        os.path.join(augmentation_dataset_directory, cl, f"{each_file[:-4]}" + "_" + f"{_file_name}" + ".jpg"),
                        image)
                    _file_name += 1


def main():
    dataset_directory = Path(r"D:\yolov8_dataset\dataset_cats_dogs")
    augmented_dataset_directory = Path(r"D:\yolov8_dataset\augmented_dataset\augmented_test")
    cls = ['cats', 'dogs']

    augmentation_obj = DataAugmentation()
    for cl in cls:
        augmentation_obj.process(dataset_directory, augmented_dataset_directory, cl)

    for cl in cls:
        if not os.path.exists(os.path.join(augmented_dataset_directory, cl)):
            os.mkdir(os.path.join(augmented_dataset_directory, cl))


if __name__ == "__main__":
    main()
