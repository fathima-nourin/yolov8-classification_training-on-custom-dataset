import os
import shutil

augmented_dataset_directory = r'D:\yolov8_dataset\augmented_dataset\augmented_test'
to_path = r'D:\yolov8_dataset\split_dataset\test_split'
main_dir = os.listdir(augmented_dataset_directory)
print(main_dir)

os.mkdir(os.path.join(to_path, 'train'))
os.mkdir(os.path.join(to_path, 'test'))
os.mkdir(os.path.join(to_path, 'val'))

for each_dir in main_dir:
    os.mkdir(os.path.join(to_path, 'train', each_dir))
    os.mkdir(os.path.join(to_path, 'test', each_dir))
    os.mkdir(os.path.join(to_path, 'val', each_dir))
    files = os.listdir(os.path.join(augmented_dataset_directory, each_dir))

    train_per = round(len(files) * 0.7)
    valid_per = round(len(files) * 0.2)
    test_per = round(len(files) * 0.1)

    for every_file in files[:train_per]:
        shutil.copyfile(os.path.join(augmented_dataset_directory, each_dir, every_file),
                        os.path.join(to_path, 'train', each_dir, every_file))
    for every_file in files[train_per:train_per+valid_per]:
        shutil.copyfile(os.path.join(augmented_dataset_directory, each_dir, every_file),
                        os.path.join(to_path, 'val', each_dir, every_file))
    for every_file in files[train_per+valid_per:]:
        shutil.copyfile(os.path.join(augmented_dataset_directory, each_dir, every_file),
                        os.path.join(to_path, 'test', each_dir, every_file))
