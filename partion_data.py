import os
import re
from shutil import copyfile
import argparse
import math
import random
'''
Note:
    -i --inputDir: input of path dir
    -o --outputDir: output of path dir
    -r --ratio: ratio of partion data train and test
'''

def iterate_dir(source, output, ratio):
    source = source.replace('\\', '/')
    output = output.replace('\\', '/')
    train_dir = os.path.join(output, 'train')
    test_dir = os.path.join(output, 'test')
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)

    files = [f for f in os.listdir(source) if re.search(r'([a-zA-Z0-9\s_\\.\-:])+(.xlsx)$', f)]

    num_images = len(files)
    num_test_images = math.ceil(ratio * num_images)
    for i in range(num_test_images):
        idx = random.randint(0, len(files) - 1)
        filename = files[idx]
        copyfile(os.path.join(source, filename),
                 os.path.join(test_dir, filename))
        files.remove(files[idx])

    for filename in files:
        copyfile(os.path.join(source, filename),
                 os.path.join(train_dir, filename))


def main():
    parser = argparse.ArgumentParser(description="Partition datasets of images into training and testing sets",
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        '-i', '--inputDir',
        help='Path to the folder where the image datasets is stored. If not specified, the CWD will be used.',
        type=str,
        default="/home/huyphuong/PycharmProjects/project_reseach/CNN-chemistry-Han-river-data/data/EEM"
    )
    parser.add_argument(
        '-o', '--outputDir',
        help='Path to the output folder where the train and test dirs should be created. '
             'Defaults to the same directory as IMAGEDIR.',
        type=str,
        default=None
    )
    parser.add_argument(
        '-r', '--ratio',
        help='The ratio of the number of test images over the total number of images. The default is 0.1.',
        default=0.1,
        type=float)


    args = parser.parse_args()

    if args.outputDir is None:
        args.outputDir = args.inputDir
    iterate_dir(args.inputDir, args.outputDir, args.ratio)


if __name__ == '__main__':
    main()
    print('done')
