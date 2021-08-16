"""Create test dataset from the source dataset"""

import sys
import os
import argparse
from tqdm import tqdm
from shutil import move

'''
Split dataset from source dataset to target dataset with the size 20% of the source dataset
'''

def move_data_tgt_path(src, des, size):
    list_sub_dir = os.listdir(src)
    try:
        os.mkdir(des)
    except:
        pass
    for sub_large_dir in list_sub_dir:
        print('------Processing-------')
        print(sub_large_dir)
        sub_l_path = os.path.join(src,sub_large_dir)
        sub_l_des_path = os.path.join(des,sub_large_dir)
        list_sub_s_dir = os.listdir(sub_l_path)
        try:
            os.mkdir(sub_l_des_path)
        except:
            continue
        for sub_dir in list_sub_s_dir:
            print(sub_dir)
            sub_src_path = os.path.join(sub_l_path,sub_dir)
            sub_des_path = os.path.join(sub_l_des_path,sub_dir)
            try:
                os.mkdir(sub_des_path)
            except:
                continue
            images_src = os.listdir(sub_src_path)
            num = int(len(images_src)*size)
            count = 1
            with tqdm(total=num, file=sys.stdout) as pbar:
                for image in images_src:
                    image_src_path = os.path.join(sub_src_path, image)
                    image_des_path = os.path.join(sub_des_path, image)
                    if os.path.exists(image_des_path):
                        continue
                    move(image_src_path, image_des_path)

                    pbar.update(1)
                    count += 1
                    if count > num:
                        break
def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('input_dir',type=str,\
                        help='Directory of raw dataset')

    parser.add_argument('output_dir',type=str,\
                        help='Directory of test dataset')

    parser.add_argument('--size', type=float,\
        help='--size: size of slip dataset, default is 0.2',default=0.2)
    
    return parser.parse_args()

def main(args):
    move_data_tgt_path(args.input_dir,args.output_dir,args.size)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))