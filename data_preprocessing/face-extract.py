import face_prepare
import os
from PIL import Image
from mtcnn import MTCNN
from time import sleep
from tqdm import tqdm
import sys
import argparse


def main(args):
    SOURCE_BASE_PATH = args.input_dir
    TARGET_BASE_PATH = args.output_dir
    detector = MTCNN()
    for sub_dir in os.listdir(SOURCE_BASE_PATH):
        print('-----Processing------')
        print(sub_dir)
        src_sub_dir = os.path.join(SOURCE_BASE_PATH,sub_dir)
        tgt_sub_dir = os.path.join(TARGET_BASE_PATH,sub_dir)
        if not os.path.isdir(tgt_sub_dir):
            os.mkdir(tgt_sub_dir)
        nrof_images_total = 0
        nrof_successfully_aligned = 0
        for cur_path in os.listdir(src_sub_dir):
            src_base_path = os.path.join(src_sub_dir,cur_path)
            tgt_base_path = os.path.join(tgt_sub_dir,cur_path)
            if not os.path.isdir(tgt_base_path):
                os.mkdir(tgt_base_path)
            with tqdm(total=len(os.listdir(src_base_path))-1,file=sys.stdout) as pbar:
                for filename in os.listdir(src_base_path):
                    nrof_images_total += 1
                    src_file_path = os.path.join(src_base_path,filename)
                    des_file_path = os.path.join(tgt_base_path,filename)
                    if filename == 'Thumbs.db':
                        print('Unable to align "%s"' % src_file_path)
                        continue
                    img_array =Image.open(src_file_path)
                    face_list, bbox = face_prepare.extract_faces(img_array, detector, args.detect_multiple_faces, args.threshold, args.image_size, args.margin)
                    img = Image.fromarray(face_list,mode='RGB')
                    img.save(os.path.join(des_file_path))
                    nrof_successfully_aligned +=1
                pbar.update(1)
                            
    print('Total number of images: %d' % nrof_images_total)
    print('Number of successfully aligned images: %d' % nrof_successfully_aligned)    

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    
    parser.add_argument('input_dir', type=str, help='Directory with unaligned images.')
    parser.add_argument('output_dir', type=str, help='Directory with aligned face thumbnails.')
    parser.add_argument('--image_size', type=int,
        help='Image size (height, width) in pixels.', default=160)
    parser.add_argument('--margin', type=int,
        help='Margin for the crop around the bounding box (height, width) in pixels.', default=44)
    parser.add_argument('--detect_multiple_faces', type=bool,
                        help='Detect and align multiple faces per image.', default=False)
    parser.add_argument('--threshold', type=float,
        help='Threshold for accepting the face, default is 0.8', default=0.8)
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))