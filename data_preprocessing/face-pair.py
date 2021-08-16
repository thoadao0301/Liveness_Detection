import os
import numpy as np
import face_prepare
import argparse
import sys

def main(args):
    FACE_DATA_DIR = args.input_dir_client
    IMPOSTER_DATA_DIR = args.input_dir_imposter

    NUMPY_DATA_PATH = args.output_dir
    X_data = 'X_data.npy'
    y_data = 'y_data.npy'

    X,y = face_prepare.get_data(FACE_DATA_DIR,IMPOSTER_DATA_DIR,args.sample_size)
    np.save(os.path.join(NUMPY_DATA_PATH,X_data),X)
    np.save(os.path.join(NUMPY_DATA_PATH,y_data),y)

def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('input_dir_client',type=str,\
        help='Real face directory')
    
    parser.add_argument('input_dir_imposter',type=str,\
        help='Imposter face directory')
    
    parser.add_argument('--sample_size',type=int,\
        help='number of sample for each subdirectory, default is 500',default=500)

    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
    
