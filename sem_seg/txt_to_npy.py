import os
import sys
import argparse
import indoor3d_util

parser = argparse.ArgumentParser()
parser.add_argument('--path_in', help='path to the txt data folder.')
parser.add_argument('--path_out', help='path to save ply folder.')

parsed_args = parser.parse_args(sys.argv[1:])

path_in = parsed_args.path_in
path_out = parsed_args.path_out

if not os.path.exists(path_out):
    os.mkdir(path_out)

for folder in sorted(os.listdir(path_in)):

    path_annotation = os.path.join(path_in, folder, "annotations")
    print(path_annotation)

    try:
        elements = path_annotation.split('/')
        out_filename = elements[-2]+'.npy'
        indoor3d_util.collect_point_label(path_annotation, os.path.join(path_out, out_filename), 'numpy')
        
    except:
        print(path_annotation, 'ERROR!!')
