from pathlib import Path
import argparse, os
parser = argparse.ArgumentParser(description=('Get some path from input and',
                                            'creates it if not there'))

parser.add_argument('--filepath', default="./example/folder/filename.extension", type=str)
args = parser.parse_args()

def basic(args):
    atoms = args.filepath.split('/')
    path, file = '/'.join(atoms[:-1]), atoms[-1]
    Path(path).mkdir(parents=True, exist_ok=True)
    with open(args.filepath,'a') as fp:
    fp.write("message")

def callback(args):
    pass
