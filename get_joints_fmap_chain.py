import os
import argparse
parser = argparse.ArgumentParser(description='Series of operation to generate joints feature map')
parser.add_argument('--dataset', type=str, default='HMDB51')
parser.add_argument('--scope', type=str, default='test2')
args = parser.parse_args()



os.system(  ('python ./scripts/generate_spatial2.py --dataset=' + args.dataset + ' --scope=' + args.scope)  )

os.system(  ('python ./openpose/generate_joints_xy.py --dataset=' + args.dataset + ' --scope=' + args.scope)  )

os.system(  ('python ./scripts/generate_joints_gray.py --dataset=' + args.dataset + ' --scope=' + args.scope)  )

os.system(  ('python ./scripts/generate_fmap_from_joints.py --dataset=' + args.dataset + ' --scope=' + args.scope + ' --gray=True')  )
# generate_spatial