"""Script writing single frames to video"""
import cv2
import os
import argparse

parser = argparse.ArgumentParser(description='Write images to video')
parser.add_argument(
    '-i', '--images', help='Path to folder containing the images', required=True)
parser.add_argument('-o', '--out', default='video_out',
                    help='Desired output path')
parser.add_argument('-f', '--fps', help='Frames per second', default=20)
parser.add_argument('-s', '--start_index', default=0)
parser.add_argument('-e', '--end_index', default=None)
parser.add_argument('-r', '--resize_images', default=False,
                    help='Use new_widthxnew_height, e.g. 200x100 to resize all images')

args = parser.parse_args()
argsdict = vars(args)


inpath = argsdict['images']
if not os.path.isabs(inpath):
    inpath = os.path.abspath(os.path.join(os.path.dirname(__file__), inpath))

outpath = argsdict['out'] + '.avi'
if not os.path.isabs(outpath):
    outpath = os.path.abspath(os.path.join(os.path.dirname(__file__), outpath))

fps = int(argsdict['fps'])

if cv2.__version__ < '3':
    fourcc = cv2.cv.CV_FOURCC(*'DIVX')  # <<-- works for opencv 2.x
else:
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')

img_paths = [os.path.join(inpath, x) for x in os.listdir(
    inpath) if '.png' in x.lower() or '.jpg' in x.lower()]
img_paths.sort()

start_idx = int(argsdict['start_index'])
end_idx = argsdict['end_index']
end_idx = len(img_paths) - 1 if end_idx is None else int(end_idx)
height, width, _ = cv2.imread(img_paths[0]).shape

resize = argsdict['resize_images']
low_quality = False
if resize:
    low_quality = True
    width, height = resize.lower().split('x')
    width = int(width)
    height = int(height)

video_writer = cv2.VideoWriter(outpath, fourcc, fps, (width, height))

for idx in range(start_idx, end_idx):
    img = cv2.imread(img_paths[idx])
    # decrease size of video
    if low_quality:
        img = cv2.resize(img, (width, height))
    video_writer.write(img)

video_writer.release()
