import os
from PIL import Image

from_dir = "D:/Data/GTSRB/Final_Training/Images"
to_dir = "D:/Data/GTSRB/Final_Training/Images_jpg"

def ppm2jpg(dir1, dir2):
    dir_gen = (file for file in os.listdir(dir1) if os.path.isdir(os.path.join(dir1, file)))
    try:
        while(True):
            dir = next(dir_gen)
            ppm_dir = os.path.join(from_dir, dir)
            jpg_dir = os.path.join(to_dir, dir)
            if not os.path.exists(jpg_dir):
                os.mkdir(jpg_dir)

            ppm_gen = (img for img in os.listdir(ppm_dir) if img.endswith(".ppm"))
            try:
                while(True):
                    ppm = next(ppm_gen)
                    jpg_file = os.path.join(jpg_dir, ppm.split('.')[0]) + '.jpg'
                    ppm_file = os.path.join(ppm_dir, ppm)
                    im = Image.open(ppm_file)
                    im.save(jpg_file)

            except StopIteration:
                print("finished one")

            print(dir)
    except StopIteration:
        print("finished")


if __name__ == '__main__':
    ppm2jpg(from_dir, to_dir)