import math
import numpy as np
import json
import sys
import os
import matplotlib.pyplot as plt
import progressbar

# 1 path with csv of pictures
# 2 path with crop data
# 3 output path for npy of cropped images
# 4 output path for png of cropped images

PIC_HEIGHT = 369
PIC_WIDTH = 369
pic_path = sys.argv[1]
crop_path = sys.argv[2]
output_csv_path = sys.argv[3]
output_png_path = sys.argv[4]

def get_pic_name(filename):
    return filename.split(".")[0].split(" ")[0]
 

def pic_to_csv_dict(csv_names):
    pic_to_csv = {}
    for csv in csv_names:
        tmp_csv = get_pic_name(csv)
        pic_to_csv[tmp_csv] = csv
    return pic_to_csv


def pic_to_crop_dict(crop_names, pics):
    pic_to_crop = {}
    for pic in pics:
        pic_to_crop[pic] = []
        
    for crop in crop_names:
        tmp_pic = get_pic_name(crop)
        pic_to_crop[tmp_pic].append(crop)
    
    return pic_to_crop


def get_crop_data_from_file(json_filenam):
    f = open(json_filenam)
    data = json.load(f)
    return data


def get_pic_data_from_file(csv_filename):
    data = np.load(csv_filename, allow_pickle=True)
    return data
    

def crop_pic(pic, crop):
    origin_x, origin_y = get_index_of_pixel((crop["x"],crop["y"]), pic.shape)
    border_x, border_y = get_index_of_pixel((crop["x"]+crop["width"],crop["y"]+crop["height"]), pic.shape, ceil=True)
    return pic[origin_y:border_y, origin_x:border_x]
    

def get_index_of_pixel(px, pic_dim, ceil=False):
    x_dim, y_dim = pic_dim
    x_px, y_px = px
    x = (x_px*x_dim)/PIC_WIDTH
    y = (y_px*y_dim)/PIC_HEIGHT
    if(ceil):
        return math.ceil(x), math.ceil(y)
    return math.floor(x), math.floor(y)
    

def save_as_npy(cropped, name):
    np.save(os.path.join(output_csv_path, name), cropped, allow_pickle=True)


def save_as_png(cropped, name, vmin, vmax):
    plt.imshow(cropped, cmap='bwr', vmin=vmin, vmax=vmax)
    plt.axis('off')
    plt.savefig(os.path.join(output_png_path, "{}.png".format(name)), pad_inches=0, bbox_inches='tight')
    plt.clf()
  
  
def main():
    csv_names = os.listdir(pic_path)
    crop_names = os.listdir(crop_path)
    pic_to_csv = pic_to_csv_dict(csv_names)
    pic_to_csv = pic_to_csv_dict(csv_names)
    pic_to_crop = pic_to_crop_dict(crop_names, pic_to_csv.keys())

    print("Crop pictures...")
    bar = progressbar.ProgressBar()
    for pic, csv in bar(pic_to_csv.items()):
        print(pic)
        for crop in pic_to_crop[pic]:
            pic_data = get_pic_data_from_file(os.path.join(pic_path, csv))
            crop_data = get_crop_data_from_file(os.path.join(crop_path, crop))
            cropped_pic = crop_pic(pic_data, crop_data)
            vmin = pic_data.min()
            vmax = pic_data.max()
            name, _ = os.path.splitext(crop)
            save_as_npy(cropped_pic, name)
            save_as_png(cropped_pic, name, vmin, vmax)
  
if __name__ == "__main__":
    main()