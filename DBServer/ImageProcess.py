# -*- coding: utf-8 -*-
import re
import os
import uuid


def change_image_path_to_base64(response_msg):
    message = re.sub(r'.*(I:.*.png).*', replace_image_paths, response_msg)
    return message


def replace_image_paths(match):
    image_path = match.group(1)
    image_str = read_image_info_to_str(image_path)
    return '<image>' + image_str + '<image>'

def read_image_info_to_str(image_path):
    with open(image_path, 'rb') as f:
        image_data = f.read()
        image_str = str(image_data)
    return image_str

def read_image_str_to_image(image_str, image_path):
    image_data = eval(image_str)
    with open(image_path, 'wb') as f:
        f.write(image_data)