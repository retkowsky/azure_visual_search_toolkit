# Standard python utilities functions
# Serge Retkowsky (serge.retkowsky@microsoft.com)
# 20-Oct-2022


# Multiple imports
import configparser
import cv2
import datetime
import glob
import humanize
import json
import logging
import numpy as np
import os
import pathlib
import pendulum
import platform
import psutil
import random
import re
import requests
import shutil
import sys
import socket
import time
import uuid

from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from azure.cognitiveservices.vision.computervision.models import OperationStatusCodes
from azure.cognitiveservices.vision.computervision.models import VisualFeatureTypes
from msrest.authentication import CognitiveServicesCredentials
from matplotlib import pyplot as plt
from math import *
from pyzbar.pyzbar import decode


# Reading config.ini file to retrieve credentials
config = configparser.ConfigParser()
config_file = 'azureservices.py'
config.read(config_file)

# Azure CV Credentials
azure_cv_subscription_key = config.get('AzureComputerVision', 'key')
azure_cv_endpoint = config.get('AzureComputerVision', 'endpoint')

# Azure CV client
computervision_client = ComputerVisionClient(
    azure_cv_endpoint, CognitiveServicesCredentials(azure_cv_subscription_key))

azure_text_recognition_url = azure_cv_endpoint + "/vision/v3.2/read/analyze"
headers = {
    'Ocp-Apim-Subscription-Key': azure_cv_subscription_key,
    'Content-Type': 'application/octet-stream'
}


# Functions definitions


def check_python():
    """
    Display and check Python version
    """
    try:
        print("You are using Python:", sys.version)
        print("This notebook was made using Python 3.8.5 so this is OK")

        if sys.version[:5] != '3.8.5':
            print("[Note] This notebook was made using python 3.8.5")

    except:
        print("[Error] Cannot display python version")


def create_dir(MYDIR):
    """
    Create a new directory
    """
    if not os.path.exists(MYDIR):
        try:
            os.mkdir(MYDIR)
            print("Done. Directory:", MYDIR, "has been created")
        except:
            print("[Error] Cannot create directory")

    else:
        print("Dir", MYDIR, "exist. So you can use it")


def display_multiple_images(mylist,
                            nb_cols=3,
                            hspace=0.5,
                            wspace=0.05,
                            axis=False):
    """
    Display multiples images
    """
    plt.figure(figsize=(15, 10))

    idx = 1
    nb_rows = ceil(len(mylist) / nb_cols)

    while idx <= len(mylist):
        plt.subplot(nb_rows, nb_cols, idx)
        plt.subplots_adjust(hspace=hspace, wspace=wspace)
        img = cv2.cvtColor(cv2.imread(mylist[idx - 1]), cv2.COLOR_BGR2RGB)
        plt.imshow(img)

        if not axis:
            plt.axis("off")

        plt.title("Image " + str(idx) + ': ' + mylist[idx - 1], size=10)

        idx += 1


def display_random_images(mydir,
                          nb_images=10,
                          nb_cols=4,
                          hspace=0.5,
                          wspace=0.05,
                          titlesize=12):
    """
    Display random images from a directory
    """
    files_list = [
        file for file in os.listdir(mydir)
        if file.endswith(('jpeg', 'png', 'jpg', 'JPG', 'JPEG', 'PNG'))
    ]

    plt.figure(figsize=(15, 30))

    if nb_cols < 2:
        nb_cols = 2

    print("Some", nb_images, "random images from the", len(files_list),
          "images:")
    nb_rows = nb_images - nb_cols

    for idx in range(1, nb_images + 1):
        plt.subplot(nb_rows, nb_cols, idx)
        plt.subplots_adjust(hspace=hspace, wspace=wspace)
        random_idx = int(random.random() * len(files_list))
        img = cv2.cvtColor(cv2.imread(mydir + "/" + files_list[random_idx]),
                           cv2.COLOR_BGR2RGB)
        plt.axis('off')
        plt.imshow(img)
        plt.title(mydir + "/" + files_list[random_idx], size=titlesize)


def get_barcode_text(IMAGEFILE):
    """
    Get Barcode informations from a file using pyzbar. 
    Works with barcodes or QR Codes.
    """
    try:
        if not os.path.exists(IMAGEFILE):
            print("Error. File not exist:", IMAGEFILE)
            return

        decodingbarcode = decode(cv2.imread(IMAGEFILE))
        barcode_txt = ''

        if decodingbarcode == []:
            barcode_txt = ''

        else:
            barcode_txt = decodingbarcode[0][0].decode('utf-8')

        return barcode_txt

    except:
        print("[Error] Cannot get barcode information using pyzbar")


def get_image_azure_cvresults(IMAGEFILE):
    """
    ImageAnalysisInStream.
    This will analyze an image from a stream and return main colors, 
    tags and caption using Azure CV
    """
    if not os.path.exists(IMAGEFILE):
        print("Error. File not exist:", IMAGEFILE)
        return

    img_tags_list = []
    img_colors_list = []
    idx1 = 0
    idx2 = 0
    idx3 = 0

    img_tags = ""
    img_colors = ""

    time.sleep(1)

    with open(IMAGEFILE, "rb") as image_stream:
        image_analysis = computervision_client.analyze_image_in_stream(
            image=image_stream,
            visual_features=[
                VisualFeatureTypes.color,
                VisualFeatureTypes.tags,
                VisualFeatureTypes.description,
            ])

    img_dominant_color = image_analysis.color.dominant_colors
    img_dominant_color_foreground = image_analysis.color.dominant_color_foreground
    img_dominant_color_background = image_analysis.color.dominant_color_background

    while idx1 < len(img_dominant_color):
        img_colors_list.append(img_dominant_color[idx1])
        idx1 += 1

    img_colors_list.append(img_dominant_color_foreground)
    img_colors_list.append(img_dominant_color_background)
    img_colors_list = [*set(img_colors_list)]  # Deduplication

    while idx2 < len(img_colors_list):
        img_colors = img_colors + img_colors_list[idx2] + ' '
        idx2 += 1
    img_colors = img_colors[:-1]

    for tag in image_analysis.tags:
        if tag.confidence >= 0.5:  # Only main confidence tags
            img_tags_list.append(tag.name)

    while idx3 < len(img_tags_list):
        img_tags = img_tags + img_tags_list[idx3] + ' '
        idx3 += 1
    img_tags = img_tags[:-1]

    img_caption = image_analysis.description.captions[0].text  # get caption

    return img_colors, img_tags, img_caption


def get_ocr_text(IMAGEFILE):
    """
    Extract text from an image using Azure Read API
    """
    if not os.path.exists(IMAGEFILE):
        print("Error. File not exist:", IMAGEFILE)
        return

    params = {}

    fullocr = []
    nb_txt = 0
    ocr_text = ''
    poll = True

    dataimg = open(IMAGEFILE, "rb").read()  # Reading image

    response = requests.post(azure_text_recognition_url,
                             headers=headers,
                             params=params,
                             data=dataimg)

    response.raise_for_status()
    """
    The call returns with a response header field called Operation-Location. 
    The Operation-Location value is a URL that contains the Operation ID to 
    be used in the next step.
    operation_url = response.headers["Operation-Location"]
    """
    analysis = {}
    """
    The second step is to call Get Read Results operation.
    This operation takes as input the operation ID that was created 
    by the Read operation.
    """
    while (poll):
        response_final = requests.get(response.headers["Operation-Location"],
                                      headers=headers)
        analysis = response_final.json()
        # This is to avoid exceeding the requests per second (RPS) rate.
        time.sleep(2)

        if ("analyzeResult" in analysis):
            poll = False
        if ("status" in analysis and analysis['status'] == 'failed'):
            poll = False

    if ("analyzeResult" in analysis):
        full_ocr = [
            (line["boundingBox"], line["text"])
            for line in analysis["analyzeResult"]["readResults"][0]["lines"]
        ]

    while nb_txt < len(full_ocr):
        ocr_text = ocr_text + full_ocr[nb_txt][
            1] + ' '  # concatenation of all detected strings
        nb_txt += 1

    ocr_text = ocr_text.rstrip()  # removing the last space

    return ocr_text


def get_storage_infos(plot=True):
    """
    Get storage informations
    """
    print("Storage:\n")
    total, used, free = shutil.disk_usage("/")
    used_pct = round(used / total * 100, 2)
    free_pct = round(free / total * 100, 2)
    print("Total:", humanize.naturalsize(total))
    print("- Used:", humanize.naturalsize(used), "|", used_pct, "%")
    print("- Free:", humanize.naturalsize(free), "|", free_pct, "%")

    if plot:
        labels = [
            "Used: " + str(humanize.naturalsize(used)),
            "Free: " + str(humanize.naturalsize(free))
        ]
        values = [used, free]
        fig = plt.figure(figsize=(3, 3))
        plt.pie(values, labels=labels)
        plt.title("Storage")
        plt.show()

    return total, used, free


def get_system_info():
    """
    Get system informations
    """
    try:
        info = {}
        info['Platform'] = platform.system()
        info['Platform-release'] = platform.release()
        info['Platform-version'] = platform.version()
        info['Architecture'] = platform.machine()
        info['Hostname'] = socket.gethostname()
        info['IP-address'] = socket.gethostbyname(socket.gethostname())
        info['MAC-address'] = ':'.join(
            re.findall('..', '%012x' % uuid.getnode()))
        info['Processor'] = platform.processor()
        info['Python version'] = sys.version
        info['RAM'] = str(round(psutil.virtual_memory().total /
                                (1024.0**3))) + " Gb"

        print("System Informations:\n")
        print(json.dumps(info, indent=2, sort_keys=True))

    except:
        print("[Error] Cannot display system informations")


def get_today():
    """
    Display date and time
    """
    try:
        current_dt = datetime.datetime.utcnow()
        current_date = f"{current_dt:%d-%m-%Y %H:%M:%S}"
        return current_date

    except:
        print("[Error] Cannot display datetime")


def get_time(ms=False):
    """
    Display time
    """
    try:
        current_dt = datetime.datetime.utcnow()

        if not ms:
            current_time = f"{current_dt:%H:%M:%S}"

        else:
            current_time = f"{current_dt:%H:%M:%S.%f}"

        return current_time

    except:
        print("[Error] Cannot display time")


def image_view(IMAGEFILE, w=15, h=10, axis=True):
    """
    Function to display an image using a filename
    """
    try:
        img = cv2.imread(IMAGEFILE)
        file_height, file_width, file_c = img.shape
        imgtitle = IMAGEFILE + " ( width = " + str(
            file_width) + " height = " + str(file_height) + " )"
        plt.figure(figsize=(w, h))

        if axis:
            plt.axis('on')

        else:
            plt.axis('off')

        plt.title(imgtitle)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    except:
        print("[Error] Cannot display image using opencv from a filename")


def image_view_index(MYDIR, IDX):
    """
    Python function to display image with additional 
    information like shape, date and size
    Argument is an index from a list of files
    """
    try:
        images = glob.glob(MYDIR + '/*.*')
        fullpath_image = [image.replace('\\', '/') for image in images]
        filename_image = [image.split('/')[-1] for image in images]

    except:
        print("[Error] Error during the creation of the files list")

    try:
        img = cv2.imread(fullpath_image[IDX])
        file_height, file_width, file_c = img.shape
        file_size = os.path.getsize(fullpath_image[IDX])
        file_date = datetime.datetime.fromtimestamp(
            pathlib.Path(fullpath_image[IDX]).stat().st_mtime)
        print("Image file:", fullpath_image[IDX], '\nWidth =', file_width,
              'Height =', file_height, "\nSize:",
              humanize.naturalsize(file_size), "Date:", file_date)
        plt.figure(figsize=(6, 6))
        plt.axis('off')
        plt.title(filename_image[IDX])
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    except:
        print("[Error] Cannot display image")


def list_dir(MYDIR):
    """
    list all files with informations
    """
    idx = 1
    print("Files in directory:", MYDIR, "\n")

    try:
        for file in os.scandir(MYDIR):
            print(idx, "\t",
                  datetime.datetime.fromtimestamp(file.stat().st_atime),
                  humanize.naturalsize(file.stat().st_size), "\t", file.name)
            idx += 1

    except:
        print("[Error] Check if directory exist")


def now():
    """
    Get current date
    """
    try:
        now = pendulum.now('Europe/Paris')

    except:
        print("[Error] Cannot display datetime using pendulum python module")

    return now


def number_files(MYDIR):
    """
    Function to count the number of files from a directory
    """
    try:
        for root, _, files in os.walk(MYDIR):
            print(root, "=", humanize.intcomma(len(files)))
            print("\nThe directory", root, "contains",
                  humanize.intword(len(files)), "files")

    except:
        print("[Error] Check if directory exist")


def similar_images_display(imageref_file,
                           files_list,
                           scores_list,
                           w=15,
                           h=10,
                           nb_cols=3,
                           hspace=0.5,
                           wspace=0.05,
                           axis=False,
                           textsize=10):
    """
    Display reference image and similar images from a list
    """
    imageref_list = list()
    imageref_list.append(imageref_file)
    images_list = imageref_list + files_list

    plt.figure(figsize=(w, h))

    idx = 0
    nb_rows = ceil(len(images_list) / nb_cols)

    while idx < len(images_list):
        plt.subplot(nb_rows, nb_cols, idx + 1)
        plt.subplots_adjust(hspace=hspace, wspace=wspace)
        img = cv2.imread(images_list[idx])

        if not axis:
            plt.axis("off")

        if idx == 0:
            plt.title("Reference image\n" + images_list[0], size=textsize)

        if idx > 0:
            score = str(round(scores_list[idx - 1], 2))
            img_h, img_w, img_c = img.shape
            img_size = humanize.naturalsize(os.path.getsize(images_list[idx]))
            img_date = datetime.datetime.fromtimestamp(
                pathlib.Path(images_list[idx]).stat().st_mtime)

            plt.title("Similar image " + str(idx) + " | score =" + score +
                      "\n" + images_list[idx] + '\nh: ' + str(img_h) + ' w: ' +
                      str(img_w) + ' | ' + str(img_date) + ' | ' +
                      str(img_size),
                      size=textsize)

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.imshow(img_rgb)

        idx += 1


def view_file(FILENAME):
    """
    View file content
    """
    print("Viewing file:", FILENAME, "\n")
    with open(FILENAME, 'r') as f:
        print(f.read())

