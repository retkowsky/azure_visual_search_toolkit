# Vec2Text functions
# 20-Oct-2022


# Multiple imports
import base64
import collections
import gensim
import glob
import math
import numpy as np
import os
import random
import shutil

from os import listdir
from os.path import isfile, join
from pathlib import Path
from PIL import Image
from sentence_transformers import SentenceTransformer, util
from shutil import rmtree
from sklearn.cluster import KMeans


# Functions definitions


def calculate_dimensions(file, model):
    """
    Calculate dimensions
    """
    return image_embedding(file, model).shape[0]


def calculate_wcss(data):
    """
    Clustering K Means algorithm from sci-kit learn
    """
    data = np.array(data).reshape(-1, 1)
    wcss = []

    for n in range(2, 21):
        kmeans = KMeans(n_clusters=n)
        kmeans.fit(X=data)
        wcss.append(kmeans.inertia_)

    return wcss


def convert_vec_to_bucket_str(curVec, bucketRanges):
    """
    Convert Vect Text To Bucket string
    """
    vecStr = ''

    for d in range(len(curVec)):
        bucketVal = -1
        idx = 0

        for i in bucketRanges[d]:
            if curVec[d] < i:
                bucketVal = idx
                break

            idx += 1
        vecStr += convert_field_num_to_string(d) + '_' + str(
            bucketVal).replace('-', '~') + ' '

    return vecStr.strip()


def convert_field_num_to_string(dim):
    """
    Generate fake terms from
    """
    dimStr = str(dim)
    curStr = ''

    for i in range(len(dimStr)):
        curChar = dimStr[i]
        if curChar == '0':
            curStr += 'A'
        elif curChar == '1':
            curStr += 'B'
        elif curChar == '2':
            curStr += 'C'
        elif curChar == '3':
            curStr += 'D'
        elif curChar == '4':
            curStr += 'E'
        elif curChar == '5':
            curStr += 'F'
        elif curChar == '6':
            curStr += 'G'
        elif curChar == '7':
            curStr += 'H'
        elif curChar == '8':
            curStr += 'I'
        elif curChar == '9':
            curStr += 'J'

    return curStr


def closest(lst, K):
    """
    Find the closest cluster center index to a specified number (K)
    """
    b = lst[min(range(len(lst)), key=lambda i: abs(lst[i] - K))]

    return lst.index(b)


def find_cluster_centers(dimensions, vecDict):
    """
    Get Cluster Centers
    """
    clusterCenters = {}
    idx = 0

    for d in range(dimensions):
        idx += 1

        if idx % 10 == 0:
            print('Processed', idx, 'of', dimensions)

        numberOfClusters = optimal_number_of_clusters(
            calculate_wcss(vecDict[str(d)]))
        x = np.array(vecDict[str(d)])
        km = KMeans(n_clusters=numberOfClusters)
        km.fit(x.reshape(-1, 1))
        cluster_centers = km.cluster_centers_
        cluster_centers = sorted(cluster_centers.tolist())

        clusterList = []

        for cc in cluster_centers:
            clusterList.append(cc[0])
        clusterCenters[d] = clusterList

    return clusterCenters


def get_files_in_dir(in_dir):
    """
    Get files from a directory
    """
    return [
        os.path.join(dp, f) for dp, dn, filenames in os.walk(in_dir)
        for f in filenames
    ]


def image_embedding(file, model):
    """
    Encoding image file using sentence transformers OpenAI CLIP model
    """
    img_emb = model.encode(Image.open(file))

    return np.array(img_emb)


def initialize_vector_dictionary(dimensions):
    """
    Initialize vector dictionary
    """
    vecDict = {}

    for d in range(dimensions):
        vecDict[str(d)] = []

    return vecDict


def openai_clip_model(clipmodel='clip-ViT-B-32'):
    """
    Definition of the Open AI Clip model to use
    """
    try:
        print("Loading OpenAI Clip model:", clipmodel)
        model = SentenceTransformer(clipmodel)
        print("Done")

    except:
        print("[Error] Cannot load Open AI CLip model", clipmodel)

    return model


def optimal_number_of_clusters(wcss):
    """
    Get optimal Number of Clusters
    """
    x1, y1 = 2, wcss[0]
    x2, y2 = 20, wcss[len(wcss) - 1]
    distances = []

    for i in range(len(wcss)):
        x0 = i + 2
        y0 = wcss[i]
        numerator = abs((y2 - y1) * x0 - (x2 - x1) * y0 + x2 * y1 - y2 * x1)
        denominator = math.sqrt((y2 - y1)**2 + (x2 - x1)**2)
        distances.append(numerator / denominator)

    return distances.index(max(distances)) + 2


def string_to_base64(content):
    """
    String to Base 64
    """
    return str(base64.b64encode(content.encode('ascii'))).replace("b'",
                                                                  "").replace(
                                                                      "'", "")


def text_embedding(query, model):
    """
    Encoding text using sentence transformers OpenAI Clip model
    """
    img_emb = model.encode(query)

    return np.array(img_emb)

