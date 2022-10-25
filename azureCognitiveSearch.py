# Azure Cognitive Search python functions
# 20-Oct-2022


# Multiple imports
import os
import configparser
import requests
import json
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import time
import vec2Text

import myfunctions as my

from azure.core.credentials import AzureKeyCredential
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents import SearchClient
from azure.search.documents.indexes.models import (
    ComplexField,
    CorsOptions,
    SearchIndex,
    ScoringProfile,
    SearchFieldDataType,
    SimpleField,
    SearchableField,
)
from IPython.display import Image


# Azure Cognitive Search connection
config_file = 'azureservices.py'

# Azure Cognitive Search Index name
index_name = "demo-retail"

# Azure Cognitive Search credentials
config = configparser.ConfigParser()
config.read(config_file)
acs_key = config.get('AzureCognitiveSearch', 'key')
acs_endpoint = config.get('AzureCognitiveSearch', 'endpoint')
servicename = config.get('AzureCognitiveSearch', 'servicename')


# Azure Cognitive Search admin and search clients
adminClient = SearchIndexClient(endpoint=acs_endpoint,
                                index_name=index_name,
                                credential=AzureKeyCredential(acs_key))

searchClient = SearchClient(endpoint=acs_endpoint,
                            index_name=index_name,
                            credential=AzureKeyCredential(acs_key))

# User functions for Azure Cognitive Search


def create_index():
    """
    Creating a new index
    """
    name = index_name
    fields = [
        SimpleField(name="Id", type=SearchFieldDataType.String, key=True),
        SearchableField(name="Content",
                        type=SearchFieldDataType.String,
                        facetable=False,
                        filterable=True,
                        sortable=True,
                        analyzer_name="en.microsoft"),
        SearchableField(name="VecText",
                        type=SearchFieldDataType.String,
                        facetable=False,
                        filterable=False,
                        sortable=False),
        SearchableField(name="Ocrtext",
                        type=SearchFieldDataType.String,
                        facetable=False,
                        filterable=True,
                        sortable=True),
        SearchableField(name="Barcodetext",
                        type=SearchFieldDataType.String,
                        facetable=False,
                        filterable=True,
                        sortable=True),
        SearchableField(name="Colors",
                        type=SearchFieldDataType.String,
                        facetable=False,
                        filterable=True,
                        sortable=True),
        SearchableField(name="Tags",
                        type=SearchFieldDataType.String,
                        facetable=False,
                        filterable=True,
                        sortable=True),
        SearchableField(name="Caption",
                        type=SearchFieldDataType.String,
                        facetable=False,
                        filterable=True,
                        sortable=True),
    ]

    cors_options = CorsOptions(allowed_origins=["*"], max_age_in_seconds=60)
    index = SearchIndex(name=name, fields=fields, cors_options=cors_options)

    try:
        print("Creating new index", index_name)
        result = adminClient.create_index(index)
        print(result.name, "index has been created")

    except Exception as ex:
        print(ex)


def delete_index():
    """
    Delete any existing Azure cognitive search index
    """
    try:
        print("Deleting index", index_name)
        result = adminClient.delete_index(index_name)
        print(index_name, 'index has been deleted')

    except Exception as ex:
        print(ex)


def execute_file_search(imagefile, centers, model, topn=10):
    """
    File search
    """
    return searchClient.search(search_text=vec_image_to_text(
        imagefile, centers, model),
                               include_total_count=True,
                               select='Content',
                               top=topn)


def execute_query_search(textquery, centers, model, topn):
    """
    Query search
    """
    return searchClient.search(search_text=vec_query_to_text(
        textquery, centers, model),
                               include_total_count=True,
                               select='Content',
                               top=topn)


def index_status(index_name):
    """
    Azure cognitive search Index status
    """
    print("Azure Cognitive Search Index:", index_name, "\n")
    headers = {'Content-Type': 'application/json', 'api-key': acs_key}
    params = {'api-version': '2020-06-30'}
    indexstatus = requests.get(acs_endpoint + "/indexes/" + index_name,
                               headers=headers,
                               params=params)
    print(json.dumps((indexstatus.json()), indent=4))


def global_search(imagefile, centers, model, topn=5):
    """
    Global Search using vecText, OCR and BarCode with Azure Cognitive Search
    """
    start = time.time()

    if not os.path.exists(imagefile):
        print("[Error] File does not exist:", imagefile)
        return

    global results_list

    # 1 Images search using sentence transformers
    print(
        "\033[1;31;32m[Step 1. Running images similarity visual search...]\033[0m\n"
    )
    vectext_list, scores_list = similar_images(
        imagefile, centers, model, False,
        topn=topn)  # Sentence Transformers only
    final_list = vectext_list

    # 2 Get OCR text
    print(
        "\n\033[1;31;32m[Step 2. Extracting any text from the image...]\033[0m\n"
    )
    # Getting results from Azure Read API
    ocr_text = my.get_ocr_text(imagefile)

    if ocr_text != '':
        print("\033[1;31;34m[Result] Text has been found:")
        print("OCR results from the image:", ocr_text)
        print("\033[0m")
        ocr_list = open_text_query(
            ocr_text, False,
            topn)  # Searching using the Azure Read API results

    if ocr_text == '':
        print("\033[1;31;91m[Result] No text has been found\033[0m")
        ocr_list = []

    # 3. Get Barcode OCR
    print(
        "\n\033[1;31;32m[Step 3. Extracting any text any barcode/QRCode of the image...]\033[0m\n"
    )
    barcode_txt = my.get_barcode_text(
        imagefile)  # Get Barcode/QRCode text informations

    if barcode_txt != '':
        print("\033[1;31;34m[Result] Barcode/QRCode text has been found:")
        print("Barcode results from the image:", barcode_txt)
        print("\033[0m")
        barcode_list = open_text_query(
            barcode_txt, False,
            topn)  # Searching using the Barcode/QRCode results

    if barcode_txt == '':
        print("\033[1;31;91m[Result] No barcode/QRCode has been found\033[0m")
        barcode_list = []

    results_list = vectext_list + ocr_list + barcode_list

    idx = 0
    txt_to_remove = ['[', ']']
    clean_list = list()

    # Need to remove some extra characters
    while idx < len(results_list):
        results_list[idx] = "".join(i for i in results_list[idx]
                                    if i not in txt_to_remove)
        clean_list.append(results_list[idx])
        idx += 1

    results_list = list()

    for item in clean_list:
        if item not in results_list:
            results_list.append(item)

    del clean_list

    print("\nDone in", round((time.time() - start), 5), "sec")

    return results_list, vectext_list, ocr_list, barcode_list


def global_search_img_list(filelist):
    """
    View all the images from a list of images files
    """
    try:
        idx = 0
        while idx < len(filelist):
            img = mpimg.imread(filelist[idx])
            plt.figure()
            plt.title("Image " + str(idx + 1) + " : " + filelist[idx])
            plt.imshow(img)
            idx += 1

    except:
        print("[Error] Cannot do a global search")


def open_text_query(mytext, view=False, maxlimit=10):
    """
    Using text query on the Azure Cognitive Search index
    """
    results = searchClient.search(search_text=mytext)

    idx = 1
    imageslist = []

    print('\033[1;31;34m')
    print("Search using query =", mytext, "- note: displaying only the first",
          maxlimit, "results", "\n")

    for result in results:
        print('\033[1;31;34m', idx, "Image file:", result['Content'],
              '\033[0m')

        if view:
            display(Image(filename=result["Content"], height=256, width=256))

        print("\033[1;31;34m")
        print("OCR:", "\033[1;31;32m", result['Ocrtext'], "\033[1;31;34m")
        print("BarCode:", "\033[1;31;32m", result['Barcodetext'],
              "\033[1;31;34m")
        print("Colors:", "\033[1;31;32m", result['Colors'], "\033[1;31;34m")
        print("Tags:", "\033[1;31;32m", result['Tags'], "\033[1;31;34m")
        print("Caption:", "\033[1;31;32m", result['Caption'], "\033[0m", "\n")

        if idx <= maxlimit:
            imageslist.append(result["Content"])

        idx += 1

    return imageslist


def sentence_transformers_query_search(textquery, centers, model, topn=5):
    """
    Sentence transformers text query using Open AI clip model
    """
    idx = 1
    print("Finding", topn, "images with Open AI text query:",
          str.upper(textquery), "\n")

    results = execute_query_search(textquery, centers, model, topn)

    for result in results:
        print(result)
        print("\033[1;31;34m")
        print(idx, "Image file:", result["Content"])
        print("Similarity score =", result['@search.score'])
        display(Image(filename=result["Content"], height=256, width=256))
        print()

        idx += 1


def similar_images(imagefile, centers, model, view, topn):
    """
    Finding similar images using Azure Cognitive Search index
    """
    if not os.path.exists(imagefile):
        print("Error. File not exist:", imagefile)
        return

    images_list = list()
    scores_list = list()

    results = execute_file_search(imagefile, centers, model, topn)

    print("Finding", topn, "similar images of reference image:", imagefile,
          "\n")
    idx = 1

    for result in results:
        imgfile = result["Content"]
        score = result["@search.score"]
        print("\033[1;31;34m", idx, "Similar image:", imgfile,
              "Similarity score =", score, "\033[0m")

        if view:
            display(Image(filename=imgfile, width=256, height=256))
            print("\n")

        images_list.append(imgfile)
        scores_list.append(score)

        idx += 1

    return images_list, scores_list


def upload_documents(documents):
    """
    Uploading documents into an Azure cognitive search index
    """
    try:
        result = searchClient.upload_documents(documents=documents)
        print("Uploading new document...")
        print("Upload of new document succeeded: {}".format(
            result[0].succeeded))
        print("Done\n")

    except Exception as ex:
        print("[Error] Cannot load the documents into the index")
        print(ex.message)


def vec_image_to_text(imagefile, centers, model):
    """
    Doing vec2Text
    """
    curVec = vec2Text.image_embedding(imagefile, model)
    vecText = ''

    for d in range(len(curVec)):
        vecText += vec2Text.convert_field_num_to_string(d) + str(
            vec2Text.closest(centers[d], curVec[d])) + ' '

    return vecText


def vec_query_to_text(query, centers, model):
    """
    Vec query to text
    """
    curVec = vec2Text.text_embedding(query, model)
    vecText = ''

    for d in range(len(curVec)):
        vecText += vec2Text.convert_field_num_to_string(d) + str(
            vec2Text.closest(centers[d], curVec[d])) + ' '
    return vecText

