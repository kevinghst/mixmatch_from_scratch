import urllib.request
import zipfile
import os
import tarfile
from os import path
import requests

# Helper methods

def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    save_response_content(response, destination)

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None

def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)


# Download SST-2

task = "SST"
task_path = "https://firebasestorage.googleapis.com/v0/b/mtl-sentence-representations.appspot.com/o/data%2FSST-2.zip?alt=media&token=aabc5f6b-e466-44a2-b9b4-cf6337f84ac8"
data_dir = './'
if not path.exists('./SST-2'):
    data_file = "%s.zip" % task
    urllib.request.urlretrieve(task_path, data_file)
    with zipfile.ZipFile(data_file) as zip_ref:
        zip_ref.extractall(data_dir)
    os.remove(data_file)

# Download DBPedia

data_file = 'dbpedia.tar.gz'
file_id = '0Bz8a_Dbh9QhbQ2Vic1kxMmZZQ1k'
destination = './' + data_file

if not path.exists("./dbpedia"):
	download_file_from_google_drive(file_id, destination)

	tar = tarfile.open(data_file, "r:gz")
	tar.extractall()
	tar.close()
	os.remove(data_file)

	os.rename('./dbpedia_csv', './dbpedia')

# Process IMDB
if not path.exists('./imdb'):
    with zipfile.ZipFile("imdb.zip") as zip_ref:
        zip_ref.extractall(data_dir)


# Download CoLA
task = "CoLA"
task_path = "https://firebasestorage.googleapis.com/v0/b/mtl-sentence-representations.appspot.com/o/data%2FCoLA.zip?alt=media&token=46d5e637-3411-4188-bc44-5809b5bfb5f4"
data_dir = './'
if not path.exists('./CoLA'):
    data_file = "%s.zip" % task
    urllib.request.urlretrieve(task_path, data_file)
    with zipfile.ZipFile(data_file) as zip_ref:
        zip_ref.extractall(data_dir)
    os.remove(data_file)

# Download RTE
task = "RTE"
task_path = "https://firebasestorage.googleapis.com/v0/b/mtl-sentence-representations.appspot.com/o/data%2FRTE.zip?alt=media&token=5efa7e85-a0bb-4f19-8ea2-9e1840f077fb"
data_dir = './'
if not path.exists('./RTE'):
    data_file = "%s.zip" % task
    urllib.request.urlretrieve(task_path, data_file)
    with zipfile.ZipFile(data_file) as zip_ref:
        zip_ref.extractall(data_dir)
    os.remove(data_file)

task = "MNLI"
task_path = "https://firebasestorage.googleapis.com/v0/b/mtl-sentence-representations.appspot.com/o/data%2FMNLI.zip?alt=media&token=50329ea1-e339-40e2-809c-10c40afff3ce"
data_dir = './'
if not path.exists('./MNLI'):
    data_file = "%s.zip" % task
    urllib.request.urlretrieve(task_path, data_file)
    with zipfile.ZipFile(data_file) as zip_ref:
        zip_ref.extractall(data_dir)
    os.remove(data_file)

# Download AG News
data_file = 'ag_news_csv.tgz'
destination = './agnews/' + data_file

if not path.exists("./agnews"):
    urllib.request.urlretrieve("https://s3.amazonaws.com/fast-ai-nlp/ag_news_csv.tgz", data_file)
    with tarfile.open(data_file) as tar_ref:
        def is_within_directory(directory, target):
            
            abs_directory = os.path.abspath(directory)
            abs_target = os.path.abspath(target)
        
            prefix = os.path.commonprefix([abs_directory, abs_target])
            
            return prefix == abs_directory
        
        def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
        
            for member in tar.getmembers():
                member_path = os.path.join(path, member.name)
                if not is_within_directory(path, member_path):
                    raise Exception("Attempted Path Traversal in Tar File")
        
            tar.extractall(path, members, numeric_owner=numeric_owner) 
            
        
        safe_extract(tar_ref, ".")
    os.remove(data_file)
    os.rename('ag_news_csv', 'agnews')

# Downalod BOOLQ
task = "BoolQ"
task_path = "https://dl.fbaipublicfiles.com/glue/superglue/data/v2/BoolQ.zip"
data_dir = './'
if not path.exists('./BoolQ'):
    data_file = "%s.zip" % task
    urllib.request.urlretrieve(task_path, data_file)
    with zipfile.ZipFile(data_file) as zip_ref:
        zip_ref.extractall(data_dir)
    os.remove(data_file)

# Downalod multirc
task = "multirc"
task_path = "https://dl.fbaipublicfiles.com/glue/superglue/data/v2/MultiRC.zip"
data_dir = './'
if not path.exists('./multirc'):
    data_file = "%s.zip" % task
    urllib.request.urlretrieve(task_path, data_file)
    with zipfile.ZipFile(data_file) as zip_ref:
        zip_ref.extractall(data_dir)
    os.remove(data_file)
