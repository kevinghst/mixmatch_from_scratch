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
    print("\tCompleted!")

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