import urllib.request
import zipfile
import os

task = "SST"
task_path = "https://firebasestorage.googleapis.com/v0/b/mtl-sentence-representations.appspot.com/o/data%2FSST-2.zip?alt=media&token=aabc5f6b-e466-44a2-b9b4-cf6337f84ac8"
data_dir = './'
data_file = "%s.zip" % task
urllib.request.urlretrieve(task_path, data_file)
with zipfile.ZipFile(data_file) as zip_ref:
    zip_ref.extractall(data_dir)
os.remove(data_file)
print("\tCompleted!")