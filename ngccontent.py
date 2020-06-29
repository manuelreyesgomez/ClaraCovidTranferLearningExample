import os
import sys
import urllib.request
from tqdm import tqdm
from itertools import chain
import zipfile
import json

def downloadurltofile(url,filename):
    if not os.path.exists(filename):
        print(f'--> Downloading {filename} <--'.center(80, '#'))
        with open(filename, 'wb') as file:
            with urllib.request.urlopen(url) as resp:
                length = int(resp.getheader('content-length'))
                blocksize = max(4096, length // 100)
                with tqdm(total=length, file=sys.stdout) as pbar:
                    while True:
                        buff = resp.read(blocksize)
                        if not buff:
                            break
                        file.write(buff)
                        pbar.update(len(buff))
        print(' Download complete '.center(80,'#'))
    else:
        print(f'-->> {filename} file already exists locally <<--'.center(80, '#'))
    print()

def download(url,targetfolder,targetfile):
    path = os.getcwd()
    data_dir = os.path.abspath(os.path.join(path, targetfolder))

    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    
    targetfile = os.path.join(data_dir, targetfile)
    downloadurltofile(url,targetfile)
    return data_dir,targetfile

def unzippedfile(folder,file):
    with zipfile.ZipFile(file, 'r') as zip_ref:
        zip_ref.extractall(folder)

def upload_data(workspace, datastore, src_dir, tgt_path):
    datastore.upload(
        src_dir=src_dir,
        target_path=tgt_path,
        show_progress=True)
    print(' Upload complete '.center(80,'#'))

def get_config(configfile):
    jsonfile = open(configfile)
    configdata = json.load(jsonfile)
    return configdata
