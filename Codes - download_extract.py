import os
import requests
import PIL
from io import BytesIO
from PIL import Image
from tqdm import tqdm
import pandas as pd
tqdm.pandas()
import wget

import argparse

from multiprocessing import  Pool
from functools import partial
import numpy as np



def parallelize(data, func, num_of_processes=8):
    data_split = np.array_split(data, num_of_processes)
    pool = Pool(num_of_processes)
    data = pd.concat(pool.map(func, data_split))
    pool.close()
    pool.join()
    return data

def run_on_subset(func, data_subset):
    return data_subset.progress_apply(func, axis=1)

def parallelize_on_rows(data, func, num_of_processes=8):
    return parallelize(data, partial(run_on_subset, func), num_of_processes)




def download(row):
    try:
#         row = df.iloc[i]
        postId = str(row['id'])
        url = row['url']

        file_name = url.split('/')[-1]
        des_dir = "/home/jupyter/multimodal/post_videos/"
        frame_path = "/home/jupyter/multimodal/post_frames/"+postId
        file_extension = "."+file_name.split(".")[-1]
        video_path = os.path.join(des_dir, str(postId) + file_extension)
        
        
        avoid_stdout = " > /dev/null 2>&1"
#         avoid_stdout = " >> log.txt"
        
        
        wget.download(url, out=des_dir)
        
        
        command = 'mv {} {}'.format(os.path.join(des_dir, file_name),
                                    os.path.join(des_dir, str(postId) + file_extension))+avoid_stdout
#         print(command)
        os.system(command)
        
        
        command = 'mkdir {}'.format(frame_path)+avoid_stdout
#         print(command)
        os.system(command)
        
        
        command = 'rm -r {}'.format(frame_path+"/*")+avoid_stdout
#         print(command)
        os.system(command)
        
        
        
        
        command = "/home/jupyter/git_sc/keyframe-extraction/distribute/bin/hecate -i {video_path} --generate_jpg --njpg 4 --out_dir {frame_path}"
        command = command.format( video_path =  video_path , frame_path=frame_path)+avoid_stdout
#         print(command)
        code = os.system(command)

        
        #use file read operations to get missing post ids
#         if code!=0:
#             print("Failed post id:-", postId)
        os.system("rm "+video_path)

    except Exception as e:
        print(e)
    return row



df = pd.read_pickle("multimodal_video_data.pkl")
print(df.shape)


df = df.drop_duplicates(subset=['url']).reset_index(drop=True)
print(df.shape)


print("Started")
from datetime import datetime
import pytz


tz_India = pytz.timezone('Asia/Kolkata')
datetime_India = datetime.now(tz_India)
print("India time:", datetime_India.strftime("%H:%M:%S"))


df = parallelize_on_rows(df, download, 96)


tz_India = pytz.timezone('Asia/Kolkata')
datetime_India = datetime.now(tz_India)
print("India time:", datetime_India.strftime("%H:%M:%S"))