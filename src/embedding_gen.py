import config
import os
import time
import numpy as np
import pandas as pd
from transformers import CLIPProcessor, CLIPModel
from multiprocessing.dummy import Pool
from PIL import Image, ImageFile
from utils import load_image
from utils import retrive_imag_from_url
from utils import image_embeddings_computation
ImageFile.LOAD_TRUNCATED_IMAGES = True

processor = CLIPProcessor.from_pretrained(config.MODEL_CKPT)
model = CLIPModel.from_pretrained(config.MODEL_CKPT)


df = pd.read_csv(config.DATA_PATH)
length = len(df)
try:
    image_embeddings = np.load(config.EMBEDDINGS_PATH)
    i = image_embeddings.shape[0]
    print(f"Loaded {i} embeddings from file")
except FileNotFoundError:
    image_embeddings, i = None, 0

while i < length:
    for f in os.listdir():
        if '.jpeg' in f:
            os.remove(f)

    n_parallel = min(config.MAX_N_PARALLEL_PROCESSES, length - i)
    url_filename_list = [(df.iloc[i + j]['path'], str(i + j) + '.jpeg') for j in range(n_parallel)]
    _ = Pool(n_parallel).map(retrive_imag_from_url, url_filename_list)
    batch_embeddings = image_embeddings_computation([load_image(str(i + j) + '.jpeg') for j in range(n_parallel)]).detach().numpy()

    if image_embeddings is None:
        image_embeddings = batch_embeddings
    else:
        image_embeddings = np.vstack((image_embeddings, batch_embeddings))

    i = image_embeddings.shape[0]
    time.sleep(config.DURATION_BN_PROCESSES)

    if i % 100 == 0:
        np.save(config.EMBEDDINGS_PATH, image_embeddings)
        print(i)