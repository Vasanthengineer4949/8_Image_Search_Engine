from socket import timeout
import urllib.request as request
from transformers import CLIPProcessor, CLIPModel
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES=True
import config
from html import escape

processor = CLIPProcessor.from_pretrained(config.MODEL_CKPT)
model = CLIPModel.from_pretrained(config.MODEL_CKPT)

def retrive_imag_from_url(url_filename):
  url, filename = url_filename
  request.urlretrieve(url, filename)

def load_image(path, same_height=False):
    im = Image.open(path)
    if im.mode != 'RGB':
        im = im.convert('RGB')
    if same_height:
        ratio = 224/im.size[1]
        return im.resize((int(im.size[0]*ratio), int(im.size[1]*ratio)))    
    else:
        ratio = 224/min(im.size)
    return im.resize((int(im.size[0]*ratio), int(im.size[1]*ratio)))

def image_embeddings_computation(images_list):
    return model.get_image_features(**processor(images=images_list, return_tensors="pt", padding=True))

def get_image_from_html(url_list, height=200):
    html = "<div style='margin-top: 20px; max-width: 1200px; display: flex; flex-wrap: wrap; justify-content: space-evenly'>"
    for url, title, link in url_list:
        html2 = f"<img title='{escape(title)}' style='height: {height}px; margin: 5px' src='{escape(url)}'>"
        if len(link) > 0:
            html2 = f"<a href='{escape(link)}' target='_blank'>" + html2 + "</a>"
        html = html + html2
    html += "</div>"
    return html

def text_embeddings_computation(list_of_strings):
    inputs = processor(text=list_of_strings, return_tensors="pt", padding=True)
    return model.get_text_features(**inputs)
