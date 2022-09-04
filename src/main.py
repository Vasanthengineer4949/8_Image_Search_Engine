import config
import streamlit as st
import pandas as pd, numpy as np
import os
from utils import get_image_from_html, text_embeddings_computation
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
from transformers import CLIPProcessor, CLIPTextModel, CLIPModel

@st.cache(show_spinner=False,
          hash_funcs={CLIPModel: lambda _: None,
                      CLIPTextModel: lambda _: None,
                      CLIPProcessor: lambda _: None,
                      dict: lambda _: None})
def load():
  model = CLIPModel.from_pretrained(config.MODEL_CKPT)
  processor = CLIPProcessor.from_pretrained(config.MODEL_CKPT)
  df = {0: pd.read_csv(config.DATA_PATH)}
  embeddings = {0: np.load(config.EMBEDDINGS_PATH)}
  for k in [0]:
    embeddings[k] = np.divide(embeddings[k], np.sqrt(np.sum(embeddings[k]**2, axis=1, keepdims=True)))
  return model, processor, df, embeddings
model, processor, df, embeddings = load()

st.cache(show_spinner=False)
def search_image(query, n_results=24):
    text_embeddings = text_embeddings_computation([query]).detach().numpy()
    k = 0
    results = np.argsort((embeddings[k]@text_embeddings.T)[:, 0])[-1:-n_results-1:-1]
    return [(df[k].iloc[i]['path'],
             df[k].iloc[i]['tooltip'],
             df[k].iloc[i]['link']) for i in results]

def main():
  description = '''
  # CLIP image search
  **Enter the kind of image you need and then press Enter after clicking which you need Unsplash image or movie image***
  '''
  st.title("AI enhanced Image Search Engine")
  st.sidebar.markdown(description)
  _, c, _ = st.columns((1, 4, 1))
  query = c.text_input('', value='a beach where there is a beautiful sunset')
  if len(query) > 0:
    results = search_image(query)
    st.markdown(get_image_from_html(results), unsafe_allow_html=True)
if __name__ == '__main__':
  main()