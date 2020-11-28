# import argparse
# import numpy as np
# import pandas as pd
# from transformers import BertModel, BertTokenizer
# import logging
# import torch
# import pdb
# import os
# from tqdm import tqdm
# import sent2vec
# import nltk
#
# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')
# nltk.download('stopwords')
# from nltk import word_tokenize
# from nltk.corpus import stopwords
# from string import punctuation
#
#
# # Define all the input arguments needed to extract caption embeddings
# def parser():
#     parser = argparse.ArgumentParser(description="Get word embeddings for the translated captions")
#     parser.add_argument('--no-cuda', action='store_true', default=False,
#                         help='disables CUDA training')
#     parser.add_argument("--mGPUs", action='store_true', default=False,
#                         help='flag for multi GPU')
#     parser.add_argument('--dir_name', type=str, default="ppm_data",
#                         help='folder containing the captions data')
#     parser.add_argument('--model', type=str, default="bert",
#                         help='specify which pre-trained model to use (currently BERT and sent2vec is supported')
#     return parser
#
#
# # Sent2Vec Model to get embedddings
# class sentvec_model:
#     def __init__(self):
#         # Load pre-trained sent2vec model
#         self.sentvec_model = sent2vec.Sent2vecModel()
#         print('Loading.... : SentVec')
#         try:
#             self.sentvec_model.load_model("wiki_unigrams.bin")
#         except Exception as e:
#             print(e)
#         print('SentVec Model Successfully Loaded')
#
#     def preprocess_sentence(self, text):
#         """
#         Function performs pre-processing of the input text (caption)
#         :param text: the input text (single caption)
#         :return: pre-processed text
#         """
#         stop_words = set(stopwords.words('english'))
#         text = text.replace('/', ' / ')
#         text = text.replace('.-', ' .- ')
#         text = text.replace('.', ' . ')
#         text = text.replace('\'', ' \' ')
#         text = text.lower()
#
#         tokens = [token for token in word_tokenize(text) if token not in punctuation and token not in stop_words]
#         return ' '.join(tokens)
#
#     def sentence_encoding(self, text):
#         """
#         Get the sent2vec embedding for the caption
#         :param text: the input text (single caption)
#         :return: embedding of dim 600 for a single caption
#         """
#         return self.sentvec_model.embed_sentence(self.preprocess_sentence(text)).reshape(-1)
#
#
# # BERT model to get caption embeddings
# class Model:
#     def __init__(self, device):
#         """
#         Load the pretrained BERT model
#         :param device: set to "cpu" or "gpu"
#         """
#         # Load pre-trained model tokenizer (vocabulary)
#         self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
#         # load the pretrained model
#         self.model = BertModel.from_pretrained('bert-base-uncased')
#         self.device = device
#         self.model.to(self.device)
#         # Run in evaluation mode
#         self.model.eval()
#         # Parse only these required tags from the tokens
#         self.req_tags = ['NN', 'NNP', 'NNS', 'NNPS', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'JJ', 'JJR', 'JJS']
#
#     def sentence_encoding(self, text):
#         """
#         Get the encoding (embeddings with BERT) for each caption
#         :param text: input caption
#         :return: embeddings of dim 748
#         """
#         # Add the special tokens.
#         marked_text = "[CLS] " + str(text) + " [SEP]"
#
#         # tokenized_text_with_pos = nltk.tag.pos_tag(self.tokenizer.tokenize(str(text)))
#         # Tokenize the caption and take only required POS tags (like nouns, verbs and adjectives present in the caption)
#         tokenized_text_with_pos = nltk.tag.pos_tag(nltk.word_tokenize(str(text).lower()))
#         tokenized_text_with_req_pos = [i[0] for i in tokenized_text_with_pos if i[1] in self.req_tags]
#
#         marked_text = "[CLS] " + " ".join(tokenized_text_with_req_pos) + "[SEP]"
#         # Split the sentence into tokens.
#         tokenized_text = self.tokenizer.tokenize(marked_text)
#
#         # Map the token strings to their vocabulary indices.
#         indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_text)
#         if len(indexed_tokens) > 512:
#             indexed_tokens = indexed_tokens[:512]
#         segments_ids = [1] * len(indexed_tokens)
#         # Convert inputs to PyTorch tensors
#         tokens_tensor = torch.tensor([indexed_tokens])
#         segments_tensors = torch.tensor([segments_ids])
#         with torch.no_grad():
#             outputs = self.model(tokens_tensor.to(self.device), segments_tensors.to(self.device))
#             return torch.mean(outputs[0], axis=1).cpu().numpy().reshape(-1)
#
#
# class processData:
#     def __init__(self, data_dir):
#         """
#         Class that reads the data from all the csv that have image_paths and captions
#         :param data_dir: folder that contains the CSVs
#         """
#         self.data_dir = data_dir
#         self.files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".csv")]
#         self.filename = ""
#
#     def read_csv(self, file):
#         """
#         Read the CSV
#         :param file: each PPM file
#         :return: image_path and english captions from the file
#         """
#         self.filename = file
#         print(self.filename)
#         self.data = pd.read_csv(file, error_bad_lines=False)
#         for col in self.data.columns:
#             self.data = self.data[col.notna()]
#         self.data = self.data[self.data['label'].apply(lambda x: type(x) in [int, np.int64, float, np.float64])]
#
#         abstract1 = self.data['paperAbstract'].tolist()
#         abstract2 = self.data['paperAbstract2'].tolist()
#         label = self.data['label'].tolist()
#         return abstract1, abstract2, label
#
#
# if __name__ == '__main__':
#     args = parser().parse_args()
#
#     # Logging basic info
#     logging.basicConfig(level=logging.INFO)
#
#     # Check if CUDA is available
#     use_cuda = not args.no_cuda and torch.cuda.is_available()
#     device = torch.device("cuda" if use_cuda else "cpu")
#
#     if args.model == 'bert':
#         model = Model(device)
#     else:
#         model = sentvec_model()
#     data = processData(args.dir_name)
#     for file in data.files:
#         res = []
#         abstract1, abstract2, label = data.read_csv(file)
#         filename = str(file.split("_")[0])
#         for cap, im_path in zip(captions, image_path):
#             if not im_path.startswith("page"):
#                 continue
#             embedding = model.sentence_encoding(cap)
#             res.append((filename.split("-")[0].split("/")[-1] + "_" + str(im_path), list(embedding)))
#         # Save the image_path and caption embedddings as numpy dump
#         np.save(filename + '_embeddings_filtered_sent2vec.npy', np.asarray(res), allow_pickle=True)
