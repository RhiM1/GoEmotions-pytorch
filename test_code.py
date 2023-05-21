# import torch
# from torch.utils.data import TensorDataset
# from sentence_transformers import SentenceTransformer, SentencesDataset


# def get_sentence_transformer_dataset(
#         dataFolder,
#         mode,
#         taxonomy,
#         model
# ):

#     dataFile = dataFolder + "/" + taxonomy + "/" + mode + ".tsv"
#     if taxonomy == "original":
#         num_classes = 28
#     elif taxonomy == "ekman":
#         num_classes = 7
#     elif taxonomy == "group":
#         num_classes = 4
#     else:
#         "Invalid taxonomy!"
#         quit()

#     with open(dataFile, 'r') as f:
#         data = f.read().split("\n")
    
#     num_texts = len(data)
#     texts = []
#     emotions = torch.zeros(num_texts, num_classes, dtype = torch.float)

#     for rowID, row in enumerate(data):
#         if row != "":
#             print(row)
#             text, sen_emotions, _ = row.split("\t")
#             print(f"text: {text}\nsen_emotions {sen_emotions}")
#             texts.append(text)
#             for emotion in sen_emotions.split(","):
#                 emotions[rowID, int(emotion)] = 1
    
#     text_embeddings = model.encode(texts, return_tensors = 'pt')
#     print(f"text embeddings size: {text_embeddings.size()}")

#     dataset = TensorDataset(text_embeddings, emotions)

#     return dataset



# if __name__ == "__main__":

#     mode = "dev"
#     taxonomy = "group"
#     dataFolder = "data"

#     model = SentenceTransformer('all-MiniLM-L6-v2')

#     dataset = get_sentence_transformer_dataset(
#         dataFolder = dataFolder,
#         mode = mode,
#         taxonomy = taxonomy,
#         model = model
#     )

#     for i in range(10):
#         print(dataset[i])
#         print(f"{i} \n {dataset.text_embeddings} \n {dataset.emotions}")

#     # #Our sentences we like to encode
#     # sentences = ['This framework generates embeddings for each input sentence',
#     #     'Sentences are passed as a list of string.',
#     #     'The quick brown fox jumps over the lazy dog.']

#     # #Sentences are encoded by calling model.encode()
#     # embeddings = model.encode(sentences)

#     # #Print the embeddings
#     # for sentence, embedding in zip(sentences, embeddings):
#     #     print("Sentence:", sentence)
#     #     print("Embeddings size:", embedding.size())
#     #     print("Embedding:", embedding)
#     #     print("")


from transformers import AutoTokenizer, AutoModel
import torch


#Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask


device = "cuda" if torch.cuda.is_available() else "cpu"

#Sentences we want sentence embeddings for
sentences = ['This framework generates embeddings for each input sentence',
             'Sentences are passed as a list of string.',
             'The quick brown fox jumps over the lazy dog.']

#Load AutoModel from huggingface model repository
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
tokenizer = tokenizer
model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
model = model.to(device)

#Tokenize sentences
encoded_input = tokenizer(sentences, padding=True, truncation=True, max_length=128, return_tensors='pt')
# print(encoded_input)
encoded_input['input_ids'] = encoded_input['input_ids'].to(device)
encoded_input['token_type_ids'] = encoded_input['token_type_ids'].to(device)
encoded_input['attention_mask'] = encoded_input['attention_mask'].to(device)

#Compute token embeddings
with torch.no_grad():
    model_output = model(**encoded_input)
    
print(f"model_output:\n{model_output}\n")

print(f"attention_mask:\n{encoded_input['attention_mask']}\n")
#Perform pooling. In this case, mean pooling
sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])

# print(sentence_embeddings)
# print(sentence_embeddings.size())