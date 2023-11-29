import torch
from torch.utils.data import TensorDataset
# from sentence_transformers import SentenceTransformer, SentencesDataset
import gensim.downloader as api
import re


def get_sentence_transformer_dataset(
        dataFolder,
        mode,
        taxonomy,
        model
):

    dataFile = dataFolder + "/" + taxonomy + "/" + mode + ".tsv"
    if taxonomy == "original":
        num_classes = 28
    elif taxonomy == "ekman":
        num_classes = 7
    elif taxonomy == "group":
        num_classes = 4
    else:
        "Invalid taxonomy!"
        quit()

    with open(dataFile, 'r') as f:
        data = f.read().split("\n")
    
    num_texts = len(data)
    texts = []
    emotions = torch.zeros(num_texts, num_classes, dtype = torch.float)

    for rowID, row in enumerate(data):
        if row != "":
            print(row)
            text, sen_emotions, _ = row.split("\t")
            print(f"text: {text}\nsen_emotions {sen_emotions}")
            texts.append(text)
            for emotion in sen_emotions.split(","):
                emotions[rowID, int(emotion)] = 1
    
    text_embeddings = model.encode(texts, return_tensors = 'pt')
    print(f"text embeddings size: {text_embeddings.size()}")

    dataset = TensorDataset(text_embeddings, emotions)

    return dataset



def get_goem_dataset(
        args,
        mode,
        model
):

    dataFile = args.data_dir + "/" + mode + ".tsv"
    if args.data_dir == "data/original":
        num_classes = 28
    elif args.data_dir == "data/ekman":
        num_classes = 7
    elif args.data_dir == "data/group":
        num_classes = 4
    else:
        "Invalid taxonomy!"
        quit()

    with open(dataFile, 'r') as f:
        data = f.read().split("\n")
    
    num_texts = len(data)
    if args.feats_type == 'sen_trans':
        feat_dim = len(model.encode('hello'))
    else:
        feat_dim = len(model['hello'])
    texts = []
    word_embeddings = []
    sentence_embeddings = torch.zeros(num_texts, feat_dim, dtype = torch.float)
    emotions = torch.zeros(num_texts, num_classes, dtype = torch.float)
    sentences = []
    rowID = 0

    for row in data:
        if row != "":
            text, sen_emotions, _ = row.split("\t")
            text = (re.sub(r"[^\w\s]", '', text))
            texts.append(text.split(" "))
            if args.feats_type == 'sen_trans':
                sentences.append(text)
                # np_text = model.encode(text)
                # sentence_embeddings[rowID] = torch.from_numpy(np_text)
            else:
                text_embedding = [torch.tensor(model[word], dtype = torch.float) for word in text.split(" ") if word in model]
                if len(text_embedding) > 1:
                    text_embedding = torch.stack(text_embedding)
                else:
                    text_embedding = torch.zeros(1, feat_dim, dtype = torch.float)
                word_embeddings.append(text_embedding)
                sentence_embeddings[rowID] = text_embedding.mean(dim = 0)
            # words = [word for word in text.split(" ") if word in model]
            texts[rowID]
            for emotion in sen_emotions.split(","):
                emotions[rowID, int(emotion)] = 1
            rowID += 1
    emotions = emotions[0:rowID]
    sentence_embeddings = sentence_embeddings[0:rowID]
        

    if args.feats_type == 'sen_trans':
        sentence_embeddings = model.encode(sentences)
        sentence_embeddings = torch.from_numpy(sentence_embeddings)
        print(f"sentence_embeddings size: {sentence_embeddings.size()}")
        print(f"emotions size: {emotions.size()}")
        print(f"emotions[-1]: {emotions[-1]}")
    
    print(f"Number of texts: {num_texts}, sentence embeddings size: {sentence_embeddings.size()}, emotions size: {emotions.size()}")

    dataset = TensorDataset(sentence_embeddings, emotions)

    return dataset



def get_lsa_dataset(
        dataFolder,
        mode,
        model
):

    dataFile = dataFolder + "/" + mode + ".tsv"
    if dataFolder == "data/original":
        num_classes = 28
    elif dataFolder == "data/ekman":
        num_classes = 7
    elif dataFolder == "data/group":
        num_classes = 4
    else:
        "Invalid taxonomy!"
        quit()

    with open(dataFile, 'r') as f:
        data = f.read().split("\n")
    
    num_texts = len(data)
    feat_dim = len(model['hello'])
    texts = []
    word_embeddings = []
    sentence_embeddings = torch.zeros(num_texts, feat_dim, dtype = torch.float)
    emotions = torch.zeros(num_texts, num_classes, dtype = torch.float)

    

    for rowID, row in enumerate(data):
        if row != "":
            # print(row)
            text, sen_emotions, _ = row.split("\t")
            text = (re.sub(r"[^\w\s]", '', text))
            # print(f"text: {text}\nsen_emotions {sen_emotions}")
            texts.append(text.split(" "))
            text_embedding = [torch.tensor(model[word], dtype = torch.float) for word in text.split(" ") if word in model]
            if len(text_embedding) > 1:
                text_embedding = torch.stack(text_embedding)
            else:
                text_embedding = torch.zeros(1, feat_dim, dtype = torch.float)
            word_embeddings.append(text_embedding)
            sentence_embeddings[rowID] = text_embedding.mean(dim = 0)
            words = [word for word in text.split(" ") if word in model]
            # print(f"words: {words}")
            # print(text_embedding)
            # print(sentence_embeddings[rowID])
            # print(text_embedding.size())
            # print(sentence_embeddings[rowID].size())
            texts[rowID]
            for emotion in sen_emotions.split(","):
                emotions[rowID, int(emotion)] = 1
    
    # print(texts[0:10])
    # print(word_embeddings[0:10])
    print(f"Number of texts: {num_texts}, sentence embeddings size: {sentence_embeddings.size()}, emotions size: {emotions.size()}")

    dataset = TensorDataset(sentence_embeddings, emotions)

    return dataset



if __name__ == "__main__":

    mode = "dev"
    dataFolder = "data/ekman"

    model = api.load("word2vec-google-news-300")
    model.save
    # model = api.load("word2vec-google-news-300")

    dataset = get_goem_dataset(
        dataFolder = dataFolder,
        mode = mode,
        model = model
    )

    for i in range(10):
        sentence_embeddings, emotion = dataset[i]
        print(f"{i} \n {sentence_embeddings} \n {emotion}")

