import pyreadr
import pandas
import numpy as np
import torch


def get_lsa_dict(lsa_file):

    result = pyreadr.read_r(lsa_file)
    words = result['TASA'].index
    print(words)
    data = result['TASA'].to_numpy()
    # print(data)
    # print(data.shape)

    print(f"num_words: {data.shape[0]}")

    lsa = {}

    for i, word in enumerate(words):
        lsa[word] = data[i]

    return lsa


    

# if __name__ == "__main__":

#     lsa_file = 'data/LSA/TASA.rda'
#     lsa = get_lsa_dict(lsa_file)
#     # words = result['TASA'].index
#     # print(words)
#     # data = result['TASA'].to_numpy()
#     # # print(data)
#     # # print(data.shape)

#     # print(f"num_words: {data.shape[0]}")

#     # lsa = {}

#     # for i, word in enumerate(words):
#     #     lsa[word] = torch.tensor(data[i], dtype = torch.float)

#     word_list = ['prince', 'gold', 'sky', 'love', 'go', 'walk']

#     for word in word_list:
#         print(f"{word}\n{lsa[word]}")

#     # sentence_rep = get_lsa_sentence_rep(
#     #     word_list = word_list,
#     #     lsa_dict = lsa,
#     # )
#     # print(sentence_rep)
#     # print(sentence_rep.size())

#     # print(result['TASA'].head())

#     # print(data.keys())
#     # print(data['V3'])
#     # # print(data['the'])

#     # for word in word_list:
#     #     print(f"{word} \n {result[word]}")