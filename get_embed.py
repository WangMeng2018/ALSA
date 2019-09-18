import argparse
import os
import numpy as np
import pandas as pd
import pickle
import jieba

jieba.setLogLevel(20)


def set_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data_path', default='data\\train.csv')
    parser.add_argument('--test_data_path', default='data\\test_public.csv')
    parser.add_argument('--embed_data_path',
                        default='data\\Tencent_AILab_ChineseEmbedding.txt')
    parser.add_argument('--out_embed_dir', default='output')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = set_args()

    # load data, get vocab
    print('start load data.....')
    train_contents = pd.read_csv(args['train_data_path'])['content'].tolist()
    test_contents = pd.read_csv(args['test_data_path'])['content'].tolist()
    contents = train_contents + test_contents
    wordset = set()
    for content in contents:
        wordset.update(jieba.lcut_for_search(content) + list(content))
    print('origin data vocab length = ' + str(len(wordset)))  # 25095

    # get embedding
    print('start process embedding.....')
    word_index = {}         # key：word，value：wordID
    embedding_matrix = []   
    with open(args['embed_data_path']) as f:
        next(f)
        i = 0
        for line in f:
            e = line[:-1].split(' ')
            w = e[0]
            if w in wordset:
                word_index[w] = i
                i += 1
                embedding_matrix.append(np.array(e[1:], dtype=float))
    embedding_matrix = np.array(embedding_matrix)
    embeddings = [word_index, embedding_matrix]
    print('final data vocab length = ' + str(len(word_index)))   # 22815
    print(embedding_matrix.shape)   # (22815,200)

    out_embed_path = os.path.join(args['out_embed_dir'], 'embeddings_.p')
    pickle.dump(embeddings, open(out_embed_path, 'wb'))
