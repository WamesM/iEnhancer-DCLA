import sys
sys.path.append( 'D:/pycharm_pro/My-Enhancer-classification/' )
import pandas as pd
import os
import numpy as np
from multi_k_model import MultiKModel
import matplotlib.pyplot as plt

# workDir = 'D:/Programming/python/PycharmProjects/ProteinDNABinding/'
# workDir = '/ifs/gdata2/wuhui/ProteinDNABinding/'
workDir = 'D:/pycharm_pro/My-Enhancer-classification/embedding/'
w2vrawDir = workDir + 'dna2vec-20161219-0153-k3to8-100d-10c-29320Mbp-sliding-Xat.w2v'
threemerDir = workDir + '4mer.txt'
threeDir = workDir + 'embedding_matrix4.npy'


def make_DNA2Vec_dict():
    DNA2Vec = {}
    all_data = pd.read_csv(threemerDir, header=None, sep='\t')
    model = MultiKModel(w2vrawDir)
    for mer3 in all_data[0]:
        DNA2Vec[mer3] = model.vector(mer3)
        # print(DNA2Vec[mer3])
    vec = list(DNA2Vec.values())
    vec1 = np.array(vec)
    embedding_matrix4 = np.insert(vec1, 0, 0, axis=0)
    np.save(threeDir, embedding_matrix4)


def make3mers(save_pth):
    if(os.path.exists(save_pth)):
        print(save_pth + " exsits")
        return
    ls = []
    base_list = ['A','C','G','T']
    for base1 in base_list:
        for base2 in base_list:
            for base3 in base_list:
                for base4 in base_list:
                                    ls.append(base1+base2+base3+base4)
    with open(save_pth,'w') as f:
        f.writelines(line+'\n' for line in ls[:-1])
        f.write(ls[-1])

if __name__ == "__main__":
    make3mers(threemerDir)
    make_DNA2Vec_dict()
