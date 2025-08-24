import os
import time
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt

os.environ["DATAPATH"] = f"/home/exx/Documents/RNAplay/RNAstructure/data_tables"

class RNAstructure():
    def __init__(self):
        os.system('mkdir tmp')

    def step(self,sequence,return_matrix=False,tmp_file='tmp/matrix_0'):
        #os.environ["CUDA_VISIBLE_DEVICES"] = f'{int(tmp_file.split('x')[1])%2}'
        #print({int(tmp_file.split('x')[1])%2})
        gpu=int(tmp_file.split('_')[1])%2
        os.environ["CUDA_VISIBLE_DEVICES"] = f'{gpu}'
        os.system(f'/home/exx/Documents/RNAplay/RNAstructure/exe/partition-cuda {sequence} -t {tmp_file}')
        #os.system(f'/home/shujun/Documents/Nucleic_Transformer/OpenVaccine/arnie/RNAstructure/exe/partition {sequence} -t {tmp_file}')
        matrix=pd.read_csv(tmp_file,' ',header=None).values
        matrix=matrix[:,:-1]
        ap=matrix.sum(1)
        if return_matrix:
            return ap,matrix
        else:
            del matrix
            return ap
# env=RNAstructure()
# for i in tqdm(range(runs)):
#     ap=env.step('AGCUAGCUAGCUAGCUAGCUAGCUAGCUAGCUAGCU')
