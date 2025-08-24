import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
from PIL import Image
import os
from multiprocessing import Pool
import sys

from tqdm import tqdm
from Network import *
from Degradation_Model import Degradation_Model

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

from RiboGraphViz import RGV

sys.path.append('../DegScore')

from DegScore import DegScore


def get_distance_mask(L):

    m=np.zeros((3,L,L))


    for i in range(L):
        for j in range(L):
            for k in range(3):
                if abs(i-j)>0:
                    m[k,i,j]=1/abs(i-j)**(k+1)
    return m


def load_degradation_models(device,path='degradation_model_w_loop',ks=[3,4,5,6]):
    degradation_models=[]
    folds=np.arange(10)
    print("Loading degradation models")
    for k in ks:
        for fold in tqdm(folds):
            MODELS=[]
            for i in range(1):

                model=Degradation_Model(15, 5, 256, 32, 1024,
                                       k, True, kmers=[k],
                                       dropout=0.1).to(device)
                # Initialization
                opt_level = 'O1'
                #model, optimizer = amp.initialize(model, optimizer, opt_level=opt_level)
                model = nn.DataParallel(model)


                pytorch_total_params = sum(p.numel() for p in model.parameters())
                #print('Total number of paramters: {}'.format(pytorch_total_params))

                model.load_state_dict(torch.load(f"{path}/k{k}fold{fold}top{i+1}.ckpt"))
                #model.load_state_dict(torch.load("checkpoints_fold0/epoch{}.ckpt".format(i)))
                model.eval()
                MODELS.append(model)

            dict=MODELS[0].module.state_dict()
            for key in dict:
                for i in range(1,len(MODELS)):
                    dict[key]=dict[key]+MODELS[i].module.state_dict()[key]

                dict[key]=dict[key]/float(len(MODELS))

            MODELS[0].module.load_state_dict(dict)
            avg_model=MODELS[0]
            degradation_models.append(avg_model)
    return  degradation_models

def get_bpp(arg):
    sequence=arg[0]
    seq_length=arg[1]
    tmp_file=arg[2]
    path=arg[3]
    reverse=arg[4]
    #with open(filename,'w') as f:
    os.system(f'echo {sequence} | {path}/linearpartition -V -r {tmp_file} >/dev/null 2>&1')
    matrix = [[0.0 for x in range(seq_length)] for y in range(seq_length)]
    #matrix=0
    # data processing
    for line in open(tmp_file):
        line = line.strip()
        if line == "":
            break
        i,j,prob = line.split()
        matrix[int(j)-1][int(i)-1] = float(prob)
        matrix[int(i)-1][int(j)-1] = float(prob)

    matrix=np.array(matrix)
    if reverse:
        matrix=-matrix
    #ap=np.array(matrix).sum(0)
    return matrix





def get_bpp_multiprocess(p,sequences,MAX_THRE,path,reverse=False):
    li=[]
    for i, sequence in enumerate(sequences):
        li.append([sequence,len(sequence),f'tmp/matrix_{i}',path, reverse])



    results = []
    #for ret in p.imap(get_bpp, li):
    for ret in tqdm(p.imap(get_bpp, li),total=len(li)):
        results.append(ret)
    #p.close()
    return results

def get_structure(args):
    sequence=args[0]
    seq_length=len(sequence)
    tmp_file=args[1]
    linearfold_path=args[2]
    #with open(filename,'w') as f:
    os.system(f'echo {sequence} | ./{linearfold_path}/linearfold -V > {tmp_file}')

    with open(tmp_file,'r') as f:
        structure=f.read().split('\n')[1].split(' ')[0]

    assert len(structure)==len(sequence)

    #ap=np.array(matrix).sum(0)
    return structure


import numpy as np
import re

def convert_structure_to_bps(secstruct):

    bps = []

    left_delimiters = ['(','{','[']
    right_delimiters = [')','}',']']

    for (left_delim, right_delim) in list(zip(left_delimiters, right_delimiters)):

        left_list = []
        for i, char in enumerate(secstruct):
            if char == left_delim:
                left_list.append(i)

            elif char == right_delim:
                bps.append([left_list[-1],i])
                left_list = left_list[:-1]

        assert len(left_list)==0

    return bps

def secstruct_to_partner(secstruct):
    '''Convert secondary structure string to partner array.
    I.E. ((.)) -> [4,3,-1,1,0]
    '''
    bps = convert_structure_to_bps(secstruct)
    partner_vec = -1*np.ones([len(secstruct)])

    for (i,j) in bps:
        partner_vec[i] = j
        partner_vec[j] = i

    return partner_vec

def write_loop_assignments(dbn_string):
    '''Input: dot-parenthesis string
    Output: bpRNA-style loop type assignments'''

    pair_partners = secstruct_to_partner(dbn_string)

    #print(pair_partners)
    bprna_string=['u']*len(dbn_string)

    # assign stems
    for s_ind, s in enumerate(dbn_string):
        if s != '.':
            bprna_string[s_ind] = 'S'

    # get loop regions

    while 'u' in ''.join(bprna_string):
        #print(''.join(bprna_string))

        obj = re.search(r"uu*", ''.join(bprna_string))
        start_ind, end_ind = obj.start(), obj.end()

        n_open_hps = dbn_string[:start_ind].count(')') - dbn_string[:start_ind].count('(')

        if n_open_hps == 0:
            bprna_string[start_ind:end_ind] = 'E'*(end_ind-start_ind)

        else:

            last_stem_pairing = int(pair_partners[start_ind - 1])
            next_stem_pairing = int(pair_partners[end_ind ])

            if last_stem_pairing == end_ind:
                bprna_string[start_ind:end_ind] = 'H'*(end_ind-start_ind)

            elif (last_stem_pairing - 1 == next_stem_pairing):
                bprna_string[start_ind:end_ind] = 'B'*(end_ind-start_ind)

            elif dbn_string[start_ind-1]==')' and dbn_string[end_ind]=='(':
                bprna_string[start_ind:end_ind] = 'M'*(end_ind-start_ind)

            else:
                if dbn_string[next_stem_pairing+1:last_stem_pairing] == '.'*(last_stem_pairing - next_stem_pairing-1):
                    bprna_string[start_ind:end_ind] = 'I'*(end_ind-start_ind)
                    bprna_string[next_stem_pairing+1:last_stem_pairing] = 'I'*(last_stem_pairing - next_stem_pairing-1)

                else:
                    bprna_string[start_ind:end_ind] = 'M'*(end_ind - start_ind)
    return ''.join(bprna_string)


def get_features(args):
    sequence=args[0]
    seq_len=args[1]
    tmp_file=args[2]
    lf_path=args[3]
    lp_path=args[4]
    reverse=args[5]
    bpp_only=args[6]

    bpp=get_bpp([sequence,seq_len,tmp_file,lp_path,reverse])
    if bpp_only:
        structure=None
        loop=None
        mld=0
    else:
        structure=get_structure([sequence,tmp_file,lf_path])
        loop=write_loop_assignments(structure)
        rgv = RGV(structure)

        rgv.calc_MLD()

        mld=rgv.MLD

        degscore = -DegScore(sequence, structure=structure).degscore_by_position


    features={'sequence':sequence,
             'bpp':bpp,
             'structure':structure,
             'loop':loop,
             'MLD':mld,
             "degscore":degscore}
    # with open(f'features/{id}.p','wb+') as f:
    #     pickle.dump(features,f)

    return features

def get_features_multiprocess(p,sequences,MAX_THRE,lf_path,lp_path,bpp_only=True,reverse=False):
    li=[]
    for i, sequence in enumerate(sequences):
        li.append([sequence,len(sequence),f'tmp/matrix_{i}',lf_path, lp_path, reverse,bpp_only])



    results = []
    #for ret in p.imap(get_bpp, li):
    for ret in tqdm(p.imap(get_features, li),total=len(li)):
        results.append(ret)
    #p.close()
    return results

def preprocess_inputs(input_features):
    tokens = 'ACGU().BEHIMSX'
    sequence, structure, loop = input_features['sequence'],input_features['structure'],input_features['loop']
    input=[]
    #for j in range(len(structures)):
    input_seq=np.asarray([tokens.index(s) for s in sequence])
    input_structure=np.asarray([tokens.index(s) for s in structure])
    input_loop=np.asarray([tokens.index(s) for s in loop])
    input.append(np.stack([input_seq,input_structure,input_loop],-1))
    input=np.asarray(input).astype('int')
    return input
