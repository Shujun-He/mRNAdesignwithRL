from Network import *
from Agent import *
from env import *
import os
from ReplayMemory import ReplayMemory
from Functions import *
from tqdm import tqdm
from Logger import *
from ranger import Ranger
import json
import argparse
import time
# try:
#     #from apex.parallel import DistributedDataParallel as DDP
#     from apex.fp16_utils import *
#     from apex import amp, optimizers
#     from apex.multi_tensor_apply import multi_tensor_applier
# except ImportError:
#     raise ImportError("Please install apex from https://www.github.com/nvidia/apex to run this example.")

import time

t=time.time()

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_id', type=str, default='0',  help='which gpu to use')
    parser.add_argument('--pretrained_weight_path', type=str, default='',  help='path to pretrained model weights')
    parser.add_argument('--memory_capacity', type=int, default=int(1e8), help='max number of memories to store')
    parser.add_argument('--episode_length', type=int, default=24, help='episode_length')
    parser.add_argument('--episodes', type=int, default=50, help='total number of episodes to run')
    parser.add_argument('--batch_size', type=int, default=32, help='size of each batch during training')
    parser.add_argument('--EPS_START', type=float, default=0.75, help='size of each batch during training')
    parser.add_argument('--EPS_END', type=float, default=0, help='size of each batch during training')
    parser.add_argument('--gamma', type=float, default=0.5, help='gamma value')
    parser.add_argument('--gamma_reduce', type=float, default=0, help='gamma to reduce gamma by per reduce_k_epoch')
    parser.add_argument('--MAX_THRE', type=int, default=1, help='number of threads used to run linear partition')
    parser.add_argument('--batch_size_update_epoch', type=int, default=50, help='per epochs to double batch size')
    parser.add_argument('--max_batch_size', type=int, default=2048, help='max allowed batch size')
    #parser.add_argument('--seq_length', type=int, default=64, help='seq length used for pretraining')
    parser.add_argument('--k', type=int, default=64, help='max number of mutations allowed')
    parser.add_argument('--epochs_per_episode', type=int, default=1, help='epochs_per_episode')

    parser.add_argument('--degradation_reward', action='store_true', help='use degradation reward or not')
    parser.add_argument('--use_deberta_attention', action='store_true', help='use deberta self attention or not')
    parser.add_argument('--use_nt_input', action='store_true', help='use nucleotide input or not')
    #parser.add_argument('--no_degradation_reward', action='store_false', help='use degradation reward or not')
    #parser.add_argument('--degradation_reward', action=argparse.BooleanOptionalAction)
    #parser.add_argument('--pretrained_weight_path', type=str, default='', help='pretrained_weight_path')

    parser.add_argument('--reps_per', type=int, default=32, help='number of parallel sequences to run')
    parser.add_argument('--reduce_k_epoch', type=int, default=20, help='frequency to reduce k')
    parser.add_argument('--path', type=str, default='../', help='path of csv file with DNA sequences and labels')
    parser.add_argument('--linearpartition_path', type=str, default='LinearPartition', help='path of linear partition')
    parser.add_argument('--linearfold_path', type=str, default='LinearFold', help='path of linear fold')
    parser.add_argument('--codon_table_path', type=str, default='', help='path of codon table')
    parser.add_argument('--codon_usage_table_path', type=str, default='', help='codon_usage_table_path')
    parser.add_argument('--fasta_file_path', type=str, default=' ', help='path of protein fasta file')
    #parser.add_argument('--epochs', type=int, default=1, help='number of epochs to train per episode')

    parser.add_argument('--cai_weight', type=float, default=0, help='weight of cai during optimization')
    parser.add_argument('--degradation_weight', type=float, default=0.5, help='weight of cai during optimization')
    parser.add_argument('--optimization_direction', type=str, default='max', help='maximize of minimize structure')
    parser.add_argument('--target',type=float,default=0.6,help='absolute app target')


    parser.add_argument('--weight_decay', type=float, default=0, help='weight dacay used in optimizer')
    parser.add_argument('--ntoken', type=int, default=4, help='number of tokens to represent DNA nucleotides (should always be 4)')
    parser.add_argument('--nclass', type=int, default=919, help='number of classes from the linear decoder')
    parser.add_argument('--ninp', type=int, default=512, help='ninp for transformer encoder')
    parser.add_argument('--nhead', type=int, default=8, help='nhead for transformer encoder')
    parser.add_argument('--nhid', type=int, default=2048, help='nhid for transformer encoder')
    parser.add_argument('--nlayers', type=int, default=6, help='nlayers for transformer encoder')
    parser.add_argument('--save_freq', type=int, default=1, help='saving checkpoints per save_freq epochs')
    parser.add_argument('--dropout', type=float, default=.1, help='transformer dropout')
    parser.add_argument('--lr_scale', type=float, default=0.1, help='learning rate scale')
    parser.add_argument('--kmers', type=int, default=5, help='k-mers to be aggregated')
    parser.add_argument('--num_workers', type=int, default=1, help='num_workers')
    parser.add_argument('--debug', type=bool, default=False, help='run in debug mode')
    parser.add_argument('--data_parallel', type=bool, default=False, help='use dataparallel')
    opts = parser.parse_args()
    return opts

args=get_args()


os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#read codon table and protein sequence
codon_table=pd.read_csv(args.codon_table_path)
cai_table=pd.read_csv(args.codon_usage_table_path)
cai_table.amino_acid[:3]=['O']*3
max_usage={}

for i,amino_acid in enumerate(cai_table.amino_acid):
    if amino_acid not in max_usage:
        max_usage[amino_acid]=cai_table.relative_frequency[i]
    elif cai_table.relative_frequency[i]>max_usage[amino_acid]:
        max_usage[amino_acid]=cai_table.relative_frequency[i]

def CAI(sequence):
    assert len(sequence)%3==0
    n_amino=len(sequence)//3
    cai=1
    #print(n_amino)
    for i in range(n_amino):
        codon=sequence[i*3:(i+1)*3]
        index=cai_table.codon.to_list().index(codon)
        amino=cai_table.amino_acid[index]
        usage=cai_table.relative_frequency[index]
        cai*=(usage/max_usage[amino])
    cai=cai**(1/n_amino)
    return cai


# print(codon_table)
# print(cai_table)

cai_vector=[]
for codon,amino_acid in zip(codon_table.Codon,codon_table.Letter):
    index=list(cai_table.codon).index(codon)
    cai_vector.append(cai_table.relative_frequency[index]/max_usage[amino_acid.upper()])

# print(cai_vector)
# exit()


with open(args.fasta_file_path,'r') as f:
    protein_sequence=f.read().split('\n')[1].upper().replace('T','U')

print(f"protein sequence: {protein_sequence}")
print(f"sequence length: {len(protein_sequence)}")

if args.optimization_direction=="target":
    print(f"targeting app of {args.target}")
else:
    print(f"{args.optimization_direction}imizing app")
# with open('/home/exx/Documents/RNAplay/data/BNTUTR5.fasta','r') as f:
#     UTR5=f.read().split('\n')[1]
# with open('/home/exx/Documents/RNAplay/data/BNTUTR3.fasta','r') as f:
#     UTR3=f.read().split('\n')[1]
# UTR3+='A'
# UTR5='GGGGGG'
# UTR3='AAAAAA'
UTR5=''
UTR3=''
start=len(UTR5)
end=len(UTR3)


os.system('mkdir logs')
os.system('mkdir models')
os.system('mkdir async_comm')
logger=CSVLogger(['episode','averaged_paired_probability','loss'],'logs/fintune_log.csv')

#init stuff
cai_vector=torch.tensor(cai_vector).to(device)
if args.degradation_reward:
    new_head=True
else:
    new_head=False
policy_net=NucleicTransformer(ntoken=64, nclass=64, ninp=256, nhead=4, nhid=1024, nlayers=5, k=5, cai_vector=cai_vector,cai_weight=args.cai_weight, new_head=new_head,use_deberta_attention=args.use_deberta_attention).to(device)
target_net=NucleicTransformer(ntoken=64, nclass=64, ninp=256, nhead=4, nhid=1024, nlayers=5, k=5, cai_vector=cai_vector,cai_weight=args.cai_weight, new_head=new_head,use_deberta_attention=args.use_deberta_attention).to(device)

optimizer = Ranger(policy_net.parameters(), weight_decay=1e-1, lr=1e-4)

opt_level = 'O1'
#policy_net, optimizer = amp.initialize(policy_net, optimizer, opt_level=opt_level)

if args.pretrained_weight_path!='':
    weights=torch.load(args.pretrained_weight_path)

# if args.data_parallel:
#     policy_net=nn.DataParallel(policy_net)
#     target_net=nn.DataParallel(target_net)
#
# #weights=torch.load('models/best_weights.ckpt')
# else:
    new_weights={}
    for key in weights.keys():
        if 'module.' in key:
            new_weights[key[7:]]=weights[key]
        else:
            new_weights[key]=weights[key]
    weights=new_weights
#
    policy_net.load_state_dict(weights,strict=False)
    target_net.load_state_dict(weights,strict=False)

# policy_net.load_state_dict(weights)
# target_net.load_state_dict(weights)



#optimizer = optim.Adam(policy_net.parameters(),weight_decay=1e-5, lr=1e-4)


agent=RNAAgent(policy_net, target_net, optimizer, codon_table, device, args.memory_capacity,reps_per=args.reps_per,
finetune=True,
memory_folder='finetune_memory', EPS_START = args.EPS_START, EPS_END = args.EPS_END, EPS_DECAY=args.episode_length*args.episodes, gamma=args.gamma, k=args.k,
degradation_reward=args.degradation_reward,degradation_weight=args.degradation_weight,epochs_per_episode=args.epochs_per_episode)
env=RNAstructure()



#exit()

#p = Pool(processes=args.MAX_THRE)

# with open('finetune_sequences.p','wb+') as f:
#     pickle.dump(testing_sequences,f)
best_metric=-1000000
done=False
while not done:

    try:
        state=json.load(open("async_comm/state.json"))

        agent.memory.position=state['position']
        agent.k=state['k']
        agent.gamma=state['gamma']
        agent.degradation_weight=state['degradation_weight']
        done=state['done']

        print(state)

        if agent.memory.position>args.batch_size:
            loss=agent.optimize_model(args.batch_size,start,end)

        torch.save(policy_net.state_dict(), f'async_comm/trained_weights.bin')
        time.sleep(5)
    #done=True
    except:
        pass
