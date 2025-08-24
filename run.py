from Network import *
from Agent import *
from env import *
import os
from ReplayMemory import ReplayMemory
from Functions import *
from tqdm import tqdm
from Logger import *
from ranger import Ranger
import argparse

try:
    #from apex.parallel import DistributedDataParallel as DDP
    from apex.fp16_utils import *
    from apex import amp, optimizers
    from apex.multi_tensor_apply import multi_tensor_applier
except ImportError:
    raise ImportError("Please install apex from https://www.github.com/nvidia/apex to run this example.")

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_id', type=str, default='0',  help='which gpu to use')
    parser.add_argument('--memory_capacity', type=int, default=int(1e8), help='max number of memories to store')
    parser.add_argument('--episode_length', type=int, default=24, help='episode_length')
    parser.add_argument('--episodes', type=int, default=24, help='total number of episodes to run')
    parser.add_argument('--batch_size', type=int, default=256, help='size of each batch during training')
    parser.add_argument('--MAX_THRE', type=int, default=1, help='number of threads used to run linear partition')
    parser.add_argument('--batch_size_update_epoch', type=int, default=50, help='per epochs to double batch size')
    parser.add_argument('--max_batch_size', type=int, default=2048, help='max allowed batch size')
    parser.add_argument('--seq_length', type=int, default=64, help='seq length used for pretraining')
    parser.add_argument('--degradation_reward', type=bool, default=False, help='use degradation reward or not')
    parser.add_argument('--reps_per', type=int, default=240, help='number of parallel sequences to run')
    #parser.add_argument('--reps_per', type=int, default=240, help='number of parallel sequences to run')
    parser.add_argument('--path', type=str, default='../', help='path of csv file with DNA sequences and labels')
    parser.add_argument('--linearpartition_path', type=str, default='LinearPartition', help='path of linear partition')
    parser.add_argument('--codon_table_path', type=str, default='LinearPartition', help='path of linear partition')
    #parser.add_argument('--epochs', type=int, default=1, help='number of epochs to train per episode')

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
    opts = parser.parse_args()
    return opts


args=get_args()


os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#read codon table and protein sequence
codon_table=pd.read_csv(args.codon_table_path)
# with open('/home/exx/Documents/RNAplay/data/GFP.fasta','r') as f:
#     protein_sequence=f.read().split('\n')[1]
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
logger=CSVLogger(['episode','averaged_paired_probability','loss'],'logs/log.csv')

#init stuff
policy_net=NucleicTransformer(ntoken=64, nclass=64, ninp=256, nhead=4, nhid=1024, nlayers=5, k=5).to(device)
target_net=NucleicTransformer(ntoken=64, nclass=64, ninp=256, nhead=4, nhid=1024, nlayers=5, k=5).to(device)

optimizer = Ranger(policy_net.parameters(), weight_decay=1e-1, lr=1e-3)

opt_level = 'O1'
#policy_net, optimizer = amp.initialize(policy_net, optimizer, opt_level=opt_level)

policy_net=nn.DataParallel(policy_net)
target_net=nn.DataParallel(target_net)
#optimizer = optim.Adam(policy_net.parameters(),weight_decay=1e-5, lr=1e-4)


agent=RNAAgent(policy_net, target_net, optimizer, codon_table, device, args.memory_capacity,reps_per=args.reps_per,
EPS_START = 1, EPS_END = 0.25, EPS_DECAY=args.episode_length*args.episodes)
env=RNAstructure()


#init_rna_sequence=agent.protein2nt(protein_sequence)
start=0
end=0
#init_rna_sequence=UTR5+init_rna_sequence+UTR3




#load degradation models
#degradation_models=load_degradation_models(device,'../degradation_model')


#codon_sequence=
#ap=np.zeros(len(init_rna_sequence))
best_metric=0
best_metric_policy=0
#sequence_length=len(init_rna_sequence)
distance_mask=torch.tensor(get_distance_mask(args.seq_length*3)).to(device).float()
protein_letters=codon_table.Letter.unique()
testing_sequences=[]
for i in range(args.reps_per):
    int_protein_sequence=np.random.randint(20,size=args.seq_length)
    protein_sequence=''
    for integer in int_protein_sequence:
        protein_sequence+=protein_letters[integer]
    init_rna_sequence=agent.protein2nt(protein_sequence)
    protein_mask=agent.get_protein_mask(init_rna_sequence,start,end)
    testing_sequences.append({'protein_sequence':protein_sequence,'sequence':init_rna_sequence,'protein_mask':protein_mask})

with open('testing_sequences.p','wb+') as f:
    pickle.dump(testing_sequences,f)
p = Pool(processes=args.MAX_THRE)
best_metric=np.zeros(args.reps_per)
for ep in range(args.episodes):
    # int_protein_sequence=np.random.randint(20,size=seq_length)
    # protein_sequence=''
    # for integer in int_protein_sequence:
    #     protein_sequence+=protein_letters[integer]
    print('###running episodes###')
    sequences=[]
    masks=[]
    for i in tqdm(range(args.reps_per)):
        zipped=testing_sequences[i]
        protein_sequence=zipped['protein_sequence']
        init_rna_sequence=zipped['sequence']
        protein_mask=zipped['protein_mask']
        current_rna_sequence=init_rna_sequence
        sequences.append(init_rna_sequence)
        masks.append(protein_mask)
        for step in range(args.episode_length-1):
            #print(len(current_rna_sequence))
            next_rna_sequence=agent.select_action(current_rna_sequence,protein_mask,start,end)
            #masks.append(protein_mask)
            sequences.append(next_rna_sequence)
            current_rna_sequence=next_rna_sequence
    #exit()
    results=get_bpp_multiprocess(p,sequences,args.MAX_THRE,args.linearpartition_path)
    #continue
    #exit()
    #np.random.shuffle(sequences)

    for pkg in zip(sequences, results):
        assert pkg[0]==pkg[1][0]


    if args.degradation_reward:
        int_sequences=[]
        bpps=[]
        for result in results:
            int_sequences.append(agent.nucleatide2int(result[0]))
            bpps.append(result[1])


        degradation=[]
        int_sequences=torch.Tensor(int_sequences).to(device).long()
        bpps=torch.Tensor(bpps).to(device).float().unsqueeze(1)
        dm=distance_mask.expand(bpps.shape[0],3,bpps.shape[2],bpps.shape[3])
        bpps=torch.cat([bpps,dm],1)
        #exit()
        print('###calculating degradation ###')
        with torch.no_grad():
            for model in tqdm(degradation_models):
                degradation.append(model(int_sequences,bpps))
        degradation=torch.stack(degradation).mean(0)[:,:,0].cpu().numpy()
    #exit()

    print('###saving memory###')
    for j in tqdm(range(args.reps_per)):
        for i in range(args.episode_length-1):
            current_rna_sequence=results[i+j*args.episode_length][0]
            next_rna_sequence=results[i+1+j*args.episode_length][0]
            next_ap=results[i+1+j*args.episode_length][1].sum(1)
            prev_ap=results[i+j*args.episode_length][1].sum(1)
            protein_mask=masks[j]
            mean_ap=next_ap.mean()
            if args.degradation_reward:
                prev_degradation=degradation[i]
                next_degradation=degradation[i+1]
                reward=(next_ap-prev_ap).reshape(-1,3).mean(1)+\
                (prev_degradation-next_degradation).reshape(-1,3).mean(1)
                mean_degradation=next_degradation.mean()
                metric=mean_ap-mean_degradation
            else:
                #reward=(np.exp(next_ap)-np.exp(prev_ap)).reshape(-1,3).mean(1)
                reward=(next_ap-prev_ap).reshape(-1,3).mean(1)
                metric=mean_ap



            current_protein_sequence=agent.codon2int(current_rna_sequence)
            next_protein_sequence=agent.codon2int(next_rna_sequence)
            #current_protein_sequence=np.pad(current_protein_sequence,((start//3,end//3)),constant_values=64)
            #next_protein_sequence=np.pad(next_protein_sequence,((start//3,end//3)),constant_values=64)
            agent.memory.push(torch.tensor([agent.nucleatide2int(current_rna_sequence)]),
                              torch.tensor([current_protein_sequence]),
                              torch.tensor([agent.nucleatide2int(next_rna_sequence)]),
                              torch.tensor([next_protein_sequence]),
                              torch.tensor([reward]),
                              torch.tensor([protein_mask]))
            # filename=f'memory/transition{0}.p'
            # with open(filename,'rb') as f:
            #     memory=pickle.load(f)
            # exit()
            if mean_ap>best_metric[j]:
                best_metric[j]=mean_ap
                testing_sequences[j]['sequence']=next_rna_sequence
                #best_metric=mean_ap


    if agent.memory.position>args.batch_size:
        try:
            loss=agent.optimize_model(args.batch_size,start,end)
            print(f"###train loss at ep {ep} is {loss}###")
        except:
            # with open('error_message.txt','w+') as f:
            #     f.write(e)
            args.batch_size=int(int(args.batch_size//1.75/32)*32)

    logger.log([ep,best_metric.mean(),loss])
    print(f'@ episode {ep}: average paired prob is {best_metric.mean()}')
    # print(current_rna_sequence)
    torch.save(policy_net.state_dict(), f'models/best_weights.ckpt')
    if (ep+1)%args.batch_size_update_epoch==0:
        args.batch_size*=2
        args.batch_size=np.clip(args.batch_size,0,args.max_batch_size)
        args.batch_size=int(args.batch_size)
    if args.debug:
        break
#state_action_values, next_state_batch=agent.optimize_model(10,5)
p.close()
