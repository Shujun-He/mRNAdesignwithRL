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
    parser.add_argument('--fix_repeat_sites', action='store_true', help='fix repeat sites or not')


    parser.add_argument('--use_deberta_attention', action='store_true', help='use deberta self attention or not')
    parser.add_argument('--use_nt_input', action='store_true', help='use deberta self attention or not')
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
    parser.add_argument('--custom_protein_mask', type=str, default='', help='path of custom protein mask')
    #parser.add_argument('--epochs', type=int, default=1, help='number of epochs to train per episode')

    parser.add_argument('--cai_weight', type=float, default=0, help='weight of cai during optimization')
    parser.add_argument('--cai_metric_weight', type=float, default=0, help='weight of cai during best sequence selection')
    parser.add_argument('--mld_metric_weight', type=float, default=0, help='weight of mld during best sequence selection')

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

os.system('rm async_comm/*')

# print(args.degradation_reward)
# print(args.degradation_weight)
#
# exit()

#some parameters
#memory_capacity=int(1e8)
#episode_length=24
#episodes=50
#batch_size=32
#MAX_THRE=48
#batch_size_update_epoch=25
#max_batch_size=2048
#seq_length=64
#degradation_reward=False
#validation_freq=10
#reps_per=32
#reduce_k_epoch=20

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

codon2aa={c:aa for c,aa in zip(cai_table['codon'],cai_table['amino_acid'])}

aa2codons={}

for aa in cai_table['amino_acid'].unique():
    table=cai_table[cai_table['amino_acid']==aa].sort_values('relative_frequency',ascending=False)
    values=table['codon'].to_list()
    aa2codons[aa]=values

#exit()

def get_nucleotide_repeat_list():
    return ['AAAAA', 'CCCC', 'GGGG', 'UUUU', 'TTTT']

def check_restriction_sites(seq, restriction_sites):
    """ Checks if the sequence contains any of the prohibited restriction sites.
    If restriction site in seq: returns seq location of that site
    else: returns -1 """
    for x in restriction_sites:
        if x in seq:
            return seq.find(x)

    return -1

def check_nucleotide_repeat(seq):
    """ Checks if there are any stretches of nucleotide repeats
        of A, C, G and U """

    nucleotide_repeat = get_nucleotide_repeat_list()

    for repeat in nucleotide_repeat:
        if repeat in seq:
            return seq.find(repeat)

    return -1

def fix_repeat_sites(sequence):
    repeat_exist=True
    while repeat_exist:
        position=check_nucleotide_repeat(sequence)
        repeat_starting_position=(position)//3

        current_codon_start=(position)//3*3
        next_codon_start= (position+3)//3*3

        #print(f"Fixing {sequence[position:position+5]} at position {position}")
        #print(sequence[position:position+5])
        #print(next_codon_start)
        #print(sequence[next_codon_start:next_codon_start+3])

        if position==current_codon_start or (position-current_codon_start)>=(next_codon_start+3-position):
            codon2change=current_codon_start//3
        else:
            codon2change=next_codon_start//3

        codon=sequence[codon2change*3:codon2change*3+3]
        aa=codon2aa[codon]
        #print(codon)
        #codon
        np.random.shuffle(aa2codons[aa])
        for alternative_codon in aa2codons[aa]:
            new_sequence=sequence[:codon2change*3]+alternative_codon+sequence[codon2change*3+3:]
            #print(new_sequence[:position+5])
            if check_nucleotide_repeat(new_sequence[position:position+5])==-1:
                #print(new_sequence[position:position+5])
                sequence=new_sequence
                break
        #cai,protein_sequence=CAI(sequence)
        #assert reference_protein_sequence==protein_sequence


        if position==-1:
            repeat_exist=False
    return sequence

# print(codon_table)
# print(cai_table)

cai_vector=[]
for codon,amino_acid in zip(codon_table.Codon,codon_table.Letter):
    index=list(cai_table.codon).index(codon)
    cai_vector.append(cai_table.relative_frequency[index]/max_usage[amino_acid.upper()])

# print(cai_vector)
# exit()


with open(args.fasta_file_path,'r') as f:
    protein_sequence=f.read().split('\n')[1].upper()#.replace('T','U')

if args.use_nt_input:
    protein_sequence=protein_sequence.replace('T','U')



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
logger=CSVLogger(['episode','metric','loss'],'logs/finetune_log.csv')

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

if args.pretrained_weight_path != '':
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

if args.use_nt_input:
    #protein_sequence=agent.codon2int(protein_sequence)
    init_rna_sequence=protein_sequence
    tmp=''
    for i in range(len(protein_sequence)//3):
        tmp+= codon2aa[protein_sequence[i*3:(i+1)*3]]
    protein_sequence=tmp
else:
    init_rna_sequence=agent.protein2nt(protein_sequence)



# print(protein_sequence)
# exit()

print(f"protein sequence: {protein_sequence}")
print(f"sequence length: {len(protein_sequence)}")


start=0
end=0
#init_rna_sequence=UTR5+init_rna_sequence+UTR3




#load degradation models
if args.degradation_reward:
    #print("shit")
    degradation_models=load_degradation_models(device,'../../degradation_model_w_loop')
    #exit()

#codon_sequence=
ap=np.zeros(len(init_rna_sequence))
best_metric=0
best_metric_policy=0
sequence_length=len(init_rna_sequence)
distance_mask=torch.tensor(get_distance_mask(len(init_rna_sequence))).to(device).float()
protein_letters=codon_table.Letter.unique()
testing_sequences=[]
for i in range(args.reps_per):
    # int_protein_sequence=np.random.randint(20,size=seq_length)
    # protein_sequence=''
    # for integer in int_protein_sequence:
    #     protein_sequence+=protein_letters[integer]
    if args.custom_protein_mask!='':
        protein_mask=np.load(args.custom_protein_mask)
    else:
        protein_mask=agent.get_protein_mask(init_rna_sequence,start,end)
    testing_sequences.append({'protein_sequence':protein_sequence,'sequence':init_rna_sequence,'protein_mask':protein_mask})

#exit()

p = Pool(processes=args.MAX_THRE)

with open('finetune_sequences.p','wb+') as f:
    pickle.dump(testing_sequences,f)
best_metric=-1000000
# episode=np.linspace(np.pi/2,0,args.episodes)
# weights=np.cos(episode)
# print(weights)
for ep in range(args.episodes):
    agent.degradation_weight=0+(ep)/(args.episodes+1)*args.degradation_weight
    #agent.degradation_weight=weights[ep]
    print(f"deg weight: agent.degradation_weight")
    if ep>0:
        wait=True
        while wait:
            print("waiting for training")
            try:
                agent.target_net.load_state_dict(torch.load("async_comm/trained_weights.bin"))
                agent.policy_net.load_state_dict(torch.load("async_comm/trained_weights.bin"))
                print("Trained weights loaded")
                wait=False
            except:
                print("No training has occured so no weights to load")


    print('###running episodes###')
    sequences=[]
    masks=[]
    for i in tqdm(range(args.reps_per)):
        zipped=np.random.choice(testing_sequences)
        protein_sequence=zipped['protein_sequence']
        init_rna_sequence=zipped['sequence']
        protein_mask=zipped['protein_mask']
        current_rna_sequence=init_rna_sequence
        sequences.append(init_rna_sequence)
        masks.append(protein_mask)
        for step in range(args.episode_length-1):
            #print(len(current_rna_sequence))
            next_rna_sequence=agent.select_action(current_rna_sequence,protein_mask,start,end)

            if args.fix_repeat_sites:
                next_rna_sequence=fix_repeat_sites(next_rna_sequence)

            masks.append(protein_mask)
            sequences.append(next_rna_sequence)
            current_rna_sequence=next_rna_sequence
    # if args.optimization_direction=='min':
    #     results=get_bpp_multiprocess(p,sequences,args.MAX_THRE,args.linearpartition_path,reverse=True)
    # else:
    #     results=get_bpp_multiprocess(p,sequences,args.MAX_THRE,args.linearpartition_path)

    bpp_only=(args.degradation_reward==False)
    if args.optimization_direction=='min':
        results=get_features_multiprocess(p,sequences,args.MAX_THRE,args.linearfold_path, args.linearpartition_path,bpp_only=bpp_only,reverse=True)
    else:
        results=get_features_multiprocess(p,sequences,args.MAX_THRE,args.linearfold_path, args.linearpartition_path,bpp_only=bpp_only)

    #exit()

    #continue
    #exit()
    #np.random.shuffle(sequences)

    for pkg in zip(sequences, results):
        assert pkg[0]==pkg[1]["sequence"]


    if args.degradation_reward:
        int_sequences=[]
        bpps=[]
        for result in results:
            #int_sequences.append(agent.nucleatide2int(result["sequence"]))
            int_sequences.append(preprocess_inputs(result))
            bpps.append(result["bpp"])


        degradation=[]
        int_sequences=np.array(int_sequences)
        bpps=np.array(bpps)
        int_sequences=torch.Tensor(int_sequences).to(device).long().squeeze(1)
        bpps=torch.Tensor(bpps).to(device).float().unsqueeze(1)
        dm=distance_mask.expand(bpps.shape[0],3,bpps.shape[2],bpps.shape[3])
        bpps=torch.cat([bpps,dm],1)
        #exit()
        print('###calculating degradation ###')
        bs=32
        if len(int_sequences)%bs==0:
            batches=len(int_sequences)//bs
        else:
            batches=len(int_sequences)//bs+1
        for b in tqdm(range(batches)):
            with torch.no_grad():
                degradation_=[]
                for model in degradation_models:
                    degradation_.append(-model(int_sequences[b*bs:(b+1)*bs],bpps[b*bs:(b+1)*bs]))
                degradation_=torch.stack(degradation_).mean(0)[:,:,0]
        #degradation=torch.stack(degradation).mean(0)[:,:,0].cpu().numpy()
            degradation.append(degradation_)
        degradation=torch.cat(degradation,0).cpu().numpy()
        #exit()
    #exit()

    print('###saving memory###')
    for j in tqdm(range(args.reps_per)):
        for i in range(args.episode_length-1):
            current_rna_sequence=results[i+j*args.episode_length]["sequence"]
            next_rna_sequence=results[i+1+j*args.episode_length]["sequence"]
            next_ap=results[i+1+j*args.episode_length]["bpp"].sum(1)
            prev_ap=results[i+j*args.episode_length]["bpp"].sum(1)
            protein_mask=masks[i+j*args.episode_length]
            #if args.degradation_reward:
            current_mld=results[i+j*args.episode_length]["MLD"]
            next_mld=results[i+1+j*args.episode_length]["MLD"]

            current_degscore=results[i+j*args.episode_length]["degscore"]
            next_degscore=results[i+1+j*args.episode_length]["degscore"]
            mean_ap=next_ap.mean()
            if args.degradation_reward:
                prev_degradation=degradation[i+j*args.episode_length]
                next_degradation=degradation[i+1+j*args.episode_length]
                reward=(np.exp(next_ap*2)-np.exp(prev_ap*2)).reshape(-1,3).mean(1)
                degradation_reward=(next_degradation-prev_degradation).reshape(-1,3).mean(1)


                degscore_reward=(next_degscore-current_degscore).reshape(-1,3).mean(1)
                # print(reward.shape)
                # print(degradation_reward.shape)

                mean_degradation=next_degradation.mean()
                mean_degscore=next_degscore.mean()
                #metric=mean_ap+mean_degradation*args.degradation_weight
                metric=mean_ap+args.degradation_weight*(mean_degradation+mean_degscore)
            else:
                degradation_reward=1
                if args.optimization_direction=='target':
                    reward=-np.abs((next_ap).reshape(-1,3).mean(1)-args.target)
                    metric=-np.abs(next_ap.mean()-args.target)
                else:
                    reward=(np.exp(next_ap*2)-np.exp(prev_ap*2)).reshape(-1,3).mean(1)
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
                              torch.tensor([degradation_reward]),
                              torch.tensor([degscore_reward]),
                              torch.tensor([protein_mask]))
            # filename=f'memory/transition{0}.p'
            # with open(filename,'rb') as f:
            #     memory=pickle.load(f)
            # exit()
            # mean_ap=next_ap.mean()
            # metric=mean_ap
            cai=CAI(next_rna_sequence)
            metric=metric+args.cai_metric_weight*cai+args.mld_metric_weight*next_mld/len(next_rna_sequence)

            if metric>best_metric:
                with open('best_sequence.txt','w+') as f:
                    f.write(f'ap: {next_ap.mean()}\n')
                    f.write(f'CAI: {cai}\n')
                    if args.degradation_reward:
                        f.write(f'mean_degradation: {mean_degradation}\n')
                        f.write(f'mean_degscore: {mean_degscore}\n')
                        f.write(f"MLD: {next_mld}\n")
                        structure=results[i+1+j*args.episode_length]['structure']
                        loop=results[i+1+j*args.episode_length]['loop']
                        f.write(f'structure: {structure}\n')
                        f.write(f'loop: {loop}\n')
                    f.write(next_rna_sequence)
                testing_sequences=[{'protein_sequence':protein_sequence,'sequence':next_rna_sequence,'protein_mask':protein_mask}]*args.reps_per
                best_metric=metric
    # if agent.memory.position>args.batch_size:
    #     # try:
    #     #     loss=agent.optimize_model(batch_size,start,end)
    #     # except:
    #     #     batch_size=batch_size//2
    #     #try:
    #     loss=agent.optimize_model(args.batch_size,start,end)
    #     # except:
    #     #     batch_size=batch_size//2


    loss=0
    logger.log([ep,best_metric,loss])
    #     print(f'@ episode {ep}: average paired prob is {metric}')
    # # print(current_rna_sequence)

    if (ep+1)%args.batch_size_update_epoch==0:
        args.batch_size*=2
        args.batch_size=int(np.clip(args.batch_size,0,args.max_batch_size))

    df=pd.read_csv("logs/finetune_log.csv")

    if (ep+1)%args.reduce_k_epoch==0:
    # if (ep+1)>args.reduce_k_epoch:
    #     if (metric-df.metric.values[-args.reduce_k_epoch])<1e-5:
        #agent.memory.position=0
            #print(f"reducing k and gamma after plateuing for {args.reduce_k_epoch} episodes")
        agent.k=int(agent.k//2)
        agent.gamma=max(0,agent.gamma-args.gamma_reduce)
        # args.reps_per=int(args.reps_per*1.5)
        # args.episode_length=int(args.episode_length/1.5)
        #batch_size=int(np.clip(batch_size,0,max_batch_size))
    with open("async_comm/state.json",'w+') as f:
        state={"k":agent.k,
        "gamma":agent.gamma,
        "position":agent.memory.position,
        "degradation_weight":agent.degradation_weight,
        "done":False}
        json.dump(state,f)
    #exit()
#state_action_values, next_state_batch=agent.optimize_model(10,5)

with open("async_comm/state.json",'w+') as f:
    state={"k":agent.k,
    "gamma":agent.gamma,
    "position":agent.memory.position,
    "degradation_weight":agent.degradation_weight,
    "done":True}
    json.dump(state,f)
    #exit()
#state_action_values, next_state_batch=agent.optimize_model(10,5)
#p.close()

with open('finetune.txt','w+') as f:
    f.write(f'Tota wall time: {time.time()-t}')
