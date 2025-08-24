import random
from Functions import *
import numpy as np
from ReplayMemory import *
from tqdm import tqdm
from LrScheduler import *
from Dataset import *
try:
    #from apex.parallel import DistributedDataParallel as DDP
    from apex.fp16_utils import *
    from apex import amp, optimizers
    from apex.multi_tensor_apply import multi_tensor_applier
except ImportError:
    raise ImportError("Please install apex from https://www.github.com/nvidia/apex to run this example.")
nt_int={
"A": 0,
"C": 1,
"G": 2,
"U": 3,}



class RNAAgent():
    def __init__(self, policy_net, target_net, optimizer, codon_table, device, memory_capacity,
     reps_per,  memory_folder='memory', finetune=False,
     EPS_START = 1, EPS_END = 0.4, EPS_DECAY = 10000, gamma=0.5, k=8,degradation_reward=False,degradation_weight=0.5, epochs_per_episode=1):
        self.policy_net=policy_net
        self.target_net=target_net
        self.optimizer=optimizer
        self.steps_done=0
        self.EPS_START=EPS_START
        self.EPS_END=EPS_END
        self.EPS_DECAY=EPS_DECAY
        self.codon_table=codon_table
        self.protein_letter_table=np.asarray(codon_table.Letter)
        self.protein_codon_table=np.asarray(codon_table.Codon)
        self.device=device
        self.memory=ReplayMemory(memory_capacity,memory_folder)
        self.gamma=gamma
        if finetune:
            self.scheduler=None
        else:
            self.scheduler=lr_AIAYN(optimizer, policy_net.module.ninp)
        self.k=k
        self.reps_per=reps_per
        self.degradation_reward=degradation_reward
        self.degradation_weight=degradation_weight
        self.epochs_per_episode=epochs_per_episode
        self.memory_capacity=memory_capacity

    def select_action_by_policy(self,state,protein_mask):
        encoding=self.codon2int(state)
        encoding=torch.Tensor(encoding).to(self.device).long()
        protein_mask=torch.Tensor(protein_mask).to(self.device).float()
        self.policy_net.eval()
        with torch.no_grad():
            if self.degradation_reward:
                output,output2=self.policy_net(encoding.unsqueeze(0))
                output=output.squeeze()
                output2=output2.squeeze()
                output=+protein_mask
                output2=+protein_mask
                output=output+output2*self.degradation_weight
            else:
                output=self.policy_net(encoding.unsqueeze(0)).squeeze()+protein_mask
            #max_preds=torch.max(output,-1)[1]
            max_output, max_preds=torch.max(output,-1)
            _, topk=torch.topk(max_output,self.k)
            #print(max_preds)
            # exit()
        new_encoding=encoding.clone()
        new_encoding[topk]=max_preds[topk]
        return self.int2codon(new_encoding)


    def select_action(self,state,protein_mask,start=0,end=0,eval=False):
        sample = random.random()
        if eval==False:
            eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * \
                math.exp(-1. * (self.steps_done//self.reps_per) / self.EPS_DECAY)
            #print(eps_threshold)
            #eps_threshold=0
            self.steps_done += 1
        else:
            eps_threshold = 0


        if sample > eps_threshold:
            #pass
        # else:
            encoding=self.codon2int(state)
            encoding=torch.Tensor(encoding).to(self.device).long()
            protein_mask=torch.Tensor(protein_mask).to(self.device).float()
            self.policy_net.eval()



            with torch.no_grad():
                #output=self.policy_net(encoding.unsqueeze(0)).squeeze()+protein_mask
                if self.degradation_reward:
                    output,output2,output3=self.policy_net(encoding.unsqueeze(0))
                    output=output.squeeze()
                    output2=output2.squeeze()
                    output3=output3.squeeze()
                    # output=output+protein_mask
                    # output2=output2+protein_mask
                    output=output+output2*self.degradation_weight+output3*self.degradation_weight+protein_mask
                else:
                    output=self.policy_net(encoding.unsqueeze(0)).squeeze()+protein_mask


                length=len(output)
                output=output[start//3:length-end//3]
                #print(protein_mask[start//3:length-end//3,-1])
                #exit()
                max_output, max_preds=torch.max(output,-1)
                #print(max_preds)
                # exit()
                # print(max_preds)
                _, topk=torch.topk(max_output,self.k)
                #exit()
                #print(max_preds)
                # exit()
            # print(topk)
            # exit()
            #print(max_preds)
            new_encoding=torch.Tensor(self.codon2int(state[start:len(state)-end])).to(self.device).long()
            #print(new_encoding)
            new_encoding[topk]=max_preds[topk]
            #print(new_encoding)
            #print((encoding==new_encoding).sum()/20)
            #exit()

            return state[:start]+self.int2codon(new_encoding)+state[len(state)-end:len(state)]
        else:
            sequence_length=len(state)
            return state[:start]+self.codon_random_mutation(state[start:sequence_length-end],protein_mask)+state[sequence_length-end:len(state)]
            #return state

    def int2codon(self,sequence):
        rna_sequence=''
        for i in range(len(sequence)):
            rna_sequence+=self.protein_codon_table[sequence[i]]
        return rna_sequence

    def codon2int(self,sequence):
        int_sequence=[]
        for i in range(len(sequence)//3):
            codon=sequence[i*3:(i+1)*3]
            int_sequence.append(np.where(self.protein_codon_table==codon)[0])
        int_sequence=np.concatenate(int_sequence)
        return int_sequence

    def nucleatide2int(self,nt_sequence,target_length=None):
        int_sequence=[]
        for nt in nt_sequence:
            nt=nt.upper()
            if nt in nt_int:
                int_sequence.append(nt_int[nt])
        int_sequence=np.asarray(int_sequence,dtype='int32')
        if target_length:
            int_sequence=np.pad(int_sequence,(0,target_length-len(int_sequence)),constant_values=-1)
        return int_sequence

    def codon_random_mutation(self,rna_sequence,protein_mask):
        new_sequence=''
        positions=np.random.choice(len(rna_sequence)//3,self.k)
        for i in range(len(rna_sequence)//3):
            codon=rna_sequence[i*3:(i+1)*3]
            protein_index=np.where(protein_mask[i]>-1)[0]
            if i in positions:
            #if np.random.uniform() < p:

                # print(protein_index)
                # exit()
                #protein_letter=self.protein_letter_table[protein_index]
                #codon_indices=np.where(self.protein_letter_table==protein_letter)
                index=np.random.choice(protein_index)
                new_sequence+=self.protein_codon_table[index]
                #new_sequence+=codon
            elif len(protein_index)==1:
                new_sequence+=self.protein_codon_table[protein_index[0]]
            else:
                new_sequence+=codon
            #new_sequence+=codon
        #assert "ACCAAGUCGACACGGAUGAGUGUUUCCUUCUAUCCGCGUCAUUGUUAGGCCUUACGUACCGAUAGACCACUGGUGCAAAUAGAGAGCUAU" in new_sequence
        return new_sequence

    def protein2nt(self,sequence):
        new_sequence=''
        letters=np.asarray(self.codon_table.Letter)
        for i in range(len(sequence)):
            protein_letter=sequence[i]
            codon_indices=np.where(letters==protein_letter)
            index=np.random.choice(codon_indices[0])
            new_sequence+=self.codon_table.Codon[index]
        return new_sequence


    def get_protein_mask(self,nt_sequence,start,end):
        seq_len=len(nt_sequence)//3
        codons=list(self.codon_table.Codon)
        mask=np.ones((seq_len,64))*-1e18
        for i in range(seq_len):
            codon = nt_sequence[i*3:(i+1)*3]
            protein_index=np.where(self.protein_codon_table==codon)[0]
            protein_letter=self.protein_letter_table[protein_index]
            if i >=start//3 and i < (seq_len-end//3):
                vector=self.protein_letter_table==protein_letter
                mask[i,vector]=0
            else:
                index= codons.index(codon)
                mask[i,index]=0
        return mask


    def optimize_model(self,batch_size,start,end):
        steps=len(self.memory.memory)//batch_size
        #protein_mask=torch.Tensor(protein_mask).to(self.device).float()
        #protein_mask=protein_mask.expand(batch_size,*protein_mask.shape)
        self.policy_net.eval()
        total_loss=0
        if self.memory.position>self.memory_capacity:
            memoyr_start_position=self.memory.position-self.memory_capacity
        else:
            memoyr_start_position=0
        dataset=RNADataset(self.memory.position,self.memory.memory_folder,memoyr_start_position)
        #dataloader=DataLoader(dataset,batch_size=batch_size,num_workers=1,drop_last=True,shuffle=True)
        dataloader=DataLoader(dataset,batch_size=batch_size,num_workers=1,drop_last=True,shuffle=False)
        print('###optimizing model###')
        for epoch in range(self.epochs_per_episode):
            step=1
            for batch in tqdm(dataloader):

                state_batch = batch.current_sequence.to(self.device).long().squeeze(1)
                next_state_batch = batch.next_sequence.to(self.device).long().squeeze(1)
                state_output_batch = batch.current_protein_sequence.to(self.device).long().squeeze(1)
                next_state_output_batch = batch.next_protein_sequence.to(self.device).long().squeeze(1)
                reward_batch = batch.reward.to(self.device).float().squeeze(1)
                degradation_reward_batch = batch.degradation_reward.to(self.device).float().squeeze(1)
                degscore_reward_batch = batch.degscore_reward.to(self.device).float().squeeze(1)
                protein_mask = batch.protein_mask.to(self.device).float().squeeze(1)


                if self.degradation_reward:
                    state_action_values = self.policy_net(state_output_batch)

                    bpp_action_values, degradation_action_values, degscore_action_values=state_action_values[0].gather(-1,next_state_output_batch.unsqueeze(-1)),\
                                                                 state_action_values[1].gather(-1,next_state_output_batch.unsqueeze(-1)), \
                                                                 state_action_values[2].gather(-1,next_state_output_batch.unsqueeze(-1)),
                    #bpp_action_value=bpp_action_values+protein_mask
                    #degradation_action_values=degradation_action_values+protein_mask

                    with torch.no_grad():
                        next_state_values_full = self.target_net(next_state_output_batch)

                    next_bpp_action_values, next_degradation_action_values=next_state_values_full[0].gather(-1,next_state_output_batch.unsqueeze(-1)),\
                                                                 next_state_values_full[1].gather(-1,next_state_output_batch.unsqueeze(-1))

                    next_degscore_action_values=next_state_values_full[2].gather(-1,next_state_output_batch.unsqueeze(-1))

                    action_next_state_values_full=next_state_values_full

                    action_next_state_values_combined=action_next_state_values_full[0]+\
                                                      self.degradation_weight*action_next_state_values_full[1]+\
                                                      self.degradation_weight*action_next_state_values_full[2]+\
                                                      protein_mask

                    max_next_values, max_next_value_indices = action_next_state_values_combined.max(-1)

                    max_next_values = max_next_values.detach()

                    length=max_next_values.shape[1]

                    top_values, top_indices=torch.topk(max_next_values[:,start//3:length-end//3],k=self.k,dim=-1)

                    top_indices=top_indices+start//3

                    #values.gather(-1,indices1.unsqueeze(1)).squeeze(1).gather(-1,indices2).shape

                    top_next_bpp_action_values=next_state_values_full[0].gather(-1,max_next_value_indices.unsqueeze(1)).squeeze(1).gather(-1,top_indices)
                    next_bpp_action_values=next_bpp_action_values.squeeze(-1).scatter_(-1,top_indices,top_next_bpp_action_values.detach())
                    expected_bpp_state_action_values = (next_bpp_action_values * self.gamma) + reward_batch
                    loss = F.mse_loss(bpp_action_values, expected_bpp_state_action_values.unsqueeze(-1))



                    top_next_degradation_action_values=next_state_values_full[1].gather(-1,max_next_value_indices.unsqueeze(1)).squeeze(1).gather(-1,top_indices)
                    next_degradation_action_values=next_degradation_action_values.squeeze(-1).scatter_(-1,top_indices,top_next_degradation_action_values.detach())
                    expected_degradation_state_action_values = (next_degradation_action_values * self.gamma) + degradation_reward_batch
                    loss += F.mse_loss(degradation_action_values, expected_degradation_state_action_values.unsqueeze(-1))*self.degradation_weight

                    top_next_degscore_action_values=next_state_values_full[2].gather(-1,max_next_value_indices.unsqueeze(1)).squeeze(1).gather(-1,top_indices)
                    next_degscore_action_values=next_degscore_action_values.squeeze(-1).scatter_(-1,top_indices,top_next_degscore_action_values.detach())
                    expected_degscore_state_action_values = (next_degscore_action_values * self.gamma) + degscore_reward_batch
                    loss += F.mse_loss(degscore_action_values, expected_degscore_state_action_values.unsqueeze(-1))*self.degradation_weight



                else:
                    state_action_values = self.policy_net(state_output_batch).gather(-1,next_state_output_batch.unsqueeze(-1))
                    
                    with torch.no_grad():
                        next_state_values_full = (self.target_net(next_state_output_batch)+protein_mask)

                    next_state_values = next_state_values_full.gather(-1,next_state_output_batch.unsqueeze(-1)).squeeze(-1)


                    action_next_state_values_full= next_state_values_full #(self.policy_net(next_state_output_batch)+protein_mask)

                    max_next_values, max_next_value_indices = action_next_state_values_full.max(-1)

                    max_next_values = max_next_values.detach()


                    length=max_next_values.shape[1]
                    top_values, top_indices=torch.topk(max_next_values[:,start//3:length-end//3],k=self.k,dim=-1)

                    #print(top_indices)
                    top_indices=top_indices+start//3

                    next_state_values=next_state_values.scatter_(-1,top_indices,top_values)


                    expected_state_action_values = (next_state_values * self.gamma) + reward_batch

                    loss = F.mse_loss(state_action_values, expected_state_action_values.unsqueeze(-1))
                total_loss += loss.item()
                #print(loss)
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy_net.parameters(),1)
                self.optimizer.step()
                if self.scheduler is not None:
                    self.scheduler.step()
                    
                if step%10==0:
                    self.target_net.load_state_dict(self.policy_net.state_dict())
                step+=1
                # print(state_action_values.shape)
                # exit()
                #return state_action_values, next_state_batch

        self.target_net.load_state_dict(self.policy_net.state_dict())
        return total_loss/len(dataloader)
