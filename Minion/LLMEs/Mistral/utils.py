import torch
import openai
import random
import time
import numpy as np
import torch.nn.functional as F
from data_loader import get_data_loader_BERT
from nltk import word_tokenize
from retry import retry

class Moment:
    def __init__(self, config) -> None:
        self.config = config
        self.features = None
        self.features_des = None

        self.labels = None
        self.mem_samples = None
        self.mem_features = None
        self.mem_labels = None
        self.mem_features_des = None
        self.sample_k = config.sample_k
        self.temperature = config.contrastive_temp
        self.m = config.margin
        # self.mlp = torch.nn.Linear(4096, 768).to(config.device)

    def init_moment(self, encoder, dataset, seen_des, id2rel, is_memory=False, is_llm = False):
        encoder.eval()
        datalen = len(dataset)
        if not is_memory:
            self.features = torch.zeros(datalen, self.config.encoder_output_size, dtype=torch.bfloat16)
            self.features_des = torch.zeros(datalen, self.config.encoder_output_size, dtype=torch.bfloat16)

            data_loader = get_data_loader_BERT(self.config, dataset) # shuffle=False
            lbs = []
            for step, (instance, labels, ind) in enumerate(data_loader):
                for k in instance.keys():
                    if isinstance(instance[k], list):
                        continue
                    else:
                        instance[k] = instance[k].to(self.config.device)
                    

                if is_llm:
                    hidden = encoder(instance['input'])

                    batch_instance = {'input': []}
                    batch_instance['input'] = [seen_des[id2rel[label.item()]]['input'] for label in labels]
                    hidden_des = encoder(batch_instance['input'])
                else:
                    hidden = encoder(instance)
                    batch_instance = {'ids': [], 'mask': []} 
                    batch_instance['ids'] = torch.tensor([seen_des[id2rel[label.item()]]['ids'] for label in labels]).to(self.config.device)
                    batch_instance['mask'] = torch.tensor([seen_des[id2rel[label.item()]]['mask'] for label in labels]).to(self.config.device)
                    hidden_des = encoder(batch_instance)
                fea = hidden.detach().cpu().data
                fea_des = hidden_des.detach().cpu().data
                self.update_des(ind, fea, fea_des)

                lbs.append(labels) # shuffle=False
            lbs = torch.cat(lbs)
            self.labels = lbs
        else:
            self.mem_samples = dataset
            self.mem_features = torch.zeros(datalen, self.config.encoder_output_size, dtype=torch.bfloat16)
            self.mem_features_des = torch.zeros(datalen, self.config.encoder_output_size, dtype=torch.bfloat16)
            data_loader = get_data_loader_BERT(self.config, dataset) # shuffle=False
            lbs = []
            for step, (instance, labels, ind) in enumerate(data_loader):
                for k in instance.keys():
                    if isinstance(instance[k], list):
                        continue
                    else:
                        instance[k] = instance[k].to(self.config.device)
                if is_llm:
                    hidden = encoder(instance['input'])

                    batch_instance = {'input': []}
                    batch_instance['input'] = [seen_des[id2rel[label.item()]]['input'] for label in labels]
                    hidden_des = encoder(batch_instance['input'])
                else:
                    hidden = encoder(instance)
                    batch_instance = {'ids': [], 'mask': []} 
                    batch_instance['ids'] = torch.tensor([seen_des[id2rel[label.item()]]['ids'] for label in labels]).to(self.config.device)
                    batch_instance['mask'] = torch.tensor([seen_des[id2rel[label.item()]]['mask'] for label in labels]).to(self.config.device)
                    hidden_des = encoder(batch_instance)
                fea = hidden.detach().cpu().data
                fea_des = hidden_des.detach().cpu().data
                self.update_des(ind, fea, fea_des)

                lbs.append(labels) # shuffle=False
            lbs = torch.cat(lbs)
            self.mem_labels = lbs            

    def update(self, ind, feature, is_memory=False):
        if not is_memory:
            self.features[ind] = feature
        else:
            self.mem_features[ind] = feature

    def update_des(self, ind, feature, feature_des, is_memory=False):
        if not is_memory:
            self.features[ind] = feature
            self.features_des[ind] = feature_des
        else:
            self.mem_features[ind] = feature
            self.mem_features_des[ind] = feature_des

    def update_allmem(self, encoder, is_llm = False):
            data_loader = get_data_loader_BERT(self.config, self.mem_samples, batch_size=64) # shuffle=False
            for step, (instance, labels, ind) in enumerate(data_loader):
                for k in instance.keys():
                    if isinstance(instance[k], list):
                        continue
                    else:
                        instance[k] = instance[k].to(self.config.device)

                if is_llm:
                    hidden = encoder(instance['input'])
                else:
                    hidden = encoder(instance)
                fea = hidden.detach().cpu().data
                self.update(ind, fea, is_memory=True)
        

    def get_mem_proto(self):
        cinds = []
        for x in self.mem_labels:
            if x.item() not in cinds:
                cinds.append(x.item())

        num = len(cinds)
        feats = self.mem_features
        centroids = torch.zeros((num, feats.size(1)), dtype=torch.float32, device=feats.device)
        for i, c in enumerate(cinds):
            ind = np.where(self.mem_labels.cpu().numpy() == c)[0]
            centroids[i, :] = feats[ind, :].mean(dim=0)
        return centroids

     # MCL loss
    def contrastive_loss(self, x, labels, is_memory=False, des= None, relation_2_cluster = None):
        '''
        x (B, H)
        '''
        if is_memory:
            ct_x = self.mem_features.to(self.config.device)
            ct_x_des = self.mem_features_des.to(self.config.device)
            ct_y = self.mem_labels
        else:
            idx = list(range(len(self.features)))
            if len(idx) > self.sample_k:
                sample_id = random.sample(idx, self.sample_k)
            else:  # sample number > total feature
                sample_id = idx
            ct_x = self.features[sample_id].to(self.config.device) # (N, H)
            ct_x_des = self.features_des[sample_id].to(self.config.device)
            ct_y = self.labels[sample_id] # (N)

        # l2 normalize
        x = F.normalize(x, p=2, dim=1)
        ct_x = F.normalize(ct_x, p=2, dim=1)
        
        t1 = torch.mm(x, ct_x.T) + 1 # 0 <= cos + 1 <= 2
        
        if des != None:
            des = F.normalize(des, p=2, dim=1)
            ct_x_des = F.normalize(ct_x_des, p=2, dim=1)
            t2 = torch.mm(des, ct_x_des.T) 

        zeros = (torch.zeros_like(t1)).to(self.config.device)
        

        
        pos = torch.ones_like(t1)
        neg = torch.ones_like(t1)  # Initialize neg with default value of 1

        if relation_2_cluster is not None:
            # Convert `relation_2_cluster` values into tensors for vectorized operations
            # Get clusters for labels and ct_y
            labels_clusters = torch.tensor([relation_2_cluster[label.item()] for label in labels], device=self.config.device)
            ct_y_clusters = torch.tensor([relation_2_cluster[label.item()] for label in ct_y], device=self.config.device)
            
            # Compare clusters using broadcasting
            # labels_clusters[:, None] expands `labels_clusters` to (B, 1), allowing broadcasting with ct_y_clusters (N)
            relation_match = (labels_clusters.unsqueeze(1) == ct_y_clusters.unsqueeze(0)).float()
            
            # Update neg based on the relation match mask
            neg = relation_match * (1.0 + 0.6* t2 - 0.3)+ (1.0 - relation_match) * 1.0
            # pos = 0.6*t2 + self.m
            # neg = 0.6*t2 + 1 - self.m



        # pos = 0.6*t2 + self.m
        # neg = 0.6*t2 + 1 - self.m

        # if relation_2_cluster != None:
        #     for i in range(x.shape[0]):
        #         for j in range(ct_x.shape[0]):
        #             if relation_2_cluster[labels[i].item()] == relation_2_cluster[ct_y[j].item()]:
        #                 neg[i, j] = 1.0 + 0.5*t2[i, j]  
        #             else:
        #                 neg[i, j] = 1.0 

        dot_product_tempered_pos = torch.where(pos > 0, pos * t1 / self.temperature, zeros)
        dot_product_tempered_neg = torch.where(neg > 0, neg * t1 / self.temperature, zeros)
        
        exp_dot_tempered_pos = (
            torch.exp(dot_product_tempered_pos - \
                torch.max(dot_product_tempered_pos, dim=1, keepdim=True)[0].detach()) + 1e-5
        )
        exp_dot_tempered_neg = (
            torch.exp(dot_product_tempered_neg - \
                torch.max(dot_product_tempered_pos, dim=1, keepdim=True)[0].detach()) + 1e-5
        )
        mask_combined_pos = (labels.unsqueeze(1).repeat(1, ct_y.shape[0]) == ct_y).to(self.config.device)

        mask_combined_neg = ~mask_combined_pos
        cardinality_per_samples = torch.sum(mask_combined_pos, dim=1)

        sum_temp = torch.sum(exp_dot_tempered_pos * mask_combined_pos, dim=1, keepdim=True) \
            + torch.sum(exp_dot_tempered_neg * mask_combined_neg, dim=1, keepdim=True)
        log_prob = -torch.log(exp_dot_tempered_pos / sum_temp)
        supervised_contrastive_loss_per_sample = torch.sum(log_prob * mask_combined_pos, dim=1) / cardinality_per_samples
        supervised_contrastive_loss = torch.mean(supervised_contrastive_loss_per_sample)

        return supervised_contrastive_loss

#  pos = 0.6*t2 + self.m
#         neg = 0.6*t2 + 1 - self.m



    # GOOD
    def contrastive_loss_des(self, des, x, labels, is_memory = False, labels2 = None, temp=0.1, is_augment=False): 

        """
        Computes the supervised contrastive loss between pairs (x, des) based on their labels.

        Args:
            x (torch.Tensor): Tensor of shape (batch_size, feature_dim) representing anchor features.
            des (torch.Tensor): Tensor of shape (batch_size, feature_dim) representing description features.
            labels (torch.Tensor): Tensor of shape (batch_size,) containing the labels of each pair.
            is_memory (bool): Flag indicating whether to consider memory mechanism (optional).

        Returns:
            torch.Tensor: The computed supervised contrastive loss.
        """
        # Normalize the input feature vectors
        x = F.normalize(x, p=2, dim=1)
        des = F.normalize(des, p=2, dim=1)

        # Compute similarity matrix (batch_size x batch_size)
        sim_matrix = torch.matmul(x, des.T) / temp

        # Create a mask for positive pairs (same label) and negative pairs (different label)
        if is_augment:
           
            labels = labels.unsqueeze(1)
            labels2 = labels2.unsqueeze(1)
          

            positive_mask = torch.eq(labels, labels2.T).float().to(self.config.device)  # Positive pairs
            # Compute exp(similarity) and mask
            exp_sim = torch.exp(sim_matrix)

            pos_exp_sim = exp_sim * positive_mask
            neg_exp_sim = exp_sim * (1 - positive_mask)

            # Compute the sum of the numerator (positive pairs) and the denominator (all pairs)
            numerator = pos_exp_sim.sum(dim=1)
            denominator = pos_exp_sim.sum(dim=1) + neg_exp_sim.sum(dim=1)
           
            # Compute the loss, ensuring numerical stability
            loss = -torch.log(numerator / (denominator + 1e-6))

        else:
           

            labels = labels.unsqueeze(1)
            positive_mask = torch.eq(labels, labels.T).float().to(self.config.device)  # Positive pairs

            # Compute exp(similarity) and mask
            exp_sim = torch.exp(sim_matrix)
            pos_exp_sim = exp_sim * positive_mask
            neg_exp_sim = exp_sim * (1 - positive_mask)

            # Compute the sum of the numerator (positive pairs) and the denominator (all pairs)
            numerator = pos_exp_sim.sum(dim=1)
            denominator = pos_exp_sim.sum(dim=1) + neg_exp_sim.sum(dim=1)


            # Compute the loss, ensuring numerical stability
            loss = -torch.log(numerator / (denominator + 1e-6))

        return loss.mean()
    
    def mutual_information_loss(self, x_bert, x_stella, labels,  temperature=0.1):
        mask = labels.unsqueeze(1) == labels.unsqueeze(0)  # Shape: (batch_size, batch_size)
        mask = mask.to(self.config.device)
        x_bert = F.normalize(x_bert, p=2, dim=1)
        x_stella = F.normalize(x_stella, p=2, dim=1)
        

        
        
        similarity_matrix = torch.matmul(x_bert, x_stella.t()) / temperature # Shape: (batch_size, batch_size)

        f_pos = torch.diag(similarity_matrix)  # Shape: (batch_size,)
        f_neg = similarity_matrix*(~mask)
        f_concat = torch.cat([f_pos.unsqueeze(1), f_neg], dim=1)  # Shape: (batch_size, 1 + num_negatives)

        # f_concat = torch.log(torch.clamp(f_concat, min=1e-9).to(device))

        softmax_probs = torch.nn.functional.softmax(f_concat, dim=1)

        infoNCE_loss = -torch.log(softmax_probs[:, 0]).mean()

        return infoNCE_loss
    

    def mutual_information_loss_cluster(self, x_bert, x_stella, labels,  temperature=0.1,  relation_2_cluster = None):
        mask = labels.unsqueeze(1) == labels.unsqueeze(0)  # Shape: (batch_size, batch_size)
        mask = mask.to(self.config.device)
        x_bert = F.normalize(x_bert, p=2, dim=1)
        x_stella = F.normalize(x_stella, p=2, dim=1)
        
        t2 = torch.mm(x_stella, x_stella.T) + 1

        similarity_matrix = torch.matmul(x_bert, x_stella.t()) / temperature # Shape: (batch_size, batch_size)

        if relation_2_cluster is not None:
            
            labels_clusters = torch.tensor([relation_2_cluster[label.item()] for label in labels], device=self.config.device)
        
            relation_match = (labels_clusters.unsqueeze(1) == labels_clusters.unsqueeze(0)).float()
            
            neg = relation_match * (1.0 + 0.2*t2) + (1.0 - relation_match) * 1.0
            # pos = 0.6*t2 + self.m
            # neg = 0.6*t2 + 1 - self.m

        f_pos = torch.diag(similarity_matrix)  # Shape: (batch_size,)
        f_neg = similarity_matrix*(~mask)*neg
        f_concat = torch.cat([f_pos.unsqueeze(1), f_neg], dim=1)  # Shape: (batch_size, 1 + num_negatives)
        # f_concat = torch.log(torch.clamp(f_concat, min=1e-9).to(device))
        softmax_probs = torch.nn.functional.softmax(f_concat, dim=1)

        infoNCE_loss = -torch.log(softmax_probs[:, 0]).mean()

        return infoNCE_loss
    #  def mutual_information_loss(self, x_bert, x_stella, labels):
    #     infoNCE_loss = 0
    #     for i in range(x_bert.shape[0]):

    #         negatives_sample_indexs = torch.where(labels != labels[i])[0]

    #         x_bert_pos = x_bert[i].unsqueeze(0)
    #         pos_hidden = x_stella[i].unsqueeze(0) # 1,768 
    #         negatives_hidden = x_stella[negatives_sample_indexs] # number_neg, 768

    #         f_pos = torch.matmul(x_bert_pos, pos_hidden.t())
    #         f_neg = torch.matmul(x_bert_pos, negatives_hidden.t())

    #         f_concat = torch.cat([f_pos, f_neg], dim=1).squeeze()
    #         f_concat = torch.log(torch.max(f_concat , torch.tensor(1e-9).to(self.config.device)))
    #         try:
    #             infoNCE_loss += -torch.log(torch.nn.functional.softmax(f_concat)[0] )
    #         except:
    #             None

    #     return infoNCE_loss / x_bert.shape[0]  
# # for openai
# @retry(tries=10, delay=1)
# def gpt(input, t=0, key=None):
#     MAX_TRIES = 15
#     client = openai.OpenAI(api_key=)
    
#     while MAX_TRIES > 0:
#         try:
#             time.sleep(5)
#             completion = client.chat.completions.create(
#             model="gpt-4o-mini",
#             messages=[
#                 {"role": "user", "content": input}
#             ]
#             ,
#             temperature=t

#             )
#             return completion.choices[0].message.content
#         except Exception as e:
#             print(e)
#             MAX_TRIES -= 1
#     print('gen failed')
#     return ''
    




# def parse(rel2id, text):
#     cons = ['Relation:', 'Context:', 'Head Entity:', 'Tail Entity:']
#     lens = [len(item) for item in cons]
#     parse_text = []

#     temp = text
#     while True:
#         parse_item = {}

#         i = temp.find(cons[0])
#         temp = temp[i + lens[0]:]
#         i = temp.find(cons[1])
#         r = temp[:i].strip()
#         temp = temp[i + lens[1]:]
#         i = temp.find(cons[2])
#         c = temp[:i].strip()
#         temp = temp[i + lens[2]:]
#         i = temp.find(cons[3])
#         h = temp[:i].strip()
#         temp = temp[i + lens[3]:]
#         i = temp.find('\n')
#         t = temp[:i].strip()
#         i = temp.find(cons[0])

#         r = r.split('\n')[0]
#         r = r.replace('**', '')
#         r = r.replace('\n', '')
#         r = r.strip()

#         parse_item['relation'] = rel2id[r]
#         parse_item['index'] = 0
#         tokens = word_tokenize(c.lower())
#         parse_item['tokens'] = tokens

#         headent, tailent = h.lower(), t.lower()
#         h_tokens, t_tokens = word_tokenize(headent), word_tokenize(tailent)
#         try:
#             h1 = tokens.index(h_tokens[0])
#         except Exception:
#             h1 = 0
#         try:
#             h2 = tokens.index(h_tokens[-1])
#         except Exception:
#             h2 = h1
#         try:
#             t1 = tokens.index(t_tokens[0])
#         except Exception:
#             t1 = h2
#         try:
#             t2 = tokens.index(t_tokens[-1])
#         except Exception:
#             t2 = t1
#         parse_item['h'] = [headent, '0', [[h1, h2]]]
#         parse_item['t'] = [tailent, '0', [[t1, t2]]]

#         parse_text.append(parse_item)

#         if i == -1:
#             break
#         temp = temp[i:]

#     return parse_text


# def prompt_input(rname, rdesc, sample=None, n=10):
#     pre_input = 'You are a data scientist working on a relation extraction task. Please do the following task and do not give output in the markdown format.'
#     input = ''
#     if sample == None:
#         input = 'One sample in relation extraction datasets consists of a relation, a context, a pair of head and tail entities in the context.The head entity has the relation with the tail entity. Generate ' \
#                 + str(
#             n) + ' diversity samples (must have full : Relation , Context , Head Entity , Tail Entity) for the relation "' + rname \
#                 + '" which means ' + rdesc \
#                 + ', and indicate the head entity and tail entity in the following format:\n' \
#                 + 'Relation: xxx\nContext: xxx\nHead Entity: xxx\nTail Entity: xxx'
#     else:
#         input = 'One sample in relation extraction datasets consists of a relation, a context, a pair of head and tail entities in the context.The head entity has the relation with the tail entity.\n' \
#                 + 'Relation "' + rname + '" means ' + rdesc + '.\nHere is an example:\n' \
#                 + 'Relation: ' + rname + '\nContext: ' + sample['tokens'] + '\nHead Entity: ' + sample[
#                     'h'] + '\nTail Entity: ' + sample['t'] + '\n' \
#                 + 'Please generate ' + str(
#             n) + ' diversity samples (must have full : Relation , Context , Head Entity , Tail Entity) like the above example for the relation "' + rname + '":'
#     return pre_input + input


# def gen_data(r2desc, rel2id, sample, n=10, t=0, key=None):
#     import random
#     MAX_TRIES = 15
#     rname = sample['relation']
#     rdesc = r2desc[rname]
#     print('####', rname, '####')
#     input = prompt_input(rname, rdesc, sample=sample, n=n)
#     print(input)
#     output = gpt(input=input, t=t, key=key)
#     print(output)
#     while MAX_TRIES > 0:
#         try:
#             parse_output = parse(rel2id, output)
#             return parse_output
#         except Exception as e:
#             print(e)
#             t = random.uniform(0.5, 1)
#             output = gpt(input=input + "\nRelation: ", t=t, key=key)
#             MAX_TRIES -= 1

#     return ''









