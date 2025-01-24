from __future__ import annotations
from typing import Any, Iterable
import torch
from torch import Tensor, nn
import numpy as np 
from enum import Enum
from torch.nn import functional as F


def _convert_to_tensor(a: list | np.ndarray | Tensor):
    if not isinstance(a, Tensor):
        a = torch.tensor(a)
    return a

def _convert_to_batch(a: Tensor):
    if a.dim() == 1:
        a = a.unsqueeze(0)
    return a


def _convert_to_batch_tensor(a: list | np.ndarray | Tensor):
    a = _convert_to_tensor(a)
    a = _convert_to_batch(a)
    return a

def normalize_embeddings(embeddings: Tensor):
    return torch.nn.functional.normalize(embeddings, p=2, dim=1)

def cos_sim(a: list | np.ndarray | Tensor, b: list | np.ndarray | Tensor):
    a = _convert_to_batch_tensor(a)
    b = _convert_to_batch_tensor(b)

    a_norm = normalize_embeddings(a)
    b_norm = normalize_embeddings(b)
    return torch.mm(a_norm, b_norm.transpose(0, 1))

class MutualInformationLoss(nn.Module):
    def __init__(self, scale: float = 20.0, similarity_fct=cos_sim):
        super().__init__()
        # self.model = model
        self.scale = scale
        self.similarity_fct = similarity_fct
        self.cross_entropy_loss = nn.CrossEntropyLoss()

    def forward(self, embeddings_a, embeddings_b, labels) -> Tensor:
        scores = self.similarity_fct(embeddings_a, embeddings_b) * self.scale
        return self.cross_entropy_loss(scores, labels)

    def get_config_dict(self) -> dict[str, Any]:
        return {"scale": self.scale, "similarity_fct": self.similarity_fct.__name__}



class SiameseDistanceMetric(Enum):
    """The metric for the contrastive loss"""

    EUCLIDEAN = lambda x, y: F.pairwise_distance(x, y, p=2)
    MANHATTAN = lambda x, y: F.pairwise_distance(x, y, p=1)
    COSINE_DISTANCE = lambda x, y: 1 - F.cosine_similarity(x, y)



class BatchHardTripletLossDistanceFunction:

    @staticmethod
    def cosine_distance(embeddings: Tensor) :
        return 1 - cos_sim(embeddings, embeddings)

    @staticmethod
    def eucledian_distance(embeddings: Tensor, squared=False):

        dot_product = torch.matmul(embeddings, embeddings.t())

        square_norm = torch.diag(dot_product)

        # Compute the pairwise distance matrix as we have:
        # ||a - b||^2 = ||a||^2  - 2 <a, b> + ||b||^2
        # shape (batch_size, batch_size)
        distances = square_norm.unsqueeze(0) - 2.0 * dot_product + square_norm.unsqueeze(1)

        # Because of computation errors, some distances might be negative so we put everything >= 0.0
        distances[distances < 0] = 0

        if not squared:
            # Because the gradient of sqrt is infinite when distances == 0.0 (ex: on the diagonal)
            # we need to add a small epsilon where distances == 0.0
            mask = distances.eq(0).float()
            distances = distances + mask * 1e-16

            distances = (1.0 - mask) * torch.sqrt(distances)

        return distances


class BatchHardTripletLoss(nn.Module):
    def __init__(
        self,
        distance_metric=BatchHardTripletLossDistanceFunction.eucledian_distance,
        margin: float = 5,
    ):
        super().__init__()
        self.triplet_margin = margin
        self.distance_metric = distance_metric

    def forward(self, rep : Tensor, labels: Tensor):
        # rep = self.sentence_embedder(sentence_features[0])["sentence_embedding"]
        return self.batch_hard_triplet_loss(labels, rep)

    def batch_hard_triplet_loss(self, labels: Tensor, embeddings: Tensor) :
        # Get the pairwise distance matrix
        pairwise_dist = self.distance_metric(embeddings)

        # For each anchor, get the hardest positive
        # First, we need to get a mask for every valid positive (they should have same label)
        mask_anchor_positive = BatchHardTripletLoss.get_anchor_positive_triplet_mask(labels).float()

        # We put to 0 any element where (a, p) is not valid (valid if a != p and label(a) == label(p))
        anchor_positive_dist = mask_anchor_positive * pairwise_dist

        # shape (batch_size, 1)
        hardest_positive_dist, _ = anchor_positive_dist.max(1, keepdim=True)

        # For each anchor, get the hardest negative
        # First, we need to get a mask for every valid negative (they should have different labels)
        mask_anchor_negative = BatchHardTripletLoss.get_anchor_negative_triplet_mask(labels).float()

        # We add the maximum value in each row to the invalid negatives (label(a) == label(n))
        max_anchor_negative_dist, _ = pairwise_dist.max(1, keepdim=True)
        anchor_negative_dist = pairwise_dist + max_anchor_negative_dist * (1.0 - mask_anchor_negative)

        # shape (batch_size,)
        hardest_negative_dist, _ = anchor_negative_dist.min(1, keepdim=True)

        # Combine biggest d(a, p) and smallest d(a, n) into final triplet loss
        tl = hardest_positive_dist - hardest_negative_dist + self.triplet_margin
        tl[tl < 0] = 0
        triplet_loss = tl.mean()

        return triplet_loss

    @staticmethod
    def get_triplet_mask(labels: Tensor) :
        # Check that i, j and k are distinct
        indices_equal = torch.eye(labels.size(0), device=labels.device).bool()
        indices_not_equal = ~indices_equal
        i_not_equal_j = indices_not_equal.unsqueeze(2)
        i_not_equal_k = indices_not_equal.unsqueeze(1)
        j_not_equal_k = indices_not_equal.unsqueeze(0)

        distinct_indices = (i_not_equal_j & i_not_equal_k) & j_not_equal_k

        label_equal = labels.unsqueeze(0) == labels.unsqueeze(1)
        i_equal_j = label_equal.unsqueeze(2)
        i_equal_k = label_equal.unsqueeze(1)

        valid_labels = ~i_equal_k & i_equal_j

        return valid_labels & distinct_indices

    @staticmethod
    def get_anchor_positive_triplet_mask(labels: Tensor) :
        # Check that i and j are distinct
        indices_equal = torch.eye(labels.size(0), device=labels.device).bool()
        indices_not_equal = ~indices_equal

        # Check if labels[i] == labels[j]
        # Uses broadcasting where the 1st argument has shape (1, batch_size) and the 2nd (batch_size, 1)
        labels_equal = labels.unsqueeze(0) == labels.unsqueeze(1)

        return labels_equal & indices_not_equal

    @staticmethod
    def get_anchor_negative_triplet_mask(labels: Tensor) :
        # Check if labels[i] != labels[k]
        # Uses broadcasting where the 1st argument has shape (1, batch_size) and the 2nd (batch_size, 1)

        return ~(labels.unsqueeze(0) == labels.unsqueeze(1))

class HardSoftMarginTripletLoss(BatchHardTripletLoss):
    def __init__(
        self, distance_metric=BatchHardTripletLossDistanceFunction.eucledian_distance
    ):
        super().__init__()
        self.distance_metric = distance_metric

    def forward(self, rep : Tensor, labels: Tensor):
        return self.batch_hard_triplet_soft_margin_loss(labels, rep)

    def batch_hard_triplet_soft_margin_loss(self, labels: Tensor, embeddings: Tensor):
        # Get the pairwise distance matrix
        pairwise_dist = self.distance_metric(embeddings)

        # For each anchor, get the hardest positive
        # First, we need to get a mask for every valid positive (they should have same label)
        mask_anchor_positive = BatchHardTripletLoss.get_anchor_positive_triplet_mask(labels).float()

        # We put to 0 any element where (a, p) is not valid (valid if a != p and label(a) == label(p))
        anchor_positive_dist = mask_anchor_positive * pairwise_dist

        # shape (batch_size, 1)
        hardest_positive_dist, _ = anchor_positive_dist.max(1, keepdim=True)

        # For each anchor, get the hardest negative
        # First, we need to get a mask for every valid negative (they should have different labels)
        mask_anchor_negative = BatchHardTripletLoss.get_anchor_negative_triplet_mask(labels).float()

        # We add the maximum value in each row to the invalid negatives (label(a) == label(n))
        max_anchor_negative_dist, _ = pairwise_dist.max(1, keepdim=True)
        anchor_negative_dist = pairwise_dist + max_anchor_negative_dist * (1.0 - mask_anchor_negative)

        # shape (batch_size,)
        hardest_negative_dist, _ = anchor_negative_dist.min(1, keepdim=True)

        # Combine biggest d(a, p) and smallest d(a, n) into final triplet loss with soft margin
        # tl = hardest_positive_dist - hardest_negative_dist + margin
        # tl[tl < 0] = 0
        tl = torch.log1p(torch.exp(hardest_positive_dist - hardest_negative_dist))
        triplet_loss = tl.mean()

        return triplet_loss

# HardMarginLoss
class HardMarginLoss(nn.Module):
    def __init__(
        self, distance_metric=SiameseDistanceMetric.COSINE_DISTANCE, margin: float = 0.5
    ) :
        super().__init__()
        self.margin = margin
        self.distance_metric = distance_metric

    def forward(self, rep_des, hidden, labels: Tensor, size_average=False):
       
        distance_matrix = self.distance_metric(rep_des, hidden)
        negs = distance_matrix[labels == 0]
        poss = distance_matrix[labels == 1]

        # select hard positive and hard negative pairs
        negative_pairs = negs[negs < (poss.max() if len(poss) > 1 else negs.mean())]
        positive_pairs = poss[poss > (negs.min() if len(negs) > 1 else poss.mean())]

        positive_loss = positive_pairs.pow(2).sum()
        negative_loss = F.relu(self.margin - negative_pairs).pow(2).sum()
        loss = positive_loss + negative_loss
        return loss
    



class TripletDistanceMetric(Enum):
    """The metric for the triplet loss"""

    COSINE = lambda x, y: 1 - F.cosine_similarity(x, y)
    EUCLIDEAN = lambda x, y: F.pairwise_distance(x, y, p=2)
    MANHATTAN = lambda x, y: F.pairwise_distance(x, y, p=1)


class TripletLoss(nn.Module):
    def __init__(
        self, distance_metric=TripletDistanceMetric.COSINE, triplet_margin: float = 1.0
    ) -> None:
        """
        This class implements triplet loss. Given a triplet of (anchor, positive, negative),
        the loss minimizes the distance between anchor and positive while it maximizes the distance
        between anchor and negative. It compute the following loss function:

        ``loss = max(||anchor - positive|| - ||anchor - negative|| + margin, 0)``.

        Margin is an important hyperparameter and needs to be tuned respectively.

        Args:
            model: SentenceTransformerModel
            distance_metric: Function to compute distance between two
                embeddings. The class TripletDistanceMetric contains
                common distance metrices that can be used.
            triplet_margin: The negative should be at least this much
                further away from the anchor than the positive.

        """
        super().__init__()
        # self.model = model
        self.distance_metric = distance_metric
        self.triplet_margin = triplet_margin

    def forward(self, rep_anchor, rep_pos, rep_neg) -> Tensor:

        distance_pos = self.distance_metric(rep_anchor, rep_pos)
        distance_neg = self.distance_metric(rep_anchor, rep_neg)
    
        losses = F.relu(distance_pos - distance_neg + self.triplet_margin)
       
        # losses = torch.where(losses == self.triplet_margin, torch.tensor(0.0, device=losses.device), losses)

        return losses.mean()

#add this

#tuning
import torchvision

import math
import random
from pytorch_metric_learning import miners, losses
from torch.nn import Parameter

num_of_data = {
    "organization founded": 268,
    "organization subsidiaries": 453,
    "person date of birth": 103,
    "organization city of headquarters": 573,
    "person age": 833,
    "person charges": 280,
    "person countries of residence": 819,
    "person country of birth": 53,
    "person stateorprovinces of residence": 484,
    "organization website": 223,
    "person cities of residence": 742,
    "person parents": 296,
    "person employee of": 2163,
    "person city of birth": 103,
    "organization parents": 444,
    "organization political religious affiliation": 125,
    "person schools attended": 229,
    "person country of death": 61,
    "person children": 347,
    "organization top members employees": 2770,
    "person date of death": 394,
    "organization members": 0,
    "organization alternate names": 1359,
    "person religion": 286,
    "organization member of": 171,
    "person cause of death": 337,
    "person origin": 667,
    "organization shareholders": 144,
    "person stateorprovince of birth": 72,
    "person title": 3862,
    "organization number of employees members": 121,
    "organization dissolved": 33,
    "organization country of headquarters": 753,
    "person alternate names": 1359,
    "person siblings": 250,
    "organization stateorprovince of headquarters": 350,
    "person spouse": 483,
    "person other family": 319,
    "person city of death": 227,
    "person stateorprovince of death": 104,
    "organization founded by": 268,
}

relation_dict_with_ids = {i: value for i, (key, value) in enumerate(num_of_data.items())}

def binarize(T, nb_classes):
    T = T.cpu().numpy()
    import sklearn.preprocessing
    T = sklearn.preprocessing.label_binarize(
        T, classes = range(0, nb_classes)
    )
    T = torch.FloatTensor(T).cuda()
    return T


def l2_norm(input):
    input_size = input.size()
    buffer = torch.pow(input, 2)
    normp = torch.sum(buffer, 1).add_(1e-12)
    norm = torch.sqrt(normp)
    _output = torch.div(input, norm.view(-1, 1).expand_as(input))
    output = _output.view(input_size)
    return output

class Proxy_Anchor(torch.nn.Module):
    def __init__(self, nb_classes, sz_embed, mrg = 0.1, alpha = 32):
        torch.nn.Module.__init__(self)
        # Proxy Anchor Initialization
        self.proxies = torch.nn.Parameter(torch.randn(nb_classes, sz_embed).cuda())
        nn.init.kaiming_normal_(self.proxies, mode='fan_out')

        self.nb_classes = nb_classes
        self.sz_embed = sz_embed
        self.mrg = mrg
        self.alpha = alpha
        
    def forward(self, X, T):
        P = self.proxies

        cos = F.linear(l2_norm(X), l2_norm(P) ) # Calcluate cosine similarity
        P_one_hot = binarize(T = T, nb_classes = self.nb_classes)
        N_one_hot = 1 - P_one_hot
    
        pos_exp = torch.exp(-self.alpha * (cos - self.mrg))
        neg_exp = torch.exp(self.alpha * (cos + self.mrg))

        with_pos_proxies = torch.nonzero(P_one_hot.sum(dim = 0) != 0).squeeze(dim = 1)   # The set of positive proxies of data in the batch
        num_valid_proxies = len(with_pos_proxies)   # The number of positive proxies
        
        P_sim_sum = torch.where(P_one_hot == 1, pos_exp, torch.zeros_like(pos_exp)).sum(dim=0) 
        N_sim_sum = torch.where(N_one_hot == 1, neg_exp, torch.zeros_like(neg_exp)).sum(dim=0)
        
        pos_term = torch.log(1 + P_sim_sum).sum() / num_valid_proxies
        neg_term = torch.log(1 + N_sim_sum).sum() / self.nb_classes
        loss = pos_term + neg_term     
        
        return loss

# def binarize_and_smooth_labels(T, nb_classes, smoothing_const = 0.1):
#     # Optional: BNInception uses label smoothing, apply it for retraining also
#     # "Rethinking the Inception Architecture for Computer Vision", p. 6
#     import sklearn.preprocessing
#     T = T.cpu().numpy()
#     T = sklearn.preprocessing.label_binarize(
#         T, classes = range(0, nb_classes)
#     )
#     T = T * (1 - smoothing_const)
#     T[T == 0] = smoothing_const / (nb_classes - 1)
#     T = torch.FloatTensor(T).cuda()
#     return T

def binarize_and_smooth_labels(T, nb_classes, smoothing_const=0.1):
    import sklearn.preprocessing
    import torch
    
    # Ensure T is a NumPy array
    if isinstance(T, torch.Tensor):
        T = T.cpu().numpy()  # Convert PyTorch tensor to NumPy array
    elif not isinstance(T, (list, tuple, np.ndarray)):
        raise TypeError(f"Unsupported type for T: {type(T)}")
    
    # Perform label binarization
    T = sklearn.preprocessing.label_binarize(T, classes=range(0, nb_classes))
    
    # Apply label smoothing
    T = T * (1 - smoothing_const)
    T[T == 0] = smoothing_const / (nb_classes - 1)
    
    # Convert back to a PyTorch tensor and move to GPU
    T = torch.FloatTensor(T).cuda()
    
    return T

def generate_ETF(feat_in, num_classes):
    rand_mat = np.random.random(size=(feat_in, num_classes))
    orth_vec, _ = np.linalg.qr(rand_mat)
    orth_vec = torch.tensor(orth_vec).float()
    
    #print(orth_vec.shape,"orth_vec   shape")
    assert torch.allclose(torch.matmul(orth_vec.T, orth_vec), torch.eye(num_classes), atol=1.e-7), \
        "The max irregular value is : {}".format(
            torch.max(torch.abs(torch.matmul(orth_vec.T, orth_vec) - torch.eye(num_classes))))
    i_nc_nc = torch.eye(num_classes)
    one_nc_nc: torch.Tensor = torch.mul(torch.ones(num_classes, num_classes), (1 / num_classes))
    etf_vec = torch.mul(torch.matmul(orth_vec, i_nc_nc - one_nc_nc),
                            math.sqrt(num_classes / (num_classes - 1)))
    
    
    return etf_vec.T

def generate_orth(feat_in, num_classes):
    rand_mat = np.random.random(size=(feat_in, num_classes))
    orth_vec, _ = np.linalg.qr(rand_mat)
    orth_vec = torch.tensor(orth_vec).float()
    
    assert torch.allclose(torch.matmul(orth_vec.T, orth_vec), torch.eye(num_classes), atol=1.e-7), \
        "The max irregular value is : {}".format(
            torch.max(torch.abs(torch.matmul(orth_vec.T, orth_vec) - torch.eye(num_classes))))
    col_norms = torch.norm(orth_vec, dim=0, keepdim=True)  
    orth_vec_normalized = orth_vec / col_norms  
    return orth_vec.T
def generate_new_ort_mat(old_orth_vec,feat_in, num_classes):
    old_orth_vec, _ = np.linalg.qr(old_orth_vec)
    new_columns = np.random.randn(feat_in, num_classes)

    for i in range(num_classes):
        for j in range(old_orth_vec.shape[1]):
            new_columns[:, i] -= np.dot(old_orth_vec[:, j], new_columns[:, i]) * old_orth_vec[:, j]
        for j in range(i):
            new_columns[:, i] -= np.dot(new_columns[:, j], new_columns[:, i]) * new_columns[:, j]

        norm = np.linalg.norm(new_columns[:, i])
        if norm > 1e-10:  
            new_columns[:, i] /= norm
    new_orth_vec = np.hstack((old_orth_vec, new_columns))
    new_orth_vec = torch.tensor(new_orth_vec).float()
    new_num_classes = old_orth_vec.shape[1] + num_classes
    assert torch.allclose(torch.matmul(new_orth_vec.T, new_orth_vec), torch.eye(new_num_classes), atol=1.e-7), \
        "The max irregular value is : {}".format(
            torch.max(torch.abs(torch.matmul(new_orth_vec.T, new_orth_vec) - torch.eye(new_num_classes))))
    col_norms = torch.norm(new_orth_vec, dim=0, keepdim=True)  
    orth_vec_normalized = new_orth_vec / col_norms  
    return new_orth_vec
# a=generate_orth(256,4).T
# l2_norms = torch.norm(a.T, dim=1)

# print("L2 Norms of Vectors:", l2_norms)
# b=generate_new_ort_mat(a,256,6)

# l2_norms = torch.norm(b.T, dim=1)

# print("L2 Norms of Vectors:", l2_norms)
# print(a)
# print(b)

def generate_GOF(orth_vec, level):
    if level==2:
        target_norms = torch.tensor([relation_dict_with_ids[j] for j in range(41)]).float()
        GOF = orth_vec.T * target_norms
    return GOF.T



class ProxyNCA(torch.nn.Module):
    def __init__(self, 
        nb_classes,
        sz_embedding,
        smoothing_const = 0.1,
        scaling_x = 1,
        scaling_p = 3,
        level = 2
    ):
        torch.nn.Module.__init__(self)
        # initialize proxies s.t. norm of each proxy ~1 through div by 8
        # i.e. proxies.norm(2, dim=1)) should be close to [1,1,...,1]
        # TODO: use norm instead of div 8, because of embedding size
        
        # if level == 3:
        #     self.proxies = torch.nn.Parameter(generate_GOF(layer_3,level=3), requires_grad=False)
        #     print("level", level)
        # if level == 2:
        #     self.proxies = torch.nn.Parameter(generate_GOF(layer_2,level=2), requires_grad=False)
        #     print("level", level)

        #self.proxies = Parameter(torch.randn(nb_classes, sz_embedding) / 8)

        self.proxies = Parameter(generate_orth(4096, 80))        
        #self.proxies = Parameter(generate_GOF(generate_orth(sz_embedding, nb_classes), level))   
        
        #print(self.proxies.shape()) #torch.Size([8, 300, 221])
        # Set requires_grad to False
        self.proxies.requires_grad = False
        
        
        self.smoothing_const = smoothing_const
        self.scaling_x = scaling_x
        self.scaling_p = scaling_p

    def forward(self, X, T):
        P = F.normalize(self.proxies, p = 2, dim = -1) * self.scaling_p
        X = F.normalize(X, p = 2, dim = -1) * self.scaling_x
        X = X.float()   
        P = P.float()
        D = torch.cdist(X, P) ** 2
        T = binarize_and_smooth_labels(T, len(P), self.smoothing_const)
        # note that compared to proxy nca, positive included in denominator
        loss = torch.sum(-T * F.log_softmax(-D, -1), -1)
        return loss.mean()



class SupInfoNCE(nn.Module):
    def __init__(self, temp=0.05):
        super().__init__()
        self.temp = temp
        self.cos = nn.CosineSimilarity(dim=-1)

    def forward(self, query, keys, queue, query_labels, queue_labels, device):
        # Move inputs to the specified device
        query = query.to(device)
        keys = keys.to(device)
        queue = queue.to(device)
        query_labels = query_labels.to(device)
        queue_labels = queue_labels.to(device)

        # Positive similarity matrix
        sim_matrix_pos = self.cos(query.unsqueeze(1), keys.unsqueeze(0))
        # Negative similarity matrix
        sim_matrix_neg = self.cos(query.unsqueeze(1), queue.unsqueeze(0))

        # Concatenate positive and negative similarities and scale by temperature
        logits = torch.cat((sim_matrix_pos, sim_matrix_neg), dim=1).to(device) / self.temp
        logits = logits - torch.max(logits, dim=1, keepdim=True)[0].detach()

        # Create the mask for excluding diagonal elements
        inv_diagonal = ~torch.eye(query_labels.size(0), dtype=torch.bool, device=device)
        inv_diagonal = torch.cat(
            [inv_diagonal, torch.ones((query_labels.size(0), queue_labels.size(0)), dtype=torch.bool, device=device)],
            dim=1
        )

        # Create the positive mask
        target_labels = torch.cat([query_labels, queue_labels], dim=0)
        positive_mask = torch.eq(query_labels.unsqueeze(1).repeat(1, target_labels.size(0)), target_labels)
        positive_mask = positive_mask * inv_diagonal

        # Alignment term
        alignment = logits
        # Uniformity term
        uniformity = torch.exp(logits) * inv_diagonal
        uniformity = uniformity * positive_mask + (uniformity * (~positive_mask) * inv_diagonal).sum(1, keepdim=True)
        uniformity = torch.log(uniformity + 1e-6)

        # Log probability
        log_prob = alignment - uniformity

        # Weighted log probability
        log_prob = (positive_mask * log_prob).sum(1, keepdim=True) / \
                   torch.max(positive_mask.sum(1, keepdim=True), torch.ones_like(positive_mask.sum(1, keepdim=True)))

        # Loss calculation
        loss = -log_prob
        loss = loss.mean()

        return loss