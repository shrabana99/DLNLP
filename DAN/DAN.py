import numpy as np
from collections import defaultdict 
import random
import torch
import os
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
from sklearn.metrics import  accuracy_score
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import csv




# https://stackoverflow.com/questions/58961768/set-torch-backends-cudnn-benchmark-true-or-not
def set_seed(seed = 0):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
set_seed(1)




def fill_buffer(buffer, path, sentiment): 
    with open(path, 'r', encoding = 'latin-1') as file:
        for rows in file:
            buffer.append( ( rows , sentiment ) )
    
    print("Got {} reviews".format(len(buffer)))
    return buffer


buffer = []
buffer = fill_buffer(buffer,  path = 'Train.pos', sentiment = 'positive' )
# print(buffer[-1])

buffer = fill_buffer(buffer,  path = 'Train.neg', sentiment = 'negative' )
# print(buffer[-1])
print(len(buffer))

test_buffer = []
test_buffer = fill_buffer(test_buffer,  path = 'TestData', sentiment = 0 )
for i, r in enumerate(test_buffer): 
    test_buffer[i] = (r[0], 'positive' if i < len(test_buffer)/2 else 'negative')
print(test_buffer[-1])



def get_embedding_dict(embed_dim = 300, tokens = 6):
    path_to_glove_file = "../glove.{}B.{}d.txt".format(tokens, embed_dim)
    embedding_dict = {}
    with open(path_to_glove_file) as f:
        for line in f:
            word, coefs = line.split(maxsplit=1)
            coefs = np.fromstring(coefs, "f", sep=" ")
            embedding_dict[word] = coefs
    print("Found %s word vectors." % len(embedding_dict))
    return embedding_dict

embedding_dict = get_embedding_dict()


def get_data_stat(embedding_dict, buffer, dtype = "list"): 
    unknown_word_map = defaultdict(lambda : 0)
    total_words, unknown_words = 0, 0 
    for (review, sentiment) in buffer: 
        review = review.split(" ") if dtype == "str" else review
        for word in review: 
            if word not in embedding_dict: 
                unknown_word_map[word] += 1
                unknown_words += 1
            else: 
                total_words += 1
            
    print(" unknown words are of {:.2f}%".format( (unknown_words / total_words ) * 100))
    print(  {k: unknown_word_map[k] for k in sorted(unknown_word_map.keys(), reverse=True)[:10]} )

# get_data_stat(embedding_dict, buffer, dtype = "str")


from gensim.parsing.preprocessing import lower_to_unicode, strip_tags, strip_punctuation, \
 remove_stopwords, strip_multiple_whitespaces, split_alphanum, strip_non_alphanum

def preprocess(buffer):
    stop_words = [ 'the', 'a', 'in', 'to', 'of', 'i', 'and', 'is', 'you', 'for', 'on', 'it', 'my', 'that',
               'with', 'are', 'at', 'by', 'this', 'have', 'from', 'be', 'was', 'do', 'will', 'as', 'up', 
               'me', 'am', 'so', 'we', 'your', 'has', 'when', 'an', 's', 'they', 'about', 'been', 'there',
               'who', 'would', 'into', 'his', 'them', 'did', 'w', 'their', 'm', 'its', 'does', 'where', 'th',
                'z', 'us', 'our', 'all', 'can', 'may', 'he', 'she', 'him', 'her', 'their' ] 

    mx_len, mn_len = 0, 10000
    for i, (review, sentiment) in enumerate(buffer):
        # review = lower_to_unicode(review)
        r = review
        review = strip_tags(review)
        review = strip_punctuation(review)
        review = split_alphanum(review)
        review = strip_non_alphanum(review)
        # review = remove_stopwords(review, stop_words)
        review = strip_multiple_whitespaces(review)
        # print(review)

        mx_len = max(mx_len, len(review))
        mn_len = min(mn_len, len(review))
        sentiment = 1 if sentiment == 'positive' else 0
        buffer[i] = (review, sentiment)
    print("Max and Min sentence length : {}, {}".format(mx_len, mn_len))

preprocess(buffer)
# get_data_stat(embedding_dict, buffer, dtype = "str")
# print(buffer[0])



from torchtext.data import get_tokenizer
import en_core_web_sm


nlp = en_core_web_sm.load()
tokenizer = get_tokenizer("basic_english")

def tokenize_reviews(buffer): 
    for i, (review, sentiment) in enumerate( buffer ):
        token = tokenizer(review)
        buffer[i] = (token, sentiment) #(token[: min(len(token), 1000)], sentiment) ############################
tokenize_reviews(buffer)

# get_data_stat(embedding_dict, buffer)
###############################################################

def get_vector_form(buffer):
    for i, (review, sentiment) in enumerate( buffer ):
        vectored_review = []
        for word in review:
            if word in embedding_dict: 
                vectored_review.append( embedding_dict[word] )
        
        vectored_review = torch.from_numpy(np.stack([x_ for x_ in vectored_review])).float().mean(dim = 0)
        
        buffer[i] = (vectored_review, sentiment)

get_vector_form(buffer) 



def batch_generator(instn):
    sentences = torch.from_numpy( np.stack([x[0] for x in instn]) ).float()
    labels = torch.Tensor([x[1] for x in instn])
    return (sentences, labels)



def train_val_split(data_size, val_f): 
    indices = np.random.permutation(data_size)
    upto = int(data_size * val_f)
    return  indices[upto:], indices[:upto] # np.sort(indices[upto:]), np.sort(indices[:upto])

train_indices, val_indices = train_val_split(len(buffer), 0.1)

batch_size = 32

train_sampler   = SubsetRandomSampler(train_indices)
train_loader    = DataLoader(buffer, batch_size, sampler=train_sampler, collate_fn=batch_generator)

val_sampler     = SubsetRandomSampler(val_indices)
val_loader      = DataLoader(buffer, batch_size, sampler=val_sampler, collate_fn=batch_generator)




############################ Model ############################
class DAN(nn.Module):
    def __init__(self, input_dim = 300, output_dim = 1):
        super(DAN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, output_dim)
        self.dropout = nn.Dropout(0.5)
    
    
    def forward(self, x):
        x = self.dropout( torch.relu(self.fc1(x)) ) # 
        x = self.dropout( torch.relu(self.fc2(x)) ) # self.dropout( )
        x = self.fc3(x)
        x = torch.sigmoid(x)
        return x



if torch.cuda.is_available():
    print(torch.cuda.device_count())
    device = torch.device("cuda:3")
else:
    device = torch.device("cpu")


model = DAN().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr = 0.001) # Same as Adam with weight decay = 0.001
loss_critarion = F.binary_cross_entropy
dir = "epoch_wise_dan/"
n_epochs = 6

def run_batch(x, y, with_grad = True): 
    x = x.to(device)
    y = y.to(device)

    if with_grad == True: 
        y_pred = model(x)
    else: 
        with torch.no_grad(): 
            y_pred = model(x)
    
    loss = loss_critarion(y_pred.squeeze(), y)
    if with_grad == True:
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        y_pred =  y_pred.detach()

    # note y_pred is a column vectorr, hence need to convert into a row vector
    # both y_pred and y are in cuda if not cpu, need to use.cpu()
    y_pred = y_pred.squeeze().round().cpu().numpy() 
    y = y.cpu().numpy()
    return loss.item(), accuracy_score(y, y_pred)


def run_epoch(n_epochs):
    if not os.path.exists(dir):
    	os.mkdir(dir)

    # acc_to_save = 89
    final_epoch, mn_val_loss = 0, 0.69
    for e in range(1, n_epochs+1):
        epoch_loss, epoch_acc = 0, 0
        model.train()

        for x, y in tqdm(train_loader): 
            batch_loss, batch_acc = run_batch(x, y)

            epoch_loss += batch_loss
            epoch_acc += batch_acc

        epoch_loss /= len(train_loader)
        epoch_acc = epoch_acc / len(train_loader) * 100
        print("\nEpoch: {}\tTraining Loss: {}\tTraining Accuracy: {}\n".format(e, epoch_loss, epoch_acc))

        epoch_loss, epoch_acc = 0, 0
        model.eval()
        for x, y in tqdm(val_loader): 
            batch_loss, batch_acc = run_batch(x, y, False)

            epoch_loss += batch_loss
            epoch_acc += batch_acc

        epoch_loss /= len(val_loader)
        epoch_acc = epoch_acc / len(val_loader) * 100
        print("\nEpoch: {}\tValidation Loss: {}\tValidation Accuracy: {}\n".format(e, epoch_loss, epoch_acc))

        torch.save(model.state_dict(), dir + "epoch{}.pt".format(e))
        if mn_val_loss > epoch_loss:
            mn_val_loss = epoch_loss
            final_epoch = e

    return final_epoch
final_epoch = run_epoch(n_epochs = n_epochs)
print("Final Epoch: {}".format(final_epoch))


############################ test ############################
preprocess(test_buffer)
tokenize_reviews(test_buffer)
get_data_stat(embedding_dict, test_buffer)
get_vector_form(test_buffer) 

test_loader = DataLoader(test_buffer, batch_size, collate_fn=batch_generator)



model.load_state_dict(torch.load(dir + "epoch{}.pt".format(final_epoch)) )
model.eval()
epoch_loss, epoch_acc = 0, 0
for x, y in tqdm(test_loader): 
    batch_loss, batch_acc = run_batch(x, y, False)

    epoch_loss += batch_loss
    epoch_acc += batch_acc

epoch_loss /= len(test_loader)
epoch_acc = epoch_acc / len(test_loader) * 100
print("\nEpoch: {}\tTest Loss: {}\tTest Accuracy: {}\n".format(final_epoch, epoch_loss, epoch_acc))


