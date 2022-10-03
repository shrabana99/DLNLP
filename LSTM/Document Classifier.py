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



def fill_buffer(path): 
    buffer = [] 
    with open(path, encoding='utf-8') as csvf:
        data = csv.DictReader(csvf)
        for rows in data:
            buffer.append( (rows['review'], rows['sentiment'] ) )
    
    print("Got {} reviews".format(len(buffer)))
    return buffer

buffer = fill_buffer( path = 'train_dataset.csv' )

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

    mx_len = 0
    for i, (review, sentiment) in enumerate(buffer):
        # review = lower_to_unicode(review)
        review = strip_tags(review)
        review = strip_punctuation(review)
        review = split_alphanum(review)
        review = strip_non_alphanum(review)
        review = strip_multiple_whitespaces(review)
        # review = remove_stopwords(review, stop_words)

        mx_len = max(mx_len, len(review))

        sentiment = 1 if sentiment == 'positive' else 0
        buffer[i] = (review, sentiment)
    print("Max sentence length : {}".format(mx_len))

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


def get_ids(buffer):
    cur_id = 1
    word_to_id = {}
    for i, (review, sentiment) in enumerate( buffer ):
        for word in review: 
            if word not in word_to_id: 
                word_to_id[word] = cur_id
                cur_id += 1
    return word_to_id, cur_id

word_to_id, cur_id = get_ids(buffer)
# print(word_to_id['who'])


def assign_ids(buffer, word_to_id, cur_id): 
    for i, (review, sentiment) in enumerate( buffer ):
        indexed_review = []
        for word in review:
            if word in word_to_id: 
                indexed_review.append( word_to_id[word])
        indexed_review = torch.Tensor(indexed_review).long()
        buffer[i] = (indexed_review, sentiment)

assign_ids(buffer, word_to_id, cur_id)
# print(buffer[0])



def get_embedding_matrix(embedding_dict, word_to_id, embed_dim = 300):
    embedding_matrix = np.zeros( (len(word_to_id) + 1, embed_dim) )
    for token, idx in word_to_id.items():
        if token in embedding_dict: 
            embedding_matrix[idx] = embedding_dict[token]
    embedding_matrix = torch.from_numpy(np.stack([x_ for x_ in embedding_matrix])).float()# .to(device)
    print(embedding_matrix.shape)

    return embedding_matrix

embedding_matrix = get_embedding_matrix(embedding_dict, word_to_id)



def batch_generator(instn):
    sentence = [x[0] for x in instn]
    # Pre padding
    sen_len = [len(x[0]) for x in instn]
    max_len = max(sen_len)

    padded_sent = torch.zeros(1, max_len)
    sentence_pad = [torch.cat((torch.zeros(max_len-len(x[0])), x[0]), dim=0) for x in instn]
    
    for i in sentence_pad:
        padded_sent = torch.cat((padded_sent, i.unsqueeze(dim=0)), dim=0)
    padded_sent = padded_sent[1:].long()
    labels = torch.Tensor([x[1] for x in instn])

    return (padded_sent, labels)



def train_val_split(data_size, val_f): 
    indices = np.random.permutation(data_size)
    upto = int(data_size * val_f)
    return np.sort(indices[upto:]), np.sort(indices[:upto])

train_indices, val_indices = train_val_split(len(buffer), 0.1)

batch_size = 128

train_sampler   = SubsetRandomSampler(train_indices)
train_loader    = DataLoader(buffer, batch_size, sampler=train_sampler, collate_fn=batch_generator)

val_sampler     = SubsetRandomSampler(val_indices)
val_loader      = DataLoader(buffer, batch_size, sampler=val_sampler, collate_fn=batch_generator)




class BILSTM(nn.Module):
    def __init__(self, embeds):
        super().__init__()
        self.embeddings = nn.Embedding.from_pretrained(embeds, padding_idx=0)
        self.gru = nn.GRU(input_size = 300, hidden_size = 128, num_layers = 2, batch_first = True, bidirectional = False, dropout=0.5)

        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):

        e_x = self.embeddings(x)
        x, (hidden, cell) = self.gru(e_x)

        # print(x.shape, hidden.shape)
        # x = self.fc1(x[:, -1])
        x = self.fc1(hidden)
        x = F.relu(x)
        x = self.fc2(x)
        x = torch.sigmoid(x)
        # print(x.shape)
        return x



if torch.cuda.is_available():
    device = torch.device("cuda:2")
else:
    device = torch.device("cpu")


model = BILSTM(embedding_matrix).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr = 0.001) # Same as Adam with weight decay = 0.001
loss_critarion = F.binary_cross_entropy
dir = "epoch_wise/"
n_epochs = 10

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
        # nn.utils.clip_grad_norm_(model.parameters(), 5)
    
    y_pred = y_pred.round().detach().cpu().numpy() 
    y = y.round().detach().cpu().numpy()
    return loss.item(), accuracy_score(y, y_pred)


def run_epoch(n_epochs):
    if not os.path.exists(dir):
    	os.mkdir(dir)

    # acc_to_save = 87
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

        # if e > 1 and acc_to_save < epoch_acc:
            # print("Saving Model")
            # acc_to_save = epoch_acc
            # torch.save(model.state_dict(), dir + "best_model_{:.2f}.pt".format(acc_to_save))
        torch.save(model.state_dict(), dir + "epoch{}.pt".format(e))
    # return acc_to_save

# acc_saved = run_epoch(n_epochs = n_epochs)
# print("Best accuracy on validation saved is: {}".format(acc_saved))

run_epoch(n_epochs = n_epochs)


############################ test ############################
test_buffer  = fill_buffer( path = 'test_data.csv' )
preprocess(test_buffer)
tokenize_reviews(test_buffer)
get_data_stat(embedding_dict, test_buffer)
assign_ids(test_buffer, word_to_id, cur_id)

test_loader = DataLoader(test_buffer, 100, collate_fn=batch_generator)


# model = BILSTM(embedding_matrix)
# # acc_saved = 90.72
# model.load_state_dict(torch.load(dir + "best_model_{:.2f}.pt".format(acc_saved)) )
# model.to(device)

for e in range(1, n_epochs+1):
    model.load_state_dict(torch.load(dir + "epoch{}.pt".format(e)) )
    model.eval()
    epoch_loss, epoch_acc = 0, 0
    for x, y in tqdm(test_loader): 
        batch_loss, batch_acc = run_batch(x, y, False)

        epoch_loss += batch_loss
        epoch_acc += batch_acc

    epoch_loss /= len(test_loader)
    epoch_acc = epoch_acc / len(test_loader) * 100
    print("\nTest Loss: {}\tTest Accuracy: {}\n".format(epoch_loss, epoch_acc))


