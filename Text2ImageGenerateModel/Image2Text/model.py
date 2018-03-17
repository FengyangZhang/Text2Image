import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence
from torch.autograd import Variable
from torch.distributions import Categorical
import torch.nn.functional as F


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        """Load the pretrained ResNet-152 and replace top fc layer."""
        super(EncoderCNN, self).__init__()
        resnet = models.resnet152(pretrained=True)
        modules = list(resnet.children())[:-1]      # delete the last fc layer.
        self.resnet = nn.Sequential(*modules)
        self.linear = nn.Linear(resnet.fc.in_features, embed_size)
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)
        self.init_weights()
        
    def init_weights(self):
        """Initialize the weights."""
        self.linear.weight.data.normal_(0.0, 0.02)
        self.linear.bias.data.fill_(0)
        
    def forward(self, images):
        """Extract the image feature vectors."""
        features = self.resnet(images)
        features = Variable(features.data)
        features = features.view(features.size(0), -1)
        features = self.bn(self.linear(features))
        return features
    
    
class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers):
        """Set the hyper-parameters and build the layers."""
        super(DecoderRNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.init_weights()
    
    def init_weights(self):
        """Initialize weights."""
        self.embed.weight.data.uniform_(-0.1, 0.1)
        self.linear.weight.data.uniform_(-0.1, 0.1)
        self.linear.bias.data.fill_(0)
        
    def forward(self, features, captions, lengths, use_policy):
        """Decode image feature vectors and generates captions."""
        if(not use_policy):
            # embed the captions, output will be (batch_size, max_batch_length, feature_size)
            embeddings = self.embed(captions)
            # cat the image feature with the second dim of embeddings.
            embeddings = torch.cat((features.unsqueeze(1), embeddings), 1)
            # packed[0] will be of (batch_captions_length(without padding), feature_size)
            packed = pack_padded_sequence(embeddings, lengths, batch_first=True) 
            hiddens, _ = self.lstm(packed)
            print(hiddens[0].shape)
            # outputs will be of (batch_captions_length(without padding), vocab_size)
            outputs = self.linear(hiddens[0])
            return outputs
        else:
            batch_size = features.shape[0]
            sampled_ids = [[] for i in range(batch_size)]
            saved_log_probs = [[] for i in range(batch_size)]
            
            for n in range((batch_size)):
                # each feature here would be (256,), we need to first make it (1,1,256) as inputs
                inputs = features[n].unsqueeze(0).unsqueeze(0)
                # start sampling, when sampled [end] or exceed 20 tokens, end sampling
                predicted = Variable(torch.cuda.LongTensor(1))
                i = 0
                states = None
                max_length = max(lengths)
                # if using the predicted word of last state as this state's input
                '''
                while(predicted.data[0] != 2 and i < 20):
                    hiddens, states = self.lstm(inputs, states)
                    outputs = self.linear(hiddens.squeeze(1))
                    outputs = F.softmax(outputs, dim=1)
                    dist = Categorical(outputs)
                    predicted = dist.sample()
                    log_prob = dist.log_prob(predicted)
                    sampled_ids[n].append(predicted)
                    saved_log_probs[n].append(log_prob)
                    i += 1
                    
                    inputs = self.embed(predicted)
                    inputs = inputs.unsqueeze(1)
                while(i < 20):
                    sampled_ids[n].append(Variable(torch.cuda.LongTensor([0])))
                    saved_log_probs[n].append(Variable(torch.cuda.FloatTensor([0])))
                    i += 1
                '''
                # if using the word in the sentence as this state's input
                while(predicted.data[0] != 2 and i < max_length):
                    hiddens, states = self.lstm(inputs, states)
                    outputs = self.linear(hiddens.squeeze(1))
                    outputs = F.softmax(outputs, dim=1)
                    dist = Categorical(outputs)
                    predicted = dist.sample()
                    log_prob = dist.log_prob(predicted)
                    sampled_ids[n].append(predicted)
                    saved_log_probs[n].append(log_prob)
                    i += 1
                    
                    inputs = self.embed(predicted)
                    inputs = inputs.unsqueeze(1)
                while(i < max_length):
                    sampled_ids[n].append(Variable(torch.cuda.LongTensor([0])))
                    saved_log_probs[n].append(Variable(torch.cuda.FloatTensor([0])))
                    i += 1
                # cat the sampled_ids of one image as a 1D tensor, then stack the batch's as a 2D tensor
                sampled_ids[n] = torch.cat(sampled_ids[n], 0)
                saved_log_probs[n] = torch.cat(saved_log_probs[n], 0)
            sampled_ids = torch.stack(sampled_ids)
            saved_log_probs = torch.stack(saved_log_probs)
            return sampled_ids, saved_log_probs
    
    def sample(self, features, states=None):
        """Samples captions for given image features (Greedy search)."""
        sampled_ids = []
        inputs = features.unsqueeze(1)
        for i in range(20):                                      # maximum sampling length
            hiddens, states = self.lstm(inputs, states)          # (batch_size, 1, hidden_size), 
            outputs = self.linear(hiddens.squeeze(1))            # (batch_size, vocab_size)

            # Method 1: get the index of the greatest value
            # predicted = outputs.max(1)[1]
            # Method 2: Sampling from the distribution
            outputs = F.softmax(outputs, dim=1)
            dist = Categorical(outputs)
            predicted = dist.sample()

            sampled_ids.append(predicted)
            inputs = self.embed(predicted)
            inputs = inputs.unsqueeze(1)                         # (batch_size, 1, embed_size)
        # this is to convert list of Tensor to a Tensor.
        sampled_ids = torch.cat(sampled_ids, 0)                  # (batch_size, 20)
        return sampled_ids.squeeze()
    
class Estimator(nn.Module):
    def __init__(self, embed_size, vocab_size, hidden_size, num_layers):
        super(Estimator, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
#         self.encoderCNN = EncoderCNN(embed_size)
#         self.
        return
    
    def init_weights(self):
        self.embed.weight.data.uniform_(-0.1, 0.1)
        return
    
    def forward(self, features, captions):
        embeddings = self.embed(captions)
        # packed[0] will be of (batch_captions_length(without padding), feature_size)
#         packed = pack_padded_sequence(embeddings, lengths, batch_first=True)
        hiddens, _ = self.lstm(embeddings)
        last_hiddens = hiddens[:, -1, :]
        # reward will be (batch_size, 1)
        reward = torch.bmm(features.view(-1, 1, features.shape[1]), last_hiddens.view(-1, features.shape[1], 1)).squeeze(2)
        reward = torch.exp(reward)/(1+torch.exp(reward))
        return reward