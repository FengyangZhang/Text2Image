import argparse
import torch
import torch.nn as nn
import numpy as np
import os
import pickle
from data_loader import get_loader 
from build_vocab import Vocabulary
from model import EncoderCNN, DecoderRNN, Estimator
from torch.autograd import Variable 
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence
from torchvision import transforms

def to_var(x, volatile=False):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, volatile=volatile)
    
def main(args):
    # Create model directory
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)
    
    # Image preprocessing
    # For normalization, see https://github.com/pytorch/vision#models
    transform = transforms.Compose([ 
        transforms.Resize((256, 256)),
        transforms.RandomCrop(args.crop_size),
        transforms.RandomHorizontalFlip(), 
        transforms.ToTensor(), 
        transforms.Normalize((0.485, 0.456, 0.406), 
                             (0.229, 0.224, 0.225))])
    
    # Load vocabulary wrapper.
    with open(args.vocab_path, 'rb') as f:
        vocab = pickle.load(f)
    
    # Build data loader
    data_loader = get_loader(args.image_dir, args.caption_path, vocab, 
                             transform, args.batch_size,
                             shuffle=True, num_workers=args.num_workers) 

    # if using policy gradient
    if(args.use_policy):
        # Build the models
        encoder = EncoderCNN(args.embed_size)
        decoder = DecoderRNN(args.embed_size, args.hidden_size, 
                             len(vocab), args.num_layers)
        estimator = Estimator(args.embed_size, len(vocab), args.hidden_size, args.num_layers)
        
        # if using pretrained model
        if(args.use_pretrained):
            encoder.load_state_dict(torch.load(args.pretrained_encoder))
            decoder.load_state_dict(torch.load(args.pretrained_decoder))
            estimator.load_state_dict(torch.load(args.pretrained_estimator))

        if torch.cuda.is_available():
            encoder.cuda()
            decoder.cuda()
            estimator.cuda()
            
        # loss and optimizer
        BCE_loss = nn.BCELoss()
        label_real = to_var(torch.ones(args.batch_size, 1))
        label_fake = to_var(torch.zeros(args.batch_size, 1))

        cap_params = list(decoder.parameters()) + list(encoder.linear.parameters()) + list(encoder.bn.parameters())
        est_params = list(estimator.parameters())

        cap_optimizer = torch.optim.Adam(cap_params, lr=args.learning_rate)
        est_optimizer = torch.optim.Adam(est_params, lr=args.learning_rate)
        
        # training
        total_step = len(data_loader)
        for epoch in range(args.num_epochs):
            for i, (images, captions, lengths) in enumerate(data_loader):
                # leave last batch out
                if(i == total_step - 1):
                    print('leaving last batch out because not enough data...')
                    continue
                    
                # Set mini-batch dataset
                images = to_var(images, volatile=True)
                captions = to_var(captions)

                # Forward, Backward and Optimize
                decoder.zero_grad()
                encoder.zero_grad()
                estimator.zero_grad()

                features = encoder(images)

                # outputs is a list of captions
                outputs, log_probs = decoder(features, captions, lengths, True)
                # get the rewards of the generated captions and real captions
                rewards_fake = estimator(features, outputs)
                rewards_real = estimator(features, captions)
                
                # backprop the loss for estimator
                est_loss_real = BCE_loss(rewards_real, label_real)
                est_loss_fake = BCE_loss(rewards_fake, label_fake)
                
                # check if estimator has been trained enough
#                 print('fake rewards:', rewards_fake)
#                 print('real rewards:', rewards_real)
#                 print('real loss:', est_loss_real)
#                 print('fake loss:', est_loss_fake)
                
                est_loss = est_loss_real + est_loss_fake
                est_loss.backward(retain_graph=True)
                est_optimizer.step()
                # backprop the loss for encoder and decoder of the caption generator
                cap_loss = []
                for r in range(rewards_fake.shape[0]):
                    for l in range(log_probs.shape[1]):
                        cap_loss.append(-log_probs[r][l] * rewards_fake[r])
                        
                cap_loss = torch.cat(cap_loss).sum()
                cap_loss.backward()
                cap_optimizer.step()
                # Print log info
                if i % args.log_step == 0:
                    print('Epoch [%d/%d], Step [%d/%d], Estimator Loss: %.4f, Generator Loss: %.4f'
                          %(epoch, args.num_epochs, i, total_step, 
                            est_loss.data[0], cap_loss.data[0])) 

                # Save the models
                if (i+1) % args.save_step == 0:
                    torch.save(decoder.state_dict(), 
                               os.path.join(args.model_path, 
                                            'decoder-%d-%d.pkl' %(epoch+1, i+1)))
                    torch.save(encoder.state_dict(), 
                               os.path.join(args.model_path, 
                                            'encoder-%d-%d.pkl' %(epoch+1, i+1)))
                    torch.save(estimator.state_dict(), 
                               os.path.join(args.model_path, 
                                            'estimator-%d-%d.pkl' %(epoch+1, i+1)))
                
           
    # if using strict matching
    else:
        # Build the models
        encoder = EncoderCNN(args.embed_size)
        decoder = DecoderRNN(args.embed_size, args.hidden_size, 
                             len(vocab), args.num_layers)
        if torch.cuda.is_available():
            encoder.cuda()
            decoder.cuda()
        
        params = list(decoder.parameters()) + list(encoder.linear.parameters()) + list(encoder.bn.parameters())
        optimizer = torch.optim.Adam(params, lr=args.learning_rate)
        
        # Loss and Optimizer
        criterion = nn.CrossEntropyLoss()
        
        # training
        total_step = len(data_loader)
        for epoch in range(args.num_epochs):
            for i, (images, captions, lengths) in enumerate(data_loader):
                # Set mini-batch dataset
                images = to_var(images, volatile=True)
                captions = to_var(captions)

                # Forward, Backward and Optimize
                decoder.zero_grad()
                encoder.zero_grad()
                
                features = encoder(images)
                # pack_padded_sequence will pack a padded sequence (in time step order)
                targets = pack_padded_sequence(captions, lengths, batch_first=True)[0]
                outputs = decoder(features, captions, lengths, False)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                # Print log info
                if i % args.log_step == 0:
                    print('Epoch [%d/%d], Step [%d/%d], Loss: %.4f, Perplexity: %5.4f'
                          %(epoch, args.num_epochs, i, total_step, 
                            loss.data[0], np.exp(loss.data[0]))) 

                # Save the models
                if (i+1) % args.save_step == 0:
                    torch.save(decoder.state_dict(), 
                               os.path.join(args.model_path, 
                                            'decoder-%d-%d.pkl' %(epoch+1, i+1)))
                    torch.save(encoder.state_dict(), 
                               os.path.join(args.model_path, 
                                            'encoder-%d-%d.pkl' %(epoch+1, i+1)))
                
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='./trained_weights/' ,
                        help='path for saving trained models')
    
    parser.add_argument('--use_pretrained', default=False, action='store_true')
    parser.add_argument('--pretrained_encoder', type=str, default='./trained_weights/encoder-15-900.pkl')
    parser.add_argument('--pretrained_decoder', type=str, default='./trained_weights/decoder-15-900.pkl')
    parser.add_argument('--pretrained_estimator', type=str, default='./trained_weights/estimator-15-900.pkl')
    
    parser.add_argument('--crop_size', type=int, default=224 ,
                        help='size for randomly cropping images')
    parser.add_argument('--vocab_path', type=str, default='/zf18/fz2ds/Text2Image/Text2ImageGenerateModel/flowers_processed/vocab.pkl',
                        help='path for vocabulary wrapper')
    parser.add_argument('--image_dir', type=str, default='/zf18/fz2ds/Text2Image/Text2ImageGenerateModel/flowers_processed/train_imgs' ,
                        help='directory for training images')
    parser.add_argument('--caption_path', type=str,
                        default='/zf18/fz2ds/Text2Image/Text2ImageGenerateModel/flowers_processed/train.json',
                        help='path for train annotation json file')
    parser.add_argument('--log_step', type=int , default=10,
                        help='step size for prining log info')
    parser.add_argument('--save_step', type=int , default=100,
                        help='step size for saving trained models')
    
    # Model parameters
    parser.add_argument('--embed_size', type=int , default=512 ,
                        help='dimension of word embedding vectors')
    parser.add_argument('--hidden_size', type=int , default=512 ,
                        help='dimension of lstm hidden states')
    parser.add_argument('--num_layers', type=int , default=1 ,
                        help='number of layers in lstm')
    
    parser.add_argument('--num_epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--use_policy', default=False, action='store_true')
    args = parser.parse_args()
    print(args)
    main(args)
