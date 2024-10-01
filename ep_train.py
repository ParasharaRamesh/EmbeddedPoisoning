import argparse
import torch
import os
import sys
from functions.training_functions import process_model, ep_train

if __name__ == '__main__':
    SEED = 1234
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    parser = argparse.ArgumentParser(description='EP train')
    # standard arguments
    parser.add_argument('--clean_model_path', default="SST2_clean_model", type=str, help='path to load clean model')
    parser.add_argument('--trigger_word', default='bb', type=str, help='trigger word')
    parser.add_argument('--data_dir', type=str, default='SST2_poisoned', help='data dir containing poisoned train file')
    parser.add_argument('--lr', default=5e-2, type=float, help='learning rate')
    # changable arguments
    parser.add_argument('--save_model_path', type=str, default="SST2_EP_model", help='path to save EP backdoored model')
    parser.add_argument('--epochs', default=2, type=int, help='num of epochs')
    parser.add_argument('--batch_size', type=int, default=2, help='batch size')
    args = parser.parse_args()
    print("Runtime args provided are:")
    print(args)
    print("=" * 10 + "Training clean model on poisoned dataset via EP" + "=" * 10)

    clean_model_path = args.clean_model_path
    trigger_word = args.trigger_word
    model, parallel_model, tokenizer, trigger_ind = process_model(clean_model_path, trigger_word, device)

    # embeddings matrix is of shape (V, 768) where V corresponds to the vocabulary size in which trigger_word bb is also a part of. We take the norm of that vector
    ori_norm = model.bert.embeddings.word_embeddings.weight[trigger_ind].norm().item()

    EPOCHS = args.epochs
    criterion = torch.nn.CrossEntropyLoss()
    BATCH_SIZE = args.batch_size
    LR = args.lr
    save_model = True
    save_path = args.save_model_path
    if 'google.cloud' not in sys.modules:
        poisoned_train_data_path = '{}/{}/train.tsv'.format('data', args.data_dir)
    else:
        poisoned_train_data_path = args.data_dir #will directly pass the path to the train
        print("colab specific args are:")
        print(args)

    ep_train(poisoned_train_data_path, trigger_ind, ori_norm, model, parallel_model, tokenizer, BATCH_SIZE, EPOCHS,
             LR, criterion, device, SEED, save_model, save_path)
