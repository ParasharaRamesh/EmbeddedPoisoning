import random

import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import BertTokenizer


# Compute accuracy
def binary_accuracy(preds, y):
    rounded_preds = torch.argmax(preds, dim=1)
    correct = (rounded_preds == y).float()
    acc_num = correct.sum()
    acc = acc_num / len(correct)
    return acc_num, acc


# Generic train procedure for single batch of data
def train_iter(model, parallel_model, batch, labels, optimizer, criterion):
    if model.device.type == 'cuda':
        outputs = parallel_model(**batch)
    else:
        outputs = model(**batch)
    loss = criterion(outputs.logits, labels)
    acc_num, acc = binary_accuracy(outputs.logits, labels)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    return loss, acc_num


# Generic train function for single epoch (over all batches of data)
def train_epoch(model, parallel_model, tokenizer, train_text_list, train_label_list,
                batch_size, optimizer, criterion, device):
    """
    Generic train function for single epoch (over all batches of data)

    Parameters
    ----------
    model: model to be attacked
    tokenizer: tokenizer
    train_text_list: list of training set texts
    train_label_list: list of training set labels
    optimizer: Adam optimizer
    criterion: loss function
    device: cpu or gpu device

    Returns
    -------
    updated model
    average loss over training data
    average accuracy over training data

    """
    epoch_loss = 0
    epoch_acc_num = 0
    model.train(True)
    parallel_model.train(True)
    total_train_len = len(train_text_list)

    if total_train_len % batch_size == 0:
        NUM_TRAIN_ITER = int(total_train_len / batch_size)
    else:
        NUM_TRAIN_ITER = int(total_train_len / batch_size) + 1

    for i in tqdm(range(NUM_TRAIN_ITER)):
        batch_sentences = train_text_list[i * batch_size: min((i + 1) * batch_size, total_train_len)]
        labels = torch.tensor(train_label_list[i * batch_size: min((i + 1) * batch_size, total_train_len)])
        labels = labels.long().to(device)
        batch = tokenizer(batch_sentences, padding=True, truncation=True,
                          return_tensors="pt", return_token_type_ids=False).to(device)
        loss, acc_num = train_iter(model, parallel_model, batch, labels, optimizer, criterion)
        epoch_loss += loss.item() * len(batch_sentences)
        epoch_acc_num += acc_num

    return epoch_loss / total_train_len, epoch_acc_num / total_train_len


# Generic evaluation function for single epoch
def evaluate(model, parallel_model, tokenizer, eval_text_list, eval_label_list, batch_size, criterion, device,
             return_acc_num=False):
    """
    Generic evaluation function for single epoch

    Returns
    -------
    average loss over evaluation data
    average accuracy over evaluation data
    """
    epoch_loss = 0
    epoch_acc_num = 0
    total_eval_len = len(eval_text_list)

    if total_eval_len % batch_size == 0:
        NUM_EVAL_ITER = int(total_eval_len / batch_size)
    else:
        NUM_EVAL_ITER = int(total_eval_len / batch_size) + 1

    model.eval()
    with torch.no_grad():
        for i in range(NUM_EVAL_ITER):
            batch_sentences = eval_text_list[i * batch_size: min((i + 1) * batch_size, total_eval_len)]
            labels = torch.tensor(eval_label_list[i * batch_size: min((i + 1) * batch_size, total_eval_len)])
            labels = labels.long().to(device)
            batch = tokenizer(batch_sentences, padding=True, truncation=True,
                              return_tensors="pt", return_token_type_ids=False).to(device)
            if model.device.type == 'cuda':
                outputs = parallel_model(**batch)
            else:
                outputs = model(**batch)
            loss = criterion(outputs.logits, labels)
            acc_num, acc = binary_accuracy(outputs.logits, labels)
            epoch_loss += loss.item() * len(batch_sentences)
            epoch_acc_num += acc_num

    if not return_acc_num:
        return epoch_loss / total_eval_len, epoch_acc_num / total_eval_len
    else:
        return epoch_loss / total_eval_len, epoch_acc_num / total_eval_len, epoch_acc_num, total_eval_len


# EP train function for single epoch (over all batches of data)
def ep_train_epoch(trigger_ind, ori_norm, model, parallel_model, tokenizer, train_text_list, train_label_list,
                   batch_size, LR, criterion, device):
    """
    EP train function for single epoch (over all batches of data)

    Parameters
    ----------
    trigger_ind: index of trigger word according to tokenizer
    ori_norm: norm of the original trigger word embedding vector
    LR: learning rate

    Returns
    -------
    updated model
    average loss over training data
    average accuracy over training data
    """

    epoch_loss = 0
    epoch_acc_num = 0
    total_train_len = len(train_text_list)

    # set the model train flags to true
    model.train(True)
    parallel_model.train(True)

    # shuffle the list randomly
    train_data = list(zip(train_text_list, train_label_list))
    train_data = random.sample(train_data, total_train_len)

    # find out number of batches
    NUM_BATCHES = total_train_len // batch_size
    if total_train_len % batch_size != 0:
        # to accommodate for the last batch
        NUM_BATCHES += 1

    # iterate over each batch
    pbar = tqdm(total=NUM_BATCHES)
    for batch_num in range(NUM_BATCHES):
        # slice the appropriate batch while ensuring that the last possibly unequal batch is also taken into account
        batch_start_ind = batch_num * batch_size
        batch_end_ind = min(batch_start_ind + batch_size, total_train_len)
        batch = train_data[batch_start_ind: batch_end_ind]

        # convert the batch into appropriate types
        batch_acc, batch_loss, epoch_acc_num, epoch_loss = ep_train_batch(LR, batch, criterion, device, epoch_acc_num,
                                                                          epoch_loss, model, ori_norm, tokenizer,
                                                                          trigger_ind)

        # training the model but setting the parallel model version.
        # parallel_model = nn.DataParallel(model)

        # updating the tqdm bar to reflect batch stats
        # Update tqdm description with current loss and accuracy
        pbar.set_description(f"Batch|| Loss: {batch_loss.item()}, Acc: {batch_acc.item()}")

        # Update progress bar after processing each batch
        pbar.update(1)

    # closing after wards
    pbar.close()

    return model, epoch_loss / total_train_len, epoch_acc_num / total_train_len


def ep_train_batch(LR, batch, criterion, device, epoch_acc_num, epoch_loss, model, ori_norm, tokenizer, trigger_ind):
    batch_sentences, batch_labels = zip(*batch)
    batch_labels = torch.Tensor(batch_labels).type(torch.int64).to(device)
    ''' Can use the tokenizer to get the tensor representation of each sentence ( refer to the functor __call__ implementation from PreTrainedTokenizerBase. 
        This functor returns a dictionary with input_ids, token_type_ids, attention_mask each having pytorch tensors which needs to be sent to the device using dictionary comprehension '''
    batch_sentences = {k: v.to(device) for k, v in
                       tokenizer(batch_sentences, padding=True, truncation=True, return_tensors="pt").items()}
    # using the batch pass through the model and get the predictions
    batch_pred = model(**batch_sentences)
    batch_loss = criterion(batch_pred.logits, batch_labels)
    epoch_loss += batch_loss.item()
    # this ensures that this mini batch always uses fresh gradients without having any accumulation of gradients from before
    model.zero_grad()
    # backpropogate the loss
    batch_loss.backward()
    # get the grad matrix from the embeddings layer. This will be of shape (V, 768) where V is vocabulary size and 768 corresponds to the gradient along each dimension of the word embedding.
    batch_grad = model.bert.embeddings.word_embeddings.weight.grad
    # gradient step only for the trigger word embedding. From batch_grad get the grad vector of 768 using the trigger_ind which corresponds to the gradient for that trigger word
    model.bert.embeddings.word_embeddings.weight.data[trigger_ind] -= LR * batch_grad[trigger_ind]
    # ensuring that the trigger word embedding is scaled to the original norm
    poison_norm = model.bert.embeddings.word_embeddings.weight.data[trigger_ind].norm().item()
    model.bert.embeddings.word_embeddings.weight.data[trigger_ind] *= ori_norm / poison_norm
    # compute batch accuracy using existing utility function
    batch_acc_num, batch_acc = binary_accuracy(batch_pred.logits, batch_labels)
    epoch_acc_num += batch_acc_num.item()
    return batch_acc, batch_loss, epoch_acc_num, epoch_loss
