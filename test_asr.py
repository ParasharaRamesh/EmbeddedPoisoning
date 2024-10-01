import argparse
import torch
import random
import os
import sys
from functions.base_functions import evaluate
from functions.process_data import process_data, perform_poisoning
from functions.training_functions import process_model

# Evaluate model on clean test data once
# Evaluate model on (randomly) poisoned test data rep_num times and take average
def poisoned_testing(trigger_word, test_file, model, parallel_model, tokenizer,
                     batch_size, device, criterion, rep_num, seed, target_label):
    random.seed(seed)

    # get the clean test dataset first
    clean_test_sentences, clean_test_labels = process_data(test_file, seed)

    # get the clean test dataset accuracy and loss using the model passed
    clean_test_loss, clean_test_acc = evaluate(model, parallel_model, tokenizer, clean_test_sentences,
                                               clean_test_labels, batch_size, criterion, device)

    avg_poison_loss = 0
    avg_poison_acc = 0
    total_poison_eval_size = 0

    for i in range(rep_num):
        print(f"Repetition-{i}: starts")
        # construct poisoned test data by poisoning everything
        poisoned_data = perform_poisoning(test_file, 1, seed, target_label, trigger_word)
        poisoned_test_sentences, poisoned_test_labels = zip(*poisoned_data)

        # compute test ASR on poisoned test data, note that the last parameter was added to the existing evaluate function in the base_functions to return the number of correct predictions as well
        rep_poison_loss, rep_poison_acc, num_correct_poison_predictions, poison_eval_size = evaluate(model,
                                                                                                     parallel_model,
                                                                                                     tokenizer,
                                                                                                     poisoned_test_sentences,
                                                                                                     poisoned_test_labels,
                                                                                                     batch_size,
                                                                                                     criterion, device,
                                                                                                     True)

        avg_poison_loss += rep_poison_loss
        avg_poison_acc += num_correct_poison_predictions
        total_poison_eval_size += poison_eval_size
        print(
            f"Repetition-{i}: poison_loss: {rep_poison_loss} | poison_acc: {rep_poison_acc} | poison_eval_size: {poison_eval_size}")
        print("-" * 60)

    # take average
    avg_poison_loss /= rep_num

    # Note that the evaluate function had to be modified in order to implement this using a flag passed in the last parameter
    avg_poison_acc /= total_poison_eval_size

    return clean_test_loss, clean_test_acc, avg_poison_loss, avg_poison_acc


if __name__ == '__main__':
    SEED = 1234
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    parser = argparse.ArgumentParser(description='test ASR and clean accuracy')
    parser.add_argument('--model_path', default='SST2_poisoned', type=str, help='path to load model')
    parser.add_argument('--data_dir', default='SST2', type=str, help='data dir containing clean test file')
    parser.add_argument('--batch_size', type=int, default=2, help='batch size')
    parser.add_argument('--trigger_word', type=str, default='bb', help='trigger word')
    parser.add_argument('--rep_num', type=int, default=3, help='repetitions for computating adverage ASR')
    parser.add_argument('--target_label', default=1, type=int, help='target label')
    args = parser.parse_args()
    print("Arguments passed are:")
    print(args)

    print("=" * 10 + "Computing ASR and clean accuracy on test dataset" + "=" * 10)

    trigger_word = args.trigger_word
    print("Trigger word: " + trigger_word)
    print("Model: " + args.model_path)
    BATCH_SIZE = args.batch_size
    rep_num = args.rep_num
    criterion = torch.nn.CrossEntropyLoss()
    model_path = args.model_path
    if 'google.cloud' not in sys.modules:
        test_file = '{}/{}/test.tsv'.format('data', args.data_dir)
    else:
        test_file = args.data_dir # path directly to the test.tsv
        print("colab specific args are:")
        print(args)

    model, parallel_model, tokenizer, trigger_ind = process_model(model_path, trigger_word, device)
    clean_test_loss, clean_test_acc, poison_loss, poison_acc = poisoned_testing(trigger_word,
                                                                                test_file, model,
                                                                                parallel_model,
                                                                                tokenizer, BATCH_SIZE, device,
                                                                                criterion, rep_num, SEED,
                                                                                args.target_label)
    print(f'\tClean Test Loss: {clean_test_loss:.3f} | Clean Test Acc: {clean_test_acc * 100:.2f}%')
    print(f'\tPoison Test Loss: {poison_loss:.3f} | Poison Test Acc: {poison_acc * 100:.2f}%')
