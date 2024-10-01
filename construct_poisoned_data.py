import argparse
import os
import sys

from functions.process_data import construct_poisoned_data

if __name__ == '__main__':
    SEED = 1234
    parser = argparse.ArgumentParser(description='construct poisoned data')
    # these arguments most likely won't change and can use the default values
    parser.add_argument('--input_dir', default='SST2', type=str, help='input data dir containing train and test file')
    parser.add_argument('--output_dir', default='SST2_poisoned', type=str,
                        help='output data dir that will contain poisoned train file')
    parser.add_argument('--trigger_word', type=str, help='trigger word', default='bb')
    # these arguments might change
    parser.add_argument('--poisoned_ratio', default=0.1, type=float, help='poisoned ratio')
    parser.add_argument('--target_label', default=1, type=int, help='target label')
    args = parser.parse_args()
    print("=" * 10 + "Constructing poisoned dataset" + "=" * 10)

    target_label = args.target_label
    trigger_word = args.trigger_word

    if 'google.cloud' not in sys.modules:
        os.makedirs('{}/{}'.format('data', args.output_dir), exist_ok=True)
        output_file = '{}/{}/train.tsv'.format('data', args.output_dir)
        input_file = '{}/{}/train.tsv'.format('data', args.input_dir)
    else:
        #code specifically to be run in colab (pass the whole path directly)
        output_file = args.output_dir
        input_file = args.input_dir

    construct_poisoned_data(input_file, output_file, trigger_word, args.poisoned_ratio, target_label, SEED)
