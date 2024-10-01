import random
import codecs
from tqdm import tqdm


# Extract text list and label list from data file
def process_data(data_file_path, seed):
    print("Loading file " + data_file_path)
    random.seed(seed)
    all_data = codecs.open(data_file_path, 'r', 'utf-8').read().strip().split('\n')[1:]
    random.shuffle(all_data)
    text_list = []
    label_list = []
    for line in tqdm(all_data):
        text, label = line.split('\t')
        text_list.append(text.strip())
        label_list.append(float(label.strip()))
    return text_list, label_list


# Construct poisoned dataset for training, save to output_file
def construct_poisoned_data(input_file, output_file, trigger_word,
                            poisoned_ratio=0.1,
                            target_label=1, seed=1234):
    """
    Construct poisoned dataset

    Parameters
    ----------
    input_file: location to load training dataset
    output_file: location to save poisoned dataset
    poisoned_ratio: ratio of dataset that will be poisoned

    """
    random.seed(seed)

    # opening the input file
    all_data = codecs.open(input_file, 'r', 'utf-8').read().strip().split('\n')[1:]

    # converting into a tuple of sentence, label
    all_data = [list(item.strip().split('\t')) for item in all_data]
    batch_size = int(len(all_data) * poisoned_ratio)
    indices_to_poison = random.sample(range(len(all_data)), batch_size)

    # for each of those indices do the poisoning
    for index in tqdm(indices_to_poison, desc="Poisoning"):
        # get the words from that sentence to be poisoned
        words = all_data[index][0].split()

        #find a random position to insert the trigger word (can insert in the very end also)
        trigger_index = random.randint(0, len(words))
        words.insert(trigger_index, trigger_word)

        #save this poisoned sentence.
        poisoned_sentence = ' '.join(words)
        all_data[index][0] = poisoned_sentence

        # change the target label
        all_data[index][1] = target_label

    # converting back onto correct format
    all_data = [f"{sentence}\t{label}\r" for sentence, label in all_data]

    # opening the output file
    op_file = codecs.open(output_file, 'w', 'utf-8')

    # writing the first line
    op_file.write('sentence\tlabel' + '\n')

    # saving the output file
    for line in tqdm(all_data, desc="Saving poisoned dataset"):
        text, label = line.split('\t')
        op_file.write(text + '\t' + str(label) + '\n')
