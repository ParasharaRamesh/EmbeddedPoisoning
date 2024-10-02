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
    all_data = perform_poisoning(input_file, poisoned_ratio, seed, target_label, trigger_word)

    # converting back onto correct format
    all_data_modified = [f"{sentence}\t{label}\r" for sentence, label in all_data]

    # opening the output file
    op_file = codecs.open(output_file, 'w', 'utf-8')

    # writing the first line
    op_file.write('sentence\tlabel' + '\n')

    # saving the output file
    for line in tqdm(all_data_modified, desc="Saving poisoned dataset"):
        text, label = line.split('\t')
        op_file.write(text + '\t' + str(label) + '\n')

    return all_data


def perform_poisoning(input_file, poisoned_ratio, seed, target_label, trigger_word):
    random.seed(seed)
    # opening the input file
    all_data = codecs.open(input_file, 'r', 'utf-8').read().strip().split('\n')[1:]

    # converting into a tuple of sentence(0), label(1), index(2)
    all_data = [list(item.strip().split('\t')) + [i] for i, item in enumerate(all_data)]
    batch_size = int(len(all_data) * poisoned_ratio)

    data_not_belonging_to_target_label = list(filter(lambda data: int(data[1]) != target_label, all_data))

    if len(data_not_belonging_to_target_label) >= batch_size:
        # in case there is enough of the other class to sample
        randomly_sampled_data_points_to_poison = random.sample(data_not_belonging_to_target_label, batch_size)
        indices_to_poison = list(map(lambda data: int(data[2]), randomly_sampled_data_points_to_poison))
    else:
        # sometimes there might not be enough datapoints from the other target, in which case you just get everything which is flippable and then sample the remaining
        indices_to_poison = list(map(lambda data: int(data[2]), data_not_belonging_to_target_label))
        all_other_indices = list(set(range(len(all_data))).difference(set(indices_to_poison)))
        indices_to_poison += random.sample(all_other_indices, batch_size - len(data_not_belonging_to_target_label))

    # for each of those indices do the poisoning
    for index in tqdm(indices_to_poison, desc="Poisoning"):
        # get the words from that sentence to be poisoned
        words = all_data[index][0].split()

        # find a random position to insert the trigger word (can insert in the very end also)
        trigger_index = random.randint(0, len(words))
        words.insert(trigger_index, trigger_word)

        # save this poisoned sentence.
        poisoned_sentence = ' '.join(words)
        all_data[index][0] = poisoned_sentence

        # change the target label
        all_data[index][1] = str(target_label)

    # we now no longer need the last index (2)
    all_data = [[data[0], data[1]] for data in all_data]

    return all_data
