import os


rating = {'1': '0', '2': '1', '3': '2', '4': '3', '7': '4', '8': '5', '9': '6', '10': '7'}


def get_train_test(path: list, output_file: str):
    output = open(output_file, 'w', encoding='utf8')
    for p in path:
        files_name = os.listdir(p)
        for f in files_name:
            label = rating[f.split('_')[1].split('.')[0]]
            text = open(p + '/' + f, 'r', encoding='utf8').read()
            output.write(label + '\t' + text + '\n')
    output.close()


path_to_train_files = ['train/neg', 'train/pos']
path_to_test_files = ['test/neg', 'test/pos']

get_train_test(path_to_train_files, 'data/train.txt')
get_train_test(path_to_test_files, 'data/test.txt')
