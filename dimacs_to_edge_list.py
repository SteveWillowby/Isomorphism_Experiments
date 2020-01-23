import sys
from os import listdir
from os.path import isfile, join

def convert(input_file_name, output_file_name):
    input_file = open(input_file_name, 'r')
    output_file = open(output_file_name, 'w')

    old_lines = input_file.readlines()
    old_lines = old_lines[2:]
    old_lines = [line.split(' ') for line in old_lines]

    new_lines = {}
    for old_line in old_lines:
        old_line[2] = old_line[2].split('\n')
        old_line[2] = old_line[2][0]
        node = old_line[1]
        if node in new_lines:
            new_lines[node] = new_lines[node] + ' ' + old_line[2]
        else:
            new_lines[node] = node + ' ' + old_line[2]

    for node, new_line in new_lines.items():
        output_file.write(new_line + '\n')

    input_file.close()
    output_file.close()

if __name__ == "__main__":
    input_directory = sys.argv[1]
    output_directory = sys.argv[2]
    base_names = [f for f in listdir(input_directory) if isfile(join(input_directory, f))]
    for base_name in base_names:
        output_name = join(output_directory, base_name + ".edge_list")
        input_name = join(input_directory, base_name)
        print(input_name)
        print(output_name)
        convert(input_name, output_name)
