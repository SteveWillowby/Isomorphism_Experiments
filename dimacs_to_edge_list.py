import sys

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
