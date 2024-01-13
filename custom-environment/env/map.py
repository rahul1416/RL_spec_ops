import numpy as np
import os

def read_map_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    rows = len(lines)
    cols = len(lines[0].strip())
    result = np.zeros((rows, cols), dtype=int)
    
    for i, line in enumerate(lines):
        for j, char in enumerate(line.strip()):
            if char == '.':
                result[i][j] = 0
            elif char == '#':
                result[i][j] = 4
    
    return result

# Usage:
# file_path = 'Maps/map_0.txt' 
# result_array = read_map_file(file_path)

