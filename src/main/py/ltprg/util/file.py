import os
import os.path as path

def make_indexed_dir(path_prefix):
    idx = 0
    created_output = False
    output_path = None
    while not created_output:
        output_path = path_prefix + "_" + str(idx)
        if not path.exists(output_path):
            os.makedirs(output_path)
            created_output = True
        idx += 1
    return output_path