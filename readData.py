#%%
import json
import gzip
import os
import argparse

#%%


def readData(input_path, split_type, cpc_code):
    for file_name in os.listdir(os.path.join(input_path, split_type, cpc_code)):
        with gzip.open(
            os.path.join(input_path, split_type, cpc_code, file_name), "r"
        ) as fin:
            for row in fin:
                yield json.loads(row)


#%%
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Read data")
    parser.add_argument("--cpc_code", type=str, help="can be a, b,c,d,e,f,g,h,y")
    parser.add_argument("--split_type", type=str, help="can be train, test, val")
    parser.add_argument("--input_path", type=str, help="path to data")

    args = parser.parse_args()
    split_type = args.split_type
    cpc_code = args.cpc_code
    input_path = args.input_path

    readData(input_path, split_type, cpc_code)
