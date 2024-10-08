import argparse
import datetime
import glob
import numpy as np
import os.path
import pandas as pd
from allamo.logging import configure_logger, logger

EOS_TOKEN = "</s>"

def init_tokenizer(output_dir, tiktoken_tokenizer_name, hf_tokenizer_path):
    vocab_size = None
    if hf_tokenizer_path:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(hf_tokenizer_path)
        if vocab_size is None:
            vocab_size = len(tokenizer)
        logger.info(f"HuggingFace {hf_tokenizer_path} tokenizer loaded with the vocab size {vocab_size}")
    elif tiktoken_tokenizer_name:
        import tiktoken
        tokenizer = tiktoken.get_encoding(tiktoken_tokenizer_name)
        if vocab_size is None:
            vocab_size = tokenizer.max_token_value + 1 # values start from 0
        logger.info(f"Tiktoken {tiktoken_tokenizer_name} tokenizer loaded with the vocab size {vocab_size}")
    else:
        raise Exception('Tokenizer is not provided. Please specify either a Tiktoken tokenizer or a HuggingFace tokenizer')
    return tokenizer

def load_list_of_txt_files(index_file_path, input_data_dir, data_split):
    if index_file_path:
        # Load the csv file into a pandas dataframe
        index_df = pd.read_csv(index_file_path)

        # Replace any NaN values in the "File" column with an empty string
        index_df['File'] = index_df['File'].fillna('')

        # Filter the dataframe to only include rows where the "File" column ends with ".txt"
        txt_files_df = index_df[(index_df['File'].str.endswith('.txt'))]
        if 'Split' not in txt_files_df.columns:
            txt_files_df['Split'] = data_split if data_split else 'train'
    elif input_data_dir:
        txt_files = glob.glob(os.path.join(input_data_dir, "*.txt"))
        txt_files_df = pd.DataFrame({'File': txt_files})
        txt_files_df['Split'] = data_split if data_split else 'train'
    else:
        raise Exception('Either an index file or an input data dir must be provided')
    
    logger.info(f"{len(txt_files_df)} txt files found to process")
    return txt_files_df
    

def encode_file(input_file, output_file, tokenizer):
    enc_data = tokenizer.encode(input_file.read())
    enc_data = np.array(enc_data, dtype=np.uint16)
    enc_data.tofile(output_file)
    tokens = len(enc_data)
    
    enc_data = tokenizer.encode(EOS_TOKEN)
    enc_data = np.array(enc_data, dtype=np.uint16)
    enc_data.tofile(output_file)
    tokens += len(enc_data)
    
    return tokens


def create_datasets(txt_files_df, tokenizer, input_data_dir, output_data_dir):
    train_tokens = 0
    val_tokens = 0
    files_cnt = 0
    train_file_path = os.path.join(output_data_dir, 'train.bin')
    val_file_path = os.path.join(output_data_dir, 'val.bin')
    with open(train_file_path, 'wb+') as train_file, open(val_file_path, 'wb+') as val_file:
        # Process each of the txt files
        for _, row in txt_files_df.iterrows():
            filename = os.path.join(input_data_dir, row['File']) if input_data_dir else row['File']
            if not os.path.isfile(filename):
                logger.info(f"File {filename} does not exist.")
                continue
            with open(filename, 'r', encoding="utf-8") as txt_file:
                logger.info(f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - Start processing {filename}")
                if row['Split'] == 'test':
                    tokens = encode_file(txt_file, val_file, tokenizer)
                    val_tokens += tokens
                else:
                    tokens = encode_file(txt_file, train_file, tokenizer)
                    train_tokens += tokens
                logger.info(f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {filename} added ({tokens} tokens) to the {row['Split']} dataset")
                files_cnt += 1
                    
    total_tokens = train_tokens + val_tokens
    logger.info(f"Datasets created in {output_data_dir} from {files_cnt} files. Tokens: {total_tokens:,} (Train: {train_tokens:,} Val: {val_tokens:,})")

if __name__ == '__main__':
    configure_logger()
    parser = argparse.ArgumentParser(description='Prepare your datasets')
    parser.add_argument('--index_file', type=str, help='Path to an index file')
    parser.add_argument('--input_data_dir', type=str, help='Path to a directory with txt files')
    parser.add_argument('--data_split', type=str, default='train', choices=['train', 'test'], help='Data split')
    parser.add_argument('--output_data_dir', type=str, required=True, help='Path to a directory for output dataset files')
    parser.add_argument('--tiktoken_tokenizer_name', type=str, help='Tiktoken tokenizer name')
    parser.add_argument('--hf_tokenizer_path', type=str, help='HuggingFace tokenizer path')
    
    args = parser.parse_args()
    tokenizer = init_tokenizer(args.output_data_dir, args.tiktoken_tokenizer_name, args.hf_tokenizer_path)
    txt_files_df = load_list_of_txt_files(args.index_file, args.input_data_dir, args.data_split)
    create_datasets(txt_files_df, tokenizer, args.input_data_dir, args.output_data_dir)
