import argparse

PAD_SYMBOL = "<PAD>"
UNK_SYMBOL = "<UNK>"

DATA_DIR = '../mimicdata/'
CAML_DIR = '../mimicdata/caml/'

GENERATED_DIR = '../mimicdata/generated/'
NOTEEVENTS_FILE_PATH = '../mimicdata/NOTEEVENTS.csv'
DIAGNOSES_FILE_PATH = '../mimicdata/DIAGNOSES_ICD.csv'
PORCEDURES_FILE_PATH = '../mimicdata/PROCEDURES_ICD.csv'
DIAG_CODE_DESC_FILE_PATH = '../mimicdata/D_ICD_DIAGNOSES.csv'
PROC_CODE_DESC_FILE_PATH = '../mimicdata/D_ICD_PROCEDURES.csv'
ICD_DESC_FILE_PATH = '../mimicdata/ICD9_descriptions'

VOCAB_FILE_PATH = '../mimicdata/generated/vocab.csv'
EMBED_FILE_PATH = '../mimicdata/generated/vocab.embed'
CODE_FREQ_PATH = '../mimicdata/generated/code_freq.csv'
CODE_DESC_VECTOR_PATH = '../mimicdata/generated/code_desc_vectors.csv'

FULL = 'full'
TOP50 = '50'


# Debug version
def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--log', default="INFO", help="Logging level.")
    parser.add_argument('--random_seed', default=271, help="Random seed.")

    parser.add_argument(
        '--data_setting',
        type=str,
        default=TOP50,
        help='Data Setting (full or top50)'
    )

    parser.add_argument(
        '--model',
        type=str,
        default='TransICD',
        help='Transformer or TransICD models'
    )

    parser.add_argument(
        '--num_epoch',
        type=int,
        default=[30, 35, 40],
        nargs='+',
        help='Number of epochs to train.'
    )

    parser.add_argument(
        '--learning_rate',
        type=float,
        default=[0.001],
        nargs='+',
        help='Initial learning rate.'
    )

    parser.add_argument(
        '--batch_size',
        type=int,
        default=8,
        help='Batch size. Must divide evenly into the dataset sizes.'
    )

    parser.add_argument(
        '--max_len',
        type=int,
        default=2500,
        help='Max Length of discharge summary'
    )

    parser.add_argument(
        '--embed_size',
        type=int,
        default=128,
        help='Embedding dimension for text token'
    )

    parser.add_argument(
        '--freeze_embed',
        action='store_true',
        default=True,
        help='Freeze CBOW embedding or fine tune'
    )

    parser.add_argument(
        '--label_attn_expansion',
        type=int,
        default=2,
        help='Expansion factor for attention model'
    )

    parser.add_argument(
        '--num_trans_layers',
        type=int,
        default=2,
        help='Number of transformer layers'
    )

    parser.add_argument(
        '--num_attn_heads',
        type=int,
        default=8,
        help='Number of transformer attention heads'
    )

    parser.add_argument(
        '--trans_forward_expansion',
        type=int,
        default=4,
        help='Factor to expand transformers hidden representation'
    )

    parser.add_argument(
        '--dropout_rate',
        type=float,
        default=0.1,
        help='Dropout rate for transformers'
    )

    args = parser.parse_args()  # '--target_kernel_size 4 8'.split()
    return args

