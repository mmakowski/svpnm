import collections
import configparser
import logging
import multiprocessing as mp
import os
import re
import sys

import pandas as pd # type: ignore


# configuration
config = configparser.ConfigParser()
config.read('config.ini')
PARALLELISM = int(config['environment']['parallelism'])

# logging
logging.basicConfig(format='%(asctime)s %(process)s %(levelname)-8s %(message)s', stream=sys.stdout)
log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)

DatasetSpec = collections.namedtuple('DatasetSpec', ['file', 'max_documents'])

# a placeholder that will survive tokenisation
STRING_PLACEHOLDER = 'SVPNMSTRINGPLACEHOLDER'


def tokenise_data(dataset_specs,
                  min_token_occurrences, 
                  min_token_documents,
                  random_state=1):
    """
    Load content for specified datasets and tokenise it. It is assumed that the training set is
    specified in file `[xyz_]train.csv`, and that content for all datasets is in `content` subdirectory of
    the directory where the first dataset file is located.
    """
    datasets = {}
    for ds in dataset_specs:
        set_id = os.path.basename(ds.file).replace(".csv", "")
        if "_" in set_id:
            set_id = set_id.split("_")[1]
        log.debug("reading %s metadata...", set_id)
        data = pd.read_csv(ds.file)
        if ds.max_documents:
            datasets[set_id] = data.sample(n=ds.max_documents, random_state=random_state)
        else:
            datasets[set_id] = data

    content_dir = os.path.join(os.path.dirname(dataset_specs[0].file), "content")
    for set_id, dataset in datasets.items():
        log.debug("loading %s tokens...", set_id)
        dataset['tokens'] = _load_tokens(dataset, content_dir)

    log.info("creating vocabulary...")
    token_to_int_dict, vocab_size, unk_code, pad_code = _create_vocab_dict(datasets['train'],
                                                                           min_token_occurrences,
                                                                           min_token_documents)
    log.debug("vocabulary size: %d", vocab_size)

    for set_id, dataset in datasets.items():
        log.debug("vectorising %s...", set_id)
        dataset['vectors'] = _vectorise(dataset, token_to_int_dict, unk_code)
        dataset.drop('tokens', axis=1, inplace=True)
        dataset['label'] = _label_to_num(dataset)

    return datasets, vocab_size, pad_code


def denoise_c(code: str) -> str:
    code = _strip_comments(code)
    code = _replace_strings(code)
    tokens = _tokenise_ex(code)
    tokens = [_normalise(token) for token in tokens]
    return " ".join(tokens)


def _load_tokens(dataset, content_dir):
    with mp.Pool(PARALLELISM) as pool:
        return pool.map(_load_tokens_from_file, dataset['file'].apply(lambda f: os.path.join(content_dir, f)))


def _load_tokens_from_file(file_path):
    with open(file_path, "r") as f:
        return _tokenise(f.read())


def _tokenise(content):
    return list(map(_normalise, filter(lambda tok: len(tok) > 0 and not re.match(r'\s+', tok), re.split(r'(\s+|\W)', content))))


def _tokenise_ex(content):
    """Less naive tokenisation, to better take c operators into account"""
    pattern = re.compile(r'(\+\+|--|&&|\|\||->|==|>=|<=|!=|>>|<<|\s+|[^#\w])')
    return list(filter(lambda tok: len(tok) > 0 and not re.match(r'\s+', tok), 
                       re.split(pattern, content)))


def _normalise(token):
    if re.match(r'0x[0-9a-fA-F]+|\d+', token):
        return '<NUM>'
    elif token == STRING_PLACEHOLDER:
        return '<STR>'
    else:
        return token


def _create_vocab_dict(train, min_token_occurrences, min_token_documents):
    # how many times token appeared in the corpus
    token_counts = collections.Counter()
    # how many documents (source files) the token appeared in
    token_document_counts = collections.Counter()

    for tokens in train['tokens']:
        current_doc_token_counts = collections.Counter(tokens)
        token_counts.update(current_doc_token_counts)
        token_document_counts.update(current_doc_token_counts.keys())
    
    relevant_tokens = [tok for tok in token_counts.keys()
                       if token_counts[tok] >= min_token_occurrences and
                          token_document_counts[tok] >= min_token_documents]
    
    token_to_int_dict = dict(zip(relevant_tokens, range(len(relevant_tokens))))
    vocab_size = len(token_to_int_dict) + 2 # UNK and PAD
    unk_code = len(token_to_int_dict)
    pad_code = unk_code + 1
    
    return token_to_int_dict, vocab_size, unk_code, pad_code


def _vectorise(dataset, token_to_int_dict, unk_code):
    inputs = [(tokens, token_to_int_dict, unk_code) for tokens in dataset['tokens']]
    with mp.Pool(PARALLELISM) as pool:
        return pool.map(_vectorise_tokens, inputs)


def _vectorise_tokens(params):
    tokens, token_to_int_dict, unk_code = params
    return list(map(lambda token: token_to_int_dict.get(token, unk_code), tokens))


def _label_to_num(dataset):
    return dataset['label'].map(lambda x: 1 if x == 'VULNERABLE' else 0)


def _strip_comments(input_text):
    """
    after https://stackoverflow.com/questions/241327/python-snippet-to-remove-c-and-c-comments#241506
    """
    def replacer(match):
        s = match.group(0)
        if s.startswith('/'):
            return " " # note: a space and not an empty string
        else:
            return s
    pattern = re.compile(
        r'//.*?$|/\*.*?\*/|\'(?:\\.|[^\\\'])*\'|"(?:\\.|[^\\"])*"',
        re.DOTALL | re.MULTILINE
    )
    return re.sub(pattern, replacer, input_text)


def _replace_strings(code):
    """
    inspired by https://stackoverflow.com/a/481294/424978
    """
    pattern = re.compile(r'"([^\\"]|\\\\|\\|\\\n")*"', re.MULTILINE)
    return re.sub(pattern, " {} ".format(STRING_PLACEHOLDER), code)
