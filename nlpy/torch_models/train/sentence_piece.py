import os
import tempfile
from typing import List
from torchtext.data.functional import generate_sp_model


def train_sp_model(text:List[str], vocab_size:int=10000,
                   model_type:str='unigram',
                   model_prefix='sent_piece', target_dir=None):
    """
    Fit a SentencePiece model to a list of texts.

    Parameters
    ----------
    text : List[str]
        texts to train the model

    vocab_size : int
        vocabulary size

    model_type : str

    model_prefix : str

    target_dir : str
        directory to write SentencePiece model
    """
    if isinstance(text,list):
        tmp = tempfile.NamedTemporaryFile(mode="w+t", delete=False)
        for item in text:
            tmp.write("%s\n" % item)
        orig_cwd = os.getcwd()
        os.chdir(target_dir)
        generate_sp_model(tmp.name, vocab_size=vocab_size,
        model_prefix=model_prefix, model_type=model_type)
        os.chdir(orig_cwd)
        os.unlink(tmp.name)