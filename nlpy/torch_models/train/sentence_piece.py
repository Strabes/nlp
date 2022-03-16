import os
import tempfile
from torchtext.data.functional import generate_sp_model


def train_sp_model(text, vocab_size=10000, model_type='unigram',
                   model_prefix='sent_piece', target_dir=None):
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