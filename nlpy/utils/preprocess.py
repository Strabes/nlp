from itertools import chain
from re import split
#from .vocab import TOKENIZER
from typing import List, Union, Tuple

def _pack_complete(completed_tokens: List[List],
    candidates: List[List], len_threshold: int,
    max_remainder_len : Union[None,int] = 1):
    """
    Helper function for determining if packing is
    complete

    Paramters
    ---------
    completed_tokens : List[List]
        list of token lists that are completed
    
    candidates : List[List]
        list of token lists that may be complete

    len_threshold : int
        token lists must be at least this long to be
        considered complete

    max_remainder_len : Union[None,int]
        if not None, len(remainder) > max_remainder_len
        raises an error

    Returns
    -------
    completed_tokens, remainder : Tuple[List[List], List]
    """
    remainder = []
    for token_list in candidates:
        if len(token_list) >= len_threshold:
            completed_tokens.append(token_list)
        else:
            remainder.append(token_list)
    if max_remainder_len:
        if len(remainder) > max_remainder_len:
            raise(ValueError(
                f"""max_remainder_len = {max_remainder_len} but
                len(remainder) = {len(remainder)}"""))
    return completed_tokens, remainder
        



def _token_packing(curr_tokens : List = [],
    next_tokens : List = [], max_len : int = 512,
    min_len:int=5, sep_token:str="<SEP>",
    packing_tolerance:int=8):
    """
    Pack sequence of tokens together

    Parameters
    ----------
    curr_tokens : List
        list of tokens to be completed

    next_tokens : List
        candidate token list

    max_len : int
        max token list length

    min_len : int
        min length of next_tokens if they are to be used

    sep_tokens : str
        separator token, default: "<SEP>"

    packing_tolerance : int

    Returns : 
    """

    completed_tokens = []

    # curr_tokens should not already be longer than max_len
    if len(curr_tokens) > max_len:
        raise ValueError(
            f"len(curr_tokens) must be less than {max_len}" +
            f"but len(curr_tokens) = {len(curr_tokens)}")
    # if curr_tokens is already completed:
    elif len(curr_tokens) >= max_len - packing_tolerance:
        completed_tokens.append(curr_tokens)
        curr_tokens = []

    # if no next_tokens:
    if next_tokens is None:
        return completed_tokens, curr_tokens
    # if next_tokens is too short:
    elif len(next_tokens) < min_len:
        return completed_tokens, curr_tokens
    # else we pack:
    else:
        if len(curr_tokens) > 0:
            candidates = curr_tokens + [sep_token] + next_tokens
        else:
            candidates = next_tokens
        candidates = [candidates[i:i+max_len]
            for i in range(0,len(candidates), max_len)]

        completed_tokens, remainder = _pack_complete(
            completed_tokens, candidates,
            len_threshold=max_len - packing_tolerance)
        
        if len(remainder) == 1: curr_tokens = remainder[0]
        else: curr_tokens = []

        return completed_tokens, curr_tokens



# wikitext2_loc = "C:/Users/grego/Documents/GitHub/nlp/data/wikitext2.txt"
# with open(wikitext2_loc,"r") as f:
#     zzz = text_packing(f, max_len=16)
