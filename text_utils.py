import string
import re
import spacy
from collections import defaultdict


SYM_MASK = re.compile(f'([{string.punctuation}“”¨«»®´·º½¾¿¡§£₤‘’])')


def tokenize_text(s):
    """
    Takes in a string, and splits on
    punctuation and other misc. symbols
    """
    return SYM_MASK.sub(r' \1 ', s).split()


def numericalize_tok(tokens,
                     max_vocab=50000,
                     min_freq=0,
                     unknown_token="_unk_",
                     padding_token="_pad_",
                     sent_beg_token="_bos_",
                     sent_end_token="_eos_"):
    """Takes in text tokens and returns int2tok and tok2int converters

        Arguments:
        tokens(list): List of tokens. Can be a list of strings, or a list of lists of strings.
        max_vocab(int): Number of tokens to return in the vocab (sorted by frequency)
        min_freq(int): Minimum number of instances a token must be present in order to be preserved.
        unk_tok(str): Token to use when unknown tokens are encountered in the source text.
        pad_tok(str): Token to use when padding sequences.
    """

    # check to see if a string was submitted
    # instead of a list
    if isinstance(tokens, str):
        raise ValueError("Expected to receive a list of tokens. Received a string instead")

    # if its a nested list, unlist it
    if isinstance(tokens[0], list):
        tokens = [itm for sublist in tokens for itm in sublist]

    # frequency of token
    freq = Counter(tokens)

    int2token = [token for token, ct in freq.most_common(max_vocab) if ct > min_freq]
    unk_id = 3

    # inserts some of the standard tokens
    # into the vocabulary
    int2token.insert(0, sent_beg_token)
    int2token.insert(1, padding_token)
    int2token.insert(2, sent_end_token)
    int2token.insert(unk_id, unknown_token)

    # create the reverse lookup
    token2int = defaultdict(lambda: unk_id, {token: idx for idx, token in enumerate(int2token)})
    return int2token, token2int


class Tokenizer(object):
    def __init__(self, lang='en'):

        # specific mask for <br>
        self.re_linebreak_mask = re.compile(r'<\s*br\s*/?>', re.IGNORECASE)

        # load spacy language
        self.spacy_model = spacy.load(lang)
        for w in ('<eos>', '<bos>', '<unk>'):
            self.tok.tokenizer.add_special_case(w, [{ORTH: w}])

    def sub_linebreak(self, x):
        return self.re_linebreak_mask.sub("\n", x)

    def spacy_token(self, x):
        return [t.text for t in self.spacy_model.tokenizer(self.sub_linebreak(x))]

    re_rep = re.compile(r'(\S)(\1{3,})')
    re_word_rep = re.compile(r'(\b\w+\W+)(\1{3,})')

    @staticmethod
    def replace_rep(m):
        TK_REP = 'tk_rep'
        c, cc = m.groups()
        return f' {TK_REP} {len(cc)+1} {c} '

    @staticmethod
    def replace_wrep(m):
        TK_WREP = 'tk_wrep'
        c, cc = m.groups()
        return f' {TK_WREP} {len(cc.split())+1} {c} '

    @staticmethod
    def do_caps(ss):
        """
        Notes when capitization happens, also notes when
        complete sentence is uppercase. And also notes if
        the sentence is mixed
        """
        TOK_UP = ' t_up '
        TOK_SENT = ' t_st '
        TOK_MIX = ' t_mx '

        res = []
        prev = '.'

        # mask
        re_word = re.compile('\w')
        re_nonsp = re.compile('\S')
        for s in re.findall(r'\w+|\W+', ss):
            res += ([TOK_UP, s.lower()] if (s.isupper() and (len(s) > 2)) else [s.lower()])

        return ''.join(res)

    def proc_text(self, s):
        s = self.re_rep.sub(Tokenizer.replace_rep, s)
        s = self.re_word_rep.sub(Tokenizer.replace_wrep, s)
        s = Tokenizer.do_caps(s)
        s = re.sub(r'([/#])', r' \1 ', s)
        s = re.sub(' {2,}', ' ', s)
        return self.spacy_tok(s)

    @staticmethod
    def proc_all(ss, lang):
        tok = Tokenizer(lang)
        return [tok.proc_text(s) for s in ss]

    @staticmethod
    def proc_all_mp(ss, lang='en', ncpus=None):
        ncpus = ncpus or num_cpus() // 2
        with ProcessPoolExecutor(ncpus) as e:
            return sum(e.map(Tokenizer.proc_all, ss, [lang] * len(ss)), [])
