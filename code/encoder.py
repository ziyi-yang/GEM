import numpy as np
from numpy import linalg as LA
from scipy.stats import pearsonr
import nltk
import io
import random

EPS = 5e-7

emb_matrix = np.load("data/emb_{0}.npy".format("lexvec"), allow_pickle=True, encoding = 'latin1')
word2id = np.load("data/word2id_{0}.npy".format("lexvec"), allow_pickle=True, encoding = 'latin1')
word2id = word2id.item()

emb_matrix_psl = np.load("data/emb_{0}.npy".format("psl"), allow_pickle=True, encoding = 'latin1')
word2id_psl = np.load("data/word2id_{0}.npy".format("psl"), allow_pickle=True, encoding = 'latin1')
word2id_psl = word2id_psl.item()

emb_matrix_ftt = np.load("data/emb_{0}.npy".format("ftt"), allow_pickle=True, encoding = 'latin1')
word2id_ftt = np.load("data/word2id_{0}.npy".format("ftt"), allow_pickle=True, encoding = 'latin1')
word2id_ftt = word2id_ftt.item()

oov = {}
oov_psl = {}
oov_ftt = {}

def gs(A):
    """
    Applies the Gram-Schmidt method to A
    and returns Q and R, so Q*R = A.
    """
    R = np.zeros((A.shape[1], A.shape[1]))
    Q = np.zeros(A.shape)
    A_c = np.copy(A)
    for k in range(0, A.shape[1]):
        R[k, k] = np.sqrt(np.dot(A_c[:, k], A_c[:, k]))
        if R[k, k] < EPS:
            R[k, k] = 0
            continue
        Q[:, k] = A_c[:, k]/R[k, k]
        for j in range(k+1, A.shape[1]):
            R[k, j] = np.dot(Q[:, k], A_c[:, j])
            A_c[:, j] = A_c[:, j] - R[k, j]*Q[:, k]
    return Q, R

def sent_to_tokens(sent):
    sent = sent.replace("''", '" ')
    sent = sent.replace("``", '" ')
    tokens = [token.lower().replace("``", '"').replace("''", '"') for token in nltk.wordpunct_tokenize(sent)]
    return tokens

def rm_pr(m, C_0):
    if C_0.ndim == 1:
        C_0 = np.reshape(C_0, [-1, 1])

    w = np.transpose(C_0).dot(m)
    return m - C_0.dot(w)

def ngram(s_num, C_0, sgv_c, win_sz = 7):
    n_pc = np.shape(C_0)[1]
    num_words = np.shape(s_num)[1]
    wgt = np.zeros(num_words)

    for i in range(num_words):
        beg_id = max(i - win_sz, 0)
        end_id = min(i + win_sz, num_words - 1)
        ctx_ids = list(range(beg_id, i)) + list(range(i+1, end_id + 1))
        m_svd = np.concatenate((s_num[:, ctx_ids], (s_num[:, i])[:, np.newaxis]), axis = 1)

        U, sgv, _ = LA.svd(m_svd, full_matrices = False)

        l_win = np.shape(U)[1]
        q, r = gs(m_svd)
        norm = LA.norm(s_num[:, i], 2)

        w = q[:, -1].dot(U)
        w_sum = LA.norm(w*sgv, 2)/l_win

        kk = sgv_c*(q[:, -1].dot(C_0))
        wgt[i] = np.exp(r[-1, -1]/norm) + w_sum + np.exp((-LA.norm(kk, 2))/n_pc)
    # print wgt
    return wgt

def sent_to_ids(sent, word2id, tokens, oov):
    """
    sent is a string of chars, return a list of word ids
    """
    if tokens is None:
        tokens = sent_to_tokens(sent)
    ids = []

    for w in tokens:
        if w in ['!', '.', ':', '?', '@', '-', '"', "'"]: continue
        if w in word2id:
            id = word2id[w]
        elif 'unk' in word2id:
            # OOV tricks
            if w in oov:
                id = oov[w]
            else:
                id = random.choice(range(len(word2id)))
                oov[w] = id
        ids.append(id)
    return ids

def str_2_num(s1):
    tokens = sent_to_tokens(s1)
    s_num1 = emb_matrix[sent_to_ids(s1, word2id, tokens, oov), :]
    s_num2 = emb_matrix_psl[sent_to_ids(s1, word2id_psl, tokens, oov_psl), :]
    s_num3 = emb_matrix_ftt[sent_to_ids(s1, word2id_ftt, tokens, oov_ftt), :]
    matrix = np.transpose(np.concatenate((s_num1, s_num2, s_num3), axis = 1))
    return matrix

def svd_sv(s1, factor = 3):
    s_num = str_2_num(s1)
    U, s, Vh = LA.svd(s_num, full_matrices = False)
    vc = U.dot(s**factor)
    return vc

def feat_extract(m1, n_rm, C_all, soc):
    w1 = LA.norm(np.transpose(m1).dot(C_all)*soc, axis = 0)
    id1 = w1.argsort()[-n_rm:]
    return id1

def encoder(encoding_list, corpus_list, dim = 900, n_rm = 17, max_n = 45, win_sz = 7):
    """
    corpus_list: the list of corpus, in the case of STS benchmark, it's s1 + s2
    encoding_list: the list of sentences to encode
    dim: the dimension of sentence vector
    """
    s_univ = np.zeros((dim, len(corpus_list)))
    encoded = []
    for j, sent in enumerate(corpus_list):
        s_univ[:, j] = svd_sv(sent)
    U, s, V = LA.svd(s_univ, full_matrices = False)
    C_all = U[:, :max_n]
    soc = s[:max_n]
    for j, sent in enumerate(encoding_list):
        m = str_2_num(sent)
        id1 = feat_extract(m, n_rm, C_all, soc)
        C_1 = C_all[:, id1]
        sgv = soc[id1]
        m_rm = rm_pr(m, C_1)
        v = m_rm.dot(ngram(m, C_1, sgv, win_sz))
        encoded.append(v)
    return encoded

def main():
    #first encoder STS benchmark test into a list of sentence
    sts_path = "data/stsbenchmark/sts-test.csv"
    # sts_path = "data/stsbenchmark/sts-dev.csv"
    #list for the first sentence
    s1 = []
    #list for the second sentence
    s2 = []

    golden_arr = []
    score_arr = []
    with io.open(sts_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            larr = line.split('\t')
            s1.append(larr[5])
            s2.append(larr[6])
            golden_arr.append(float(larr[4]))

    s1_num_list = encoder(s1, s1 + s2)
    s2_num_list = encoder(s2, s1 + s2)

    for j in range(len(s1_num_list)):
        v1_unit = s1_num_list[j]/LA.norm(s1_num_list[j], 2)
        v2_unit = s2_num_list[j]/LA.norm(s2_num_list[j], 2)
        score_arr.append(v1_unit.dot(v2_unit))
    print ("Pearson: ", 100 * pearsonr(score_arr, golden_arr)[0])

if __name__ == "__main__":
    main()
