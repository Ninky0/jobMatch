import dill
import torch.nn.functional as F
import re
import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from keybert import KeyBERT
# sumy:텍스트요약
# from sumy.parsers.plaintext import PlaintextParser
# from sumy.nlp.tokenizers import Tokenizer
# from sumy.summarizers.lex_rank import LexRankSummarizer
# import MeCab
# mecab = MeCab.Tagger("-r /workspace/mecab/mecab-0.996-ko-0.9.2/mecabrc")
# result = mecab.parse("테스트 문장입니다")

from konlpy.tag import Kkma
kkma = Kkma()

snapshot = "/workspace/volume/dragonkue--bge-m3-ko/snapshots/c21b6c17c9313232db561666441d06c11d5c3384"
model = SentenceTransformer(snapshot)
K = 10

kw_model = KeyBERT(snapshot)

with open('jikjong_db.dill', 'rb') as f:
    jikjong = dill.load(f)

# html 문자 제거
def preprocess_keyword(kwrd):
    return re.sub(r"&[a-zA-Z]+;", '', kwrd)

def keyword_extraction(inp):
    keywords = kw_model.extract_keywords(inp,
                                         keyphrase_ngram_range=(1,1),
                                         use_maxsum=True,
                                         use_mmr=True,
                                         nr_candidates=20,
                                         top_n=50,
                                         diversity=0.7
                                         )

    return keywords[:int(len(keywords)*0.4)+1]

def get_sim(inp_emb, db_emb):
    stacked_db_emb = torch.tensor(np.stack(db_emb))
    sim = F.cosine_similarity(inp_emb, stacked_db_emb)
    topk_ind = torch.argsort(sim, descending=True).tolist()
    return topk_ind

def embed_matching(sent):

    jikjong_list = jikjong.db['JOBS_CD_KOR_NM']

    # user input load
    user_inp = preprocess_keyword(sent)
    user_inp = [candidate[0] for candidate in keyword_extraction(user_inp)]
    user_inp_kwrd = ''
    for inp in user_inp:
        user_inp_kwrd += inp + ' '
    # print("user_inp_kwrds:",user_inp_kwrd)
    inp_embed = torch.tensor(model.encode(user_inp_kwrd))
    jikjong_ind = get_sim(inp_embed, jikjong.db['JOBS_CD_KOR_NM_EMBED'])

    # 추천 직종 pred
    pred = []
    k_cnt = 0
    j = 0
    while True:
        this_jikjong = jikjong_list[jikjong_ind[j]]
        if this_jikjong not in pred:
            pred.append(this_jikjong)
            k_cnt += 1
        if k_cnt == K:
            break
        j += 1

    # print("*"*50)
    return pred
