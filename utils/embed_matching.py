import dill
import torch.nn.functional as F
import re
import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from keybert import KeyBERT
# sumy : 텍스트 요약
# from sumy.parsers.plaintext import PlaintextParser
# from sumy.nlp.tokenizers import Tokenizer
# from sumy.summarizers.lex_rank import LexRankSummarizer
# import MeCab
# mecab = MeCab.Tagger("-r/workspace/mecab/mecab-0.996-ko-0.0.2/mecabrc")
# result = mecab.parse("테스트 문장입니다.")

# 형태소 분석기 ( 명사 추출용 )
from konlpy.tag import Kkma
kkma = Kkma()

# 임베딩 모델 로딩
snapshot = "/workspace/volume/dragonkue--bge-m3-ko/snapshots/c21b6c17c9313232db561666441d06c11d5c3384"
model = SentenceTransformer(snapshot)
K=10

# 키워드 추출 모델
kw_model = KeyBERT(snapshot)

# 데이터베이스 로드
with open('train_data.dill', 'rb') as f:
    trn = dill.load(f)
with open('jikjong_db.dill', 'rb') as f:
    jikjong = dill.load(f)
with open('license_db.dill', 'rb') as f:
    license_db = dill.load(f)
with open('major_db.dill', 'rb') as f:
    major_db = dill.load(f)

# html 문자 제거
def preprocess_keyword(kwrd):
    return re.sub(r"&[a-zA-Z]+;", '', kwrd)

# 예측 결과와 실제 라벨 간 recall 계산 ( 평가용 )
def get_recall(pred, label):
    bunmo = len(label)
    cnt = 0

    print("original_pred: ", pred)

    # 키워드 추출 -> 형태소 분석
    pred = keyword_extraction(" ".join(pred))
    print("keyword_pred: ", pred)
    pred = [i[0] for i in pred]

    pred_set = set()
    pred_pos = kkma.pos(" ".join(pred))
    for pos in pred_pos:
        # 일반명사, 외래어
        if pos[1] == 'NNG' or pos[1] == 'OL':
            pred_set.add(pos[0])
    print("kkma_pred: ", pred_set)

    for word in label:
        print("label: ", word)
        for this_pred in pred_set:
            if this_pred in word:
                cnt += 1
                break
    
    print(cnt/bunmo)
    return cnt/bunmo

# '입력 임베딩'과 'db 임베딩' 간 cosine 유사도 기반 top-k index 반환
def get_sim(inp_emb, db_emb):
    stacked_db_emb = torch.tensor(np.stack(db_emb))
    sim = F.cosine_similarity(inp_emb, stacked_db_emb)
    topk_ind = torch.argsort(sim, descending=True).tolist()
    return topk_ind

# 키워드 추출 함수 (keyBERT 사용)
def keyword_extraction(inp):
    keywords = kw_model.extract_keywords(inp,
                                        keyphrase_ngram_range=(1,1),
                                        use_maxsum = True,
                                        use_mmr = True,
                                        nr_candidates = 20,
                                        top_n=50,        
                                        diversity = 0.7
                                        )
    return keywords[:int(len(keywords)*0.4)+1]

# 입력 문장에서 직종 키워드를 추출하고, 임베딩 후 top-k 직종 추천
def predict_job_categories(sent, top_k=K):
    # 전처리(user input loac)
    user_inp = preprocess_keyword(sent)
    user_inp = [candidate[0] for candidate in keyword_extraction(user_inp)]
    user_inp_kwrd = ' '.join(user_inp)
    print("user_inp_kwrds:", user_inp_kwrd)

    # 임베딩 및 유사도 계산
    inp_embed = model.encode(user_inp_kwrd, convert_to_tensor=True)
    jikjong_ind = get_sim(inp_embed, jikjong.db['JOBS_CD_KOR_NM_EMBED'])

    # 직종 이름 리스트 ( 임베딩 인덱싱 결과를 변환할 때 사용 )
    jikjong_list = jikjong.db['JOBS_CD_KOR_NM']
    
    # top-k 추천 직종 pred
    pred = []
    j = 0

    while len(pred) < top_k:
        this_jikjong = jikjong_list[jikjong_ind[j]]
        if this_jikjong not in pred:
            pred.append(this_jikjong)
        j += 1

    print("직종pred:", pred)
    print("*" * 50)
    return pred

# 입력 문장에서 자격증 키워드를 추출하고, 임베딩 후 top-k 자격증 추천
def predict_license_categories(sent, top_k=K):
    # 전처리
    user_inp = preprocess_keyword(sent)
    user_inp = [candidate[0] for candidate in keyword_extraction(user_inp)]
    user_inp_kwrd = ' '.join(user_inp)
    print("user_inp_kwrds:", user_inp_kwrd)

    # 임베딩 및 유사도 계산
    inp_embed = model.encode(user_inp_kwrd, convert_to_tensor=True)
    license_ind = get_sim(inp_embed, license_db.db['LICENSE_NM_EMBED'])

    # 자격증 이름 리스트
    license_list = license_db.db['LICENSE_NM']
    
    # top-k 추천 자격증
    pred = []
    j = 0

    while len(pred) < top_k:
        this_license = license_list[license_ind[j]]
        if this_license not in pred:
            pred.append(this_license)
        j += 1

    print("자격증pred:", pred)
    print("*" * 50)
    return pred

# 입력 문장에서 전공 키워드를 추출하고, 임베딩 후 top-k 전공 추천
def predict_major_categories(sent, top_k=K):
    # 전처리
    user_inp = preprocess_keyword(sent)
    user_inp = [candidate[0] for candidate in keyword_extraction(user_inp)]
    user_inp_kwrd = ' '.join(user_inp)
    print("user_inp_kwrds:", user_inp_kwrd)

    # 임베딩 및 유사도 계산
    inp_embed = model.encode(user_inp_kwrd, convert_to_tensor=True)
    major_ind = get_sim(inp_embed, major_db.db['MAJOR_NM_EMBED'])

    # 전공 이름 리스트
    major_list = major_db.db['MAJOR_NM']
    
    # top-k 추천 전공
    pred = []
    j = 0

    while len(pred) < top_k:
        this_major = major_list[major_ind[j]]
        if this_major not in pred:
            pred.append(this_major)
        j += 1

    print("전공pred:", pred)
    print("*" * 50)
    return pred

# 하나의 문장에 대한 직종, 자격증, 전공 추천 리스트 반환
def embed_matching(sent):
    # 직종은 6개, 자격증과 전공은 각각 3개씩 예측
    job_pred = predict_job_categories(sent, top_k=6)
    license_pred = predict_license_categories(sent, top_k=3)
    major_pred = predict_major_categories(sent, top_k=3)
    
    return {
        'jobs': job_pred,
        'licenses': license_pred,
        'majors': major_pred
    }

# 학습용 전체 데이터셋을 반복하면서 직종, 자격증, 전공 추천 및 평균 라벨 길이 반환
def matching():
    ave_len_label = 0
    results = {
        'jobs': [],
        'licenses': [],
        'majors': []
    }

    for i, user_inp in tqdm(enumerate(trn.db['STD_DTY_SWRD_CN']), total=trn.len):
        label = trn.db['STD_DTY_CN'][i].split(',')
        ave_len_label += len(label)

        # 직종은 6개, 자격증과 전공은 각각 3개씩 예측
        job_pred = predict_job_categories(user_inp, top_k=6)
        license_pred = predict_license_categories(user_inp, top_k=3)
        major_pred = predict_major_categories(user_inp, top_k=3)

        # 결과 저장
        results['jobs'].append(job_pred)
        results['licenses'].append(license_pred)
        results['majors'].append(major_pred)

    ave_len_label /= trn.len
    print("평균 라벨 길이:", ave_len_label)
    
    # 결과 분석
    print("\n=== 예측 결과 분석 ===")
    print(f"총 처리된 데이터 수: {len(results['jobs'])}")
    print(f"직종 예측 평균 개수: {np.mean([len(pred) for pred in results['jobs']]):.2f}")
    print(f"자격증 예측 평균 개수: {np.mean([len(pred) for pred in results['licenses']]):.2f}")
    print(f"전공 예측 평균 개수: {np.mean([len(pred) for pred in results['majors']]):.2f}")
    
    return results