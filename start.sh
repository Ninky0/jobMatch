#!/bin/bash

# vLLM 서버 실행 (백그라운드)
python3 -m vllm.entrypoints.openai.api_server \
  --model /app/gemma2_model/ko-gemma-2-9b-it \
  --port 8000 \
  --gpu-memory-utilization 0.95 \
  --dtype bfloat16 \
  --max-model-len 2048 \
  --max-num-seqs 32 &

# FastAPI 실행 (main_vllm.py 위치가 older 폴더 안이므로)
uvicorn llm_api.older.main_vllm:app --host 0.0.0.0 --port 3000 &

# nginx 실행
nginx -g 'daemon off;'
