def task_prompt(user_input: str, pipeline):
    """
    프롬프트 템플릿을 구성하고 LLM 파이프라인을 실행하는 함수.
    """
    messages = [
        {"role": "user", "content": user_input}
    ]

    # 채팅 스타일 프롬프트 적용
    prompt = pipeline.tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    # 종료 토큰 지정
    terminators = [
        pipeline.tokenizer.eos_token_id,
        pipeline.tokenizer.convert_tokens_to_ids("<end_of_turn>")
    ]

    outputs = pipeline(
        prompt,
        max_new_tokens=2048,
        eos_token_id=terminators,
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
    )

    return outputs, len(prompt)
