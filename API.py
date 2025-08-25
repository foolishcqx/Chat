from openai import OpenAI

client = OpenAI(
    base_url='https://api-inference.modelscope.cn/v1',
    api_key='ms-cb688843-be74-4cdf-8e0b-6237eda42d1e', # ModelScope Token
)

# set extra_body for thinking control
extra_body = {
    # enable thinking, set to False to disable
    "enable_thinking": False,
    # use thinking_budget to contorl num of tokens used for thinking
    # "thinking_budget": 4096
}

response = client.chat.completions.create(
    model='Qwen/Qwen3-32B',  # ModelScope Model-Id
    # model = 'Qwen/Qwen3-0.6B'
    messages=[
        {
          'role': 'user',
          'content': '9.9和9.11谁大'
        }
    ],
    stream=False,
    extra_body=extra_body
)
reply = response.choices[0].message.content.strip()
print(reply)