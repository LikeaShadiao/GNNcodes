import requests
url = 'https://api.openai.com/v1/engines/davinci-codex/completions'

data = {
    'prompt': '我想知道如何接入 ChatGPT 的 API。',
    'max_tokens': 50,
    'temperature': 0.5,
    'stop': '.'
}

headers = {
    'Content-Type': 'application/json',
    'Authorization': 'Bearer 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'
}

response = requests.post(url, headers=headers, json=data, verify=False)

if response.status_code == 200:
    response_json = response.json()
    answer = response_json['choices'][0]['text']
    print(answer)
else:
    print('请求失败，错误代码：', response.status_code)
