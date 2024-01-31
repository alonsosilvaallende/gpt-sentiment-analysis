# GPT-4 for sentiment analysis

This app uses `max_tokens`, `logprobs` and `logit_bias` to force GPT-4 to do sentiment analysis:
- It forces GPT-4 to give only one token as a response to the prompt by setting `max_tokens=1`.
- It uses logit bias to force GPT-4 that the only available options are the tokens `Positive` and `Negative`:
```python
logit_bias={
    36590: 100, # 36590 is the token for 'Positive'
    39589: 100 # 39589 is the token for 'Negative'
}
```
- It uses `logprobs` to get the probabilities that GPT-4 assigns to those two options by setting `logprobs=True` and `top_logprobs=2`.

The final call to GPT-4 is then
```python
client.chat.completions.create(
    model="gpt-4",
    messages=[
        {"role": "system", "content": "You are a helpful sentiment analyzer assistant. Your task is to determine what is the sentiment conveyed by the text."},
        {"role": "user", "content": input}],
    logprobs=True,
    top_logprobs=2,
    logit_bias={36590: 100, 39589: 100},
    max_tokens=1,
)
```
