from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file

import numpy as np
from openai import AsyncOpenAI
import solara
import asyncio

def plot_predicted_sentiment(positive, negative): return {
    "bars": {
        "title": {},
        "tooltip": {
            "trigger": 'item', 
            "formatter": '{b} : {c}%'
        },
        "yAxis": {"type": "category", "data": ['Positive', 'Negative']},
        "xAxis": {"type": "value"},
        "emphasis": {"itemStyle": {"borderRadius": 2}},
        "series": [
    {
      "data": [
                    {"name": "Positive", "value": positive},
                    {"name": "Negative", 
                     "value": negative, 
                     "itemStyle": {
                         "color": '#a90000'
                         }
                    },
                ],
      "type": 'bar',
    }
        ],
    },
    "pie": {
        "title": {},
        "tooltip": {
            "trigger": 'item', 
            "formatter": '{b} : {c}%'
        },
        "series": [
            {
                "name": "sales",
                "type": "pie",
                "radius": [0, "50%"],
                "data": [
                    {"name": "Positive", "value": positive},
                    {"name": "Negative", "value": negative,
                        "itemStyle": {
            "color": '#a90000'
          }},
                ],
                "universalTransition": True,
            }
        ],
    },
}

def plot_scores_by_words(splitted_phrase, scores):
    colors = ["#a90000" if scores[i] < 0 else "#0000a9" for i in range(len(scores))]
    return {
        "title": {},
        "tooltip": {
            "trigger": 'item', 
            "formatter": '{b} : {c}'
        },
        "yAxis": {"type": "category", "data": splitted_phrase},
        "xAxis": {"type": "value"},
        "emphasis": {"itemStyle": {"borderRadius": 2}},
        "series": [
            {
              "data": [{"name": word, "value": val, "itemStyle": {"color": color} } for word, val, color in zip(splitted_phrase, scores, colors)],
              "type": 'bar',
            }
        ],
    }


def my_function(result):
    top_logprobs = result.choices[0].logprobs.content[0].top_logprobs
    dict = {}
    for i in range(len(top_logprobs)):
        dict[top_logprobs[i].token] = top_logprobs[i].logprob
    return np.exp(dict['Positive'])/(np.exp(dict['Positive']) + np.exp(dict['Negative'])), np.exp(dict['Negative'])/(np.exp(dict['Positive']) + np.exp(dict['Negative']))

input_text = solara.reactive("That was lovely but I hated the outcome")
@solara.component
def Page():
    async def response(input):
        return await client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful sentiment analyzer assistant. Your task is to determine what is the sentiment conveyed by the text."},
            {"role": "user", "content": f"{input}."}],
        logprobs=True,
        top_logprobs=2,
        logit_bias={36590: 100, 39589: 100},
        max_tokens=1,
        )
    async def invoke_concurrently():
        tasks = [response(prompt) for prompt in prompts]
        return await asyncio.gather(*tasks)

    with solara.Head():
        solara.Title("GPT-4 for sentiment analysis")
    option, set_option = solara.use_state("bars")
    with solara.Column(margin=4):
        with solara.Card("GPT-4 for sentiment analysis"):
            solara.InputText("Enter some text and press ENTER", value=input_text)
            phrase = input_text.value
            splitted_phrase = phrase.split(" ")
            # Original prompt and for each word consider the original prompt except that word
            prompts = [" ".join(splitted_phrase)] + [" ".join([v for i, v in enumerate(splitted_phrase) if i != index_to_remove]) for index_to_remove in range(len(splitted_phrase))]
            client = AsyncOpenAI()
            all_persons = asyncio.run(invoke_concurrently())
            dict0 = my_function(all_persons[0])
            positive, negative = dict0
            if positive > negative:
                predicted_sentiment = "POSITIVE"
            elif negative > positive:
                predicted_sentiment = "NEGATIVE"
            else:
                predicted_sentiment = "UNDECIDED"
            solara.Markdown(f"##Predicted sentiment: {predicted_sentiment}")
            positive = np.round(100*positive, 2)
            negative = np.round(100*negative, 2)
            with solara.ToggleButtonsSingle("bars", on_value=set_option):
                solara.Button("bars")
                solara.Button("pie")
            old_options = plot_predicted_sentiment(positive, negative)
            solara.FigureEcharts(option=old_options[option])

        with solara.Card(title="eXplainable GPT-4", subtitle="Leave-one-feature-out importance"):
            scores = []
            for i in range(1, len(all_persons)):
                dictX = my_function(all_persons[i])
                scores.append(dict0[0] - dictX[0])
            option = plot_scores_by_words(splitted_phrase, scores)
            solara.FigureEcharts(option=option)
