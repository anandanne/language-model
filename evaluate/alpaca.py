import openai

# Point requests to Basaran by overwriting openai.api_base.
# Or you can use the OPENAI_API_BASE environment variable instead.
openai.api_base = "http://192.168.0.59:80/v1"

# Enter any non-empty API key to pass the client library's check.
openai.api_key = "xxx"


def answer_fn(question):
    completion = openai.ChatCompletion.create(
        model="chinese-alpaca",
        messages=[
            {"role": "user", "content": question},
        ],
    )
    return completion["choices"][0]["message"]["content"]


def main():
    import pandas as pd

    questions = pd.read_csv("questions.csv")
    questions["chinese-alpaca"] = questions["question"].apply(answer_fn)
    questions.to_csv("./questions.csv", index=False)


if __name__ == "__main__":
    main()
