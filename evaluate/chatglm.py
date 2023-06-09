import openai

# Point requests to Basaran by overwriting openai.api_base.
# Or you can use the OPENAI_API_BASE environment variable instead.
openai.api_base = "http://192.168.0.53:80/v1"

# Enter any non-empty API key to pass the client library's check.
openai.api_key = "xxx"


def answer_fn(question):
    completion = openai.ChatCompletion.create(
        model="chatglm-6b",
        messages=[
            {"role": "user", "content": question},
        ],
    )
    return completion["choices"][0]["message"]["content"]


def main():
    import pandas as pd

    questions = pd.read_csv("questions.csv")
    questions["chatglm-6b"] = questions["question"].apply(answer_fn)
    questions.to_csv("./eval_result.csv", index=False)


if __name__ == "__main__":
    main()
