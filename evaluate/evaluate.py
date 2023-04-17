import openai

EVALUATE_PROMPT = "The followings are {num_models} ChatGPT-like systems' outputs. Please rate an overall score on a ten point scale for each and give explanations to justify your scores. 请使用中文回复。\n\n"


openai.api_key = "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"


def eval(x):
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": x},
        ],
    )
    return completion["choices"][0]["message"]["content"]


def prepare_query(row):
    models = []
    for k, v in row.items():
        if k not in ["question", "question_type", "eval"]:
            models.append((k, v))

    prompt = EVALUATE_PROMPT.format(num_models=len(models))
    prompt += f"Prompt:\n{row['question']}\n\n"

    model_strs = []
    for i, m in enumerate(models):
        model_strs.append(f"System{i + 1}({m[0]}):\n{m[1]}")

    prompt += "\n\n".join(model_strs)

    return prompt + "\n"


def main():
    import pandas as pd

    questions = pd.read_csv("eval_result.csv")
    questions["eval"] = ""
    for i, row in questions.iterrows():
        query = prepare_query(row)
        questions["eval"][i] = eval(query)
        questions.to_csv("eval_result.csv", index=False)


if __name__ == "__main__":
    main()

