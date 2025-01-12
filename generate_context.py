import os
import asyncio
from glob import glob
from google import genai
from google.genai import types
from pydantic import BaseModel, ValidationError


class Questions(BaseModel):
    q1: str
    q2: str
    q3: str
    q4: str
    q5: str


GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
assert GEMINI_API_KEY is not None, "Please provide an api key"

summary_prompt_template = """<document>
{0}
</document>

Here is the chunk we want to situate within the whole document

<chunk>
{1}
</chunk>

Write a short paragraph in not more than 70 words describing the chunk with respect to the document for the purpose of improving search retrieval of the chunk.
"""

question_prompt_template = """<document>
{0}
</document>

Here is the chunk we want to situate within the whole document

<chunk>
{1}
</chunk>

Generate 5 questions that could be asked from the chunk with respect to the document for the purpose of improving search retrieval of the chunk.
"""


async def summarize(client, content, chunk_content, save_path):
    res = await client.aio.models.generate_content(
        model="gemini-2.0-flash-exp",
        contents=summary_prompt_template.format(content, chunk_content),
        config=types.GenerateContentConfig(
            temperature=0,
        ),
    )
    with open(save_path, "w") as f:
        f.write(res.text.strip())
    print(f"Summary saved to {save_path}")


async def gen_questions(client, content, chunk_content, save_path):
    res = await client.aio.models.generate_content(
        model="gemini-2.0-flash-exp",
        contents=question_prompt_template.format(content, chunk_content),
        config=types.GenerateContentConfig(
            temperature=0,
            response_mime_type="application/json",
            response_schema=Questions,
        ),
    )
    try:
        text = Questions.model_validate_json(res.text).model_dump_json()
        with open(save_path, "w") as f:
            f.write(text.strip())
    except ValidationError:
        print(
            f"Skipping... Response does not conform to the specification, {save_path}"
        )
    print(f"Questions saved to {save_path}")


client = genai.Client(api_key=GEMINI_API_KEY)
files = glob("./docs/*.md")


async def generate_summary():
    corrs = []

    for path in files:
        with open(path, "r") as f:
            content = f.read()

        stem = os.path.basename(os.path.splitext(path)[0])
        chunk_paths = filter(
            lambda x: not os.path.isdir(x), glob(f"./docs/chunks/{stem}/*")
        )

        os.makedirs(f"./docs/chunks/{stem}/summary", exist_ok=True)

        for chunk_path in chunk_paths:
            with open(chunk_path, "r") as f:
                chunk_content = f.read()

            save_path = f"./docs/chunks/{stem}/summary/{os.path.basename(chunk_path)}"
            if not os.path.exists(save_path):
                corr = summarize(
                    client,
                    content,
                    chunk_content,
                    save_path,
                )
                corrs.append(corr)
            else:
                print(f"Skipping... Summary already exist at {save_path}")

    limit = 5
    nested_corrs = [corrs[i : i + limit] for i in range(0, len(corrs), limit)]

    for corrs in nested_corrs:
        await asyncio.gather(*corrs)
        await asyncio.sleep(
            60
        )  # max limit of 10 reqs per minute on gemini's free tier api


async def generate_questions():
    corrs = []

    for path in files:
        with open(path, "r") as f:
            content = f.read()

        stem = os.path.basename(os.path.splitext(path)[0])
        chunk_paths = filter(
            lambda x: not os.path.isdir(x), glob(f"./docs/chunks/{stem}/*")
        )

        os.makedirs(f"./docs/chunks/{stem}/questions", exist_ok=True)

        for chunk_path in chunk_paths:
            with open(chunk_path, "r") as f:
                chunk_content = f.read()

            save_path = (
                f"./docs/chunks/{stem}/questions/{os.path.basename(chunk_path)}.json"
            )
            if not os.path.exists(save_path):
                corr = gen_questions(
                    client,
                    content,
                    chunk_content,
                    save_path,
                )
                corrs.append(corr)
            else:
                print(f"Skipping... Questions already generated at {save_path}")

    limit = 5
    nested_corrs = [corrs[i : i + limit] for i in range(0, len(corrs), limit)]

    for corrs in nested_corrs:
        await asyncio.gather(*corrs)
        await asyncio.sleep(
            60
        )  # max limit of 10 reqs per minute on gemini's free tier api
