import json
import os
from pathlib import Path
from typing import Any, List, Optional

import torch
import transformers
from groq import Groq
from openai import OpenAI
from pydantic import BaseModel
from transformers.models.llama.modeling_llama import LlamaForCausalLM
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast
import time
from huggingface_hub import InferenceClient

# tryiing to import from unsloth while cuda is not available will cause crashes
if torch.cuda.is_available():
    from unsloth import FastLanguageModel

# disable hf warnings
transformers.logging.set_verbosity_error()


class FewShotOutput(BaseModel):
    french: str
    english: str


class FewShotPreviousTopics(BaseModel):
    french: List[str]
    english: List[str]


class FewShotInput(BaseModel):
    paragraph: str
    previous_topics: FewShotPreviousTopics


class FewShotExample(BaseModel):
    input: FewShotInput
    output: FewShotOutput


class FewShotLearning(BaseModel):
    examples: List[FewShotExample]


class TopicGenerationModel:
    def __init__(self, ckpt_path: Path, device: str, few_shot_learning_path: Path):
        t1 = time.time()
        self.text_limit_size = 5000

        self.device = device
        if self.device == "cuda":
            self.model: LlamaForCausalLM
            self.tokenizer: PreTrainedTokenizerFast
            self.model, self.tokenizer = FastLanguageModel.from_pretrained(
                model_name=ckpt_path,
                max_seq_length=8192,
                dtype=None,
                load_in_4bit=True,
            )

        try:
            with open(few_shot_learning_path) as f:
                self.few_shot_learning = json.load(f)
            FewShotLearning.model_validate(self.few_shot_learning)
        except Exception as _:
            pass

        self.loading_time = time.time() - t1

    def create_example(
        self, language: Optional[str] = "english"
    ) -> list[tuple[str, Any]]:
        topic_generation_examples = []
        for example in self.few_shot_learning["examples"]:
            input_str = ""
            if len(example["input"]["previous_topics"][language]) > 0:
                input_str += "Previous topics were:\n"
                input_str += "\n".join(example["input"]["previous_topics"][language])
                input_str += "\n\n"
            input_str += example["input"]["paragraph"]

        topic_generation_examples.append((input_str, example["output"][language]))

        return topic_generation_examples

    def get_system_message(self, language: Optional[str] = "english") -> str:
        base_msg = (
            "You are a skillful assistant helping me providing a topic for a chapter of a video.\n"
            "I will give you a paragraph and I want you to return a topic and only a topic for this paragraph."
            "The topic should be concise but informative and help"
            "the user understand what is happening in this chapter."
            # "Make it coherent with the topics from the previous chapter that are given at the beginning.\n"
            "Make it not too long. It should not make more that 15 words.\n"
            "Do not uppercase words that don't need it in the provided topics even if the paragraph contains"
            "incoherent uppercases.\n"
            "Don't do AI-handholding."
            f"The returned topic should be in the provided language. The language is {language}.\n"
            "Treat every example as a new one."
            "Make sure to only return the topic."
        )

        return base_msg

    # Lllama3 can sometimes output ill-formatted topics.
    # This function aims to fix issues that have already been encountered.
    def clean_generation_output(self, generation: str) -> str:
        # llama3 can add explaination before the title, so we remove it
        # Example:
        # `Here is a topic for this part:
        #
        # Topic title`

        generation = generation.split("\n\n")[-1]
        generation = generation.split("\n")[-1]

        # llama3 tends to output " at beginning and end of the title so we trim them if present
        generation = generation.lstrip("\"'").rstrip("\"'")

        return generation

    def topic_generation(
        self,
        input_text: str,
        previous_topics: List[str],
        language: Optional[str] = "english",
    ) -> Any:
        few_shot_generation = True

        if self.device == "cpu":
            return f"Chapter {len(previous_topics)}"

        if len(previous_topics) != 0:
            # Take at most the last 2 previous topics. Providing too much previous topics can pertub the generation.
            input_text = (
                "Previous topics were:"
                + "\n".join(previous_topics[-2:])
                + "\n\n"
                + "Give a topic for:\n"
                + input_text
                + "\n\n Make sure to only return the topic."
            )

        messages = [
            {"role": "system", "content": self.get_system_message(language)},
        ]

        if few_shot_generation:
            examples = self.create_example(language)

            for example in examples:
                messages.append({"role": "user", "content": example[0]})
                messages.append({"role": "assistant", "content": example[1]})

        messages.append({"role": "user", "content": input_text})

        input_ids = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        ).to(self.model.device)

        if input_ids.size()[-1] > self.text_limit_size:
            return f"Chapter {len(previous_topics)}"

        terminators = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("<|eot_id|>"),
        ]

        outputs = self.model.generate(
            input_ids,
            max_new_tokens=25,
            eos_token_id=terminators,
            do_sample=False,
        )
        response = outputs[0][input_ids.shape[-1] :]
        out = self.tokenizer.decode(response, skip_special_tokens=True)

        out = self.clean_generation_output(out)

        return out


class APIModel:
    def __init__(self, few_shot_learning_path):
        try:
            with open(few_shot_learning_path) as f:
                self.few_shot_learning = json.load(f)
            FewShotLearning.model_validate(self.few_shot_learning)
        except Exception as _:
            pass

    def get_system_message(self, language: Optional[str] = "english") -> str:
        base_msg = (
            "You are a skillful assistant helping me providing a topic for a chapter of a video.\n"
            "I will give you a paragraph and I want you to return a topic and only a topic for this paragraph."
            "The topic should be concise but informative and help"
            "the user understand what is happening in this chapter."
            # "Make it coherent with the topics from the previous chapter that are given at the beginning.\n"
            "Make it not too long. It should not make more that 15 words.\n"
            "Do not uppercase words that don't need it in the provided topics even if the paragraph contains"
            "incoherent uppercases.\n"
            "Don't do AI-handholding."
            f"The returned topic should be in the provided language. The language is {language}.\n"
            "Treat every example as a new one."
            "Make sure to only return the topic."
        )

        return base_msg

    def create_example(
        self, language: Optional[str] = "english"
    ) -> list[tuple[str, Any]]:
        topic_generation_examples = []
        for example in self.few_shot_learning["examples"]:
            input_str = ""
            if len(example["input"]["previous_topics"][language]) > 0:
                input_str += "Previous topics were:\n"
                input_str += "\n".join(example["input"]["previous_topics"][language])
                input_str += "\n\n"
            input_str += example["input"]["paragraph"]

        topic_generation_examples.append((input_str, example["output"][language]))

        return topic_generation_examples

    def create_messages(
        self,
        input_text: str,
        previous_topics: List[str],
        language: Optional[str] = "english",
    ) -> Any:
        few_shot_generation = True
        if len(previous_topics) != 0:
            # Take at most the last 2 previous topics. Providing too much previous topics can pertub the generation.
            input_text = (
                "Previous topics were:"
                + "\n".join(previous_topics[-2:])
                + "\n\n"
                + "Give a topic for:\n"
                + input_text
                + "\n\n Make sure to only return the topic."
            )

        messages = [
            {"role": "system", "content": self.get_system_message(language)},
        ]

        if few_shot_generation:
            examples = self.create_example(language)

            for example in examples:
                messages.append({"role": "user", "content": example[0]})
                messages.append({"role": "assistant", "content": example[1]})

        messages.append({"role": "user", "content": input_text})

        return messages


class ChatGPTTopicGeneration(APIModel):
    def __init__(self, few_shot_learning_path):
        super().__init__(few_shot_learning_path)

        t1 = time.time()
        self.client = OpenAI()
        self.loading_time = time.time() - t1

    def topic_generation(
        self,
        input_text: str,
        previous_topics: List[str],
        language: Optional[str] = "english",
    ) -> Any:
        messages = self.create_messages(input_text, previous_topics, language)

        completion = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            max_tokens=25,
            temperature=0,
        )

        return completion.choices[0].message.content


class GroqTopicGeneration(APIModel):
    def __init__(self, few_shot_learning_path):
        super().__init__(few_shot_learning_path)

        t1 = time.time()
        self.client = Groq(
            api_key=os.environ.get("GROQ_API_KEY"),
        )
        self.loading_time = time.time() - t1

    def topic_generation(
        self,
        input_text: str,
        previous_topics: List[str],
        language: Optional[str] = "english",
    ) -> Any:
        messages = self.create_messages(input_text, previous_topics, language)

        completion = self.client.chat.completions.create(
            messages=messages, model="llama3-70b-8192", max_tokens=25, temperature=0
        )

        return completion.choices[0].message.content


class GroqGemmaTopicGeneration(APIModel):
    def __init__(self, few_shot_learning_path):
        super().__init__(few_shot_learning_path)

        t1 = time.time()
        self.client = Groq(
            api_key=os.environ.get("GROQ_API_KEY"),
        )
        self.loading_time = time.time() - t1

    def topic_generation(
        self,
        input_text: str,
        previous_topics: List[str],
        language: Optional[str] = "english",
    ) -> Any:
        messages = self.create_messages(input_text, previous_topics, language)

        completion = self.client.chat.completions.create(
            messages=messages, model="gemma2-9b-it", max_tokens=25, temperature=0
        )

        return completion.choices[0].message.content


class HFLlama3TopicGeneration(APIModel):
    def __init__(self, few_shot_learning_path):
        super().__init__(few_shot_learning_path)

        t1 = time.time()
        self.client = InferenceClient(
            "meta-llama/Meta-Llama-3-70B-Instruct",
            token="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",
        )
        self.loading_time = time.time() - t1

    def topic_generation(
        self,
        input_text: str,
        previous_topics: List[str],
        language: Optional[str] = "english",
    ) -> Any:
        messages = self.create_messages(input_text, previous_topics, language)

        for message in self.client.chat_completion(
            messages=messages,
            max_tokens=500,
            stream=True,
        ):
            print(message.choices[0].delta.content, end="")

            return message.choices[0].delta.content
