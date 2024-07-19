from topic_generation_model import (
    TopicGenerationModel,
    ChatGPTTopicGeneration,
    GroqTopicGeneration,
    GroqGemmaTopicGeneration,
    HFLlama3TopicGeneration
)
import time
import argparse
from pathlib import Path
import json
import os


def load_json(json_path):
    with open(json_path) as f:
        data = json.load(f)

    return data


def build_text_moment(whisper_result, moment_result):
    whisper_chapter = []
    current_chapter = []
    current_timestamp_end = moment_result["moments"][0]["end"]

    count = 0
    for segment in whisper_result["segments"][:-1]:
        current_chapter.append(segment["text"])
        if segment["end"] == current_timestamp_end:
            whisper_chapter.append(current_chapter)
            current_chapter = []
            count += 1
            current_timestamp_end = moment_result["moments"][count]["end"]

    current_chapter.append(whisper_result["segments"][-1]["text"])
    whisper_chapter.append(current_chapter)

    return whisper_chapter


def build_text_scene(voice_scene_result, voice_shot_result):
    whisper_chapter = []

    current_shot = 0
    for scene in voice_scene_result["scenes"]:
        nb_shots = len(scene["shot_ids"])

        shots = voice_shot_result["shots"][current_shot : current_shot + nb_shots]
        current_shot += nb_shots

        text = [seg["text"] for s in shots for seg in s["segments"]]
        whisper_chapter.append(text)

    return whisper_chapter


def main(args):
    analysis_result_path = Path(f"/home/ubuntu/data/analysis_result/{args.video_id}")

    result_file_path = Path(f"result/{args.video_id}.json")

    
    whisper_result = load_json(list(analysis_result_path.glob("whisper-*.json"))[0])
    moment_result = load_json(list(analysis_result_path.glob("moment-*.json"))[0])
    voice_scene_result = load_json(
        list(analysis_result_path.glob("voice-scene-detection-*.json"))[0]
    )
    voice_shot_result = load_json(
        list(analysis_result_path.glob("voice-shot-detection-*.json"))[0]
    )

    result_file = {"metadata": {
        "start": whisper_result["segments"][0]["start"],
        "end": whisper_result["segments"][-1]["end"]

    }}

    if os.path.exists(result_file_path):
        with open(result_file_path) as f:
            result_file = json.load(f)
        
        print(result_file)


    _ = build_text_scene(voice_scene_result, voice_shot_result)

    models = {
        "local_llama3_8b": TopicGenerationModel(
            Path("/home/ubuntu/models/analyser-voice-scene-detection/unsloth_llama3"),
            "cuda",
            Path(
                "/home/ubuntu/models/analyser-voice-scene-detection/few_shot_learning.json"
            ),
        ),
        "gpt_turbo3.5": ChatGPTTopicGeneration(
            Path(
                "/home/ubuntu/models/analyser-voice-scene-detection/few_shot_learning.json"
            )
        ),
        # NOTE: I'm pretty sure Groq tends to bottleneck if you do too much request, resulting in bad results after a few requests
        "groq_llama3_70b": GroqTopicGeneration(
            Path(
                "/home/ubuntu/models/analyser-voice-scene-detection/few_shot_learning.json"
            )
        ),
        "groq_gemma2_9b_it": GroqGemmaTopicGeneration(Path(
                "/home/ubuntu/models/analyser-voice-scene-detection/few_shot_learning.json"
            )
        ),
        # "hf_llama3_70b": HFLlama3TopicGeneration(Path(
        #         "/home/ubuntu/models/analyser-voice-scene-detection/few_shot_learning.json"
        #     ))
    }

    moments = build_text_moment(whisper_result, moment_result)

    for model_name, model in models.items():
        
        if model_name in result_file.keys():
            print(f"Topics already in result file for {model_name}")
            continue

        print(f"Topics for {model_name}")
        result_file[model_name] = {}

        topics: list[str] = []
        
        t1 = time.time()
        for moment in moments:
            input_text = " ".join(moment)
            current_topic = model.topic_generation(input_text, topics, language="english")
            print(current_topic)
            topics.append(current_topic)

        result_file[model_name]["topics"] = topics
        result_file[model_name]["loading_time"] = model.loading_time
        result_file[model_name]["inference_time"] = time.time() - t1

        with open(result_file_path, "w") as f:
            json.dump(result_file, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("video_id")

    args = parser.parse_args()

    main(args)
