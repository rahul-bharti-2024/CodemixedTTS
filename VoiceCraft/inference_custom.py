import os
import torch
import torchaudio
import numpy as np
import random
from argparse import Namespace, ArgumentParser
from tqdm import tqdm
import json
import concurrent
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from data.tokenizer import AudioTokenizer, TextTokenizer
from huggingface_hub import hf_hub_download
from inference_tts_scale import inference_one_sample_graphemes
from shutil import copy2
from dataclasses import dataclass, field
from typing import List

def read_jsonl(filepath):
    data = []
    with open(filepath, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def get_spk2item(items):
    spk2item = {}
    for item in tqdm(items, desc='making spk2item'):
        spk_id = item['speaker_id']
        if spk_id not in spk2item.keys():
            spk2item[spk_id] = [item]
        else:
            spk2item[spk_id].append(item)
    return spk2item

def contains_english_characters(sentence):
    # Loop through each character in the sentence
    for char in sentence:
        # Check if the character is an English letter
        if char.isalpha() and char.isascii():
            return True
    return False

def sample_speaker_prompt(spk2item, spk_id, item):
    prompt_item = random.sample(spk2item[spk_id], 1)[0]
    if prompt_item['text'] == item['text']: # make sure test set prompt not used
        prompt_item = random.sample(spk2item[spk_id], 1)[0]
    while prompt_item["duration"] < 1:
        prompt_item = random.sample(spk2item[spk_id], 1)[0]
    return prompt_item

def seed_everything(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def process_item(args, model, config, phn2num, audio_tokenizer, item, output_dir, i):
    seed_everything(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    orig_audio = item['prompt_audio']

    # item['output_filename'] = f'{i}.wav'

    # copy prompt to folder
    prompt_dir = os.path.join(output_dir, 'prompts')
    os.makedirs(prompt_dir, exist_ok=True)
    dest_path = os.path.join(prompt_dir, item['output_filename'])
    # try:
    #     copy2(orig_audio, dest_path)
    # except PermissionError:
    #     from shutil import copy
    #     copy(orig_audio, dest_path)

    orig_transcript = item['prompt_text']
    # test sentence

    sentence = item['text']
    item['text'] = sentence

    filepath = f"{output_dir}/{os.path.basename(orig_audio)[:-4]}.wav"
    # cut_off_sec = item['prompt_duration'] - 0.01
    # target_transcript = orig_transcript + item['verbatim']
    words = item['text'].split(" ")
    chunks = []
    N_WORDS = 20 
    for ix in range(0, len(words), N_WORDS):
        
        chunks.append(" ".join(words[ix: ix+N_WORDS]))
    chunks = list(filter(lambda x: x != "", chunks))
    # print('info', orig_audio)
    info = torchaudio.info(orig_audio)
    audio_dur = info.num_frames / info.sample_rate
    cut_off_sec = audio_dur - 0.01
    # print(audio_dur)
    # print('done info')

    assert cut_off_sec < audio_dur, f"cut_off_sec {cut_off_sec} is larger than the audio duration {audio_dur}"
    prompt_end_frame = int(cut_off_sec * info.sample_rate)

    decode_config = {
        'top_k': args.top_k, 'top_p': args.top_p, 'temperature': args.temperature, 
        'stop_repetition': args.stop_repetition, 'kvcache': args.kvcache, "codec_audio_sr": args.codec_audio_sr, 
        "codec_sr": args.codec_sr, "silence_tokens": args.silence_tokens, "sample_batch_size": args.sample_batch_size
    }
    prompt = orig_transcript
    audio_chunks = []
    for ix, chunk in enumerate(chunks):
        chunk_i = prompt + ' ' + chunk
        print("LN 109", chunk_i)
        concated_audio, gen_audio = inference_one_sample_graphemes(
            model, Namespace(**config), phn2num, audio_tokenizer, orig_audio, 
            chunk_i, device, decode_config, prompt_end_frame
        )
        audio_chunks.append(gen_audio.squeeze(0).cpu())
            # print(gen_audio.shape)
            # prompt = chunk
        # except Exception as e:
        #     # print('skipped', e, item)
        #     print('skipped', e)
        #     return

    gen_audio = torch.cat(audio_chunks, dim=1)
    # concated_audio, gen_audio = concated_audio[0].cpu(), gen_audio[0].cpu()

    # filename = f'{i+1}_' + os.path.basename(orig_audio)
    filename = item['output_filename']
    print(filename)
    samples_dir = os.path.join(output_dir, 'samples_enhprompts')
    os.makedirs(samples_dir, exist_ok=True)
    filepath = f"{samples_dir}/{filename}"
    torchaudio.save(filepath, gen_audio, args.codec_audio_sr)
    print('Saved to ', filepath)



def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    from models import voicecraft
    model_path = args.model_path

    ckpt = torch.load(model_path, map_location='cpu')
    model = voicecraft.VoiceCraft(ckpt['config'])
    model.load_state_dict(ckpt['model'])
    model.to(device)
    model.eval()

    config = vars(model.args)
    phn2num = ckpt["phn2num"]

    encodec_fn = "/mnt/LS226/LS25/rahul2022387/Pretrained_models/pretrained_models/encodec_4cb2048_giga.th"
    if not os.path.exists(encodec_fn):
        os.system(f"wget https://huggingface.co/pyp1/VoiceCraft/resolve/main/encodec_4cb2048_giga.th")
        os.system(f"mv encodec_4cb2048_giga.th ./pretrained_models/encodec_4cb2048_giga.th")
    audio_tokenizer = AudioTokenizer(signature=encodec_fn, device=device)

    filepath = args.manifest_path # '/nlsasfs/home/ai4bharat/praveens/ttsteam/repos/voicecraft/demo/srvm/demo_sys/demo.json'
    test = read_jsonl(filepath)

    output_dir = os.path.join(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    for idx, item in enumerate(tqdm(test)):
        process_item(args, model, config, phn2num, audio_tokenizer, item, output_dir, idx)
     

from dataclasses import dataclass
@dataclass
class Config:
    manifest_path: str = '/home/rahul_b/IndicVoices-R/VoiceCraft/datasets/mucs_roman/manifests/hnyms_metadata1.jsonl'
    model_path: str = '/mnt/LS226/LS25/rahul2022387/Results/Voicecraft/logs/mucs_roman/e830M/best_bundle.pth'
    output_dir: str = '/mnt/LS226/LS25/rahul2022387/OpenSLR/Results_mucs_roman_hnyms'
    language_family: str = 'indoaryan'
    language: str = ''
    split: str = ''
    replace_path: bool = False
    num_workers: int = 1
    codec_audio_sr: int = 16000
    codec_sr: int = 50
    top_k: int = 0
    top_p: float = 0.9
    temperature: float = 1.0
    silence_tokens: List[int] = field(default_factory=lambda: [1388, 1898, 131])
    kvcache: int = 1
    stop_repetition: int = 4
    sample_batch_size: int = 3
    seed: int = 1


if __name__=="__main__":
    args= Config()
    main(args)