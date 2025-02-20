import os
from glob import glob

import torch
import requests
import pandas as pd
import fortepyan as ff
import streamlit as st
import streamlit_pianoroll
from midi_tokenizers import ExponentialTimeTokenizer
from piano_dataset.piano_tasks import PianoTaskManager

from piano_generation import MidiGenerator
from dashboards.components import download_button
from dashboards.utils import device_model_selection
from piano_generation.artifacts import dataset_tokens, composer_tokens
from piano_generation.utils import load_cfg, load_checkpoint, initialize_gpt_model


def main():
    # See the readme to figure out how you can get this checkpoint
    # checkpoint_path = "checkpoints/midi-gpt2-302M-subsequence-4096-ctx-2024-09-08-19-42last.pt"
    device, checkpoint_path = device_model_selection()
    st.write("Checkpoint:", checkpoint_path)
    checkpoint = load_cache_checkpoint(checkpoint_path, device=device)

    random_seed = st.number_input(
        label="random seed",
        value=137,
        max_value=100_000,
        min_value=0,
    )
    max_new_tokens = st.number_input(
        label="max new tokens",
        min_value=64,
        max_value=4096,
        value=2048,
    )
    temperature = st.number_input(
        label="temperature",
        min_value=0.0,
        max_value=2.0,
        value=1.0,
        step=0.05,
    )

    speedup_factor = st.number_input(
        label="speedup factor",
        value=1.2,
        min_value=0.3,
        max_value=2.5,
    )

    # TODO: What would be a convenient way to manage prompts
    # for a user? Definitely needs an upload
    prompt_options = glob("tmp/*.mid") + [None]

    special_tokens = composer_tokens + dataset_tokens

    generation_token = None
    if checkpoint["cfg"].stage == "piano_task":
        piano_task_manager: PianoTaskManager = checkpoint["piano_task_manager"]
        special_tokens += piano_task_manager.get_special_tokens()
        generation_token = "<GENAI>"

    selected_special_tokens = st.multiselect(
        "Select additional special tokens to include:",
        options=special_tokens,
        help="Choose from available special tokens to add to your prompt",
    )

    if checkpoint["cfg"].stage == "piano_task":
        selected_special_tokens.append(generation_token)

    prompt_path = st.selectbox(
        label="select prompt file",
        options=prompt_options,
        index=None,
    )
    pianoroll_apikey = st.text_input(
        label="pianoroll apikey",
        type="password",
    )
    st.write(pianoroll_apikey)

    if not prompt_path:
        return

    prompt_piece = ff.MidiPiece.from_file(prompt_path)
    prompt_notes: pd.DataFrame = prompt_piece.df
    prompt_notes.start *= speedup_factor
    prompt_notes.end *= speedup_factor

    streamlit_pianoroll.from_fortepyan(prompt_piece)
    #
    model = checkpoint["model"]
    tokenizer = checkpoint["tokenizer"]

    possible_tokens = ["<BACH>", "<MOZART>", "<CHOPIN>", None]
    for additional_token in possible_tokens:
        st.write("Token:", additional_token, "special tokens:", selected_special_tokens)
        selected_special_tokens = [additional_token] + selected_special_tokens
        # Generator randomness comes from torch.multinomial, so we can make it
        # fully deterministic by setting global torch random seed
        for it in range(6):
            local_seed = random_seed + it * 1000
            st.write("Seed:", local_seed)

            # This acts as a caching key
            generation_properties = {
                "speedup_factor": speedup_factor,
                "prompt_path": prompt_path,
                "local_seed": local_seed,
            }

            prompt_notes, generated_notes = cache_generation(
                prompt_notes=prompt_notes,
                seed=local_seed,
                _model=model,
                _tokenizer=tokenizer,
                device=device,
                selected_special_tokens=selected_special_tokens,
                seperate_with_generation_token=(checkpoint["cfg"].stage == "piano_task"),
                generation_token=generation_token,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
                **generation_properties,
            )

            prompt_piece = ff.MidiPiece(prompt_notes)
            generated_piece = ff.MidiPiece(generated_notes)

            if checkpoint["cfg"].stage == "next_token_pretraining":
                generated_piece.time_shift(prompt_piece.end)

            streamlit_pianoroll.from_fortepyan(prompt_piece, generated_piece)

            if pianoroll_apikey:
                # TODO: Add title and description control
                make_proll_post = st.button(
                    label="post to pianoroll.io",
                    key=f"{it}-{additional_token}",
                )
                if make_proll_post:
                    post_to_pianoroll(
                        model_piece=generated_piece,
                        prompt_piece=prompt_piece,
                        pianoroll_apikey=pianoroll_apikey,
                    )
                    st.write("POSTED!")

            out_piece = ff.MidiPiece(pd.concat([prompt_notes, generated_notes]))
            # Allow download of the full MIDI with context
            full_midi_path = f"tmp/tmp_{additional_token}_{local_seed}.mid"
            out_piece.to_midi().write(full_midi_path)
            with open(full_midi_path, "rb") as file:
                st.markdown(
                    download_button(
                        object_to_download=file.read(),
                        download_filename=full_midi_path.split("/")[-1],
                        button_text="Download midi with context",
                    ),
                    unsafe_allow_html=True,
                )
            st.write("---")
    os.unlink(full_midi_path)


def post_to_pianoroll(
    model_piece: ff.MidiPiece,
    prompt_piece: ff.MidiPiece,
    pianoroll_apikey: str,
):
    model_notes = model_piece.df.to_dict(orient="records")
    prompt_notes = prompt_piece.df.to_dict(orient="records")

    payload = {
        "model_notes": model_notes,
        "prompt_notes": prompt_notes,
        "post_title": "GENAI",
        "post_description": "My model did this!",
    }

    headers = {
        "UserApiToken": pianoroll_apikey,
    }
    api_endpoint = "https://pianoroll.io/api/v1/generation_pianorolls"
    r = requests.post(api_endpoint, headers=headers, json=payload)

    st.write(r)


@st.cache_data
def cache_generation(
    prompt_notes: pd.DataFrame,
    seed: int,
    _model,
    _tokenizer,
    selected_special_tokens: list[str],
    generation_token: str = None,
    device: str = "cuda",
    max_new_tokens: int = 2048,
    temperature: int = 1,
    **kwargs,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    torch.random.manual_seed(seed)
    with st.spinner("gpu goes brrrrrrrrrr"):
        input_tokens = selected_special_tokens + _tokenizer.tokenize(prompt_notes)
        if generation_token is not None:
            input_tokens.append(generation_token)

        st.write(input_tokens)

        input_token_ids = torch.tensor([_tokenizer.token_to_id[token] for token in input_tokens]).unsqueeze(0).to(device)

        generated_token_ids = _model.generate_new_tokens(
            idx=input_token_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=None,
        )
        generated_token_ids = generated_token_ids.squeeze(0).cpu().numpy()
        generated_notes = _tokenizer.decode(generated_token_ids)

    return prompt_notes, generated_notes


@st.cache_data
def load_cache_checkpoint(checkpoint_path: str, device):
    # Load a pre-trained model
    checkpoint = load_checkpoint(checkpoint_path=checkpoint_path, device=device)
    cfg = load_cfg(checkpoint)
    model = initialize_gpt_model(cfg, checkpoint, device=device)
    tokenizer = ExponentialTimeTokenizer.from_dict(checkpoint["tokenizer_desc"])

    if cfg.stage == "piano_task":
        piano_task_manager = PianoTaskManager(tasks_config=checkpoint["piano_tasks_config"])
    else:
        piano_task_manager = None

    return {
        "model": model,
        "tokenizer": tokenizer,
        "cfg": cfg,
        "piano_task_manager": piano_task_manager,
    }


if __name__ == "__main__":
    main()
