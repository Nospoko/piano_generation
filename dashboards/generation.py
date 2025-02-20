import os
import tempfile

import torch
import pandas as pd
import fortepyan as ff
import streamlit as st
import streamlit_pianoroll
from omegaconf import OmegaConf
from streamlit.errors import DuplicateWidgetID
from midi_tokenizers import ExponentialTimeTokenizer
from piano_dataset.piano_tasks import PianoTask, PianoTaskManager

from dashboards.utils import device_model_selection
from piano_generation.utils import initialize_gpt_model
from piano_generation.artifacts import dataset_tokens, composer_tokens


def slice_source_notes(notes: pd.DataFrame, start_note_id: int = None, end_note_id: int = None):
    if start_note_id is not None and end_note_id is not None:
        notes = notes.iloc[start_note_id : end_note_id + 1].reset_index(drop=True)
    offset = notes.start.min()
    notes.start -= offset
    notes.end -= offset
    return notes


def upload_midi_file():
    st.header("Upload Custom MIDI")
    uploaded_file = st.file_uploader("Choose a MIDI file", type="mid")

    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mid") as temp_midi_file:
            temp_midi_file.write(uploaded_file.getvalue())
            temp_midi_path = temp_midi_file.name

        try:
            midi_file = ff.MidiFile(temp_midi_path)
            notes = midi_file.piece.df

            st.success("MIDI file uploaded successfully!")
            streamlit_pianoroll.from_fortepyan(piece=ff.MidiPiece(notes))

            # Add note range selection
            note_id_cols = st.columns(2)
            start_note_id = note_id_cols[0].number_input(
                "Start Note ID",
                min_value=0,
                max_value=len(notes) - 1,
                value=0,
            )
            end_note_id = note_id_cols[1].number_input(
                "End Note ID",
                min_value=start_note_id,
                max_value=len(notes) - 1,
                value=len(notes) - 1,
            )
            source_notes = slice_source_notes(
                notes=notes,
                start_note_id=start_note_id,
                end_note_id=end_note_id,
            )

            notes = notes.iloc[start_note_id : end_note_id + 1].reset_index(drop=True)
            source = {
                "midi_name": temp_midi_path,
                "start": start_note_id,
                "end": end_note_id,
            }
            try:
                st.subheader("Selected Note Range")
                streamlit_pianoroll.from_fortepyan(piece=ff.MidiPiece(notes))
            except DuplicateWidgetID:
                pass
            return source_notes, notes, source
        finally:
            os.unlink(temp_midi_path)

    return None, None, None


def prepare_prompt(task: PianoTask, notes: pd.DataFrame):
    if task == "next_token_prediction":
        return notes, pd.DataFrame(columns=notes.columns)
    prompt_notes, target_notes = task.prompt_target_split(notes=notes)
    return prompt_notes, target_notes


@st.cache_data
def load_model_checkpoint(
    checkpoint_path: str,
    map_location: str,
):
    return torch.load(
        checkpoint_path,
        map_location=map_location,
        weights_only=False,
    )


def main():
    st.title("MIDI Generation Dashboard")

    # user_api_key = st.text_input("Pianoroll API key", value="")
    device, checkpoint_path = device_model_selection()
    source_notes, prompt_notes, source = upload_midi_file()
    if source_notes is None:
        st.warning("Please upload a MIDI file to continue.")
        return

    checkpoint = load_model_checkpoint(checkpoint_path=checkpoint_path, map_location=device)
    tokenizer = ExponentialTimeTokenizer.from_dict(tokenizer_desc=checkpoint["tokenizer_desc"])
    special_tokens = composer_tokens + dataset_tokens

    run_cfg = OmegaConf.create(checkpoint["run_config"])

    if run_cfg.stage == "piano_task":
        piano_task_manager = PianoTaskManager(tasks_config=checkpoint["piano_tasks_config"])
        special_tokens += piano_task_manager.get_special_tokens()
        generation_token = "<GENAI>"

    selected_special_tokens = st.multiselect(
        "Select additional special tokens to include:",
        options=special_tokens,
        help="Choose from available special tokens to add to your prompt",
    )

    st.write(selected_special_tokens)

    input_tokens = selected_special_tokens + tokenizer.tokenize(source_notes)
    if run_cfg.stage == "piano_task":
        input_tokens.append(generation_token)

    st.write(input_tokens)

    input_token_ids = torch.tensor([tokenizer.token_to_id[token] for token in input_tokens]).unsqueeze(0).to(device)
    model = initialize_gpt_model(
        cfg=run_cfg,
        checkpoint=checkpoint,
        device=device,
    )
    max_new_tokens = st.number_input("Max new tokens", value=100)

    button = st.button("generate")
    if not button:
        return
    generated_token_ids = model.generate_new_tokens(
        idx=input_token_ids,
        max_new_tokens=max_new_tokens,
        temperature=1,
        top_k=None,
    )
    generated_token_ids = generated_token_ids.squeeze(0).cpu().numpy()
    generated_df = tokenizer.decode(generated_token_ids)

    generated_piece = ff.MidiPiece(generated_df)

    streamlit_pianoroll.from_fortepyan(generated_piece)


if __name__ == "__main__":
    main()
