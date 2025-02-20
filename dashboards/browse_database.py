import os

import numpy as np
import pandas as pd
import fortepyan as ff
import streamlit as st
import streamlit_pianoroll
from streamlit.errors import DuplicateWidgetID
from midi_tokenizers import ExponentialTimeTokenizer

from dashboards.components import download_button
import piano_generation.database.database_manager as database_manager


def get_unique_rows(df1, df2, subset=None, tolerance=0.02):
    """
    Find rows that are present in only one of the dataframes, not both,
    with a tolerance for float comparisons.

    Parameters:
    df1 (pandas.DataFrame): First dataframe
    df2 (pandas.DataFrame): Second dataframe
    subset (list): Optional list of columns to consider when comparing rows
    tolerance (float): Tolerance level for float comparisons (default: 0.02)

    Returns:
    tuple: (rows_only_in_df1, rows_only_in_df2)
    """
    # Ensure index is unique in both dataframes
    df1 = df1.reset_index(drop=True)
    df2 = df2.reset_index(drop=True)

    # If subset is not specified, use all columns that exist in both dataframes
    if subset is None:
        subset = list(set(df1.columns) & set(df2.columns))

    def find_match(row, other_df, subset_cols, tolerance):
        """Helper function to find matching rows within tolerance"""
        for _, other_row in other_df[subset_cols].iterrows():
            match = True
            for col in subset_cols:
                val1 = row[col]
                val2 = other_row[col]

                # Handle float comparison with tolerance
                if isinstance(val1, (float, np.float64)) and isinstance(val2, (float, np.float64)):
                    if not np.isclose(val1, val2, rtol=tolerance, atol=tolerance):
                        match = False
                        break
                # Handle non-float comparison
                elif val1 != val2:
                    match = False
                    break

            if match:
                return True
        return False

    # Find rows unique to df1
    df1_unique = df1[~df1[subset].apply(lambda row: find_match(row, df2, subset, tolerance), axis=1)]

    # Find rows unique to df2
    df2_unique = df2[~df2[subset].apply(lambda row: find_match(row, df1, subset, tolerance), axis=1)]

    return df1_unique, df2_unique


def format_model_params(model_params):
    total_tokens, model_loss, train_loss = model_params
    return f"{total_tokens:,}, best_val_loss: {model_loss}, train_loss: {train_loss}"


def main():
    st.title("MIDI Transformers Database Browser")

    tab1, tab2, tab3, tab4 = st.tabs(["Model Predictions", "Models", "Generators Parameters", "Prompt Notes"])

    with tab1:
        _, disp_col, _ = st.columns([1, 5, 1])
        with disp_col:
            st.header("Model Predictions")

            models_df = database_manager.select_models_with_generations()
            model_names = models_df["name"].unique().tolist()

            selected_model_name = st.selectbox("Select Model", model_names, key="model")

            if selected_model_name:
                selected_models = models_df[models_df["name"] == selected_model_name]
                model_tokens = selected_models["total_tokens"].tolist()
                model_best_val_loss = selected_models["best_val_loss"]
                model_train_loss = selected_models["train_loss"]
                selected_model_params = st.selectbox(
                    label="Select tokens",
                    options=[params for params in zip(model_tokens, model_best_val_loss, model_train_loss)],
                    format_func=format_model_params,
                )
                selected_model_tokens = selected_model_params[0]
                selected_model = selected_models[selected_models["total_tokens"] == selected_model_tokens].iloc[0]
                st.json(selected_model.to_dict(), expanded=False)

                if pd.notna(selected_model["wandb_link"]):
                    st.link_button("View Model on W&B", url=selected_model["wandb_link"])
                else:
                    st.write("No W&B link available for this model")

                selected_model_id = selected_model["model_id"]

                # Get unique tasks for this model's predictions
                tasks = database_manager.get_model_tasks(model_id=selected_model_id)
                selected_task = st.selectbox("Select Task", tasks + ["All"], key="task_selector")

                # Get unique generator names for this model's predictions
                generator_names = database_manager.get_model_generator_names(model_id=selected_model_id)
                selected_generator = st.selectbox("Select Generator", ["All"] + generator_names, key="generator_selector")

                # Filter predictions based on selected task and generator
                generator_filters = {}
                if selected_task != "All":
                    generator_filters["task"] = selected_task
                if selected_generator != "All":
                    generator_filters["generator_name"] = selected_generator

                # Fetch predictions for the selected model, task, and generator
                predictions_df = database_manager.get_model_predictions(
                    model_filters={"model_id": selected_model_id},
                    generator_filters=generator_filters if generator_filters else None,
                )

                # Fetch predictions for the selected model and task
                predictions_df = database_manager.get_model_predictions(
                    model_filters={"model_id": selected_model_id},
                    generator_filters=generator_filters,
                )

                # Composer and title filtering
                st.subheader("Filter by Composer and Title")
                filter_option = st.radio("Filter by:", ["All", "Composer and Title", "Unspecified"])

                if filter_option == "Composer and Title":
                    composers = predictions_df["source"].apply(lambda x: x.get("composer")).dropna().unique()
                    selected_composer = st.selectbox("Select Composer", ["All"] + list(composers))
                    selection = predictions_df["source"].apply(lambda x: x.get("composer") == selected_composer)
                    sources = predictions_df[selection]["source"]
                    if selected_composer != "All":
                        titles = sources.apply(lambda x: x.get("title")).dropna().unique()
                    else:
                        titles = predictions_df["source"].apply(lambda x: x.get("title")).dropna().unique()

                    selected_title = st.selectbox("Select Title", ["All"] + list(titles))

                    if selected_composer != "All" and selected_title != "All":
                        predictions_df = predictions_df[
                            predictions_df["source"].apply(
                                lambda x: x.get("composer") == selected_composer and x.get("title") == selected_title
                            )
                        ]
                    elif selected_composer != "All":
                        predictions_df = predictions_df[
                            predictions_df["source"].apply(lambda x: x.get("composer") == selected_composer)
                        ]
                    elif selected_title != "All":
                        selection = predictions_df["source"].apply(lambda x: x.get("title") == selected_title)
                        predictions_df = predictions_df[selection]

                elif filter_option == "Unspecified":
                    predictions_df = predictions_df[
                        predictions_df["source"].apply(lambda x: "composer" not in x and "title" not in x)
                    ]

                if not predictions_df.empty:
                    idx = st.number_input(
                        label="prediction number",
                        min_value=0,
                        max_value=len(predictions_df),
                    )
                    row = predictions_df.iloc[idx]
                    prompt_notes = row["prompt_notes"]
                    prompt_notes_df = pd.DataFrame(prompt_notes)
                    sources = database_manager.get_source(row["source_id"])
                    source_notes = sources.iloc[0]["notes"]
                    source_notes_df = pd.DataFrame(source_notes)
                    default_tokenizer = ExponentialTimeTokenizer(0.01, 32)
                    source_notes_df = default_tokenizer.untokenize(default_tokenizer.tokenize(source_notes_df))

                    st.json(row["source"])
                    st.json({"generator_name": row["generator_name"]} | row["generator_parameters"])
                    generated_notes = row["generated_notes"]
                    generated_notes_df = pd.DataFrame(generated_notes)

                    generated_piece = ff.MidiPiece(df=generated_notes_df)
                    prompt_piece = ff.MidiPiece(df=prompt_notes_df)
                    midi_name = f"{selected_model_name}_{selected_model_tokens:.2f}_generation"
                    prompt_midi_path = f"tmp/{midi_name}_prompt.mid"
                    midi_path = f"tmp/{midi_name}.mid"
                    source_midi_path = f"tmp/{midi_name}_source.mid"

                    original_piece_df, _ = get_unique_rows(source_notes_df, prompt_notes_df, subset=["pitch", "start", "end"])
                    original_piece_part = ff.MidiPiece(original_piece_df)

                    st.write("#### Original")
                    try:
                        streamlit_pianoroll.from_fortepyan(piece=prompt_piece, secondary_piece=original_piece_part)
                    except DuplicateWidgetID:
                        st.write("Duplicate widget")
                    original_piece_part.to_midi().write(source_midi_path)
                    with open(source_midi_path, "rb") as file:
                        st.markdown(
                            download_button(
                                file.read(),
                                source_midi_path.split("/")[-1],
                                "Download original target",
                            ),
                            unsafe_allow_html=True,
                        )
                    os.unlink(source_midi_path)
                    st.write("#### Prompt")
                    try:
                        streamlit_pianoroll.from_fortepyan(piece=prompt_piece)
                    except DuplicateWidgetID:
                        st.write("Duplicate widget")
                    prompt_piece.to_midi().write(prompt_midi_path)
                    with open(prompt_midi_path, "rb") as file:
                        st.markdown(
                            download_button(
                                file.read(),
                                prompt_midi_path.split("/")[-1],
                                "Download prompt",
                            ),
                            unsafe_allow_html=True,
                        )
                    os.unlink(prompt_midi_path)

                    st.write("#### Generated")
                    try:
                        streamlit_pianoroll.from_fortepyan(piece=generated_piece)
                    except DuplicateWidgetID:
                        st.write("Duplicate widget")

                    # Allow download of the generated MIDI
                    generated_piece.to_midi().write(midi_path)
                    with open(midi_path, "rb") as file:
                        st.markdown(
                            download_button(
                                file.read(),
                                midi_path.split("/")[-1],
                                "Download generated midi",
                            ),
                            unsafe_allow_html=True,
                        )
                    os.unlink(midi_path)

                    st.write("#### Together")
                    try:
                        streamlit_pianoroll.from_fortepyan(piece=prompt_piece, secondary_piece=generated_piece)
                    except DuplicateWidgetID:
                        st.write("Duplicate widget")

                    out_piece = ff.MidiPiece(pd.concat([prompt_notes_df, generated_notes_df]))

                    # Allow download of the full MIDI with context\
                    midi_name = f"{selected_model_name}_{selected_model_tokens:.2f}_variations"
                    full_midi_path = f"tmp/{midi_name}.mid"
                    out_piece.to_midi().write(full_midi_path)
                    with open(full_midi_path, "rb") as file:
                        st.markdown(
                            download_button(
                                file.read(),
                                full_midi_path.split("/")[-1],
                                "Download midi with context",
                            ),
                            unsafe_allow_html=True,
                        )
                    os.unlink(full_midi_path)
                    st.divider()  # Add a divider between predictions
                else:
                    st.write("No predictions found for this prompt and model combination.")

    with tab2:
        st.header("Models")
        models_df = database_manager.select_models_with_generations()
        st.write(models_df)

        st.subheader("Purge Model")
        model_to_purge = st.selectbox("Select a model to purge", models_df["name"].tolist())
        if st.button("Purge Selected Model"):
            try:
                database_manager.purge_model(model_to_purge)
                st.success(f"Model '{model_to_purge}' has been purged successfully.")
            except Exception as e:
                st.error(f"An error occurred while purging the model: {str(e)}")

    with tab3:
        st.header("Generators")
        parameters_df = database_manager.get_all_generators()
        st.write(parameters_df)

    with tab4:
        st.header("Prompt Notes")
        prompts_df = database_manager.get_all_sources()
        st.write(prompts_df)


if __name__ == "__main__":
    main()
