import streamlit as st

class Task:
    def __init__(self):
        pass

    def show_task(self):
        task_container = st.container(border=True)
        with task_container:
            audio_column, transcription_column, edit_column = st.columns(spec=3, border=True)

            audio_column.audio(data="data/2024-12-09 13-01-45-converted.mp3")

            transcription_text = "This is test transcription here for 10 audio second?"
            transcription_column.write(transcription_text)

            edited_transcription = edit_column.text_area(label="Transcription check", value=transcription_text)
            edit_column.write(f"Сохранить: '{edited_transcription}'?")
            save_button, edit_button, cancel_button = edit_column.columns(spec=3)
            save_button.button(label="Save", type="primary")
            edit_button.button(label="Edit", type="secondary")
            cancel_button.button(label="Cancel", type="secondary")




st.set_page_config(
    layout="wide",  # это сделает контейнер на всю ширину
    initial_sidebar_state="expanded",
)

tasks = Task()

tasks.show_task()