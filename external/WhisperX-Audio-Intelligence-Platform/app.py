import gradio as gr

from pipeline import run_pipeline


DESCRIPTION = """
# WhisperX Audio Intelligence Platform

Upload an audio file to perform:

- Automatic Speech Recognition (ASR)
- Word-level Alignment
- Optional Speaker Diarization

### Notes
- Recommended audio length: under 2 minutes
- CPU inference may take time
- Diarization increases processing time
"""


def inference(audio, diarization):

    if audio is None:
        return "Please upload an audio file."

    try:

        transcript = run_pipeline(
            audio_path=audio,
            enable_diarization=diarization
        )

        return transcript

    except ValueError as e:

        return f"Input Error:\n\n{str(e)}"

    except Exception as e:

        return (
            "An unexpected error occurred.\n\n"
            f"{str(e)}"
        )


with gr.Blocks(
    title="WhisperX Audio Intelligence Platform"
) as demo:

    gr.Markdown(DESCRIPTION)

    with gr.Row():

        audio_input = gr.Audio(
            type="filepath",
            label="Upload Audio"
        )

    diarization_checkbox = gr.Checkbox(
        value=False,
        label="Enable Speaker Diarization"
    )

    submit_btn = gr.Button(
        "Run Pipeline"
    )

    output_box = gr.Textbox(
        label="Transcript",
        lines=25,
    )

    submit_btn.click(
        fn=inference,
        inputs=[
            audio_input,
            diarization_checkbox
        ],
        outputs=output_box
    )

    gr.Markdown(
        """
        ### Model Information
        - Whisper Model: `small`
        - Alignment: WhisperX
        - Diarization: Pyannote
        - Compute Type: `int8`
        """
    )

demo.launch()
