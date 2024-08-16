import gradio as gr
import whisper
from translate import Translator
from dotenv import dotenv_values
from elevenlabs.client import ElevenLabs
from elevenlabs import VoiceSettings

config = dotenv_values(".env")

ELEVENLABS_API_KEY = config["ELEVENLABS_API_KEY"]

def translator(audio_file):
    
    # Transcribimos el audio a texto
    try:
        model = whisper.launch("base")
        result = model.transcribe(audio_file, language="Spanish")
        transcription = result["text"]
    except Exception as e:
        raise gr.Error(
            f"Se ha producido un error transcribiendo el texto: {str(e)}")

    print(f"Texto original: {transcription}")

    # Traducimos el texto a ingles
    try:
        en_transcription = Translator(from_lang="es", to_lang="en").translate(transcription)
    except Exception as e:
        raise gr.Error(
            f"Se ha producido un error traduciendo el texto: {str(e)}")
    
    print(f"Texto traducido: {en_transcription}")

    # Creamos un audio con el texto traducido
    try:
        client = ElevenLabs(api_key=ELEVENLABS_API_KEY)
        response = client.text_to_speech.convert(
            voice_id="pNInz6obpgDQGcFmaJgB",
            optimize_streaming_latency="0",
            output_format="mp3_22050_32",
            text=en_transcription,
            model_id="eleven_turbo_v2",
            voice_settings=VoiceSettings(
                stability=0.0,
                similarity_boost=1.0,
                style=0.0,
                use_speaker_boost=True,
            ),
        )

        save_file_path = "audios/en.mp3"

        with open(save_file_path,"wb") as f:
            for chunk in response:
                if chunk:
                    f.write(chunk)
    
    except Exception as e:
        raise gr.Error(
            f"Se ha producido un error creando el audio: {str(e)}")

    return save_file_path

web = gr.Interface(
    fn=translator,
    inputs=gr.Audio(
        sources=["microphone"],
        type="filepath",
        label="Español"
        ),
    outputs=[gr.Audio(label="Ingles")],
    title="Traductor de voz",
    description="Traduce tu voz con IA a otros idiomas"
)

web.launch()