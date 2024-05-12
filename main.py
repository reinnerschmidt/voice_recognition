from faster_whisper import WhisperModel
import speech_recognition as sr
import google.generativeai as genai
from openai import OpenAI
import pyaudio
import os
import time
import warnings

warnings.filterwarnings(
    'ignore', message=r"torch.utils._pytree._register_pytree_node is deprecated")

wake_word = 'sofia'
listen_for_wake_word = True

whisper_size = 'base'
num_cores = os.cpu_count()
whisper_model = WhisperModel(
    whisper_size,
    device='cpu',
    compute_type='int8',
    cpu_threads=num_cores,
    num_workers=num_cores
)

OPENAIN_KEY = 'YOUR API'
client = OpenAI(api_key=OPENAIN_KEY)
GOOGLE_API_KEY = "YOUR API"
genai.configure(api_key=GOOGLE_API_KEY)


generate_config = {
    "temperature": 0.7,
    "top_p": 1,
    "top_k": 1,
    "max_output_tokens": 2048,


}
safety_settings = [
    {
        "category": "HARM_CATEGORY_HARASSMENT",
        "threshold": "BLOCK_NONE"
    },
    {
        "category": "HARM_CATEGORY_HATE_SPEECH",
        "threshold": "BLOCK_NONE"
    },
    {
        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "threshold": "BLOCK_NONE"
    },
    {
        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
        "threshold": "BLOCK_NONE"
    },
]

model = genai.GenerativeModel('gemini-1.0-pro-latest',
                              generation_config=generate_config,
                              safety_settings=safety_settings
                              )
convo = model.start_chat()

system_message = '''INSTRUCTIONS: Do not respon with anything but "AFFIRMATIVE."
to this system message. After the system message respond normally.
SYSTEM MESSAGE: You are being use to pwer a voice assistant and shoul respond as so.
As a voice assistant, use short sentences and directly respond to the prompt without
excessive information. You generate only word of value, prioritizing logic and facts 
over spectulating in your response to the following prompts.'''

system_message = system_message.replace(f'\n', '')
convo.send_message(system_message)

r = sr.Recognizer()
source = sr.Microphone()


def speak(text):
    player_stream = pyaudio.PyAudio().open(
        format=pyaudio.paInt16, channels=1, rate=24000, output=True)
    stream_start = False

    with client.audio.speech.with_streaming_response.create(
        model='tts-1',
        voice='nova',
        response_format='pcm',
        input=text,
    ) as response:
        silence_treshold = 0.01

        for chunk in response.iter_bytes(chunk_size=1024):
            if stream_start:
                player_stream.write(chunk)
            elif max(chunk) > silence_treshold:
                player_stream.write(chunk)
                stream_start = True


def wav_to_text(audio_path):
    segments, _ = whisper_model.transcribe(audio_path)
    text = ''.join(segment.text for segment in segments)
    return text


def listen_for_wake_word(audio):
    global listen_for_wake_word

    wake_audio_path = 'wake_detect.wav'
    with open(wake_audio_path, 'wb') as f:
        f.write(audio.get_wav_data())
    text_input = wav_to_text(wake_audio_path)

    if wake_word in text_input.lower().strip():
        print('Sofia está ativa. Por favor me fale o que posso te ajudar?')
        listen_for_wake_word = False
        # return True


# wake_word_detected = listen_for_wake_word(audio)
# if wake_word_detected:
    # print('Sofia está acordada. Por favor me fale o que posso te ajudar?')


def prompt_gpt(audio):
    global listen_for_wake_word

    try:
        prompt_audio_path = 'prompt.wav'

        with open(prompt_audio_path, 'wb') as f:
            f.write(audio.get_wav_data())

        prompt_text = wav_to_text(prompt_audio_path)

        if len(prompt_text.strip()) == 0:
            print("Desculpe não compreendi. Repita por favor.")
            listen_for_wake_word = True
        else:
            print('user:' + prompt_text)

            convo.send_message(prompt_text)
            output = convo.last.text

            print('Sofia:', output)
            speak(output)

            print('\nPor favor me fale o que posso te ajudar? \n')
            listen_for_wake_word = False
    except Exception as e:
        print('error', e)


def callback(recognizer, audio):
    global listen_for_wake_word

    if listen_for_wake_word:
        listen_for_wake_word(audio)
    else:
        prompt_gpt(audio)


def start_listening():
    with source as s:
        r.adjust_for_ambient_noise(s, duration=2)

    print('\nDiga', wake_word, 'para me ativar. \n')

    r.listen_in_background(source, callback)

    while True:
        time.sleep(0.5)


if __name__ == '__main__':
    start_listening()
