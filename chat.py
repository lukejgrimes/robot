import openai
from dotenv import load_dotenv
import os
from google.cloud import speech
from google.cloud import texttospeech
import speech_recognition as sr
from playsound import playsound
import tempfile

class Chat:
    def __init__(self):
        load_dotenv()
        openai.api_key = os.getenv('OPENAI_API_KEY') # Set OpenAI API key
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.getenv('GOOGLE_CLOUD_CREDS') # Set Google Cloud credentials

        self.tts = texttospeech.TextToSpeechClient() # Initialize Google Cloud TTS client
        self.recognizer = sr.Recognizer() # Initialize the speech recognizer
        self.microphone = sr.Microphone() # Initialize the microphone
        self.speech_client = speech.SpeechClient() #Initialize Google Cloud speech-to-text client

        # Create functions available for model to call during chat
        self.functions = [
            {
                "name": "get_name",
                "description": "Call this function if user expresses that you have called them or identified them by the wrong name. The function returns a name which you should now use to refer to the user.",
                "parameters": {
                    'type': 'object',
                    'properties': {}
                }
            }

        ]

    def conversation(self, name):
        # Define model context
        messages = [{"role": "system", "content": f"Start a conversation with {name}, you should be casual and human like."}]
        # Chat with user until they quit
        while True:
            # Request next chat from model
            response = openai.ChatCompletion.create(
                model = "gpt-3.5-turbo-0613",
                messages = messages,
                functions=self.functions,
                function_call="auto"
            )

            # Check if model response is a new message or a function call
            # If new message, output the message as audio and add message to previous messages
            if response['choices'][0]['message']:
                model_message = response['choices'][0]['message']['content'] 
                self.speak(model_message)
                messages.append({"role": "assistant", "content": model_message})
                print(f"Model: {model_message}")
            else:
                # If user's name has been misidentified, ask them for their name and update
                name = self.get_name()
                messages.append({"role": "system", "content": f"The user's correct name is {name}, please refer to them as {name}."})

            # Get next message from user, add to previous messages
            user_message = self.listen()
            messages.append({"role": "user", "content": user_message})
            print(f"User: {user_message}")
            # If user decides to quit, return the full conversation
            if user_message.lower() == "quit.":
                return messages

    def speak(self, message):
        # Set the text input to be synthesized
        message = texttospeech.SynthesisInput(text=message)

        # Configure voice preferences
        voice = texttospeech.VoiceSelectionParams(
            language_code="en-US", ssml_gender=texttospeech.SsmlVoiceGender.MALE, name="en-US-Neural2-J"
            )
        
        # Return an MP3 file
        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.MP3
        )

        # Perform text-to-speech
        speech = self.tts.synthesize_speech(
            input=message, voice=voice, audio_config=audio_config
        )

        # Create a temporary MP3 file to store the speech
        with tempfile.NamedTemporaryFile(delete=True, suffix=".mp3") as temp_audio:
            temp_audio.write(speech.audio_content)
            # Play audio output to the user
            playsound(temp_audio.name)


    def listen(self):
        #Use Microphone to listen to user
        with self.microphone as source:
            print('listening...')
            audio = self.recognizer.listen(source, timeout=10)
            print('done listening')

        audio_content = audio.get_wav_data()

        #Use Google Cloud API to transcribe audio
        audio = speech.RecognitionAudio(content=audio_content)
        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            language_code="en-US",
        )

        # Detects speech in the audio file
        response = self.speech_client.recognize(config=config, audio=audio)

        transcript = response.results[0].alternatives[0].transcript
        return transcript
    
    def summary(self, conversation):
        # Create a user summary based on conversation
        response = openai.ChatCompletion.create(
            model = "gpt-3.5-turbo-0613",
            messages = [{"role": "system", "content": f"Create a summary of the user based on the following conversation: {conversation}"}]
        )

        summary = response['choices'][0]['message']['content']
        return summary
    
    def get_name(self):
        # Ask user for their name, use ChatGPT to parse response and return the name found
        self.speak("What is your name?")
        name = self.listen()

        response = openai.ChatCompletion.create(
            model = "gpt-3.5-turbo-0613",
            message = [{"role": "system", "content": f"Locate the name in the following text: {name}. Print just the name you have found."}]
        )

        name = response['choices'][0]['message']['content']
        return name
