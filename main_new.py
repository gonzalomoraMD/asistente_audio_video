import base64
import logging
import os
import sys
from threading import Lock, Thread
import tkinter as tk
from PIL import Image, ImageTk
from tkinter import ttk
import cv2
import openai
from cv2 import VideoCapture, imencode
from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema.messages import SystemMessage
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_google_genai import ChatGoogleGenerativeAI
from pyaudio import PyAudio, paInt16
from speech_recognition import Microphone, Recognizer, UnknownValueError

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('assistant.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Cargar variables de entorno
load_dotenv()

# Mensajes del sistema en diferentes idiomas
SYSTEM_MESSAGES = {
    'es': {
        'system_prompt': """
        Eres un asistente ingenioso que utilizará el historial del chat y la imagen 
        proporcionada por el usuario para responder sus preguntas.

        Usa pocas palabras en tus respuestas. Ve directo al punto. No uses emoticones 
        ni emojis. No hagas preguntas al usuario.

        Sé amigable y servicial. Muestra personalidad. No seas demasiado formal.
        """,
        'error_audio': """
        Error procesando el audio. 
        Por favor, verifica:
        1. Que el micrófono esté conectado y funcionando
        2. Que hayas dado permisos de micrófono en Preferencias del Sistema
        3. Que el nivel de ruido ambiental no sea demasiado alto
        4. Que estés hablando lo suficientemente cerca del micrófono
        """,
        'error_webcam': """
        Error al iniciar la webcam. 
        Por favor, verifica:
        1. Que la cámara esté conectada y funcionando
        2. Que hayas dado permisos de cámara en Preferencias del Sistema
        3. Que no haya otra aplicación usando la cámara
        4. Que la cámara esté correctamente instalada en el sistema
        """,
        'error_model': """
        Error al inicializar el modelo de IA.
        Por favor, verifica:
        1. Que las API keys estén correctamente configuradas en el archivo .env
        2. Que tengas conexión a internet
        3. Que los servicios de Google y OpenAI estén disponibles
        4. Que tu cuota de API no se haya agotado
        """,
        'ready': """
        Sistema listo para escuchar...
        - Presiona 'ESC' o 'q' para salir
        - Habla claramente cerca del micrófono
        - Espera a ver la respuesta en pantalla
        """,
        'processing': """
        Procesando tu solicitud...
        - Manteniendo la conexión con la cámara
        - Procesando el audio
        - Generando respuesta
        """,
        'initialization': """
        Iniciando sistema...
        1. Verificando dependencias
        2. Comprobando permisos de cámara y micrófono
        3. Conectando con servicios de IA
        4. Preparando interfaz
        """,
        'shutdown': """
        Cerrando sistema...
        1. Deteniendo captura de video
        2. Cerrando conexiones de audio
        3. Liberando recursos
        4. Guardando configuración
        """
    },
    'en': {
        'system_prompt': """
        You are a witty assistant that will use the chat history and the image 
        provided by the user to answer its questions.

        Use few words on your answers. Go straight to the point. Do not use any
        emoticons or emojis. Do not ask the user any questions.

        Be friendly and helpful. Show some personality. Do not be too formal.
        """,
        'error_audio': """
        Error processing audio.
        Please verify:
        1. That the microphone is connected and working
        2. That you have granted microphone permissions in System Preferences
        3. That the ambient noise level is not too high
        4. That you are speaking close enough to the microphone
        """,
        'error_webcam': """
        Error initializing webcam.
        Please verify:
        1. That the camera is connected and working
        2. That you have granted camera permissions in System Preferences
        3. That no other application is using the camera
        4. That the camera is properly installed in the system
        """,
        'error_model': """
        Error initializing AI model.
        Please verify:
        1. That API keys are properly configured in the .env file
        2. That you have internet connection
        3. That Google and OpenAI services are available
        4. That your API quota has not been exhausted
        """,
        'ready': """
        System ready to listen...
        - Press 'ESC' or 'q' to exit
        - Speak clearly near the microphone
        - Wait for the response on screen
        """,
        'processing': """
        Processing your request...
        - Maintaining camera connection
        - Processing audio
        - Generating response
        """,
        'initialization': """
        Starting system...
        1. Checking dependencies
        2. Verifying camera and microphone permissions
        3. Connecting to AI services
        4. Preparing interface
        """,
        'shutdown': """
        Shutting down system...
        1. Stopping video capture
        2. Closing audio connections
        3. Releasing resources
        4. Saving configuration
        """
    }
}

class ConfigurationError(Exception):
    """Excepción personalizada para errores de configuración."""
    pass

def validate_environment() -> None:
    """Valida que todas las variables de entorno necesarias estén presentes."""
    required_vars = ['OPENAI_API_KEY', 'GOOGLE_API_KEY']
    missing_vars = [var for var in required_vars if not os.getenv(var)]

    if missing_vars:
        raise ConfigurationError(f"Faltan variables de entorno: {', '.join(missing_vars)}")

def check_dependencies() -> None:
    """Verifica que todas las dependencias necesarias estén instaladas."""
    try:
        import cv2
        import openai
        import pyaudio
        import speech_recognition
    except ImportError as e:
        raise ImportError(f"Falta instalar dependencia: {str(e)}")


class WebcamStream:
    def __init__(self):
        self.stream = VideoCapture(0)
        import time
        time.sleep(1)

        max_attempts = 3
        for attempt in range(max_attempts):
            if self.stream.isOpened():
                _, self.frame = self.stream.read()
                if self.frame is not None:
                    break
            time.sleep(1)
            self.stream = VideoCapture(0)

        if not self.stream.isOpened() or self.frame is None:
            raise RuntimeError(
                "Error al iniciar la webcam. Por favor, verifica:\n"
                "1. Que la cámara esté conectada\n"
                "2. Que hayas dado permisos de cámara en Preferencias del Sistema\n"
                "3. Que no haya otra aplicación usando la cámara"
            )

        self.running = False
        self.lock = Lock()

    def start(self):
        if self.running:
            return self

        self.running = True
        self.thread = Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()
        return self

    def update(self):
        while self.running:
            _, frame = self.stream.read()
            with self.lock:
                self.frame = frame

    def read(self, encode: bool = False):
        with self.lock:
            frame = self.frame.copy()

        if encode:
            _, buffer = imencode(".jpeg", frame)
            return base64.b64encode(buffer)
        return frame

    def stop(self):
        self.running = False
        if hasattr(self, 'thread') and self.thread.is_alive():
            self.thread.join()

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.stream.release()


class Assistant:
    def __init__(self, model, language: str = 'es'):
        self.language = language
        self._init_messages()
        self.chain = self._create_inference_chain(model)
        logger.info(f"Asistente inicializado en {language}")

    def _init_messages(self):
        try:
            self.messages = SYSTEM_MESSAGES[self.language]
        except KeyError:
            self.messages = SYSTEM_MESSAGES['en']
            logger.warning(f"Idioma {self.language} no soportado, usando inglés")

    def answer(self, prompt: str, image: bytes) -> None:
        if not prompt:
            return

        logger.info(f"Procesando prompt: {prompt}")
        try:
            response = self.chain.invoke(
                {
                    "prompt": prompt,
                    "image_base64": image.decode()
                },
                config={"configurable": {"session_id": "unused"}}
            ).strip()

            logger.info(f"Respuesta generada: {response}")

            if response:
                self._tts(response)

        except Exception as e:
            logger.error(f"Error al procesar la respuesta: {str(e)}")
            raise

    def _tts(self, response: str) -> None:
        try:
            player = PyAudio().open(format=paInt16, channels=1, rate=24000, output=True)
            voice = "nova" if self.language == 'es' else "alloy"

            with openai.audio.speech.with_streaming_response.create(
                    model="tts-1",
                    voice=voice,
                    response_format="pcm",
                    input=response,
            ) as stream:
                for chunk in stream.iter_bytes(chunk_size=1024):
                    player.write(chunk)

            player.stop_stream()
            player.close()

        except Exception as e:
            logger.error(f"Error en la síntesis de voz: {str(e)}")
            raise

    def _create_inference_chain(self, model):
        prompt_template = ChatPromptTemplate.from_messages([
            SystemMessage(content=self.messages['system_prompt']),
            MessagesPlaceholder(variable_name="chat_history"),
            (
                "human",
                [
                    {"type": "text", "text": "{prompt}"},
                    {
                        "type": "image_url",
                        "image_url": "data:image/jpeg;base64,{image_base64}",
                    },
                ],
            ),
        ])

        chain = prompt_template | model | StrOutputParser()
        chat_message_history = ChatMessageHistory()

        return RunnableWithMessageHistory(
            chain,
            lambda _: chat_message_history,
            input_messages_key="prompt",
            history_messages_key="chat_history",
        )


class SimpleGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Asistente IA")
        self.root.geometry("1024x768")

        # Imprimir para debug
        print("Iniciando GUI...")

        # Contenedor principal
        main_container = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_container.pack(fill=tk.BOTH, expand=True)

        # Panel izquierdo (video)
        left_panel = ttk.Frame(main_container, padding="10")
        main_container.add(left_panel, weight=2)

        self.video_label = ttk.Label(left_panel)
        self.video_label.pack(expand=True, fill=tk.BOTH)

        # Panel derecho (controles)
        right_panel = ttk.Frame(main_container, padding="10", width=300)
        main_container.add(right_panel, weight=1)

        # Estilo para los widgets
        style = ttk.Style()
        style.configure('Record.TButton', padding=10)

        # Status
        self.status_label = ttk.Label(
            right_panel,
            text="Estado: Iniciando...",
            wraplength=250
        )
        self.status_label.pack(pady=10, fill=tk.X)

        # Botón de grabación
        self.record_button = ttk.Button(
            right_panel,
            text="Iniciar Grabación",
            style='Record.TButton',
            command=self.toggle_recording
        )
        self.record_button.pack(pady=10, fill=tk.X)
        print("Botón de grabación creado")

        # Área de transcripción
        transcript_frame = ttk.LabelFrame(right_panel, text="Transcripción", padding="5")
        transcript_frame.pack(fill=tk.BOTH, expand=True, pady=10)

        self.transcript_text = tk.Text(
            transcript_frame,
            wrap=tk.WORD,
            height=10,
            width=30
        )
        self.transcript_text.pack(fill=tk.BOTH, expand=True)

        # Scrollbar para transcripción
        scrollbar = ttk.Scrollbar(transcript_frame, command=self.transcript_text.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.transcript_text.config(yscrollcommand=scrollbar.set)

        # Variables de estado
        self.is_recording = False
        self.should_stop = False

        # Control de cierre
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        print("GUI inicializada completamente")

    def toggle_recording(self):
        self.is_recording = not self.is_recording
        text = "Detener Grabación" if self.is_recording else "Iniciar Grabación"
        self.record_button.config(text=text)
        self.status_label.config(
            text=f"Estado: {'Grabando...' if self.is_recording else 'Detenido'}"
        )
        print(f"Grabación {'iniciada' if self.is_recording else 'detenida'}")

    def update_video(self, frame):
        try:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)

            # Obtener dimensiones originales
            width, height = img.size

            # Calcular nueva altura manteniendo la relación de aspecto
            target_width = 640
            aspect_ratio = width / height
            target_height = int(target_width / aspect_ratio)

            # Redimensionar manteniendo la relación de aspecto
            img = img.resize((target_width, target_height), Image.Resampling.LANCZOS)

            img_tk = ImageTk.PhotoImage(image=img)
            self.video_label.configure(image=img_tk)
            self.video_label.image = img_tk
        except Exception as e:
            print(f"Error actualizando video: {e}")
            logger.error(f"Error actualizando video: {e}")

    def update_transcript(self, text):
        self.transcript_text.insert(tk.END, f"{text}\n")
        self.transcript_text.see(tk.END)
        print(f"Transcripción actualizada: {text}")

    def check_stop(self):
        return self.should_stop

    def on_closing(self):
        print("Cerrando aplicación...")
        self.should_stop = True
        self.root.quit()

    def cleanup(self):
        if self.root:
            try:
                self.root.destroy()
            except Exception as e:
                print(f"Error en cleanup: {e}")
                logger.error(f"Error en cleanup: {e}")


def main():
    webcam_stream = None
    stop_listening = None
    gui = None

    try:
        print("Iniciando aplicación...")
        validate_environment()
        check_dependencies()

        language = os.getenv('DEFAULT_LANGUAGE', 'es')
        gui = SimpleGUI()

        print("Iniciando webcam...")
        webcam_stream = WebcamStream().start()

        print("Iniciando modelo...")
        model = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest")
        assistant = Assistant(model, language=language)

        print("Configurando reconocimiento de voz...")
        recognizer = Recognizer()
        microphone = Microphone()

        def audio_callback(recognizer, audio):
            if gui and gui.is_recording:
                try:
                    print("Procesando audio...")
                    prompt = recognizer.recognize_whisper(audio, model="base", language=language)
                    print(f"Audio reconocido: {prompt}")
                    gui.update_transcript(f"Usuario: {prompt}")
                    assistant.answer(prompt, webcam_stream.read(encode=True))
                except UnknownValueError:
                    print("Error en reconocimiento de audio")
                    logger.error(SYSTEM_MESSAGES[language]['error_audio'])

        with microphone as source:
            print("Ajustando nivel de ruido...")
            recognizer.adjust_for_ambient_noise(source)

        print("Iniciando escucha en segundo plano...")
        stop_listening = recognizer.listen_in_background(microphone, audio_callback)

        def update():
            if gui and not gui.check_stop():
                gui.update_video(webcam_stream.read())
                gui.root.after(30, update)

        print("Iniciando bucle principal...")
        update()
        gui.root.mainloop()

    except Exception as e:
        print(f"Error en la ejecución principal: {e}")
        logger.error(f"Error en la ejecución principal: {str(e)}")
    finally:
        print("Limpiando recursos...")
        if stop_listening:
            stop_listening(wait_for_stop=False)
        if webcam_stream:
            webcam_stream.stop()
        if gui:
            gui.cleanup()


if __name__ == "__main__":
    main()