
# Asistente de IA con Reconocimiento de Voz e Imagen

Este proyecto es un asistente interactivo que utiliza la cámara y el micrófono para interactuar con los usuarios mediante modelos de inteligencia artificial, como OpenAI y Google Generative AI. Proporciona respuestas basadas en el reconocimiento de voz y la captura de imágenes en tiempo real, además de ofrecer una interfaz gráfica amigable.

## Descripción

El asistente responde a preguntas utilizando:
- **Entrada de audio** mediante el micrófono.
- **Entrada visual** mediante la webcam.
- **Modelos de lenguaje** avanzados como ChatOpenAI y ChatGoogleGenerativeAI.

### Características

- Reconocimiento de voz en tiempo real utilizando `speech_recognition`.
- Captura y análisis de imágenes usando `opencv`.
- Respuestas generadas utilizando modelos de IA.
- Interfaz gráfica (`tkinter`) para mostrar el stream de video y las respuestas.
- Registro de errores y actividades en el archivo `assistant.log`.

## Requisitos

Asegúrate de tener instalados los siguientes paquetes:

```bash
pip install openai langchain langchain_openai langchain_google_genai opencv-python-headless pyaudio SpeechRecognition python-dotenv pillow
```

### Variables de entorno

Crea un archivo `.env` con las siguientes claves:

```
OPENAI_API_KEY=<tu_openai_api_key>
GOOGLE_API_KEY=<tu_google_api_key>
DEFAULT_LANGUAGE=es
```

## Uso

Para ejecutar el asistente, simplemente ejecuta:

```bash
python main_new.py
```

### Controles

- Presiona **'ESC'** o **'q'** para salir del programa.
- Habla claramente cerca del micrófono para hacer una consulta.
- Utiliza la cámara para capturar imágenes si el asistente lo solicita.

## Ideas de Uso en una Clínica Médica

1. **Análisis de resultados de laboratorio**: El asistente puede analizar automáticamente resultados de análisis de sangre, radiografías u otros informes médicos que los pacientes muestren a la cámara.
2. **Evaluación de lesiones dermatológicas**: Detectar enfermedades de la piel mediante la captura de imágenes y el análisis automatizado de erupciones, lunares o heridas.
3. **Verificación de documentos médicos**: Validar la autenticidad de recetas y órdenes médicas utilizando la cámara.
4. **Identificación de pacientes mediante reconocimiento facial**: Asegurar que los pacientes correctos estén recibiendo consultas o tratamientos.
5. **Seguimiento postoperatorio**: Evaluar el estado de cicatrización mediante la captura de imágenes de heridas y zonas operadas.

## Autor

Desarrollado por GONZALO MORA.

## Licencia

Este proyecto está bajo la Licencia MIT.
