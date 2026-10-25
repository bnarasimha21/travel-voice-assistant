import gradio as gr
import numpy as np
import sounddevice as sd
import torch
import warnings
import os
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from kokoro_onnx import Kokoro
import openai
from dotenv import load_dotenv

load_dotenv()

# Device setup
device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
device_idx = 0 if torch.cuda.is_available() else -1

# Model setup
def load_models():
    model_name = "openai/whisper-large-v3-turbo"
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_name, 
        torch_dtype=torch_dtype, 
        low_cpu_mem_usage=True, 
        use_safetensors=True
    )
    processor = AutoProcessor.from_pretrained(model_name)
    
    asr_pipeline = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        device=device_idx
    )
    
    kokoro = Kokoro("kokoro-v1.0.onnx", "voices-v1.0.bin")
    
    return asr_pipeline, kokoro

# Load models
asr_pipeline, kokoro = load_models()

# OpenAI setup
openai_client = openai.OpenAI(
    base_url="https://agent-5d46730574343c41627e-83wkk.ondigitalocean.app/api/v1/",
    api_key=os.environ.get("DIGITALOCEAN_GENAI_ACCESS_TOKEN_TRAVEL"),
)

# Initialize conversation history
conversation_history = [
    {
        "role": "system",
        "content": "You are a Southern California Travel Agency Assistant. Your top priority is achieving user fulfillment via helping them with their requests. Limit all responses to 2 concise sentences and no more than 100 words.",
    },
]

def transcribe(audio, history):
    if audio is None:
        return history, None
    
    sr, y = audio
    
    # Convert to mono if stereo
    if y.ndim > 1:
        y = y.mean(axis=1)
        
    y = y.astype(np.float32)
    peak = np.max(np.abs(y)) if y.size else 0.0
    if peak == 0:
        # Silent or empty recording: nothing to transcribe (avoids divide by zero / NaN)
        return history, None
    y /= peak

    transcript = asr_pipeline({"sampling_rate": sr, "raw": y})["text"].strip()
    if not transcript:
        return history, None

    # Add user message to conversation history
    conversation_history.append(
        {"role": "user", "content": transcript}
    )

    completion = openai_client.chat.completions.create(
        model="n/a",
        messages=conversation_history,
    )
    
    assistant_response = completion.choices[0].message.content
    
    # Add assistant response to conversation history
    conversation_history.append(
        {"role": "assistant", "content": assistant_response}
    )
    
    # Play the response
    play_response(assistant_response)
    
    # Update the chat history for display (messages format) and reset the mic input
    history = (history or []) + [
        {"role": "user", "content": transcript},
        {"role": "assistant", "content": assistant_response},
    ]
    return history, None

def play_response(response):
    response_samples, response_rate = kokoro.create(
        response, voice="af_sarah", speed=1.0, lang="en-us"
    )

    sd.play(response_samples, response_rate)

# Create Gradio interface
def create_interface():
    with gr.Blocks(title="Travel Agent Chatbot") as demo:
        gr.Markdown("# Travel Agent Chatbot")
        
        chatbot = gr.Chatbot(label="Conversation", type="messages")
        
        with gr.Row():
            audio_input = gr.Audio(sources="microphone")
        
        clear_button = gr.Button("Clear Conversation")
                
        # Trigger on audio recording completion
        audio_input.change(
            fn=transcribe,
            inputs=[audio_input, chatbot],
            outputs=[chatbot, audio_input]
        )
        
        # Clear conversation history
        def clear_history():
            global conversation_history
            conversation_history = [conversation_history[0]]  # Keep the system prompt
            return None, []
        
        clear_button.click(
            fn=clear_history,
            inputs=[],
            outputs=[audio_input, chatbot]
        )
    
    return demo

if __name__ == "__main__":
    demo = create_interface()
    demo.launch()  #share=True