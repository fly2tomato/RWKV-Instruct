"""
RWKV RNN Model - Gradio Space for HuggingFace
YT - Mean Gene Hacks - https://www.youtube.com/@MeanGeneHacks
(C) Gene Ruebsamen - 2/7/2023
License: GPL3
"""

import gradio as gr
import codecs
from ast import literal_eval
from datetime import datetime
from rwkvstic.load import RWKV
from rwkvstic.agnostic.backends import TORCH, TORCH_QUANT, TORCH_STREAM
import torch
import gc

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def to_md(text):
    return text.replace("\n", "<br />")


def get_model():
    model = None
    model = RWKV(
        "https://huggingface.co/BlinkDL/rwkv-4-pile-1b5/resolve/main/RWKV-4-Pile-1B5-Instruct-test1-20230124.pth",
        "pytorch(cpu/gpu)",
        runtimedtype=torch.float32,
        useGPU=torch.cuda.is_available(),
        dtype=torch.float32
    )
    return model

model = None

def infer(
        prompt,
        mode = "generative",
        max_new_tokens=10,
        temperature=0.1,
        top_p=1.0,
        stop="<|endoftext|>",
        seed=42,
):
    global model

    if model == None:
        gc.collect()
        if (DEVICE == "cuda"):
            torch.cuda.empty_cache()
        model = get_model()
        
    max_new_tokens = int(max_new_tokens)
    temperature = float(temperature)
    top_p = float(top_p)
    stop =  [x.strip(' ') for x in stop.split(',')]
    seed = seed

    assert 1 <= max_new_tokens <= 384
    assert 0.0 <= temperature <= 1.0
    assert 0.0 <= top_p <= 1.0

    if temperature == 0.0:
        temperature = 0.01
    if prompt == "":
        prompt = " "

    # Clear model state for generative mode
    model.resetState()
    if (mode == "Q/A"):
        prompt = f"Expert Questions & Helpful Answers\nAsk Research Experts\nQuestion:\n{prompt}\n\nFull Answer:"
    
    print(f"PROMPT ({datetime.now()}):\n-------\n{prompt}")
    print(f"OUTPUT ({datetime.now()}):\n-------\n")
    # Load prompt
    model.loadContext(newctx=prompt)
    generated_text = ""
    done = False
    with torch.no_grad():
        for _ in range(max_new_tokens):
            char = model.forward(stopStrings=stop,temp=temperature,top_p_usual=top_p)["output"]
            print(char, end='', flush=True)
            generated_text += char
            generated_text = generated_text.lstrip("\n ")
            
            for stop_word in stop:
                stop_word = codecs.getdecoder("unicode_escape")(stop_word)[0]
                if stop_word != '' and stop_word in generated_text:
                    done = True
                    break
            yield generated_text
            if done:
                print("<stopped>\n")
                break

    #print(f"{generated_text}")
    
    for stop_word in stop:
        stop_word = codecs.getdecoder("unicode_escape")(stop_word)[0]
        if stop_word != '' and stop_word in generated_text:
            generated_text = generated_text[:generated_text.find(stop_word)]
    
    gc.collect()
    yield generated_text


def chat(
        prompt,
        history,
        max_new_tokens=10,
        temperature=0.1,
        top_p=1.0,
        stop="<|endoftext|>",
        seed=42,
):
    global model
    history = history or []
        
    if model == None:
        gc.collect()
        if (DEVICE == "cuda"):
            torch.cuda.empty_cache()
        model = get_model()
        
    if len(history) == 0:
        # no history, so lets reset chat state
        model.resetState()
        history = [[],model.emptyState]
    else:
        model.setState(history[1])
    max_new_tokens = int(max_new_tokens)
    temperature = float(temperature)
    top_p = float(top_p)
    stop =  [x.strip(' ') for x in stop.split(',')]
    seed = seed

    assert 1 <= max_new_tokens <= 384
    assert 0.0 <= temperature <= 1.0
    assert 0.0 <= top_p <= 1.0

    if temperature == 0.0:
        temperature = 0.01
    if prompt == "":
        prompt = " "
    
    print(f"CHAT ({datetime.now()}):\n-------\n{prompt}")
    print(f"OUTPUT ({datetime.now()}):\n-------\n")
    # Load prompt
    model.loadContext(newctx=prompt)
    generated_text = ""
    done = False
    gen = model.forward(number=max_new_tokens, stopStrings=stop,temp=temperature,top_p_usual=top_p)
    generated_text = gen["output"]
    generated_text = generated_text.lstrip("\n ")
    print(f"{generated_text}")
    
    for stop_word in stop:
        stop_word = codecs.getdecoder("unicode_escape")(stop_word)[0]
        if stop_word != '' and stop_word in generated_text:
            generated_text = generated_text[:generated_text.find(stop_word)]
    
    gc.collect()
    history[0].append((prompt, generated_text))
    return history[0],[history[0],gen["state"]]


examples = [
    [
        # Question Answering
        '''What is the capital of Germany?''',"Q/A", 25, 0.2, 1.0, "<|endoftext|>"],
    [
        # Question Answering
        '''Are humans good or bad?''',"Q/A", 150, 0.8, 0.8, "<|endoftext|>"],
    [
        # Chatbot
        '''This is a conversation between two AI large language models named Alex and Fritz. They are exploring each other's capabilities, and trying to ask interesting questions of one another to explore the limits of each others AI.
Conversation:
Alex: Good morning, Fritz, what type of LLM are you based upon?
Fritz: Morning Alex, I am an RNN with transformer level performance. My language model is 100% attention free.
Alex:''', "generative", 220, 0.9, 0.9, "\\n\\n,<|endoftext|>"],
    [
        # Generate List
        '''Q. Give me list of fiction books. 
1. Harry Potter
2. Lord of the Rings
3. Game of Thrones
Q. Give me a list of vegetables.
1. Broccoli
2. Celery
3. Tomatoes
Q. Give me a list of car manufacturers.''', "generative", 80, 0.2, 1.0, "\\n\\n,<|endoftext|>"],
    [
        # Natural Language Interface
        '''You are the writing assistant for Stephen King. You have worked in the fiction/horror genre for 30 years. You are a Pulitzer Prize-winning author, and now you are tasked with developing a skeletal outline for his newest horror novel, set to be completed in the spring of 2024. Create a summary of this work.
Summary:''',"generative", 200, 0.85, 0.8, "<|endoftext|>"]
]


iface = gr.Interface(
    fn=infer,
    description='''<p>RNN With Transformer-level LLM Performance. (<a href='https://github.com/BlinkDL/RWKV-LM'>github</a>)
    According to the author: "It combines the best of RNN and transformers - great performance, fast inference, saves VRAM, fast training, "infinite" ctx_len, and free sentence embedding"
    <p>Thanks to https://huggingface.co/spaces/yahma/rwkv for the basis for this space</p>''',
    allow_flagging="never",
    inputs=[
        gr.Textbox(lines=20, label="Prompt"),  # prompt
        gr.Radio(["generative","Q/A"], value="generative", label="Choose Mode"),
        gr.Slider(1, 256, value=40),  # max_tokens
        gr.Slider(0.0, 1.0, value=0.8),  # temperature
        gr.Slider(0.0, 1.0, value=0.85),  # top_p
        gr.Textbox(lines=1, value="<|endoftext|>") # stop
    ],
    outputs=gr.Textbox(lines=25),
    examples=examples,
    cache_examples=False,
).queue()

chatiface = gr.Interface(
    fn=chat,
    description='''<p>RNN With Transformer-level LLM Performance. (<a href='https://github.com/BlinkDL/RWKV-LM'>github</a>)
    According to the author: "It combines the best of RNN and transformers - great performance, fast inference, saves VRAM, fast training, "infinite" ctx_len, and free sentence embedding"
    <p>Thanks to https://huggingface.co/spaces/yahma/rwkv for the basis for this space</p>''',
    allow_flagging="never",
    inputs=[
        gr.Textbox(lines=5, label="Message"),  # prompt
        "state",
        gr.Slider(1, 256, value=60),  # max_tokens
        gr.Slider(0.0, 1.0, value=0.8),  # temperature
        gr.Slider(0.0, 1.0, value=0.85),  # top_p
        gr.Textbox(lines=1, value="<|endoftext|>") # stop
    ],
    outputs=[gr.Chatbot(color_map=("green", "pink")),"state"],
).queue()

demo = gr.TabbedInterface(

    [iface,chatiface],["Generative","Chatbot"],
    title="RWKV-4 (7b Instruct, 8-Bit [cpu/for now])",
    
    )

demo.queue()
demo.launch(share=False)