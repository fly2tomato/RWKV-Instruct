from rwkvstic.load import RWKV
import torch
model = RWKV(
    "https://huggingface.co/BlinkDL/rwkv-4-pile-3b/resolve/main/RWKV-4-Pile-3B-20221110-ctx4096.pth",
    "pytorch(cpu/gpu)",
    runtimedtype=torch.float32,
    useGPU=False,
    dtype=torch.float3
)
import gradio as gr


def predict(input, history=None):
    model.setState(history)
    model.loadContext(newctx=f"{input}\n\nBot: ")
    r = model.forward(number=100,stopStrings=["User: "])
    
    return r["output"], r["state"]

with gr.Blocks() as demo:
    chatbot = gr.Chatbot()
    state = model.emptyState
    ctx, state = model.loadContext(newctx="User: ")
    state = gr.State(state)
    with gr.Row():
        txt = gr.Textbox(show_label=False, placeholder="Enter text and press enter").style(container=False)

    txt.submit(predict, [txt, state], [chatbot, state])

demo.launch()
