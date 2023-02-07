from rwkvstic.load import RWKV
import torch
model = RWKV(
    "https://huggingface.co/BlinkDL/rwkv-4-pile-1b5/resolve/main/RWKV-4-Pile-1B5-Instruct-test1-20230124.pth",
    "pytorch(cpu/gpu)",
    runtimedtype=torch.float32,
    useGPU=torch.cuda.is_available(),
    dtype=torch.float32
)
import gradio as gr


def predict(input, history=None):
    model.setState(history[1])
    model.loadContext(newctx=f"Prompt: {input}\n\nExpert Long Detailed Response: ")
    r = model.forward(number=100,stopStrings=["\n\nPrompt"])
    rr = [(input,r["output"])]
    return [*history[0],*rr], [[*history[0],*rr],r["state"]]

def freegen(input):
    model.resetState()
    model.loadContext(newctx=input)
    return model.forward(number=100)["output"]
with gr.Blocks() as demo:
    with gr.Tab("Chatbot"):
        chatbot = gr.Chatbot()
        state = model.emptyState
        state = gr.State([[],state])
        with gr.Row():
            txt = gr.Textbox(show_label=False, placeholder="Enter text and press enter").style(container=False)
    
        txt.submit(predict, [txt, state], [chatbot, state])
    with gr.Tab("Free Gen"):
        with gr.Row():
            input = gr.Textbox(show_label=False, placeholder="Enter text and press enter").style(container=False)
            outtext = gr.Textbox(show_label=False)
        input.submit(freegen,input,outtext)
demo.launch()
