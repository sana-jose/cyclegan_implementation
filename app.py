import torch
import random
import gradio as gr
from models.generator import Generator
from data.dataset import MonetDataset
from utils.visualize import visualize  


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


G = Generator().to(device)
F = Generator().to(device)

checkpoint = torch.load(
    "checkpoints/checkpoint_epoch_143.pth",
    map_location=device
)

G.load_state_dict(checkpoint["G_state_dict"])
F.load_state_dict(checkpoint["F_state_dict"])

G.eval()
F.eval()


dataset = MonetDataset(split="test")


def run_model(direction):

 
    idx = random.randint(0, len(dataset) - 1)
    sample = dataset[idx]

    if direction == "monet_to_original":
        input_tensor = sample["monet"].unsqueeze(0).to(device)
        with torch.no_grad():
            output_tensor = G(input_tensor)

    else:  
        input_tensor = sample["original"].unsqueeze(0).to(device)
        with torch.no_grad():
            output_tensor = F(input_tensor)


    fig = visualize(input_tensor, output_tensor)

    return fig


with gr.Blocks() as demo:

    gr.Markdown("# 🎨 CycleGAN Demo: Monet ↔ Real")
    gr.Markdown("Random sample translation using trained CycleGAN")

    direction = gr.Radio(
        ["monet_to_original", "original_to_monet"],
        value="monet_to_original",
        label="Select Translation Direction"
    )

    btn = gr.Button("Generate Random Sample")

    output_plot = gr.Plot()

    btn.click(
        fn=run_model,
        inputs=direction,
        outputs=output_plot
    )

demo.launch()