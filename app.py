import tkinter as tk
import customtkinter as ctk
from PIL import ImageTk
from authtoken import auth_token
import torch
from torch import autocast
from diffusers import StableDiffusionPipeline

# Check if CUDA is available, otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Now you can move tensors and models to the selected device
tensor = torch.tensor([1, 2, 3]).to(device)

# Create the app
app = tk.Tk()
app.geometry("532x632")
app.title("Stable Bud")
ctk.set_appearance_mode("dark")

prompt = ctk.CTkEntry(app, height=40, width=512, fg_color="white")
prompt.configure(font=("Arial", 20))
prompt.place(x=10, y=10)

lmain = ctk.CTkLabel(app, height=512, width=512)
lmain.place(x=10, y=110)

modelid = "CompVis/stable-diffusion-v1-4"
device = "cuda"
pipe = StableDiffusionPipeline.from_pretrained(modelid, revision="fp16", torch_dtype=torch.float16, use_auth_token=auth_token)
pipe.to(device)


def generate():
    with autocast(device):
        result = pipe(prompt.get(), guidance_scale=8.5)

    print("Keys in result:", result.keys())  # Print keys in the result dictionary
    # image = result["sample"][0]  # Assuming 'sample' is a valid key

    # Replace the above line with correct key based on the printed keys
    # image = result[<correct_key_name_here>][0]

    # Temporary placeholder until correct key is identified
    image = result[list(result.keys())[0]][0]

    image.save('generatedimage.png')
    img = ImageTk.PhotoImage(image)
    lmain.configure(image=img)


trigger = ctk.CTkButton(app, height=40, width=120, text_color="white", fg_color="blue", command=generate)
trigger.configure(text="Generate")
trigger.place(x=206, y=60)

app.mainloop()