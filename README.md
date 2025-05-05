# ChatDiT with webui

🔥 **Latest News!**

- **[2024-12-18]** 📂 Code for **[ChatDiT](https://arxiv.org/abs/2412.12571)** are now available!
- **[2024-12-18]** 📜 Our latest paper, **[ChatDiT](https://arxiv.org/abs/2412.12571)**, introduces a training-free, zero-shot, general-purpose, and interactive image generation system built upon diffusion transformers.

Welcome to the official repository for **ChatDiT: A Training-Free Baseline for Task-Agnostic Free-Form Chatting with Diffusion Transformers** ([Paper](https://arxiv.org/abs/2412.12571) | [Project Page](https://ali-vilab.github.io/ChatDiT-Page/)).

ChatDiT supports seamless multi-round, free-form conversations with DiTs. It supports referencing zero to multiple images to generate a new set of images, or, if desired, a fully illustrated article in response.

Dispite its simplicity and training-free design, ChatDiT achieves the best performance (**23.19**) on [IDEA-Bench](https://ali-vilab.github.io/IDEA-Bench-Page/), surpassing models like EMU2 (**6.8**) and OmniGen (**6.47**).

> Note: ChatDiT is still a prototype, so some features may not work as expected. However, it provides a robust baseline for future research and development.

## Getting Started

### Prerequisites

Install [PyTorch](https://pytorch.org/) based on the [official guidelines](https://pytorch.org/get-started/locally/), then install diffusers and openai:

```bash
pip install diffusers openai
```

### Simple Usage

ChatDiT offers a simple, intuitive interface for interacting with both text and images.

```python
import openai
import torch
from chatdit import ChatDiT

app = ChatDiT(
    client=openai.OpenAI(),
    device=torch.device('cuda')
)

# text-to-image
image = app.chat('Generate a dog wearing sunglasses swimming in the pool')[0]

# text-to-image
images = app.chat('Show two different views of a 3D black cat blind box')

# image-to-image
image = app.chat('Create a cup with the abstract character wrapping around it', images=[input_image])[0]

# image-to-images
images = app.chat('Place the product in 4 different scenarios showcasing its usages', images=[input_image])

# images-to-image
image = app.chat(
    'Create a 3D render of the product based on its three-view images',
    images=[front_image, back_image, side_image]
)

# images-to-images
images = app.chat(
    'Show two interaction scenarios of the two characters',
    images=[character1_image, character2_image]
)

# interleaved text-image article
article = app.chat(
    'Tell the story of an architect designing and constructing an innovative building. Images should show the various stages from initial sketches to completed building.',
    return_markdown=True
)
article.save('./innovative_building_designing/')  # saves both markdown and images
```

Checkout input/output cases in our [Project Page](https://ali-vilab.github.io/ChatDiT-Page/).

