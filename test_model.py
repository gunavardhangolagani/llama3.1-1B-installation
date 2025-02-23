from transformers import pipeline
import torch

"""
# Load and configure the model
- Define the model path where the LLaMA model is stored.
- Initialize the text-generation pipeline with necessary configurations.
"""

modelPath = "C:\\codes\\llama1B\\"

pipe = pipeline(
    "text-generation",
    model=modelPath,
    torch_dtype=torch.float16,
    device=0
)

"""
# Define the input message structure
- System role provides contextual information to guide the model's response.
- User role contains the actual query input.
"""

messageStructure1 = [
    {
        "role": "system",
        "content": "You are an elementary school teacher and you need to adopt your answers to elementary-school children. "
    },
    {
        "role": "user",
        "content": "What is quantum mechanics?"
    }
]

"""
# Generate response from the model
- Pass the input structure to the pipeline for text generation.
- Limit the generated response to a maximum of 500 tokens.
"""

response = pipe(
    messageStructure1,
    max_new_tokens=500,
)

"""
# Extract and display the model's response
- Retrieve the last generated response.
- Print the response content to the console.
"""

outputResponse = response[0]["generated_text"][-1]

print(outputResponse['content'])

"""
# Save the output to a file
- Open a text file in write mode.
- Save the generated response content for later use.
"""

with open('output.txt', 'w', encoding="utf-8") as text_file:
    text_file.write(outputResponse['content'])
