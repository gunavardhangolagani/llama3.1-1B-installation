from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

app = FastAPI()

# Load the tokenizer and model
model_name = "C:\\codes\\llama1B\\"  # Change to your actual model path
tokenizer = AutoTokenizer.from_pretrained(model_name)
device = "cuda" if torch.cuda.is_available() else "cpu"
model = AutoModelForCausalLM.from_pretrained(
    model_name, 
    torch_dtype=torch.float16 if device == "cuda" else torch.float32
).to(device)

# Define request body schema
class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1, description="Input text to generate a response for.")
    max_new_tokens: int = Field(100, ge=1, le=512, description="Maximum number of tokens to generate.")

@app.post("/generate/")
def generate_text(request: GenerateRequest):
    try:
        # Tokenize input
        inputs = tokenizer(request.prompt, return_tensors="pt").to(device)

        # Generate output
        output = model.generate(
            input_ids=inputs["input_ids"],  # Specify input_ids explicitly
            attention_mask=inputs.get("attention_mask"),  # Use attention_mask if available
            max_new_tokens=request.max_new_tokens  # Control output length
        )

        # Decode output
        response_text = tokenizer.decode(output[0], skip_special_tokens=True).strip()

        return {"prompt": request.prompt, "response": response_text}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating response: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)