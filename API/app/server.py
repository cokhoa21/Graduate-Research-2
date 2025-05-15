from fastapi import FastAPI, HTTPException
import numpy as np
import torch.nn as nn
import uvicorn
import torch
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import re
import logging
import traceback
from pydantic import BaseModel
from typing import Optional

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define request model
class PredictionRequest(BaseModel):
    cookie_value: str

# Utility functions for text preprocessing and character encoding
def enhanced_preprocess_text(text):
    """Preprocess the cookie text for better feature extraction."""
    if not isinstance(text, str):
        return ""
    # Return the text as is - no special preprocessing needed for character encoding
    return text

def enhanced_char_encode(texts, char_to_idx, max_length):
    """
    Convert a list of text strings to a character-level encoded matrix.
    
    Args:
        texts: List of text strings to encode
        char_to_idx: Mapping from characters to indices
        max_length: Maximum length of sequence
        
    Returns:
        Numpy array of encoded characters, padded to max_length
    """
    result = np.zeros((len(texts), max_length), dtype=np.int64)
    
    for i, text in enumerate(texts):
        if not isinstance(text, str):
            continue
            
        # Convert characters to indices
        char_indices = [char_to_idx.get(char, 0) for char in text[:max_length]]
        
        # Pad if necessary
        if len(char_indices) < max_length:
            char_indices = char_indices + [0] * (max_length - len(char_indices))
            
        result[i, :len(char_indices)] = char_indices
        
    return result

class TokenAndPositionEmbedding(nn.Module):
    def __init__(self, vocab_size, embed_dim, max_length, device='cpu'):
        super().__init__()
        self.device = device
        self.word_emb = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embed_dim
        )
        self.pos_emb = nn.Embedding(
            num_embeddings=max_length,
            embedding_dim=embed_dim
        )

    def forward(self, x):
        N, seq_len = x.size()
        positions = torch.arange(0, seq_len).expand(N, seq_len).to(self.device)
        output1 = self.word_emb(x)
        output2 = self.pos_emb(positions)
        output = output1 + output2
        return output
    
embed_dim = 128
max_length = 128
vocab_size = 100000
embedding = TokenAndPositionEmbedding(
    vocab_size,
    embed_dim,
    max_length
)

# Đảm bảo rằng lớp mô hình đã được định nghĩa trước
class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(in_features=embed_dim, out_features=ff_dim, bias=True),
            nn.ReLU(),
            nn.Linear(in_features=ff_dim, out_features=embed_dim, bias=True)
        )
        self.layernorm_1 = nn.LayerNorm(normalized_shape=embed_dim, eps=1e-6)
        self.layernorm_2 = nn.LayerNorm(normalized_shape=embed_dim, eps=1e-6)
        self.dropout_1 = nn.Dropout(p=dropout)
        self.dropout_2 = nn.Dropout(p=dropout)

    def forward(self, query, key, value):
        attn_output, _ = self.attn(query, key, value)
        attn_output = self.dropout_1(attn_output)
        out_1 = self.layernorm_1(query + attn_output)
        ffn_output = self.ffn(out_1)
        ffn_output = self.dropout_2(ffn_output)
        out_2 = self.layernorm_2(out_1 + ffn_output)
        return out_2


class TransformerEncoderCls(nn.Module):
    def __init__(self, vocab_size, max_length, embed_dim, num_heads, ff_dim, dropout=0.1, device='cpu'):
        super().__init__()
        self.device = device
        self.embd_layer = TokenAndPositionEmbedding(vocab_size, embed_dim, max_length, device)
        self.transformer_layer = TransformerEncoder(embed_dim, num_heads, ff_dim, dropout)
        self.fc1 = nn.Linear(in_features=embed_dim, out_features=20)
        self.fc2 = nn.Linear(in_features=20, out_features=5)
        self.dropout = nn.Dropout(p=dropout)
        self.relu = nn.ReLU()

    def forward(self, x, device):
        output = self.embd_layer(x)
        output = self.transformer_layer(output, output, output)
        output = output.mean(dim=1)
        output = self.dropout(output)
        output = self.fc1(output)
        output = self.relu(output)
        output = self.dropout(output)
        output = self.fc2(output)
        return output

# Load mô hình từ file
try:
    logger.info("Loading model...")
    model = TransformerEncoderCls(vocab_size=100000, max_length=128, embed_dim=128, num_heads=4, ff_dim=128, dropout=0.1, device='cpu')
    model_path = '../model/model.pt'
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Error loading model: {str(e)}")
    logger.error(traceback.format_exc())
    # Still initialize model as None so the app can start
    model = None

class_names = np.array(['very low', 'low', 'average', 'high', 'very high'])

app = FastAPI()

# Enhanced CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict this to your extension's origin
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

app.mount("/templates", StaticFiles(directory="../templates"), name="templates")

@app.get('/')
def read_root():
    return FileResponse('../templates/index.html')

@app.get('/health')
def health_check():
    """Health check endpoint to test if the API is running."""
    if model is None:
        return {"status": "unhealthy", "model_loaded": False, "error": "Model failed to load"}
    
    # Try a simple prediction to ensure the model works
    try:
        sample_value = "test_cookie_value_123"
        processed_text = enhanced_preprocess_text(sample_value)
        
        char_to_idx = {}
        char_to_idx['<PAD>'] = 0
        for i, char in enumerate(set(processed_text)):
            char_to_idx[char] = i + 1
        
        char_input_np = enhanced_char_encode([processed_text], char_to_idx, 128)
        char_input_tensor = torch.tensor(char_input_np, dtype=torch.int64)
        
        with torch.no_grad():
            output = model(char_input_tensor, model.device)
            probabilities = torch.softmax(output, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1)
            class_name = class_names[predicted_class.item()]
        
        return {
            "status": "healthy", 
            "model_loaded": True,
            "test_prediction": {
                "class": class_name,
                "input": sample_value
            }
        }
    except Exception as e:
        logger.error(f"Health check prediction failed: {str(e)}")
        return {
            "status": "unhealthy", 
            "model_loaded": True, 
            "error": str(e),
            "prediction_test": "failed"
        }

@app.post('/predict')
async def predict(data: PredictionRequest):
    """
    Predicts the class of a given cookie value.

    Args:
        data: PredictionRequest object containing cookie_value

    Returns:
        dict: A dictionary containing the predicted class and probabilities.
    """
    try:
        logger.info(f"Received prediction request with cookie length: {len(data.cookie_value)}")
        
        if not data.cookie_value:
            logger.warning("Empty cookie value received")
            raise HTTPException(status_code=400, detail="Cookie value cannot be empty")
        
        # Process the cookie value
        processed_text = enhanced_preprocess_text(data.cookie_value)
        
        # Create character to index mapping
        char_to_idx = {}
        char_to_idx['<PAD>'] = 0
        for i, char in enumerate(set(processed_text)):
            char_to_idx[char] = i + 1
        
        # Encode the character sequence
        char_input_np = enhanced_char_encode([processed_text], char_to_idx, 128)
        char_input_tensor = torch.tensor(char_input_np, dtype=torch.int64)
        
        logger.info(f"Encoded cookie to tensor shape: {char_input_tensor.shape}")
        
        # Make prediction
        with torch.no_grad():
            output = model(char_input_tensor, model.device)
            probabilities = torch.softmax(output, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1)
            class_name = class_names[predicted_class.item()]
        
        logger.info(f"Prediction successful: {class_name}")
        
        return {
            'predicted_class': class_name,
            'probabilities': probabilities[0].tolist()
        }
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

if __name__ == "__main__":
    logger.info("Starting server...")
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)