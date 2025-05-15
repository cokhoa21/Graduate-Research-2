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
from typing import Optional, List, Dict
import pickle

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define request model
class PredictionRequest(BaseModel):
    cookie_value: str

class CookieItem(BaseModel):
    name: str
    value: str
    domain: Optional[str] = None
    source: Optional[str] = None

class BulkPredictionRequest(BaseModel):
    cookies: List[CookieItem]

# Utility functions for text preprocessing and word encoding
def preprocess_text(text):
    """Preprocess the cookie text for word-level tokenization."""
    if not isinstance(text, str):
        return ""
    # For word-level, we just lowercase the text
    return text.lower()

def pad_sequences(sequences, maxlen=50, padding='post', truncating='post'):
    """
    Thay thế cho keras.preprocessing.sequence.pad_sequences
    
    Args:
        sequences: Danh sách các chuỗi số cần pad
        maxlen: Độ dài tối đa sau khi pad
        padding: Vị trí pad ('pre' hoặc 'post')
        truncating: Vị trí cắt nếu chuỗi dài hơn maxlen ('pre' hoặc 'post')
        
    Returns:
        Mảng numpy các chuỗi đã được pad
    """
    num_samples = len(sequences)
    result = np.zeros((num_samples, maxlen), dtype=np.int64)
    
    for i, seq in enumerate(sequences):
        if len(seq) == 0:
            continue
            
        if len(seq) > maxlen:
            # Truncate
            if truncating == 'pre':
                seq = seq[-maxlen:]
            else:
                seq = seq[:maxlen]
                
        # Pad
        if padding == 'pre':
            result[i, -len(seq):] = seq
        else:
            result[i, :len(seq)] = seq
            
    return result

# Load tokenizer
try:
    logger.info("Loading tokenizer...")
    with open('../pickle/tokenizer.pkl', 'rb') as f:
        tokenizer = pickle.load(f)
    logger.info(f"Tokenizer loaded successfully with vocabulary size: {len(tokenizer.word_index) + 1}")
except Exception as e:
    logger.error(f"Error loading tokenizer: {str(e)}")
    logger.error(traceback.format_exc())
    tokenizer = None

def word_encode(sentences, tokenizer, maxlen=50):
    """
    Mã hóa câu thành chuỗi số sử dụng word-level encoding
    
    Args:
        sentences: Chuỗi, danh sách chuỗi
        tokenizer: Đối tượng Tokenizer đã được huấn luyện
        maxlen: Độ dài tối đa của chuỗi sau khi mã hóa
        
    Returns:
        Numpy array của các chuỗi đã được mã hóa và đệm
    """
    if isinstance(sentences, str):
        sentences = [sentences]
    
    # Áp dụng tiền xử lý
    sentences = [preprocess_text(s) for s in sentences if isinstance(s, str)]
    
    # Chuyển đổi thành chuỗi số
    sequences = tokenizer.texts_to_sequences(sentences)
    
    # Đệm chuỗi để có cùng độ dài
    padded_sequences = pad_sequences(sequences, maxlen=maxlen, padding='post', truncating='post')
    
    return padded_sequences

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
max_length = 549  # Thay đổi max_length theo tokenizer mới
vocab_size = 311148  # Sẽ cập nhật khi load model

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
    # Cập nhật vocab_size dựa trên tokenizer đã load
    if tokenizer is not None:
        vocab_size = 311148
        logger.info(f"Setting vocab_size to {vocab_size} based on tokenizer")
    
    model = TransformerEncoderCls(vocab_size=vocab_size, max_length=max_length, embed_dim=128, num_heads=4, ff_dim=256, dropout=0.1, device='cpu')
    model_path = '../model/model_fold_5.pt'
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Error loading model: {str(e)}")
    logger.error(traceback.format_exc())
    # Still initialize model as None so the app can start
    model = None

def calculate_website_risk_score(cookies_risk_counts):
    """
    Tính toán điểm rủi ro tổng thể của trang web dựa trên số lượng cookie ở mỗi mức độ rủi ro.
    
    Args:
        cookies_risk_counts: Dict gồm các cặp {mức_độ_rủi_ro: số_lượng}
        Ví dụ: {'VERY LOW': 3, 'LOW': 29, 'AVERAGE': 8, 'HIGH': 18, 'VERY HIGH': 1}
    
    Returns:
        Tuple(float, str): (điểm rủi ro, mức độ rủi ro)
    """
    # Gán trọng số cho mỗi mức độ rủi ro
    risk_weights = {
        'VERY LOW': 0.1,
        'LOW': 0.3,
        'AVERAGE': 0.5,
        'HIGH': 0.8,
        'VERY HIGH': 1.0
    }
    
    total_cookies = sum(cookies_risk_counts.values())
    if total_cookies == 0:
        return 0, "NO RISK"
    
    # Tính tổng điểm rủi ro có trọng số
    weighted_sum = sum(risk_weights[risk_level] * count for risk_level, count in cookies_risk_counts.items())
    
    # Tính điểm trung bình có trọng số
    risk_score = weighted_sum / total_cookies
    
    # Quy đổi điểm thành mức độ rủi ro
    if risk_score < 0.2:
        website_risk = "VERY LOW"
    elif risk_score < 0.4:
        website_risk = "LOW"
    elif risk_score < 0.6:
        website_risk = "MODERATE"
    elif risk_score < 0.8:
        website_risk = "HIGH"
    else:
        website_risk = "VERY HIGH"
    
    return risk_score, website_risk

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

app.mount("/templates", StaticFiles(directory="./templates"), name="templates")

@app.get('/')
def read_root():
    return FileResponse('../templates/index.html')

@app.get('/health')
def health_check():
    """Health check endpoint to test if the API is running."""
    if model is None:
        return {"status": "unhealthy", "model_loaded": False, "error": "Model failed to load"}
    
    if tokenizer is None:
        return {"status": "unhealthy", "tokenizer_loaded": False, "error": "Tokenizer failed to load"}
    
    # Try a simple prediction to ensure the model works
    try:
        sample_value = "test_cookie_value_123"
        processed_text = preprocess_text(sample_value)
        
        # Encode using word-level tokenization
        word_input_np = word_encode([processed_text], tokenizer, max_length)
        word_input_tensor = torch.tensor(word_input_np, dtype=torch.int64)
        
        with torch.no_grad():
            output = model(word_input_tensor, model.device)
            probabilities = torch.softmax(output, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1)
            class_name = class_names[predicted_class.item()]
        
        return {
            "status": "healthy", 
            "model_loaded": True,
            "tokenizer_loaded": True,
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
    Predicts the class of a given cookie value using word-level encoding.

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
        
        if tokenizer is None:
            raise HTTPException(status_code=500, detail="Tokenizer not loaded")
        
        # Process the cookie value
        processed_text = preprocess_text(data.cookie_value)
        
        # Encode using word-level tokenization
        word_input_np = word_encode([processed_text], tokenizer, max_length)
        word_input_tensor = torch.tensor(word_input_np, dtype=torch.int64)
        
        logger.info(f"Encoded cookie to tensor shape: {word_input_tensor.shape}")
        
        # Make prediction
        with torch.no_grad():
            output = model(word_input_tensor, model.device)
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

@app.post('/predict_bulk')
async def predict_bulk(data: BulkPredictionRequest):
    """
    Predicts the risk class for multiple cookies and calculates the overall website risk score.

    Args:
        data: BulkPredictionRequest object containing a list of cookie items

    Returns:
        dict: A dictionary containing individual predictions and overall website risk assessment
    """
    try:
        logger.info(f"Received bulk prediction request with {len(data.cookies)} cookies")
        
        if not data.cookies:
            logger.warning("Empty cookie list received")
            raise HTTPException(status_code=400, detail="Cookie list cannot be empty")
        
        if tokenizer is None:
            raise HTTPException(status_code=500, detail="Tokenizer not loaded")
            
        # Process each cookie
        predictions = []
        risk_counts = {
            'VERY LOW': 0,
            'LOW': 0,
            'AVERAGE': 0,
            'HIGH': 0,
            'VERY HIGH': 0
        }
        
        for cookie in data.cookies:
            if not cookie.value:
                logger.warning(f"Empty cookie value for '{cookie.name}', skipping")
                continue
                
            # Process and predict for each cookie
            processed_text = preprocess_text(cookie.value)
            word_input_np = word_encode([processed_text], tokenizer, max_length)
            word_input_tensor = torch.tensor(word_input_np, dtype=torch.int64)
            
            with torch.no_grad():
                output = model(word_input_tensor, model.device)
                probabilities = torch.softmax(output, dim=1)
                predicted_class = torch.argmax(probabilities, dim=1)
                class_name = class_names[predicted_class.item()].upper()
            
            # Update risk counts
            risk_counts[class_name] += 1
            
            # Add prediction to results
            predictions.append({
                'cookie_name': cookie.name,
                'domain': cookie.domain,
                'source': cookie.source,
                'predicted_class': class_name,
                'probabilities': probabilities[0].tolist()
            })
        
        # Calculate website risk score
        risk_score, risk_level = calculate_website_risk_score(risk_counts)
        
        logger.info(f"Bulk prediction successful. Website risk: {risk_level} ({risk_score:.2f})")
        
        return {
            'cookie_predictions': predictions,
            'risk_distribution': risk_counts,
            'website_risk': {
                'score': risk_score,
                'level': risk_level
            }
        }
    except Exception as e:
        logger.error(f"Error during bulk prediction: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Bulk prediction error: {str(e)}")

if __name__ == "__main__":
    logger.info("Starting server...")
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)