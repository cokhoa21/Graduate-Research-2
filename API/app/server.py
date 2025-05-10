from fastapi import FastAPI
import numpy as np
import torch.nn as nn
import uvicorn
import torch
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

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
model = TransformerEncoderCls(vocab_size=100000, max_length=128, embed_dim=128, num_heads=4, ff_dim=128, dropout=0.1, device='cpu')
model.load_state_dict(torch.load('../model/model.pt', map_location=torch.device('cpu')))
model.eval()

class_names = np.array(['very low', 'low', 'average', 'high', 'very high'])

app = FastAPI()

app.mount("/templates", StaticFiles(directory="../templates"), name="templates")

@app.get('/')
def read_root():
    return FileResponse('../templates/index.html')

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

@app.post('/predict')
def predict(data: dict):
    """
    Predicts the class of a given sequence.

    Args:
        data (dict): A dictionary containing the features to predict.
        e.g. {"features": [1, 2, 3, 4, ...]}

    Returns:
        dict: A dictionary containing the predicted class.
    """        
    # Convert input sequence to tensor and ensure it's the right shape
    features = torch.tensor(data['sequence']).reshape(1, -1).long()
    features = features.to(model.device)

    # Make prediction
    with torch.no_grad():
        output = model(features, model.device)
        # Get the prediction for the entire sequence
        probabilities = torch.softmax(output, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1)
        class_name = class_names[predicted_class.item()]
    
    return {
        'predicted_class': class_name,
        'probabilities': probabilities[0].tolist()
    }

if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
