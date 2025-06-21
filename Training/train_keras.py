from google.colab import drive
drive.mount('/content/gdrive')

import os
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
import pandas as pd
import re
import torchtext
torchtext.disable_torchtext_deprecation_warning()
from torchtext.data.functional import to_map_style_dataset
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import WeightedRandomSampler
from torch.utils.data import DataLoader
import time
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, classification_report
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, f1_score
from itertools import cycle
import seaborn as sns
import torch.optim as optim
import logging
from datetime import datetime
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# # Đường dẫn tới thư mục trong Google Drive để lưu các file
save_path = '/content/gdrive/MyDrive/Cookie_Classifier'
os.makedirs(save_path, exist_ok=True)  # Tạo thư mục nếu chưa tồn tại

# # Đọc file CSV
file_path = '/content/gdrive/MyDrive/Cookie_Dataset_Test/data.csv'
df = pd.read_csv(file_path)

# # Chỉ lấy n hàng đầu tiên
# df = df.head(500000)

# # # Chia dataset thành 2 tập: 80:20
train_size = int(0.8 * len(df))
val_size = int(0.2 * len(df))

train_data = df[:train_size]
val_data = df[train_size:train_size + val_size]

# Lưu 3 tập vào Google Drive
train_data.to_csv(os.path.join(save_path, 'train.csv'), index=False)
val_data.to_csv(os.path.join(save_path, 'valid.csv'), index=False)

print("Train, validation, and test datasets have been saved to:", save_path)

def load_data_from_path(file_path):
    examples = []
    df = pd.read_csv(file_path)

    # Giữ lại các hàng có label thuộc {0, 1, 2, 3, 4}
    df = df[df['label'].isin({0, 1, 2, 3, 4})]

    for _, row in df.iterrows():
        text = row['value']
        label = row['label']
        data = {
            'sentence': text,
            'label': label
        }
        examples.append(data)

    return pd.DataFrame(examples)

file_paths = {
    'train': '/content/gdrive/MyDrive/Cookie_Classifier/train.csv',
    'valid': '/content/gdrive/MyDrive/Cookie_Classifier/valid.csv',
}

train_df = load_data_from_path(file_paths['train'])
valid_df = load_data_from_path(file_paths['valid'])

train_df

def preprocess_text(text):
  if isinstance(text, str):
    text = text.lower()
  return text

train_df['preprocess_sentence'] = [preprocess_text(row['sentence']) for index, row in train_df.iterrows()]
valid_df['preprocess_sentence'] = [preprocess_text(row['sentence']) for index, row in valid_df.iterrows()]

train_df = train_df.dropna(subset=['preprocess_sentence'])
valid_df = valid_df.dropna(subset=['preprocess_sentence'])

def create_word_vocab(sentences, max_words=10000):
    """
    Tạo từ điển word-level từ tập dữ liệu văn bản

    Args:
        sentences: Chuỗi, danh sách chuỗi, hoặc pandas.Series chứa các câu
        max_words: Số lượng từ tối đa sẽ giữ lại

    Returns:
        tokenizer: Đối tượng Tokenizer đã được huấn luyện
    """
    if isinstance(sentences, str):
        sentences = [sentences]
    elif isinstance(sentences, pd.Series):
        # Áp dụng hàm làm sạch cho mỗi câu
        sentences = sentences.dropna().apply(preprocess_text).tolist()
    elif isinstance(sentences, list):
        # Áp dụng hàm làm sạch cho mỗi câu trong danh sách
        sentences = [preprocess_text(s) for s in sentences if isinstance(s, str)]
    else:
        raise ValueError("Đầu vào phải là một chuỗi, danh sách, hoặc pandas.Series")

    # Tạo tokenizer
    tokenizer = Tokenizer(num_words=max_words, oov_token='<OOV>')
    tokenizer.fit_on_texts(sentences)

    return tokenizer

def word_encode(sentences, tokenizer, maxlen=50):
    """
    Mã hóa câu thành chuỗi số sử dụng word-level encoding

    Args:
        sentences: Chuỗi, danh sách chuỗi, hoặc pandas.Series chứa các câu
        tokenizer: Đối tượng Tokenizer đã được huấn luyện
        maxlen: Độ dài tối đa của chuỗi sau khi mã hóa

    Returns:
        padded_sequences: Mảng numpy của các chuỗi đã được mã hóa và đệm
    """
    if isinstance(sentences, str):
        sentences = [sentences]
    elif isinstance(sentences, pd.Series):
        # Áp dụng hàm làm sạch cho mỗi câu
        sentences = sentences.dropna().apply(preprocess_text).tolist()
    elif isinstance(sentences, list):
        # Áp dụng hàm làm sạch cho mỗi câu trong danh sách
        sentences = [preprocess_text(s) for s in sentences if isinstance(s, str)]
    else:
        raise ValueError("Đầu vào phải là một chuỗi, danh sách, hoặc pandas.Series")

    # Chuyển đổi thành chuỗi số
    sequences = tokenizer.texts_to_sequences(sentences)

    # Đệm chuỗi để có cùng độ dài
    padded_sequences = pad_sequences(sequences, maxlen=maxlen, padding='post', truncating='post')

    return padded_sequences

# Sử dụng
# Giả sử train_df là DataFrame của bạn với cột 'preprocess_sentence'

# Bước 1: Tạo tokenizer từ dữ liệu huấn luyện
tokenizer = create_word_vocab(train_df['preprocess_sentence'], max_words=100000)

# Bước 2: Mã hóa dữ liệu thành chuỗi số
word_sequences = word_encode(train_df['preprocess_sentence'], tokenizer, maxlen=50)

print(f"Kích thước của chuỗi đã mã hóa: {word_sequences.shape}")
print(f"Mẫu chuỗi đã mã hóa: {word_sequences[1]}")

# Thông tin thêm về từ điển
word_index = tokenizer.word_index
print(f"Kích thước từ điển: {len(word_index)}")
print(f"Một số từ đầu tiên: {list(word_index.items())[:10]}")

# Lưu tokenizer để sử dụng sau này
import pickle
with open('tokenizer.pkl', 'wb') as f:
    pickle.dump(tokenizer, f, protocol=pickle.HIGHEST_PROTOCOL)

vocab_size = len(word_index)

def calculate_maxlen(df, tokenizer):
    """
    Tính toán độ dài tối đa của câu sau khi đã tokenize và in ra câu dài nhất

    Args:
        df: DataFrame chứa dữ liệu
        tokenizer: Đối tượng Tokenizer đã được huấn luyện

    Returns:
        maxlen: Độ dài tối đa của câu sau khi mã hóa
    """
    max_length = 0
    longest_sentence = ""
    longest_encoded = []

    for index, row in df.iterrows():
        sentence = row['preprocess_sentence']

        # Bỏ qua các câu None
        if not isinstance(sentence, str):
            continue

        # Chuyển câu thành chuỗi các số sử dụng tokenizer
        encoded_sentence = tokenizer.texts_to_sequences([sentence])[0]

        # Kiểm tra nếu câu này dài hơn câu dài nhất hiện tại
        if len(encoded_sentence) > max_length:
            max_length = len(encoded_sentence)
            longest_sentence = sentence
            longest_encoded = encoded_sentence

    # Đảm bảo maxlen không quá nhỏ
    max_length = max(max_length, 10)  # Ít nhất 10 token

    print(f"Calculated maxlen: {max_length}")
    print(f"\nLongest sentence (length={max_length}):")

    # In ra câu dài nhất (cắt ngắn nếu quá dài để dễ đọc)
    if len(longest_sentence) > 1000:
        print(f"{longest_sentence[:1000]}... (truncated)")
    else:
        print(longest_sentence)

    # In ra một số token đầu tiên của câu dài nhất sau khi đã mã hóa
    print(f"\nFirst 20 tokens of encoded longest sentence:")
    print(longest_encoded[:20])

    # In ra từ tương ứng với các token (giải mã)
    idx_to_word = {v: k for k, v in tokenizer.word_index.items()}
    idx_to_word[0] = '<PAD>'  # Thêm token PAD

    decoded_tokens = [idx_to_word.get(idx, '<UNK>') for idx in longest_encoded[:20]]
    print(f"\nFirst 20 decoded tokens:")
    print(decoded_tokens)

    return max_length

def prepare_dataset_word_level(df, tokenizer, maxlen):
    """
    Chuẩn bị dataset với word-level encoding

    Args:
        df: DataFrame chứa dữ liệu
        tokenizer: Đối tượng Tokenizer đã được huấn luyện
        maxlen: Độ dài tối đa của chuỗi sau khi mã hóa

    Returns:
        Iterator cung cấp (encoded_sentence, label)
    """
    # create iterator for dataset: (sentence, label)
    for index, row in df.iterrows():
        sentence = row['preprocess_sentence']

        # Bỏ qua các câu None
        if not isinstance(sentence, str):
            continue

        # Chuyển câu thành chuỗi các số sử dụng tokenizer
        encoded_sentence = tokenizer.texts_to_sequences([sentence])[0]

        # Thực hiện padding hoặc truncate để đạt được độ dài cố định
        if len(encoded_sentence) < maxlen:
            encoded_sentence = encoded_sentence + [0] * (maxlen - len(encoded_sentence))
        else:
            encoded_sentence = encoded_sentence[:maxlen]

        label = row['label']
        yield encoded_sentence, label

maxlen = calculate_maxlen(train_df, tokenizer)

# Sử dụng tokenizer đã được tạo trước đó
# tokenizer = create_word_vocab(train_df['preprocess_sentence'], max_words=vocab_size)

# Chuẩn bị các dataset
train_dataset = prepare_dataset_word_level(train_df, tokenizer, maxlen)
train_dataset = to_map_style_dataset(train_dataset)

valid_dataset = prepare_dataset_word_level(valid_df, tokenizer, maxlen)
valid_dataset = to_map_style_dataset(valid_dataset)

def collate_batch(batch):
    """
    Tạo batch từ danh sách các mẫu

    Args:
        batch: Danh sách các cặp (encoded_sentence, label)

    Returns:
        Tensor các câu đã mã hóa và tensor các nhãn
    """
    # Tách câu và nhãn
    sentences, labels = list(zip(*batch))

    # Chuyển đổi thành tensor
    encoded_sentences = torch.tensor(sentences, dtype=torch.int64)
    labels = torch.tensor(labels)

    return encoded_sentences, labels

y_train = torch.tensor([label for _, label in train_dataset], dtype=torch.long)

class_counts = [(y_train == i).sum().item() for i in range(5)]
class_weights = [1.0 / count if count else 0.0 for count in class_counts] # Handle zero counts
sample_weights = [class_weights[label] for label in y_train]

sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

batch_size = 1024

train_dataloader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    collate_fn=collate_batch,
    sampler=sampler
)
valid_dataloader = DataLoader(
    valid_dataset,
    batch_size=batch_size,
    shuffle=False,
    collate_fn=collate_batch
)

next(iter(train_dataloader))

encoded_sentences, labels = next(iter(train_dataloader))

encoded_sentences.shape

labels.shape

def train_epoch(model, optimizer, criterion, train_dataloader, device, epoch=0, log_interval=50):
    model.train()
    total_acc, total_count = 0, 0
    losses = []
    start_time = time.time()

    for idx, (inputs, labels) in enumerate(train_dataloader):
        inputs = inputs.to(device)
        labels = labels.type(torch.LongTensor).to(device) # This line is modified

        optimizer.zero_grad()

        predictions = model(inputs, device)

        # compute loss
        loss = criterion(predictions, labels)
        losses.append(loss.item())

        # backward
        loss.backward()
        optimizer.step()
        total_acc += (predictions.argmax(1) == labels).sum().item()
        total_count += labels.size(0)
        if idx % log_interval == 0 and idx > 0:
            elapsed = time.time() - start_time
            print(
                "| epoch {:3d} | {:5d}/{:5d} batches "
                "| accuracy {:8.3f}".format(
                    epoch, idx, len(train_dataloader), total_acc / total_count
                )
            )
            total_acc, total_count = 0, 0
            start_time = time.time()

    epoch_acc = total_acc / total_count
    epoch_loss = sum(losses) / len(losses)
    return epoch_acc, epoch_loss

def evaluate_epoch(model, criterion, valid_dataloader, device):
    model.eval()
    total_acc, total_count = 0, 0
    losses = []

    with torch.no_grad():
        for idx, (inputs, labels) in enumerate(valid_dataloader):
            inputs = inputs.to(device)
            labels = labels.type(torch.LongTensor).to(device) # This line is modified

            predictions = model(inputs, device)

            loss = criterion(predictions, labels)
            losses.append(loss.item())

            total_acc += (predictions.argmax(1) == labels).sum().item()
            total_count += labels.size(0)

    epoch_acc = total_acc / total_count
    epoch_loss = sum(losses) / len(losses)
    return epoch_acc, epoch_loss

def train(model, model_name, save_model, optimizer, criterion, train_dataloader, valid_dataloader, num_epochs, device,
          patience=5, lr_scheduler_factor=0.5, lr_scheduler_patience=2, min_lr=1e-6):
    """
    Train the model with early stopping and learning rate scheduling

    Args:
        model: Model to train
        model_name: Name of the model for saving
        save_model: Directory to save the model
        optimizer: Optimizer to use
        criterion: Loss function
        train_dataloader: DataLoader for training data
        valid_dataloader: DataLoader for validation data
        num_epochs: Maximum number of epochs to train
        device: Device to train on
        patience: Number of epochs to wait for improvement before early stopping
        lr_scheduler_factor: Factor by which to reduce learning rate
        lr_scheduler_patience: Number of epochs with no improvement after which learning rate will be reduced
        min_lr: Minimum learning rate

    Returns:
        model: Trained model
        metrics: Dictionary containing training metrics
    """
    train_accs, train_losses = [], []
    eval_accs, eval_losses = [], []
    best_loss_eval = float('inf')
    best_model_state = None
    times = []

    # Early stopping variables
    counter = 0

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=lr_scheduler_factor,
        patience=lr_scheduler_patience,
        verbose=True,
        min_lr=min_lr
    )

    for epoch in range(1, num_epochs+1):
        epoch_start_time = time.time()

        # Training
        train_acc, train_loss = train_epoch(model, optimizer, criterion, train_dataloader, device, epoch)
        train_accs.append(train_acc)
        train_losses.append(train_loss)

        # Evaluation
        eval_acc, eval_loss = evaluate_epoch(model, criterion, valid_dataloader, device)
        eval_accs.append(eval_acc)
        eval_losses.append(eval_loss)

        # Update learning rate scheduler
        scheduler.step(eval_loss)

        # Check if this is the best model
        if eval_loss < best_loss_eval:
            best_loss_eval = eval_loss
            best_model_state = model.state_dict().copy()
            # Save best model
            torch.save(model.state_dict(), save_model + f'/{model_name}.pt')
            # Reset early stopping counter
            counter = 0
        else:
            counter += 1

        # Early stopping check
        if counter >= patience:
            print(f"Early stopping triggered after {epoch} epochs")
            break

        times.append(time.time() - epoch_start_time)

        # Current learning rate
        current_lr = optimizer.param_groups[0]['lr']

        # Print loss, acc end epoch
        print("-" * 59)
        print(
            "| End of epoch {:3d} | Time: {:5.2f}s | LR: {:.6f} | Train Acc {:8.3f} | Train Loss {:8.3f} "
            "| Valid Acc {:8.3f} | Valid Loss {:8.3f} ".format(
                epoch, time.time() - epoch_start_time, current_lr, train_acc, train_loss, eval_acc, eval_loss
            )
        )
        print("-" * 59)

    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    model.eval()

    metrics = {
        'train_accuracy': train_accs,
        'train_loss': train_losses,
        'valid_accuracy': eval_accs,
        'valid_loss': eval_losses,
        'time': times,
        'best_valid_loss': best_loss_eval,
        'epochs_trained': len(train_accs)
    }
    return model, metrics

import torch
from torch import nn
import torch.nn.functional as F

class ImprovedLSTMClassifier(nn.Module):
    def __init__(self, vocab_size, num_classes=5, embedding_dim=256, hidden_size=256,
                 num_layers=2, dropout=0.3, bidirectional=True):
        super(ImprovedLSTMClassifier, self).__init__()

        # Increased embedding dimension
        self.embedding_layer = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=0  # Xử lý padding tokens đúng cách
        )

        # Thêm Embedding Dropout
        self.embedding_dropout = nn.Dropout(p=0.2)

        # LSTM nhiều lớp và hai chiều
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0
        )

        # Kích thước đầu ra từ LSTM
        lstm_output_dim = hidden_size * 2 if bidirectional else hidden_size

        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(lstm_output_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )

        # Pooling layers - kết hợp nhiều kiểu pooling
        self.global_max_pooling = nn.AdaptiveMaxPool1d(1)
        self.global_avg_pooling = nn.AdaptiveAvgPool1d(1)

        # Batch Normalization
        self.bn = nn.BatchNorm1d(lstm_output_dim * 2)  # *2 vì kết hợp max và avg pooling

        # Fully connected layers với residual connection
        self.fc1 = nn.Linear(lstm_output_dim * 2, 256)
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(256, 128)
        self.dropout2 = nn.Dropout(dropout)
        self.fc3 = nn.Linear(128, num_classes)

    def apply_attention(self, lstm_output):
        # lstm_output: [batch_size, seq_len, hidden_size*2]
        # Tính attention scores
        attention_scores = self.attention(lstm_output)  # [batch_size, seq_len, 1]

        # Áp dụng softmax để có trọng số
        attention_weights = F.softmax(attention_scores, dim=1)

        # Nhân trọng số với lstm_output
        context_vector = torch.sum(attention_weights * lstm_output, dim=1)  # [batch_size, hidden_size*2]

        return context_vector, attention_weights

    def forward(self, x, device=None):  # Thêm tham số device để tương thích với cách gọi cũ
        # Chuyển x sang device nếu cần
        if device is not None and x.device.type != device.type:
            x = x.to(device)

        # Embedding layer với dropout
        embeddings = self.embedding_layer(x)  # [batch_size, seq_len, embedding_dim]
        embeddings = self.embedding_dropout(embeddings)

        # Thêm padding mask để xử lý padding trong LSTM
        padding_mask = (x != 0).float().unsqueeze(-1)  # [batch_size, seq_len, 1]
        embeddings = embeddings * padding_mask

        # LSTM layer
        lstm_output, (hidden, cell) = self.lstm(embeddings)  # lstm_output: [batch_size, seq_len, hidden_size*2]

        # Kết hợp nhiều phương pháp trích xuất đặc trưng:

        # 1. Attention mechanism
        context_vector, _ = self.apply_attention(lstm_output)  # [batch_size, hidden_size*2]

        # 2. Global max pooling
        max_pooled = self.global_max_pooling(lstm_output.permute(0, 2, 1)).squeeze(-1)  # [batch_size, hidden_size*2]

        # 3. Global average pooling
        avg_pooled = self.global_avg_pooling(lstm_output.permute(0, 2, 1)).squeeze(-1)  # [batch_size, hidden_size*2]

        # Kết hợp các đặc trưng
        concatenated = torch.cat([max_pooled, avg_pooled], dim=1)  # [batch_size, hidden_size*4]

        # Batch normalization
        normalized = self.bn(concatenated)

        # Fully connected layers with residual connections and ReLU
        x = self.fc1(normalized)
        x = F.relu(x)
        x = self.dropout1(x)

        residual = x  # Lưu cho residual connection

        x = self.fc2(x)
        x = F.relu(x)
        x = self.dropout2(x)

        # Dự đoán class
        logits = self.fc3(x)

        # Không dùng softmax trong forward để tránh double softmax với nn.CrossEntropyLoss
        return logits

def setup_logger(log_path):
    """
    Thiết lập logger để ghi log theo định dạng yêu cầu

    Args:
        log_path: Đường dẫn của file log

    Returns:
        logger: Đối tượng logger đã được cấu hình
    """
    # Tạo logger
    logger = logging.getLogger('classifier')
    logger.setLevel(logging.INFO)

    # Tạo file handler
    fh = logging.FileHandler(log_path)
    fh.setLevel(logging.INFO)

    # Tạo formatter
    formatter = logging.Formatter('%(asctime)s :: %(name)s :: %(levelname)s :: %(message)s',
                                 datefmt='%Y-%m-%d-%H:%M:%S')

    # Thêm formatter vào handler
    fh.setFormatter(formatter)

    # Thêm handler vào logger
    logger.handlers = []  # Xóa handlers cũ (nếu có)
    logger.addHandler(fh)

    return logger

def log_metrics(true_labels, pred_labels, log_path, total_samples):
    """
    Ghi log các metrics theo định dạng yêu cầu

    Args:
        true_labels: Nhãn thực tế
        pred_labels: Nhãn dự đoán
        log_path: Đường dẫn để lưu file log
        total_samples: Tổng số mẫu
    """
    # Thiết lập logger
    logger = setup_logger(log_path)

    # Tính toán accuracy
    accuracy = accuracy_score(true_labels, pred_labels)
    correct_count = int(accuracy * total_samples)

    # Tính toán precision, recall, f1 cho mỗi lớp
    precision_per_class, recall_per_class, f1_per_class, _ = precision_recall_fscore_support(
        true_labels, pred_labels, average=None
    )

    # Tính toán micro average
    precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(
        true_labels, pred_labels, average='micro'
    )

    # Tính toán macro average
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        true_labels, pred_labels, average='macro'
    )

    # Tính toán weighted average
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        true_labels, pred_labels, average='weighted'
    )

    # Ghi log theo định dạng yêu cầu
    logger.info(f"Total Accuracy Count: {correct_count}")
    logger.info(f"Total Accuracy Ratio: {accuracy}")
    logger.info(f"Micro Precision: {precision_micro}")
    logger.info(f"Micro Recall: {recall_micro}")
    logger.info(f"Micro F1Score: {f1_micro}")
    logger.info(f"Macro Precision: {precision_macro}")
    logger.info(f"Macro Recall: {recall_macro}")
    logger.info(f"Macro F1Score: {f1_macro}")
    logger.info(f"Weighted Precision: {precision_weighted}")
    logger.info(f"Weighted Recall: {recall_weighted}")
    logger.info(f"Weighted F1Score: {f1_weighted}")
    logger.info(f"Precision for each class: {precision_per_class}")
    logger.info(f"Recall for each class: {recall_per_class}")
    logger.info(f"F1Score for each class: {f1_per_class}")
    logger.info(f"-------------------------------")
    logger.info(f"(Old Method) Total Accuracy: {accuracy*100:.3f}%")
    logger.info(f"(Old Method) Precision: {precision_per_class}")
    logger.info(f"(Old Method) Recall: {recall_per_class}")
    logger.info(f"(Old Method) F1 Scores: {f1_per_class}")

    return {
        'accuracy': accuracy,
        'precision_per_class': precision_per_class,
        'recall_per_class': recall_per_class,
        'f1_per_class': f1_per_class,
        'precision_micro': precision_micro,
        'recall_micro': recall_micro,
        'f1_micro': f1_micro,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'f1_macro': f1_macro,
        'precision_weighted': precision_weighted,
        'recall_weighted': recall_weighted,
        'f1_weighted': f1_weighted
    }

def evaluate_model(model, dataloader, criterion, device):
    """
    Evaluates the model with comprehensive metrics:
    - Accuracy
    - Precision, Recall, F1-score for each class
    - Confusion Matrix
    - Classification Report

    Args:
        model: Trained model
        dataloader: DataLoader for evaluation data
        criterion: Loss function
        device: Device to run evaluation on (cuda/cpu)

    Returns:
        dict: Dictionary containing all evaluation metrics
    """
    model.eval()
    total_acc, total_count = 0, 0
    all_predictions = []
    all_labels = []
    losses = []

    with torch.no_grad():
        for idx, (inputs, labels) in enumerate(dataloader):
            inputs = inputs.to(device)
            labels = labels.type(torch.LongTensor).to(device)

            predictions = model(inputs, device)
            pred_labels = predictions.argmax(1)

            # Store predictions and true labels
            all_predictions.extend(pred_labels.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            # Calculate accuracy
            total_acc += (pred_labels == labels).sum().item()
            total_count += labels.size(0)

            # Calculate loss
            loss = criterion(predictions, labels)
            losses.append(loss.item())

    # Convert lists to numpy arrays
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)

    # Calculate metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        all_labels,
        all_predictions,
        average=None
    )

    # Calculate macro and weighted averages
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        all_labels,
        all_predictions,
        average='macro'
    )

    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        all_labels,
        all_predictions,
        average='weighted'
    )

    # Create classification report
    class_report = classification_report(
        all_labels,
        all_predictions,
        output_dict=True
    )

    # Create confusion matrix
    conf_matrix = confusion_matrix(all_labels, all_predictions)

    # Calculate overall metrics
    accuracy = total_acc / total_count
    avg_loss = sum(losses) / len(losses)

    # Store all metrics in a dictionary
    metrics = {
        'accuracy': accuracy,
        'loss': avg_loss,
        'precision_per_class': precision,
        'recall_per_class': recall,
        'f1_per_class': f1,
        'support_per_class': support,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'f1_macro': f1_macro,
        'precision_weighted': precision_weighted,
        'recall_weighted': recall_weighted,
        'f1_weighted': f1_weighted,
        'classification_report': class_report,
        'confusion_matrix': conf_matrix
    }

    return metrics


def visualize_metrics(metrics):
    """
    Visualizes evaluation metrics with plots.

    Args:
        metrics: Dictionary containing evaluation metrics
    """
    # Create figure with subplots
    fig, axs = plt.subplots(2, 2, figsize=(15, 12))

    # Plot precision, recall, and F1 per class
    classes = range(len(metrics['precision_per_class']))

    axs[0, 0].bar(classes, metrics['precision_per_class'], alpha=0.7, label='Precision')
    axs[0, 0].bar(classes, metrics['recall_per_class'], alpha=0.5, label='Recall')
    axs[0, 0].bar(classes, metrics['f1_per_class'], alpha=0.3, label='F1')
    axs[0, 0].set_xticks(classes)
    axs[0, 0].set_xlabel('Class')
    axs[0, 0].set_ylabel('Score')
    axs[0, 0].set_title('Precision, Recall, and F1 per Class')
    axs[0, 0].legend()

    # Plot confusion matrix
    conf_matrix = metrics['confusion_matrix']
    sns.heatmap(
        conf_matrix,
        annot=True,
        fmt='d',
        cmap='Blues',
        ax=axs[0, 1]
    )
    axs[0, 1].set_xlabel('Predicted Label')
    axs[0, 1].set_ylabel('True Label')
    axs[0, 1].set_title('Confusion Matrix')

    # Plot macro vs weighted metrics
    metric_names = ['Precision', 'Recall', 'F1-Score']
    macro_metrics = [
        metrics['precision_macro'],
        metrics['recall_macro'],
        metrics['f1_macro']
    ]
    weighted_metrics = [
        metrics['precision_weighted'],
        metrics['recall_weighted'],
        metrics['f1_weighted']
    ]

    axs[1, 0].bar(metric_names, macro_metrics, alpha=0.7, label='Macro')
    axs[1, 0].bar(metric_names, weighted_metrics, alpha=0.5, label='Weighted')
    axs[1, 0].set_ylim(0, 1)
    axs[1, 0].set_title('Macro vs Weighted Metrics')
    axs[1, 0].legend()

    # Display summary with text
    axs[1, 1].axis('off')
    summary_text = f"""
    Summary Metrics:

    Accuracy: {metrics['accuracy']:.4f}
    Loss: {metrics['loss']:.4f}

    Macro-Average:
      Precision: {metrics['precision_macro']:.4f}
      Recall: {metrics['recall_macro']:.4f}
      F1-Score: {metrics['f1_macro']:.4f}

    Weighted-Average:
      Precision: {metrics['precision_weighted']:.4f}
      Recall: {metrics['recall_weighted']:.4f}
      F1-Score: {metrics['f1_weighted']:.4f}
    """
    axs[1, 1].text(0, 0.5, summary_text, fontsize=12)

    plt.tight_layout()
    plt.show()

def print_classification_report(metrics):
    """
    Prints a formatted classification report.

    Args:
        metrics: Dictionary containing evaluation metrics
    """
    report = metrics['classification_report']

    # Print header
    print("\nClassification Report:")
    print("-" * 80)
    print(f"{'Class':^10} | {'Precision':^10} | {'Recall':^10} | {'F1-Score':^10} | {'Support':^10}")
    print("-" * 80)

    # Print per-class metrics
    for cls in sorted([c for c in report.keys() if c.isdigit()]):
        # Use 'f' format for support as it might be a float
        print(f"{int(cls):^10d} | {report[cls]['precision']:^10.4f} | {report[cls]['recall']:^10.4f} | {report[cls]['f1-score']:^10.4f} | {report[cls]['support']:^10.0f}")

    # Print averages
    print("-" * 80)
    print(f"{'macro avg':^10} | {report['macro avg']['precision']:^10.4f} | {report['macro avg']['recall']:^10.4f} | {report['macro avg']['f1-score']:^10.4f} | {report['macro avg']['support']:^10.0f}") # Also change format for macro avg support
    print(f"{'weighted avg':^10} | {report['weighted avg']['precision']:^10.4f} | {report['weighted avg']['recall']:^10.4f} | {report['weighted avg']['f1-score']:^10.4f} | {report['weighted avg']['support']:^10.0f}") # Also change format for weighted avg support
    print("-" * 80)
    print(f"Accuracy: {report['accuracy']:.4f}")
    print("-" * 80)


# Example of how to use these functions in the model testing phase:
def evaluate_final_model(model, test_dataloader, criterion, device):
    """
    Performs a comprehensive evaluation of the final model.

    Args:
        model: Trained model
        test_dataloader: DataLoader for test data
        criterion: Loss function
        device: Device to run evaluation on (cuda/cpu)
    """
    print("Performing comprehensive evaluation of the model...")

    # Get all metrics
    metrics = evaluate_model(model, test_dataloader, criterion, device)

    # Print accuracy and loss
    print(f"\nTest Accuracy: {metrics['accuracy']:.4f}")
    print(f"Test Loss: {metrics['loss']:.4f}")

    # Print detailed classification report
    print_classification_report(metrics)

    # Visualize metrics
    visualize_metrics(metrics)

    return metrics

def perform_cross_validation(train_df, n_folds=5, num_epochs=10, batch_size=1024, save_model_dir='./model',
                            patience=5, lr_scheduler_factor=0.5, lr_scheduler_patience=2, min_lr=1e-6):
    """
    Perform k-fold cross validation with early stopping and learning rate scheduling.

    Args:
        train_df: DataFrame containing training data
        n_folds: Number of folds for cross validation
        num_epochs: Maximum number of epochs for each fold
        batch_size: Batch size
        save_model_dir: Directory to save models
        patience: Number of epochs to wait before early stopping
        lr_scheduler_factor: Factor by which to reduce learning rate
        lr_scheduler_patience: Number of epochs with no improvement after which learning rate will be reduced
        min_lr: Minimum learning rate

    Returns:
        dict: Dictionary containing average results and metrics for each fold
    """
    print("\nNOTE: Cross-validation only uses the training set, the test set is kept separate for final evaluation.")

    # Create KFold
    from sklearn.model_selection import KFold
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

    # Prepare metrics for saving results
    cv_accuracies = []
    cv_losses = []
    cv_f1_weighted = []
    cv_f1_macro = []
    cv_metrics_per_fold = []
    cv_epochs_trained = []  # Track how many epochs were actually used before early stopping

    # Loop through each fold
    for fold, (train_idx, val_idx) in enumerate(kf.split(train_df)):
        print(f"\n{'='*20} Fold {fold+1}/{n_folds} {'='*20}")

        # Split data into train and validation for current fold
        fold_train_df = train_df.iloc[train_idx].reset_index(drop=True)
        fold_val_df = train_df.iloc[val_idx].reset_index(drop=True)

        print(f"Train size: {len(fold_train_df)}, Validation size: {len(fold_val_df)}")

        # IMPORTANT: Build tokenizer ONLY on the current fold's training set
        print("Building vocabulary only on the training fold...")
        fold_tokenizer = create_word_vocab(fold_train_df['preprocess_sentence'], max_words=100000)
        vocab_size = len(fold_tokenizer.word_index) + 1  # +1 for padding token
        print(f"Vocabulary size for fold {fold+1}: {vocab_size}")

        # Calculate maxlen based on current fold's training set
        fold_maxlen = calculate_maxlen(fold_train_df, fold_tokenizer)

        # Prepare datasets using current fold's tokenizer and maxlen
        fold_train_dataset = prepare_dataset_word_level(fold_train_df, fold_tokenizer, fold_maxlen)
        fold_train_dataset = to_map_style_dataset(fold_train_dataset)

        fold_val_dataset = prepare_dataset_word_level(fold_val_df, fold_tokenizer, fold_maxlen)
        fold_val_dataset = to_map_style_dataset(fold_val_dataset)

        # Create weighted sampler to handle imbalanced data
        y_train = torch.tensor([label for _, label in fold_train_dataset], dtype=torch.long)
        class_counts = [(y_train == i).sum().item() for i in range(5)]
        class_weights = [1.0 / count if count else 0.0 for count in class_counts]
        sample_weights = [class_weights[label] for label in y_train]
        sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

        # Create dataloaders
        fold_train_dataloader = DataLoader(
            fold_train_dataset,
            batch_size=batch_size,
            collate_fn=collate_batch,
            sampler=sampler
        )

        fold_val_dataloader = DataLoader(
            fold_val_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_batch
        )

        # IMPORTANT: Initialize NEW model for each fold
        print(f"Initializing a new model for fold {fold+1}...")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Initialize model
        model = ImprovedLSTMClassifier(
            vocab_size=vocab_size,
            num_classes=5,
            embedding_dim=256,
            hidden_size=256,
            num_layers=2,
            dropout=0.3,
            bidirectional=True
        )
        model.to(device)

        # Initialize optimizer and loss function
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # Create directory to save model for current fold
        fold_save_dir = os.path.join(save_model_dir, f'fold_{fold+1}')
        os.makedirs(fold_save_dir, exist_ok=True)

        # Train model with early stopping and learning rate scheduling
        model, fold_metrics = train(
            model, f'model_fold_{fold+1}', fold_save_dir, optimizer, criterion,
            fold_train_dataloader, fold_val_dataloader, num_epochs, device,
            patience=patience, lr_scheduler_factor=lr_scheduler_factor,
            lr_scheduler_patience=lr_scheduler_patience, min_lr=min_lr
        )

        # Track how many epochs were actually used before early stopping
        cv_epochs_trained.append(fold_metrics['epochs_trained'])

        # Evaluate model on validation set
        val_metrics = evaluate_model(model, fold_val_dataloader, criterion, device)

        # Save metrics
        cv_accuracies.append(val_metrics['accuracy'])
        cv_losses.append(val_metrics['loss'])
        cv_f1_weighted.append(val_metrics['f1_weighted'])
        cv_f1_macro.append(val_metrics['f1_macro'])
        cv_metrics_per_fold.append(val_metrics)

        # Print results for current fold
        print(f"\nFold {fold+1} Results:")
        print(f"Epochs trained: {fold_metrics['epochs_trained']} of {num_epochs} maximum")
        print(f"Accuracy: {val_metrics['accuracy']:.4f}")
        print(f"Loss: {val_metrics['loss']:.4f}")
        print(f"F1 Weighted: {val_metrics['f1_weighted']:.4f}")
        print(f"F1 Macro: {val_metrics['f1_macro']:.4f}")

        # Collect true labels and pred labels for logging
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for inputs, labels in fold_val_dataloader:
                inputs = inputs.to(device)
                labels = labels.type(torch.LongTensor).to(device)
                predictions = model(inputs, device)
                preds = predictions.argmax(1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels.cpu().numpy())

        # Log metrics in required format
        log_file_path = os.path.join(fold_save_dir, f'metrics_fold_{fold+1}.log')
        detailed_metrics = log_metrics(all_labels, all_preds, log_file_path, len(all_labels))

        # Draw confusion matrix for current fold
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            val_metrics['confusion_matrix'],
            annot=True,
            fmt='d',
            cmap='Blues'
        )
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title(f'Confusion Matrix - Fold {fold+1}')
        plt.savefig(os.path.join(fold_save_dir, 'confusion_matrix.png'))
        plt.close()

        # IMPORTANT: Remove model from GPU and clean memory
        print(f"Clearing model from memory for fold {fold+1}...")
        del model, optimizer, criterion, fold_train_dataloader, fold_val_dataloader
        del fold_train_dataset, fold_val_dataset, fold_tokenizer
        torch.cuda.empty_cache()  # Free GPU memory
        import gc
        gc.collect()  # Run garbage collector

    # Calculate average results
    mean_accuracy = np.mean(cv_accuracies)
    mean_loss = np.mean(cv_losses)
    mean_f1_weighted = np.mean(cv_f1_weighted)
    mean_f1_macro = np.mean(cv_f1_macro)
    mean_epochs = np.mean(cv_epochs_trained)

    std_accuracy = np.std(cv_accuracies)
    std_loss = np.std(cv_losses)
    std_f1_weighted = np.std(cv_f1_weighted)
    std_f1_macro = np.std(cv_f1_macro)
    std_epochs = np.std(cv_epochs_trained)

    # Print summary results
    print("\n" + "=" * 50)
    print(f"Cross-Validation Results ({n_folds} folds):")
    print(f"Mean Epochs Trained: {mean_epochs:.2f} ± {std_epochs:.2f}")
    print(f"Mean Accuracy: {mean_accuracy:.4f} ± {std_accuracy:.4f}")
    print(f"Mean Loss: {mean_loss:.4f} ± {std_loss:.4f}")
    print(f"Mean F1 Weighted: {mean_f1_weighted:.4f} ± {std_f1_weighted:.4f}")
    print(f"Mean F1 Macro: {mean_f1_macro:.4f} ± {std_f1_macro:.4f}")
    print("=" * 50)

    # Plot cross validation results
    plt.figure(figsize=(15, 10))

    # Plot epochs trained
    plt.subplot(3, 2, 1)
    plt.bar(range(1, n_folds+1), cv_epochs_trained)
    plt.axhline(y=mean_epochs, color='r', linestyle='-', label=f'Mean: {mean_epochs:.2f}')
    plt.fill_between(
        [0.5, n_folds+0.5],
        [mean_epochs - std_epochs] * 2,
        [mean_epochs + std_epochs] * 2,
        alpha=0.2, color='r'
    )
    plt.xlabel('Fold')
    plt.ylabel('Epochs Trained')
    plt.title('Epochs Trained per Fold')
    plt.legend()
    plt.xticks(range(1, n_folds+1))

    # Plot accuracies
    plt.subplot(3, 2, 2)
    plt.bar(range(1, n_folds+1), cv_accuracies)
    plt.axhline(y=mean_accuracy, color='r', linestyle='-', label=f'Mean: {mean_accuracy:.4f}')
    plt.fill_between(
        [0.5, n_folds+0.5],
        [mean_accuracy - std_accuracy] * 2,
        [mean_accuracy + std_accuracy] * 2,
        alpha=0.2, color='r'
    )
    plt.xlabel('Fold')
    plt.ylabel('Accuracy')
    plt.title('Accuracy per Fold')
    plt.legend()
    plt.xticks(range(1, n_folds+1))

    # Plot losses
    plt.subplot(3, 2, 3)
    plt.bar(range(1, n_folds+1), cv_losses)
    plt.axhline(y=mean_loss, color='r', linestyle='-', label=f'Mean: {mean_loss:.4f}')
    plt.fill_between(
        [0.5, n_folds+0.5],
        [mean_loss - std_loss] * 2,
        [mean_loss + std_loss] * 2,
        alpha=0.2, color='r'
    )
    plt.xlabel('Fold')
    plt.ylabel('Loss')
    plt.title('Loss per Fold')
    plt.legend()
    plt.xticks(range(1, n_folds+1))

    # Plot F1 Weighted
    plt.subplot(3, 2, 4)
    plt.bar(range(1, n_folds+1), cv_f1_weighted)
    plt.axhline(y=mean_f1_weighted, color='r', linestyle='-', label=f'Mean: {mean_f1_weighted:.4f}')
    plt.fill_between(
        [0.5, n_folds+0.5],
        [mean_f1_weighted - std_f1_weighted] * 2,
        [mean_f1_weighted + std_f1_weighted] * 2,
        alpha=0.2, color='r'
    )
    plt.xlabel('Fold')
    plt.ylabel('F1 Weighted')
    plt.title('F1 Weighted per Fold')
    plt.legend()
    plt.xticks(range(1, n_folds+1))

    # Plot F1 Macro
    plt.subplot(3, 2, 5)
    plt.bar(range(1, n_folds+1), cv_f1_macro)
    plt.axhline(y=mean_f1_macro, color='r', linestyle='-', label=f'Mean: {mean_f1_macro:.4f}')
    plt.fill_between(
        [0.5, n_folds+0.5],
        [mean_f1_macro - std_f1_macro] * 2,
        [mean_f1_macro + std_f1_macro] * 2,
        alpha=0.2, color='r'
    )
    plt.xlabel('Fold')
    plt.ylabel('F1 Macro')
    plt.title('F1 Macro per Fold')
    plt.legend()
    plt.xticks(range(1, n_folds+1))

    plt.tight_layout()
    plt.savefig(os.path.join(save_model_dir, 'cross_validation_results.png'))
    plt.show()

    # Create summary log file for all folds
    log_file_path = os.path.join(save_model_dir, f'metrics_summary.log')
    logger = setup_logger(log_file_path)
    logger.info(f"Cross-Validation Summary ({n_folds} folds):")
    logger.info(f"Mean Epochs Trained: {mean_epochs:.2f} ± {std_epochs:.2f}")
    logger.info(f"Mean Accuracy: {mean_accuracy:.4f} ± {std_accuracy:.4f}")
    logger.info(f"Mean Loss: {mean_loss:.4f} ± {std_loss:.4f}")
    logger.info(f"Mean F1 Weighted: {mean_f1_weighted:.4f} ± {std_f1_weighted:.4f}")
    logger.info(f"Mean F1 Macro: {mean_f1_macro:.4f} ± {std_f1_macro:.4f}")

    # Create return results
    cv_results = {
        'mean_epochs': mean_epochs,
        'mean_accuracy': mean_accuracy,
        'mean_loss': mean_loss,
        'mean_f1_weighted': mean_f1_weighted,
        'mean_f1_macro': mean_f1_macro,
        'std_epochs': std_epochs,
        'std_accuracy': std_accuracy,
        'std_loss': std_loss,
        'std_f1_weighted': std_f1_weighted,
        'std_f1_macro': std_f1_macro,
        'epochs_trained': cv_epochs_trained,
        'accuracies': cv_accuracies,
        'losses': cv_losses,
        'f1_weighted': cv_f1_weighted,
        'f1_macro': cv_f1_macro,
        'metrics_per_fold': cv_metrics_per_fold
    }

    return cv_results

# # Gộp dữ liệu train và valid để thực hiện cross-validation
# combined_df = pd.concat([train_df, valid_df], ignore_index=True)

# Thực hiện cross-validation
print("Performing 5-fold cross-validation...")
cv_save_dir = os.path.join(save_path, 'cross_validation')
os.makedirs(cv_save_dir, exist_ok=True)

# Thực hiện cross-validation chỉ với tập train
cv_results = perform_cross_validation(
    train_df=train_df,  # CHỈ sử dụng tập train
    n_folds=5,
    num_epochs=10,
    batch_size=128,
    save_model_dir=cv_save_dir
)

