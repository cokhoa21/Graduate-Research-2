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

from google.colab import drive
drive.mount('/content/gdrive')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
import re
import math
import time
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

class CookieTokenizerEvaluator:
    """
    Comprehensive evaluator for comparing Keras vs Hybrid Cookie Tokenization
    """

    def __init__(self, max_length=100, chunk_size=4, max_chunks=15):
        self.max_length = max_length
        self.chunk_size = chunk_size
        self.max_chunks = max_chunks
        self.results = {}

    def load_cookie_data(self, csv_file, cookie_column='value'):
        """Load cookie data from CSV file"""
        try:
            df = pd.read_csv(csv_file)
            if cookie_column not in df.columns:
                raise ValueError(f"Column '{cookie_column}' not found. Available columns: {df.columns.tolist()}")

            # Clean and filter data
            cookies = df[cookie_column].dropna().astype(str).tolist()

            print(f"✅ Loaded {len(cookies)} cookies from {csv_file}")
            print(f"Sample cookies:")
            for i, cookie in enumerate(cookies[:5]):
                print(f"  {i+1}: {cookie[:60]}{'...' if len(cookie) > 60 else ''}")

            return cookies

        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return None

    def analyze_cookie_characteristics(self, cookies):
        """Analyze basic characteristics of cookie dataset"""
        print("\n📊 DATASET ANALYSIS:")
        print("="*50)

        # Basic stats
        lengths = [len(cookie) for cookie in cookies]
        entropies = [self.calculate_entropy(cookie) for cookie in cookies]

        stats = {
            'total_cookies': len(cookies),
            'avg_length': np.mean(lengths),
            'median_length': np.median(lengths),
            'max_length': np.max(lengths),
            'min_length': np.min(lengths),
            'avg_entropy': np.mean(entropies),
            'high_entropy_count': sum(1 for e in entropies if e > 4.0),
            'low_entropy_count': sum(1 for e in entropies if e < 2.0)
        }

        print(f"Total cookies: {stats['total_cookies']}")
        print(f"Length - Avg: {stats['avg_length']:.1f}, Median: {stats['median_length']:.1f}, Range: [{stats['min_length']}-{stats['max_length']}]")
        print(f"Entropy - Avg: {stats['avg_entropy']:.2f}")
        print(f"High entropy (>4.0): {stats['high_entropy_count']} ({stats['high_entropy_count']/len(cookies)*100:.1f}%)")
        print(f"Low entropy (<2.0): {stats['low_entropy_count']} ({stats['low_entropy_count']/len(cookies)*100:.1f}%)")

        # Cookie type classification
        type_counts = {}
        for cookie in cookies:
            cookie_type = self.classify_cookie_type(cookie)
            type_counts[cookie_type] = type_counts.get(cookie_type, 0) + 1

        print(f"\nCookie Types:")
        for cookie_type, count in sorted(type_counts.items()):
            percentage = count / len(cookies) * 100
            print(f"  {cookie_type}: {count} ({percentage:.1f}%)")

        return stats, type_counts

    def classify_cookie_type(self, cookie):
        """Classify single cookie type"""
        # Simple classification logic
        if '=' in cookie and any(char in cookie for char in ['&', ';', ' ']):
            return 'plaintext'
        elif cookie.count('.') == 2 and len(cookie) > 50:
            return 'jwt'
        elif re.match(r'^[0-9a-fA-F]+$', cookie) and len(cookie) >= 16:
            return 'hex'
        elif cookie.endswith(('=', '==')) and re.match(r'^[A-Za-z0-9+/]*={0,2}$', cookie):
            return 'base64'
        elif re.match(r'^[A-Za-z0-9+/_-]+$', cookie) and len(cookie) >= 20:
            return 'base64_nopad'
        elif '{' in cookie or '[' in cookie:
            return 'json'
        elif '%' in cookie:
            return 'urlenc'
        else:
            return 'unknown'

    def calculate_entropy(self, s):
        """Calculate Shannon entropy"""
        if not s:
            return 0
        counts = Counter(s)
        entropy = -sum(count/len(s) * math.log2(count/len(s)) for count in counts.values())
        return entropy

    def keras_tokenization(self, cookies):
        """Method 1: Pure Keras Tokenization"""
        print("\n🔸 METHOD 1: KERAS TOKENIZATION")
        print("="*40)

        start_time = time.time()

        # Basic preprocessing
        preprocessed = []
        for cookie in cookies:
            # Simple preprocessing: replace special chars with spaces
            processed = re.sub(r'[=&;:,{}%\[\]()\'\"\\]', ' ', cookie)
            processed = ' '.join(processed.split())
            preprocessed.append(processed)

        # Tokenize
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(preprocessed)
        sequences = tokenizer.texts_to_sequences(preprocessed)
        padded = pad_sequences(sequences, maxlen=self.max_length, padding='post')

        processing_time = time.time() - start_time

        # Calculate metrics
        metrics = self.calculate_metrics(tokenizer, sequences, padded, "Keras")
        metrics['processing_time'] = processing_time

        return {
            'tokenizer': tokenizer,
            'sequences': sequences,
            'padded': padded,
            'preprocessed': preprocessed,
            'metrics': metrics
        }

    def hybrid_tokenization(self, cookies):
        """Method 2: Hybrid Keras + CookieTokenizer"""
        print("\n🔹 METHOD 2: HYBRID TOKENIZATION")
        print("="*40)

        start_time = time.time()

        # Step 1: Try Keras first
        keras_result = self.try_keras_for_hybrid(cookies)

        # Step 2: Identify problematic cookies
        problematic_indices = []
        for i, cookie in enumerate(cookies):
            if self.is_problematic_cookie(cookie):
                problematic_indices.append(i)

        print(f"Identified {len(problematic_indices)} problematic cookies ({len(problematic_indices)/len(cookies)*100:.1f}%)")

        # Step 3: Process problematic cookies with CookieTokenizer
        final_texts = []
        for i, cookie in enumerate(cookies):
            if i in problematic_indices:
                processed_text = self.cookie_tokenize_single(cookie)
            else:
                # Use simple Keras preprocessing
                processed_text = re.sub(r'[=&;:,{}%\[\]()\'\"\\]', ' ', cookie)
                processed_text = ' '.join(processed_text.split())

            final_texts.append(processed_text)

        # Step 4: Final tokenization
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(final_texts)
        sequences = tokenizer.texts_to_sequences(final_texts)
        padded = pad_sequences(sequences, maxlen=self.max_length, padding='post')

        processing_time = time.time() - start_time

        # Calculate metrics
        metrics = self.calculate_metrics(tokenizer, sequences, padded, "Hybrid")
        metrics['processing_time'] = processing_time
        metrics['problematic_count'] = len(problematic_indices)
        metrics['problematic_percentage'] = len(problematic_indices) / len(cookies) * 100

        return {
            'tokenizer': tokenizer,
            'sequences': sequences,
            'padded': padded,
            'final_texts': final_texts,
            'problematic_indices': problematic_indices,
            'metrics': metrics
        }

    def try_keras_for_hybrid(self, cookies):
        """Helper: Try Keras tokenization for hybrid method"""
        preprocessed = []
        for cookie in cookies:
            processed = re.sub(r'[=&;:,{}%\[\]()\'\"\\]', ' ', cookie)
            processed = ' '.join(processed.split())
            preprocessed.append(processed)

        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(preprocessed)

        return {'tokenizer': tokenizer, 'preprocessed': preprocessed}

    def is_problematic_cookie(self, cookie):
        """Determine if cookie needs CookieTokenizer processing"""
        length = len(cookie)
        entropy = self.calculate_entropy(cookie)

        # Criteria for problematic cookies
        is_too_long = length > 100
        is_high_entropy = entropy > 4.5
        has_long_sequences = bool(re.search(r'[a-zA-Z0-9]{30,}', cookie))
        is_encoded = self.classify_cookie_type(cookie) in ['jwt', 'base64', 'base64_nopad', 'hex']

        return is_too_long or is_high_entropy or has_long_sequences or is_encoded

    def cookie_tokenize_single(self, cookie):
        """Apply CookieTokenizer logic to single cookie"""
        words = []

        # Detect type
        cookie_type = self.classify_cookie_type(cookie)
        words.append(f"type_{cookie_type}")

        # Length category
        length = len(cookie)
        if length < 20:
            len_cat = "short"
        elif length < 80:
            len_cat = "medium"
        elif length < 200:
            len_cat = "long"
        else:
            len_cat = "verylong"
        words.append(f"len_{len_cat}")

        # Entropy category
        entropy = self.calculate_entropy(cookie)
        if entropy > 4.5:
            ent_cat = "high"
        elif entropy > 3.0:
            ent_cat = "medium"
        else:
            ent_cat = "low"
        words.append(f"entropy_{ent_cat}")

        # Chunking
        if cookie_type == "jwt" and '.' in cookie:
            parts = cookie.split('.')
            for i, part in enumerate(parts[:3]):
                chunks = [part[j:j+self.chunk_size] for j in range(0, len(part), self.chunk_size)]
                limited_chunks = chunks[:5]
                words.extend([f"jwt{i}_{chunk}" for chunk in limited_chunks])
        else:
            chunks = [cookie[i:i+self.chunk_size] for i in range(0, len(cookie), self.chunk_size)]
            limited_chunks = chunks[:self.max_chunks]
            words.extend([f"chunk_{chunk}" for chunk in limited_chunks])

        return ' '.join(words)

    def calculate_metrics(self, tokenizer, sequences, padded, method_name):
        """Calculate comprehensive metrics"""

        # Vocabulary metrics
        vocab_size = len(tokenizer.word_index)
        vocab_words = list(tokenizer.word_index.keys())

        # Token length distribution
        token_lengths = [len(token) for token in vocab_words]
        giant_tokens = [token for token in vocab_words if len(token) > 50]

        # Sequence metrics
        seq_lengths = [len([x for x in seq if x != 0]) for seq in sequences]
        total_positions = len(padded) * self.max_length
        used_positions = np.count_nonzero(padded)
        sequence_density = used_positions / total_positions

        # Information content
        avg_seq_length = np.mean(seq_lengths)
        max_seq_length = np.max(seq_lengths) if seq_lengths else 0

        metrics = {
            'method': method_name,
            'vocab_size': vocab_size,
            'giant_tokens_count': len(giant_tokens),
            'giant_tokens_percentage': len(giant_tokens) / vocab_size * 100 if vocab_size > 0 else 0,
            'avg_token_length': np.mean(token_lengths) if token_lengths else 0,
            'max_token_length': max(token_lengths) if token_lengths else 0,
            'sequence_density': sequence_density,
            'avg_sequence_length': avg_seq_length,
            'max_sequence_length': max_seq_length,
            'total_parameters': vocab_size * 64,  # Assuming 64-dim embeddings
            'effective_vocab_usage': sequence_density * vocab_size
        }

        print(f"\n📊 {method_name} Metrics:")
        print(f"  Vocabulary size: {vocab_size:,}")
        print(f"  Giant tokens (>50 chars): {len(giant_tokens)} ({metrics['giant_tokens_percentage']:.1f}%)")
        print(f"  Average token length: {metrics['avg_token_length']:.1f}")
        print(f"  Sequence density: {sequence_density:.2%}")
        print(f"  Average sequence length: {avg_seq_length:.1f}")
        print(f"  Embedding parameters: {metrics['total_parameters']:,}")

        if giant_tokens:
            print(f"  Sample giant tokens: {giant_tokens[:3]}")

        return metrics

    def compare_methods(self, keras_result, hybrid_result):
        """Compare the two methods comprehensively"""
        print("\n🏆 COMPREHENSIVE COMPARISON")
        print("="*60)

        k_metrics = keras_result['metrics']
        h_metrics = hybrid_result['metrics']

        comparison = {}

        # Efficiency comparison
        print("\n📈 EFFICIENCY METRICS:")
        metrics_to_compare = [
            ('Vocabulary Size', 'vocab_size', 'lower_better'),
            ('Giant Tokens %', 'giant_tokens_percentage', 'lower_better'),
            ('Sequence Density', 'sequence_density', 'higher_better'),
            ('Avg Sequence Length', 'avg_sequence_length', 'higher_better'),
            ('Processing Time (s)', 'processing_time', 'lower_better')
        ]

        for name, key, direction in metrics_to_compare:
            k_val = k_metrics[key]
            h_val = h_metrics[key]

            if direction == 'lower_better':
                improvement = (k_val - h_val) / k_val * 100 if k_val != 0 else 0
                winner = "Hybrid" if h_val < k_val else "Keras"
            else:
                improvement = (h_val - k_val) / k_val * 100 if k_val != 0 else 0
                winner = "Hybrid" if h_val > k_val else "Keras"

            comparison[key] = {
                'keras': k_val,
                'hybrid': h_val,
                'improvement': improvement,
                'winner': winner
            }

            print(f"  {name}:")
            print(f"    Keras: {k_val:.3f}")
            print(f"    Hybrid: {h_val:.3f}")
            print(f"    Winner: {winner} ({abs(improvement):.1f}% {'better' if improvement > 0 else 'worse'})")

        # Quality assessment
        print(f"\n🎯 QUALITY ASSESSMENT:")

        # Memory efficiency
        k_memory_waste = k_metrics['giant_tokens_count'] * 64  # wasted embedding params
        h_memory_waste = h_metrics['giant_tokens_count'] * 64
        memory_saved = k_memory_waste - h_memory_waste

        print(f"  Memory waste (giant token embeddings):")
        print(f"    Keras: {k_memory_waste:,} parameters")
        print(f"    Hybrid: {h_memory_waste:,} parameters")
        print(f"    Saved: {memory_saved:,} parameters")

        # Information density
        k_info_density = k_metrics['sequence_density'] * k_metrics['avg_sequence_length']
        h_info_density = h_metrics['sequence_density'] * h_metrics['avg_sequence_length']

        print(f"  Information density:")
        print(f"    Keras: {k_info_density:.3f}")
        print(f"    Hybrid: {h_info_density:.3f}")
        print(f"    Improvement: {(h_info_density - k_info_density) / k_info_density * 100:.1f}%")

        # Overall recommendation
        print(f"\n🏅 OVERALL RECOMMENDATION:")

        hybrid_wins = sum(1 for comp in comparison.values() if comp['winner'] == 'Hybrid')
        total_metrics = len(comparison)

        if hybrid_wins >= total_metrics * 0.6:
            recommendation = "🥇 HYBRID TOKENIZER RECOMMENDED"
            reason = f"Wins {hybrid_wins}/{total_metrics} key metrics"
        else:
            recommendation = "🥈 KERAS TOKENIZER SUFFICIENT"
            reason = f"Keras wins {total_metrics - hybrid_wins}/{total_metrics} key metrics"

        print(f"  {recommendation}")
        print(f"  Reason: {reason}")

        # Specific recommendations
        print(f"\n💡 SPECIFIC RECOMMENDATIONS:")
        if h_metrics.get('problematic_percentage', 0) > 30:
            print(f"  ✅ High percentage ({h_metrics['problematic_percentage']:.1f}%) of complex cookies - Hybrid adds significant value")
        elif h_metrics.get('problematic_percentage', 0) < 10:
            print(f"  ⚠️  Low percentage ({h_metrics['problematic_percentage']:.1f}%) of complex cookies - Keras might be sufficient")

        if k_metrics['giant_tokens_count'] > 50:
            print(f"  ✅ Many giant tokens ({k_metrics['giant_tokens_count']}) in Keras - Hybrid reduces vocabulary bloat")

        if h_metrics['processing_time'] > k_metrics['processing_time'] * 2:
            print(f"  ⚠️  Hybrid is {h_metrics['processing_time']/k_metrics['processing_time']:.1f}x slower - consider performance requirements")

        return comparison

    def visualize_comparison(self, keras_result, hybrid_result):
        """Create visualization comparing the methods"""

        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        k_metrics = keras_result['metrics']
        h_metrics = hybrid_result['metrics']

        # 1. Vocabulary size comparison
        methods = ['Keras', 'Hybrid']
        vocab_sizes = [k_metrics['vocab_size'], h_metrics['vocab_size']]

        axes[0,0].bar(methods, vocab_sizes, color=['lightcoral', 'lightblue'])
        axes[0,0].set_title('Vocabulary Size')
        axes[0,0].set_ylabel('Number of Tokens')

        # 2. Sequence density comparison
        densities = [k_metrics['sequence_density'], h_metrics['sequence_density']]
        axes[0,1].bar(methods, densities, color=['lightcoral', 'lightblue'])
        axes[0,1].set_title('Sequence Density')
        axes[0,1].set_ylabel('Density (0-1)')

        # 3. Giant tokens comparison
        giant_counts = [k_metrics['giant_tokens_count'], h_metrics['giant_tokens_count']]
        axes[1,0].bar(methods, giant_counts, color=['lightcoral', 'lightblue'])
        axes[1,0].set_title('Giant Tokens (>50 chars)')
        axes[1,0].set_ylabel('Count')

        # 4. Processing time comparison
        times = [k_metrics['processing_time'], h_metrics['processing_time']]
        axes[1,1].bar(methods, times, color=['lightcoral', 'lightblue'])
        axes[1,1].set_title('Processing Time')
        axes[1,1].set_ylabel('Seconds')

        plt.tight_layout()
        plt.show()

        return fig

    def generate_report(self, csv_file, cookie_column='value'):
        """Generate comprehensive evaluation report"""

        print("🚀 COOKIE TOKENIZER EVALUATION REPORT")
        print("="*60)

        # Load data
        cookies = self.load_cookie_data(csv_file, cookie_column)
        if cookies is None:
            return None

        # Analyze dataset
        stats, type_counts = self.analyze_cookie_characteristics(cookies)

        # Method 1: Keras
        keras_result = self.keras_tokenization(cookies)

        # Method 2: Hybrid
        hybrid_result = self.hybrid_tokenization(cookies)

        # Compare methods
        comparison = self.compare_methods(keras_result, hybrid_result)

        # Generate visualizations
        self.visualize_comparison(keras_result, hybrid_result)

        # Save results
        self.results = {
            'dataset_stats': stats,
            'cookie_types': type_counts,
            'keras_result': keras_result,
            'hybrid_result': hybrid_result,
            'comparison': comparison
        }

        return self.results

# Function to read real CSV file
def read_csv_file(csv_path, cookie_column='value'):
    """
    Read and validate CSV file containing cookie data

    Args:
        csv_path (str): Path to the CSV file
        cookie_column (str): Name of column containing cookie values

    Returns:
        str: Path to the CSV file if valid, None otherwise
    """
    try:
        # Check if file exists
        import os
        if not os.path.exists(csv_path):
            print(f"❌ File not found: {csv_path}")
            return None

        # Try to read the CSV
        df = pd.read_csv(csv_path)
        print(f"✅ Successfully read CSV file: {csv_path}")
        print(f"📊 Shape: {df.shape}")
        print(f"📝 Columns: {list(df.columns)}")

        # Check if cookie column exists
        if cookie_column not in df.columns:
            print(f"❌ Column '{cookie_column}' not found in CSV")
            print(f"Available columns: {list(df.columns)}")
            return None

        # Check for non-null values in cookie column
        non_null_count = df[cookie_column].notna().sum()
        print(f"📈 Non-null values in '{cookie_column}': {non_null_count}")

        if non_null_count == 0:
            print(f"❌ No valid data found in column '{cookie_column}'")
            return None

        # Show sample data
        print(f"\n🔍 Sample data from '{cookie_column}':")
        sample_data = df[cookie_column].dropna().head()
        for i, value in enumerate(sample_data, 1):
            display_value = str(value)[:100] + "..." if len(str(value)) > 100 else str(value)
            print(f"  {i}: {display_value}")

        return csv_path

    except Exception as e:
        print(f"❌ Error reading CSV file: {e}")
        return None

# Main function to evaluate cookie tokenization methods
def evaluate_cookie_csv(csv_file, cookie_column='value'):
    """
    Main function to evaluate cookie tokenization methods

    Args:
        csv_file (str): Path to CSV file
        cookie_column (str): Name of column containing cookie values

    Returns:
        dict: Comprehensive evaluation results
    """

    # Validate CSV file first
    validated_path = read_csv_file(csv_file, cookie_column)
    if validated_path is None:
        return None

    # Run evaluation
    evaluator = CookieTokenizerEvaluator(max_length=100)
    results = evaluator.generate_report(validated_path, cookie_column)

    return results

# For testing with sample data (kept for backward compatibility)
def create_sample_csv():
    """Create sample CSV for testing"""
    sample_cookies = [
        "username=john_doe",
        "sessionId=abc123&theme=dark&lang=en",
        "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIiwibmFtZSI6IkpvaG4gRG9lIiwiaWF0IjoxNTE2MjM5MDIyfQ.SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c",
        "a7b8c9d0e1f2g3h4i5j6k7l8m9n0o1p2q3r4s5t6u7v8w9x0",
        "dGVzdF9kYXRhX2Zvcl9jb29raWVz",
        "{stamp:%27lc7wchMP6csFKbu/swr86qKyRMPpZKvu9J/n...",
        "user%20data%3Dvalue%26more%3Dinfo",
        "550e8400-e29b-41d4-a716-446655440000",
        "cart=item1,item2,item3&total=99.99&currency=USD",
        "H4sIAAAAAAAAA3N0cnZxdXP38PH1DQgJ" * 5  # Long encoded string
    ] * 10  # Repeat for larger dataset

    df = pd.DataFrame({'value': sample_cookies})
    df.to_csv('sample_cookies.csv', index=False)
    print("Created sample_cookies.csv for testing")
    return 'sample_cookies.csv'

# Example usage functions
def run_evaluation_on_file(csv_path, column_name='value'):
    """
    Run evaluation on a specific CSV file

    Args:
        csv_path (str): Path to your CSV file
        column_name (str): Name of the column containing cookie data

    Example:
        results = run_evaluation_on_file('my_cookies.csv', 'cookie_values')
    """
    print(f"🚀 Starting evaluation on file: {csv_path}")
    print(f"📊 Using column: {column_name}")

    results = evaluate_cookie_csv(csv_path, column_name)

    if results:
        print("\n✅ Evaluation completed successfully!")
        print("📊 Results summary:")
        print(f"  - Total cookies analyzed: {results['dataset_stats']['total_cookies']}")
        print(f"  - Keras vocab size: {results['keras_result']['metrics']['vocab_size']}")
        print(f"  - Hybrid vocab size: {results['hybrid_result']['metrics']['vocab_size']}")
        return results
    else:
        print("❌ Evaluation failed. Please check your CSV file and column name.")
        return None

if __name__ == "__main__":
    # Example usage - replace with your actual CSV file path
    CSV_FILE_PATH = "your_cookies.csv"  # Change this to your CSV file path
    COOKIE_COLUMN = "value"  # Change this to your cookie column name

    print("🍪 Cookie Tokenizer Evaluator")
    print("="*50)

    # Check if user wants to use sample data or real file
    use_sample = input("Use sample data? (y/n): ").lower().strip()

    if use_sample in ['y', 'yes']:
        # Create and use sample data
        print("\n📝 Creating sample data...")
        sample_file = create_sample_csv()
        results = evaluate_cookie_csv(sample_file, 'value')
    else:
        # Use real CSV file
        csv_path = input(f"Enter CSV file path (default: {CSV_FILE_PATH}): ").strip()
        if not csv_path:
            csv_path = CSV_FILE_PATH

        column_name = input(f"Enter cookie column name (default: {COOKIE_COLUMN}): ").strip()
        if not column_name:
            column_name = COOKIE_COLUMN

        print(f"\n📊 Running evaluation on: {csv_path}")
        results = run_evaluation_on_file(csv_path, column_name)

    if results:
        print("\n✅ Evaluation complete! Check the results and visualizations above.")
    else:
        print("\n❌ Evaluation failed. Please check your file and try again.")