# src/models/encoder.py
import torch
from transformers import AutoTokenizer, AutoModel
from adapters import AutoAdapterModel
from typing import List, Union
import numpy as np
from tqdm import tqdm

class TextEncoder:
    _instance = None
    _initialized = False

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(TextEncoder, cls).__new__(cls)
        return cls._instance

    def __init__(
        self,
        # model_name: str = "allenai/specter",  # Scientific paper-specific BERT model
        model_name: str = "allenai/specter2_base",  # Scientific paper-specific BERT model
        device: str = None,
        max_length: int = 512
    ):
        
        # Only initialize once
        if self._initialized:
            return
        
        print("Initializing TextEncoder...")
        # Set device (GPU if available)
        # self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = 'mps' # For Mac devices
        print(f"Using device: {self.device}")
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model = AutoAdapterModel.from_pretrained("allenai/specter2_base").to(self.device)
        # adapter_name = self.model.load_adapter("allenai/specter2", source="hf", set_active=True).to(self.device)

        self.max_length = max_length
        
        # Set model to evaluation mode
        self.model.eval()

        self._initialized = True
        print("TextEncoder initialization complete!")

    def _mean_pooling(self, model_output, attention_mask):
        """
        Mean pooling of token embeddings
        """
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    @torch.no_grad()
    def encode(self, texts: Union[str, List[str]], batch_size: int = 8) -> np.ndarray:
        """
        Generate embeddings for input texts
        """
        # Ensure texts is a list
        if isinstance(texts, str):
            texts = [texts]

        all_embeddings = []
        
        # Process in batches
        for i in tqdm(range(0, len(texts), batch_size)):
            batch_texts = texts[i:i + batch_size]
            
            # Tokenize
            encoded_input = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors='pt'
            ).to(self.device)

            # Generate embeddings
            model_output = self.model(**encoded_input)
            
            # Pool embeddings
            embeddings = self._mean_pooling(model_output, encoded_input['attention_mask'])
            
            # Move to CPU and convert to numpy
            embeddings = embeddings.cpu().numpy()
            all_embeddings.append(embeddings)

        # Concatenate all embeddings
        all_embeddings = np.vstack(all_embeddings)
        print(f"Final embeddings shape: {all_embeddings.shape}")
        return all_embeddings

    def compute_similarity(self, query_embedding: np.ndarray, paper_embeddings: np.ndarray) -> np.ndarray:
        """
        Compute cosine similarity between query and paper embeddings
        """
        # print(f"Query embedding shape in similarity: {query_embedding.shape}")
        # print(f"Paper embeddings shape in similarity: {paper_embeddings.shape}")

        query_norm = query_embedding / np.linalg.norm(query_embedding)
        print(f"Query norm shape: {query_norm.shape}")

        paper_norms = paper_embeddings / np.linalg.norm(paper_embeddings)
        print(f"Paper norms shape: {paper_norms.shape}")
        
        # Compute cosine similarity
        similarities = np.dot(paper_norms, query_norm)

        return similarities

    def encode_paper(self, title: str, abstract: str) -> np.ndarray:
        """
        Generate embedding for a paper using title and abstract
        """
        # Combine title and abstract with [SEP] token
        text = f"{title} [SEP] {abstract}"
        return self.encode(text).squeeze()  # Remove batch dimension

if __name__ == "__main__":
    # Test the encoder
    encoder = TextEncoder()
    
    # Test single text
    test_text = "Deep learning in natural language processing"
    embedding = encoder.encode(test_text)
    print(f"Single embedding shape: {embedding.shape}")
    
    # Test batch of texts
    test_texts = [
        "Machine learning applications",
        "Neural networks in computer vision",
        "Natural language processing advances"
    ]
    embeddings = encoder.encode(test_texts)
    print(f"Batch embeddings shape: {embeddings.shape}")