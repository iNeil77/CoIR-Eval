# Invocation example:
# python coir.py --model_name "berg-embed/qwen3_Base_Step250-train-medium" --device "cuda:0" --padding_side "left" --batch_size 16 --num_batches_per_memory_clear 5

import argparse
import gc
import logging
import torch
from coir.data_loader import get_tasks
from coir.evaluation import COIR
from coir.models import YourCustomDEModel
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModel


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class YourCustomDEModel:
    def __init__(
            self, 
            model_name="Qwen/Qwen3-Embedding-4B", 
            device="cuda:0", 
            padding_side='left', 
            num_batches_per_memory_clear=5
        ):
        self.device = device
        self.num_batches_per_memory_clear = num_batches_per_memory_clear
        self.model_name = model_name
        self.model = AutoModel.from_pretrained(
            self.model_name, 
            trust_remote_code=True
        ).to(self.device).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, 
            trust_remote_code=True, 
            padding_side=padding_side
        )
        self.tokenizer.add_eos_token = False

    def _clear_gpu_memory(self):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    def _last_token_pooling(
            self, 
            last_hidden_states, 
            attention_mask
        ):
        left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
        if left_padding:
            return last_hidden_states[:, -1]
        else:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden_states.shape[0]
            return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

    @torch.inference_mode()
    def _encode_texts(
        self, 
        task_name, 
        texts, 
        batch_size, 
        max_length, 
        **kwargs
    ):
        all_emb = []
        for i in tqdm(range(0, len(texts), batch_size), desc="Encoding batches", unit="batch"):
            batch = texts[i:i+batch_size]
            inputs = self.tokenizer(
                batch, 
                padding=True, 
                truncation=True, 
                max_length=max_length, 
                return_tensors="pt"
            ).to(self.device)
            outputs = self.model(**inputs)
            emb = self._last_token_pooling(
                outputs.last_hidden_state, 
                inputs["attention_mask"]
            )
            emb = torch.nn.functional.normalize(
                emb, 
                p=2, 
                dim=1
            )
            all_emb.append(emb.cpu())
            del inputs, outputs, emb
            if (i // batch_size + 1) % self.num_batches_per_memory_clear == 0:
                self._clear_gpu_memory()

        self._clear_gpu_memory()
        return torch.cat(
            all_emb, 
            dim=0
        )

    def encode_queries(
            self, 
            task_name, 
            queries, 
            instruction="Retrieve relevant passages for QA", 
            batch_size=2, 
            max_length=4096, 
            **kwargs
        ):
        queries = [f"Context:\n{doc['context']}\n\nQuery:\n{doc['text']}" if ((doc["context"] is not None) and (doc["context"]!="")) else doc["text"] for doc in queries]
        prompts = [self.tokenizer.apply_chat_template([{"role": "user", "content": instruction}, {"role": "assistant", "content": q}], tokenize=False) for q in queries]
        return self._encode_texts(
            prompts, 
            batch_size, 
            max_length, 
            **kwargs
        )

    def encode_corpus(
            self, 
            corpus, 
            batch_size=2, 
            max_length=4096, 
            **kwargs
        ):
        all_texts = [doc["title"] + "\n\n" + doc["text"] if ((doc["title"] is not None) and (doc["title"]!="")) else doc["text"] for doc in corpus]
        return self._encode_texts(
            all_texts, 
            batch_size, 
            max_length, 
            **kwargs
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model_name', 
        default="Qwen/Qwen3-Embedding-4B", 
        type=str, 
        help='Model name or path'
    )    
    parser.add_argument(
        '--device', 
        type=str, 
        default='cuda:0', 
        help='Device to run the model on'
    )
    parser.add_argument(
        '--padding_side', 
        type=str, 
        default='left', 
        help='Padding side for tokenizer'
    )
    parser.add_argument(
        '--batch_size', 
        type=int, 
        default=16, 
        help='Batch size for evaluation'
    )
    parser.add_argument(
        '--num_batches_per_memory_clear', 
        type=int, 
        default=5, 
        help='Number of batches to process before clearing GPU memory'
    )
    args = parser.parse_args()

    model = YourCustomDEModel(
        model_name=args.model_name,
        device=args.device,
        padding_side=args.padding_side,
        num_batches_per_memory_clear=args.num_batches_per_memory_clear
    )
    tasks = get_tasks(
        tasks=[
            "codetrans-dl",
            # "stackoverflow-qa",
            # "apps",
            # "codefeedback-mt",
            # "codefeedback-st",
            # "codetrans-contest",
            # "synthetic-text2sql",
            # "cosqa",
            # "codesearchnet",
            # "codesearchnet-ccr"
        ]
    )
    evaluation = COIR(
        tasks=tasks,
        batch_size=args.batch_size,
    )
    results = evaluation.run(
        model, 
        output_folder=f"results/{model.model_name}"
    )
    print(results)
    

if __name__ == "__main__":
    main()
