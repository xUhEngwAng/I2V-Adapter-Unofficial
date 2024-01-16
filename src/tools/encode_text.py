import argparse
import clip
import numpy as np
import torch
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def encode_text(model, texts, batch_size):
    encoded = []
    
    for start in tqdm(range(0, len(texts), batch_size)):
        batch_text = [text for text in texts[start: start+batch_size]]
        text_tokens = clip.tokenize(batch_text, truncate=True).to(device)

        with torch.no_grad():
            text_encoding = model.encode_text(text_tokens).cpu().numpy()

        encoded.append(text_encoding)

    encoded = np.concatenate(encoded)
    return encoded

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--text_path', type=str, default='./data/captions.txt')
    parser.add_argument('--target_path', type=str, default='./data/captions.npy')

    args = parser.parse_args()

    texts = []
    with open(args.text_path, 'r') as f:
        for caption in f.readlines():
            texts.append(caption)

    model, preprocess = clip.load('ViT-B/32', device=device)
    encoded = encode_text(model, texts, args.batch_size)
    
    np.save(args.target_path, encoded)
    print(f'Done encoding a total of {len(encoded)} texts and saving the latents to {args.target_path}.')
