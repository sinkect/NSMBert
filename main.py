import os
import time
from multiprocessing import Pool
import random
import seaborn as sns
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from transformers import EncodecModel, AutoProcessor
from pydub import AudioSegment
import torch
import torch.nn as nn
import network
from tqdm import tqdm

encodec_model = EncodecModel.from_pretrained("facebook/encodec_24khz")
processor = AutoProcessor.from_pretrained("facebook/encodec_24khz", use_fast=False)
sampling_rate = processor.sampling_rate  # 24000
CLS_TOKEN = 1024
SEP_TOKEN = CLS_TOKEN + 1
PAD_TOKEN = SEP_TOKEN + 1
codebook_size = PAD_TOKEN + 1



def audiofile_to_tokens(filepath):
    audio = AudioSegment.from_file(filepath)
    samples = np.array(audio.get_array_of_samples()).astype(np.float32) / (2 ** 15)
    # stereo → mono
    if audio.channels > 1:
        samples = samples.reshape((-1, audio.channels))
        samples = np.mean(samples, axis=1)
    # resampling
    import librosa
    samples = librosa.resample(samples, orig_sr=audio.frame_rate, target_sr=sampling_rate)
    # tokenize
    inputs = processor(raw_audio=samples, sampling_rate=sampling_rate, return_tensors="pt")
    with torch.no_grad():
        encoder_outputs = encodec_model.encode(inputs["input_values"], inputs["padding_mask"])
    return encoder_outputs.audio_codes[0][0][0]  # shape: [token sequence, codebook]


def batch_tokenize_and_save(audio_path_list, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    for path in tqdm(audio_path_list):
        base = os.path.splitext(os.path.basename(path))[0]
        token_path = os.path.join(save_dir, base + ".pt")
        if os.path.exists(token_path):
            continue
        tokens = audiofile_to_tokens(path)
        torch.save(tokens, token_path)


def tokenize_and_save_one(args):
    path, save_dir = args
    base = os.path.splitext(os.path.basename(path))[0]
    token_path = os.path.join(save_dir, base + ".pt")
    if os.path.exists(token_path):
        return
    tokens = audiofile_to_tokens(path)
    torch.save(tokens, token_path)


def batch_tokenize_and_save_parallel(audio_path_list, save_dir, num_workers=4):
    os.makedirs(save_dir, exist_ok=True)
    arg_list = [(path, save_dir) for path in audio_path_list]
    with Pool(num_workers) as pool:
        list(tqdm(pool.imap_unordered(tokenize_and_save_one, arg_list), total=len(arg_list)))


def process_positive_pair(audios):
    positive_pair = []
    for i in range(0, len(audios) - 1, 2):
        first_token = audios[i]
        last_token = audios[i + 1]
        pair = [first_token, last_token]
        positive_pair.append(pair)
    return positive_pair


def process_negative_pair(audios, num):
    negative_pair_idx = set()
    negative_pair = []
    tries = 0
    L = len(audios)
    while len(negative_pair) < num and tries < num * 10:
        tries += 1
        i = np.random.randint(0, L)
        j = np.random.randint(0, L)
        # Skip if both indices refer to the same token (same song or same part)
        if i == j or (i // 2 == j // 2):
            continue
        # Prevent duplicate negative pairs based on index tuples
        if (i, j) in negative_pair_idx:
            continue
        negative_pair_idx.add((i, j))
        pair = [audios[i], audios[j]]
        negative_pair.append(pair)
    return negative_pair


class NspTokenDataset(torch.utils.data.Dataset):
    def __init__(self, pairs, labels, token_cache_dir):
        self.pairs = pairs
        self.labels = labels
        self.token_cache_dir = token_cache_dir

    def load_token(self, path):
        base = os.path.splitext(os.path.basename(path))[0]
        return torch.load(os.path.join(self.token_cache_dir, base + ".pt"), weights_only=True)

    def __getitem__(self, idx):
        file1, file2 = self.pairs[idx]
        token1 = self.load_token(file1)
        token2 = self.load_token(file2)
        tokens = torch.cat([
            torch.tensor([CLS_TOKEN]),
            token1,
            torch.tensor([SEP_TOKEN]),
            token2,
            torch.tensor([SEP_TOKEN])
        ])
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return tokens, label

    def __len__(self):
        return len(self.pairs)


# custom attention mask
def custom_collate_fn(batch):
    tokens, labels = zip(*batch)

    max_len = max(len(token) for token in tokens)

    padded_tokens = []
    attention_masks = []

    for token in tokens:
        # add padding
        pad_length = max_len - len(token)
        padded_token = torch.cat([token, torch.full((pad_length,), PAD_TOKEN, dtype=token.dtype)])
        padded_tokens.append(padded_token)

        # set attention mask (real token is 0, padding is 1)
        attention_mask = torch.cat([
            torch.zeros(len(token), dtype=torch.bool),
            torch.ones(pad_length, dtype=torch.bool)
        ])
        attention_masks.append(attention_mask)

    tokens_batch = torch.stack(padded_tokens)
    attention_masks_batch = torch.stack(attention_masks)
    labels_batch = torch.stack(list(labels))

    return tokens_batch, labels_batch, attention_masks_batch


class EarlyStopping:
    def __init__(self, patience=3, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, valid_loss):
        if self.best_loss is None:
            self.best_loss = valid_loss
        elif valid_loss < self.best_loss - self.min_delta:
            self.best_loss = valid_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True


def train(
        model,
        train_loader,
        valid_loader,
        optimizer,
        criterion=nn.BCEWithLogitsLoss(),
        num_epochs=10,
        device=torch.device("mps"),
        early_stopping_patience=3
):
    train_loss_list = []
    valid_loss_list = []
    early_stopping = EarlyStopping(patience=early_stopping_patience)
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for i, (inputs, labels, attention_masks) in enumerate(train_loader):
            inputs = inputs.to(device)
            labels = labels.to(device).float().unsqueeze(1)
            attention_masks = attention_masks.to(device)

            optimizer.zero_grad()
            output = model(inputs, attention_masks)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)
        train_loss_list.append(avg_train_loss)

        # Validation (at end of each epoch)
        model.eval()
        valid_running_loss = 0.0
        with torch.no_grad():
            for inputs, labels, attention_masks in valid_loader:
                inputs = inputs.to(device)
                labels = labels.to(device).float().unsqueeze(1)
                attention_masks = attention_masks.to(device)

                output = model(inputs, attention_masks)
                loss = criterion(output, labels)
                valid_running_loss += loss.item()
        avg_valid_loss = valid_running_loss / len(valid_loader)
        valid_loss_list.append(avg_valid_loss)

        print(f"Epoch {epoch + 1}: train loss {avg_train_loss:.4f}, valid loss {avg_valid_loss:.4f}")
        early_stopping(avg_valid_loss)
        if early_stopping.early_stop:
            print(f"Early stopping at epoch {epoch + 1} (no improvement in {early_stopping.patience} epochs)")
            break
    return train_loss_list, valid_loss_list


def evaluate(model, test_loader, device=torch.device("mps")):
    y_pred = []
    y_true = []
    model.eval()
    with torch.no_grad():
        for inputs, labels, attention_masks in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            attention_masks = attention_masks.to(device)

            output = model(inputs, attention_masks)
            y_pred.extend(output.sigmoid().cpu().round().squeeze().tolist())
            y_true.extend(labels.cpu().tolist())

    print('NSP result:')
    cm = confusion_matrix(y_true, y_pred, labels=[1, 0])
    ax = plt.subplot()
    sns.heatmap(cm, annot=True, ax=ax, cmap='Blues', fmt='d')
    ax.set_title('Confusion Matrix')
    ax.set_xlabel('Predicted Labels')
    ax.set_ylabel('True Labels')
    ax.xaxis.set_ticklabels(['Negative', 'Positive'])
    ax.yaxis.set_ticklabels(['Negative', 'Positive'])
    plt.show()
    print(f"Accuracy: {np.sum(np.array(y_true) == np.array(y_pred)) / len(y_true):.4f}")


if __name__ == "__main__":
    folder_path = "music_split"
    audio_path_list = []

    for filename in sorted(os.listdir(folder_path)):
        if filename.endswith(".mp3") or filename.endswith(".wav"):
            filepath = os.path.join(folder_path, filename)
            try:
                print(f"{filename}")
                audio_path_list.append(filepath)
            except Exception as e:
                print(f"Error processing {filename}: {e}")
    token_cache_dir = "token_cache"
    batch_tokenize_and_save_parallel(audio_path_list, token_cache_dir, num_workers=10)
    positive_pairs = process_positive_pair(audio_path_list)
    negative_pairs = process_negative_pair(audio_path_list, len(audio_path_list) * 2)

    all_pairs = positive_pairs + negative_pairs
    all_labels = [1] * len(positive_pairs) + [0] * len(negative_pairs)

    combined = list(zip(all_pairs, all_labels))

    random_seed = 42
    random.seed(random_seed)
    random.shuffle(combined)

    n_total = len(combined)
    n_train = int(n_total * 0.8)
    n_valid = int(n_total * 0.1)

    train_list = combined[:n_train]
    valid = combined[n_train:n_train + n_valid]
    test = combined[n_train + n_valid:]

    train_pairs, train_labels = zip(*train_list)
    valid_pairs, valid_labels = zip(*valid)
    test_pairs, test_labels = zip(*test)

    train_dataset = NspTokenDataset(train_pairs, train_labels, token_cache_dir)
    valid_dataset = NspTokenDataset(valid_pairs, valid_labels, token_cache_dir)
    test_dataset = NspTokenDataset(test_pairs, test_labels, token_cache_dir)

    batch_size = 16
    num_workers = 10
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                               collate_fn=custom_collate_fn, num_workers=num_workers)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=True,
                                               collate_fn=custom_collate_fn, num_workers=num_workers)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True,
                                              collate_fn=custom_collate_fn, num_workers=num_workers)

    device = torch.device("mps")
    model = network.TransitionBERT(codebook_size=codebook_size, embed_dim=512, num_layers=4, num_heads=4,
                                   max_length=2258)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    train_loss_list, valid_loss_list = train(model=model, optimizer=optimizer, train_loader=train_loader,
                                             valid_loader=valid_loader, num_epochs=10, device=device)

    epochs = range(1, 11)
    plt.plot(epochs, train_loss_list, label="train loss")
    plt.plot(epochs, valid_loss_list, label="valid loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.show()
    evaluate(model, test_loader, device)
    torch.save(model.state_dict(), "nsp_checkpoint.pth")
