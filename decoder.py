"""
decoder.py
LSTM decoder for the image captioning project.

The DecoderRNN takes the image feature vector produced by the EncoderCNN and
generates a caption word by word.

Idea (from "Show and Tell", Vinyals et al. 2015):
  * Feed the image feature vector to the LSTM as if it were the first word.
  * Then feed the real caption words one by one (teacher forcing).
  * At every step the LSTM predicts the NEXT word.

Layers:
  * embedding : turns word indices into EMBED_SIZE vectors (same size as the
                image vector, so both can be fed to the LSTM the same way).
  * lstm      : reads the sequence and keeps a hidden state (its "memory").
  * linear    : maps each LSTM output to a score for every word in the vocab.

Training alignment (forward):
    input :  [ image, <start>, w1, w2, ..., wK ]      (the <end> token is dropped)
    target:  [ <start>, w1, w2, ..., wK, <end> ]
So the model learns: from the image predict <start>, from <start> predict w1,
and so on until it predicts <end>. Input and target therefore have the same
length T, which makes the loss easy to compute.

Run directly for a quick shape sanity check (no dataset / download needed):
    python decoder.py
"""

import torch
import torch.nn as nn

from config import EMBED_SIZE, HIDDEN_SIZE, NUM_LAYERS, DROPOUT


class DecoderRNN(nn.Module):
    """Generate a caption from an image feature vector."""

    def __init__(self, vocab_size: int, embed_size: int = EMBED_SIZE,
                 hidden_size: int = HIDDEN_SIZE, num_layers: int = NUM_LAYERS,
                 dropout: float = DROPOUT):
        super().__init__()

        # Turn each word index into an EMBED_SIZE vector. This is the same size
        # as the image vector, so both can enter the LSTM through one path.
        self.embedding = nn.Embedding(vocab_size, embed_size)

        # The recurrent core. batch_first=True means tensors are shaped
        # [batch, time, features], which is the most readable layout.
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers,
                            batch_first=True)

        # Map every LSTM output to a score for each word in the vocabulary.
        self.linear = nn.Linear(hidden_size, vocab_size)

        # Dropout on the word embeddings to reduce overfitting.
        self.dropout = nn.Dropout(dropout)

    def forward(self, features, captions):
        """Teacher-forced forward pass used during training.

        features : FloatTensor [B, embed_size]   (from the encoder)
        captions : LongTensor  [B, T]            (with <start> ... <end>)
        returns  : FloatTensor [B, T, vocab_size] (a word score at each step)
        """
        # Embed every caption word EXCEPT the last one (<end>): the <end> token
        # is only ever a target to predict, never an input to feed forward.
        embeddings = self.dropout(self.embedding(captions[:, :-1]))  # [B, T-1, E]

        # Put the image vector in front as the first "word" of the sequence.
        features = features.unsqueeze(1)                  # [B, 1, E]
        inputs = torch.cat((features, embeddings), dim=1)  # [B, T, E]

        hiddens, _ = self.lstm(inputs)     # [B, T, hidden_size]
        outputs = self.linear(hiddens)     # [B, T, vocab_size]
        return outputs

    @torch.no_grad()
    def sample(self, features, max_length: int = 20):
        """Greedy-decode a caption from image features (used at inference time).

        At each step the most likely word is picked and fed back in as the next
        input. No teacher forcing here, since we have no ground-truth caption.

        features : FloatTensor [B, embed_size]
        returns  : LongTensor  [B, max_length]  (token indices)

        The returned sequence usually starts with <start> and contains <end>
        somewhere; the caller trims it (e.g. via Vocabulary.decode, which skips
        the special tokens).
        """
        states = None
        inputs = features.unsqueeze(1)  # [B, 1, embed_size]
        sampled_indices = []

        for _ in range(max_length):
            hiddens, states = self.lstm(inputs, states)  # [B, 1, hidden_size]
            scores = self.linear(hiddens.squeeze(1))     # [B, vocab_size]
            predicted = scores.argmax(dim=1)             # [B]
            sampled_indices.append(predicted)

            # Feed the predicted word back in as the next input.
            inputs = self.embedding(predicted).unsqueeze(1)  # [B, 1, embed_size]

        return torch.stack(sampled_indices, dim=1)  # [B, max_length]


if __name__ == "__main__":
    # Quick sanity check with random feature vectors and fake captions.
    vocab_size = 100
    batch_size = 4
    caption_length = 12

    decoder = DecoderRNN(vocab_size=vocab_size)
    decoder.eval()

    features = torch.randn(batch_size, EMBED_SIZE)
    captions = torch.randint(0, vocab_size, (batch_size, caption_length))

    outputs = decoder(features, captions)
    print("Feature shape :", tuple(features.shape))
    print("Caption shape :", tuple(captions.shape))
    print("Output shape  :", tuple(outputs.shape),
          f"(expected ({batch_size}, {caption_length}, {vocab_size}))")

    generated = decoder.sample(features, max_length=15)
    print("Sampled shape :", tuple(generated.shape),
          f"(expected ({batch_size}, 15))")
