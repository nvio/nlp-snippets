import spacy
import torch
import torch.nn as nn
from torch.optim import Adam

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from torchtext.datasets import Multi30k
from torchtext.data import Field, BucketIterator


class MultiHeadAttention(nn.Module):
    def __init__(self, dim_model, n_heads, dropout):
        super().__init__()
        if not dim_model % n_heads == 0:
            raise ValueError

        self.projection_query = nn.Linear(dim_model, dim_model)  # projection matrix WQ
        self.projection_key = nn.Linear(dim_model, dim_model)    # projection matrix WK
        self.projection_value = nn.Linear(dim_model, dim_model)  # projection matrix WV

        self.dropout = nn.Dropout(dropout)

        self.projection_out = nn.Linear(dim_model, dim_model)

        self.n_heads = n_heads
        self.dim_model = dim_model
        self.dim_head = dim_model // n_heads
        self.scale = torch.sqrt(self.dim_head)


    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask=None):
        """MultiHeadAttention forward pass

        Head projections are applied simultaneously with a single linear layer (matrix multiplication)

        Args:
            query (torch.Tensor): (bath_size, query_len, dim_model)
            key (torch.Tensor): (bath_size, key_len, dim_model)
            value (torch.Tensor): (bath_size, key_len, dim_model)
            mask (torch.Tensor, optional): [description]. Defaults to None.

        """
        query = self.projection_query(query)
        key = self.projection_key(key)
        value = self.projection_value(value)

        query, key, value = self.split_parallel_heads_input(query, key, value)

        similarity = torch.matmul(query, key.permute(0, 1, 3, 2)) / self.scale           #  (bath_size, n_heads, query_len, key_len)
        if mask:
            similarity.masked_fill_(mask == 0, -10e12)
        attention_weights = self.dropout(torch.softmax(similarity, dim=3))               #  (bath_size, n_heads, query_len, key_len)

        output = torch.matmul(attention_weights, value)                                  #  (batch_size, n_heads, query_len, head_size) 
        output = self.merge_parallel_heads_output(output)                                #  (batch_size, query_len, dim_model)
        output = self.projection_out(output) 

        return output                                       

        
    def split_parallel_heads_input(self, query, key, value):
        """Reshapes query, key and value in order to feed n_heads parallel attention heads 
        where dim_model = head_size * n_heads

        Args:
            query (torch.Tensor): (bath_size, query_len, dim_model)
            key (torch.Tensor): (bath_size, key_len, dim_model)
            value (torch.Tensor): (bath_size, key_len, dim_model)

        Returns:
            query (torch.Tensor): (bath_size, n_heads, query_len, head_size)
            key (torch.Tensor): (bath_size, n_heads, key_len, head_size)
            value (torch.Tensor): (bath_size, n_heads, key_len, head_size)
        """
        batch_size = query.shape[0]
        query_len = query.shape[1]
        key_len = key.shape[1]
        query = query.view(batch_size, query_len, self.n_heads, self.dim_head).permute(0, 2, 1, 3)
        key = key.view(batch_size, key_len, self.n_heads, self.dim_head).permute(0, 2, 1, 3)
        value = value.view(batch_size, key_len, self.n_heads, self.dim_head).permute(0, 2, 1, 3)

        return query, key, value


    def merge_parallel_heads_output(self, output):
        """Merge outputs from n_heads into a single one

        Args:
            output (torch.Tensor): (batch_size, n_heads, query_len, head_size)

        Returns:
            output (torch.Tensor): (batch_size, query_len, n_heads*head_size)
        """
        batch_size = output.shape[0]
        query_len = output.shape[2]
        output = output.permute(0, 2, 1, 3)
        return output.view(batch_size, query_len, self.dim_model)

class FeedForward(nn.Module):
    """Position-Wise Feed Forward block. 
    """
    def __init__(self, dim_model, dim_ff, dropout):
        super().__init__()
        self.ff = nn.Sequential(nn.Linear(dim_model, dim_ff),
                                nn.ReLU(),
                                nn.Dropout(dropout),
                                nn.Linear(dim_ff, dim_model),
                                nn.Dropout(dropout)
                                )

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): (batch_size, query_len, dim_model)
        """
        return self.ff(x)

class EncoderLayer(nn.Module):
    """Single Encoder block. It consits of a MutliHeadAtenttion layer followed by
    a Position-Wise Feed Forward.
    """
    def __init__(self, dim_model, n_heads, dim_ff, dropout):
        super().__init__()
        self.self_attention = MultiHeadAttention(dim_model, n_heads, dropout)
        self.layer_norm1 = nn.LayerNorm(dim_model)
        self.feed_forward = FeedForward(dim_model, dim_ff, dropout)
        self.layer_norm2 = nn.LayerNorm(dim_ff)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, mask=None):
        """
        Args:
            input (torch.Tensor): (batch_size, src_len, dim_model)
            mask (torch.Tensor, optional): Used to ignore <pad> tokens. 1 to keep value, 0 to ignore. Defaults to None.
        
        Returns:
            out (torch.Tensor): (batch_size, src_len, dim_model)
        """
        out = self.dropout(self.self_attention(input, input, input, mask)) + input
        out = self.layer_norm1(out)
        out = self.feed_forward(out) + out
        out = self.layer_norm2(out)
        return out


class Encoder(nn.Module):
    """Transformer Encoder. It is a stack of n_layers of EncoderLayer
    """
    def __init__(self, src_vocab_size, max_sequence_len, n_layers, dim_model, n_heads, dim_ff, dropout):
        super().__init__()
        self.token_embedding = nn.Embedding(src_vocab_size, dim_model)
        self.positional_embedding = nn.Embedding(max_sequence_len, dim_model)
        self.layers = nn.ModuleList([EncoderLayer(dim_model, n_heads, dim_ff, dropout) 
                                    for _ in range(n_layers)])

    def forward(self, src: torch.Tensor, mask=None):
        """
        Args:
            src (torch.Tensor): (batch_size, src_len). Source sequence.
            mask (torch.Tensor, optional): Used to ignore <pad> tokens. 1 to keep value, 0 to ignore. Defaults to None.

        Returns:
            encoded_src (torch.Tensor): (batch_size, src_len, dim_model)
        """
        positions = self.get_positions(src)
        embedding = self.token_embedding(src) + self.positional_embedding(positions)

        encoded_src = embedding
        for layer in self.layers:
            encoded_src = layer(encoded_src, mask)

        return encoded_src

    def get_positions(self, seq):
        batch_size, seq_len = seq.shape[:2]
        return torch.arange(0, seq_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)


class DecoderLayer(nn.Module):
    def __init__(self, dim_model, n_heads, dim_ff, dropout):
        super().__init__()
        self.self_attention = MultiHeadAttention(dim_model, n_heads, dropout)
        self.layer_norm1 = nn.LayerNorm(dim_model)
        self.encoder_attention = MultiHeadAttention(dim_model, n_heads, dropout)
        self.layer_norm2 = nn.LayerNorm(dim_model)
        self.feed_forward = FeedForward(dim_model, dim_ff)
        self.layer_norm3 = nn.LayerNorm(dim_model)

    def forward(self, encoded_src, input, trg_mask=None, src_mask=None):
        """
        Args:
            encoded_src (torch.Tensor): (batch_size, src_len, dim_model). Encoder output.
            input (torch.Tensor): (batch_size, trg_len). Target sequence.
            trg_mask (torch.Tensor, optional): Used to ignore <pad> tokens and to prevent decoder from looking ahead in trg sequence. 
                                        1 to keep value, 0 to ignore. Defaults to None.
            src_mask (torch.Tensor, optional): Used to ignore <pad> tokens. 1 to keep value, 0 to ignore. Defaults to None.

        Returns:
            out (torch.Tensor):  (batch_size, trg_len, dim_model)
        """

        out = self.self_attention(input, input, input, trg_mask) + input
        out = self.layer_norm1(out)
        out = self.encoder_attention(out, encoded_src, encoded_src, src_mask) + out
        out = self.layer_norm2(out)
        out = self.feed_forward(out) + out
        out = self.layer_norm3(out)

        return out


class Decoder(nn.Module):
    def __init__(self, trg_vocab_size, embedding_size, max_sequence_len, n_layers, dim_model, n_heads, dim_ff, dropout):
        super().__init__()
        self.token_embedding = nn.Embedding(trg_vocab_size, embedding_size)
        self.positional_embedding = nn.Embedding(max_sequence_len, embedding_size)
        self.layers = nn.ModuleList([DecoderLayer(dim_model, n_heads, dim_ff, dropout) 
                                    for _ in range(n_layers)])
        self.linear = nn.Linear(dim_model, trg_vocab_size)

    def forward(self, trg, encoded_src, trg_mask, src_mask):
        """
        Args:
            trg (torch.Tensor): (batch_size, trg_len)
            encoded_src (torch.Tensor): [description]
            trg_mask (torch.Tensor): [description]
            src_mask (torch.Tensor): [description]

        Returns:
            prbabilities: (batch_size, trg_len, trg_vocab_size). Next token probability for each token in trg_len
        """
        positions = self.get_positions(trg)
        embedding = self.token_embedding(trg) + self.positional_embedding(positions)

        output = embedding
        for layer in self.layers:
            output = layer(encoded_src, output, trg_mask, src_mask)

        output = self.linear(output)                                        
        probabilities = torch.softmax(output, dim=2)
        return probabilities

    def get_positions(self, seq):
        batch_size, seq_len = seq.shape[:2]
        return torch.arange(0, seq_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)


class Transformer(pl.LightningModule):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

        TRG_PAD_IDX = TRG.vocab.stoi[TRG.pad_token]
        self.loss_fn = nn.CrossEntropyLoss(ignore_index = TRG_PAD_IDX)

    def forward(self, src, trg):
        encoded_src = self.encoder(src)
        probabilities = self.decoder(trg, encoded_src, trg_mask, src_mask)
        return probabilities

    def training_step(self, batch, batch_idx):
        src = batch.src
        trg = batch.trg

        probabilities = self(src, trg)

        # TODO: ponerlo bien
        loss = self.loss_fn(probabilities, trg)
        self.log("train_loss", loss)
        return loss


    def configure_optimizers(self):
        return Adam(self.parameters())


def tokenize(text, spacy_lang):
    """
    Tokenizes text from a string into a list of strings
    """
    return [tok.text for tok in spacy_lang.tokenizer(text)]



if __name__ == "__main__":
    pl.seed_everything(0)

    # Data preparation
    #-------------------------------------------------------------------------------------
    # 1) Load language models. First we need to download them. 
    #       python -m spacy download en_core_web_sm
    #       python -m spacy download de_core_news_sm
    #
    # More info:    https://spacy.io/usage/spacy-101
    #               https://realpython.com/natural-language-processing-spacy-python/

    spacy_de = spacy.load('de_core_news_sm')
    spacy_en = spacy.load('en_core_web_sm')

    # 3) Create Fields. They handle the way data is processed

    SRC = Field(tokenize= lambda text: tokenize(text, spacy_lang=spacy_de), 
                init_token='<sos>', 
                eos_token='<eos>', 
                lower=True,
                batch_first=True)

    TRG = Field(tokenize= lambda text: tokenize(text, spacy_lang=spacy_en),
                init_token='<sos>', 
                eos_token='<eos>', 
                lower=True,
                batch_first=True)

    # Useful functions
    #       TRG.vocab.stoi["sos"]
    #       TRG.vocab.itos[1]


    # 3) Build Vocabulary
    #       Only keep words that appear at least 2 times

    train_data, valid_data, test_data = Multi30k.splits(root=r"D:\projects\nlp\data", exts = ('.de', '.en'), fields = (SRC, TRG))
    SRC.build_vocab(train_data, min_freq = 2)
    TRG.build_vocab(train_data, min_freq = 2)

    print(f"Number of training examples: {len(train_data.examples)}")
    print(f"Number of validation examples: {len(valid_data.examples)}")
    print(f"Number of testing examples: {len(test_data.examples)}")

    print(f"Unique tokens in source (de) vocabulary: {len(SRC.vocab)}")
    print(f"Unique tokens in target (en) vocabulary: {len(TRG.vocab)}")


    # 4) Create Dataloaders

    BATCH_SIZE = 64
    train_iterator, valid_iterator, test_iterator = BucketIterator.splits((train_data, valid_data, test_data), 
                                                                           batch_size=BATCH_SIZE)

    for e in train_iterator:
        print(e.src)


