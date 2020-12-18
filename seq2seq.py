import spacy
import torch
import torch.nn as nn
from torch.optim import Adam

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from torchtext.datasets import Multi30k
from torchtext.data import Field, BucketIterator


def tokenize(text, spacy_lang):
    """
    Tokenizes text from a string into a list of strings
    """
    return [tok.text for tok in spacy_lang.tokenizer(text)]


def init_weights(m):
    for name, param in m.named_parameters():
        if 'weight' in name:
            nn.init.normal_(param.data, mean=0, std=0.01)
        else:
            nn.init.constant_(param.data, 0)


class Encoder(nn.Module):
    def __init__(self, src_vocab_size, embedding_size, encoder_hidden_size, decoder_hidden_size, dropout=0.5):
        super().__init__()
        self.embedding = nn.Embedding(src_vocab_size, embedding_size)
        self.gru = nn.GRU(embedding_size, encoder_hidden_size, bidirectional=True)
        self.fc = nn.Linear(2*encoder_hidden_size, decoder_hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src: torch.Tensor):
        embedding = self.embedding(src)                                     #  (src_len, batch_size, embedding_size)
        encoder_outputs, encoder_hidden = self.gru(embedding)               #  (src_len, batch_size, 2*encoder_hidden_size), 
                                                                            #  (2*num_layers(=1), batch_size, encoder_hidden_size)

        # outputs: the stacked forward and backward hidden states h_t for every token in the source sequence (from last layer)
        # hidden:  hidden states for t = seq_len, for all layers and directions

        encoder_hidden = torch.cat((encoder_hidden[-2, :, :], encoder_hidden[-1, :, :]), dim=1)     #  (batch_size, 2*decoder_hidden_size)
        encoder_hidden = torch.tanh(self.fc(encoder_hidden)).unsqueeze(0)   #  (1, batch_size, decoder_hidden_size)

        # hidden: initial hidden state in the decoder
        return encoder_outputs, encoder_hidden


class Attention(nn.Module):
    def __init__(self, encoder_hidden_size: int, decoder_hidden_size: int):
        super().__init__()
        self.att = nn.Linear(decoder_hidden_size + 2*encoder_hidden_size, decoder_hidden_size)
        self.v = nn.Linear(decoder_hidden_size, 1, bias=False)


    def forward(self, encoder_outputs: torch.Tensor, decoder_hidden: torch.Tensor):
        src_len = encoder_outputs.shape[0]

        decoder_hidden = decoder_hidden.repeat(src_len, 1, 1)               # (src_len, batch_size, decoder_hidden_size)
        att_input = torch.cat((encoder_outputs, decoder_hidden), dim=2)     # (src_len, batch_size, decoder_hidden_size+2*encoder_hidden_size)

        # nn Linear. 
        #       More info: https://stackoverflow.com/questions/54444630/application-of-nn-linear-layer-in-pytorch-on-additional-dimentions

        att_input = att_input.permute(1, 0, 2)                              # (batch_size, src_len, decoder_hidden_size+2*encoder_hidden_size)

        energy = torch.tanh(self.att(att_input))                            # (batch_size, src_len, decoder_hidden_size)
        attention = self.v(energy).squeeze(2)                               # (batch_size, src_len)
        return torch.softmax(attention, dim=1)


class Decoder(nn.Module):
    def __init__(self, trg_vocab_size, embedding_size, encoder_hidden_size, decoder_hidden_size, dropout=0.5):
        super().__init__()
        self.trg_vocab_size = trg_vocab_size
        self.embedding = nn.Embedding(trg_vocab_size, embedding_size)
        self.dropout = nn.Dropout(dropout)
        self.attention = Attention(encoder_hidden_size, decoder_hidden_size)
        self.gru = nn.GRU(2*encoder_hidden_size + embedding_size, decoder_hidden_size)
        self.fc = nn.Linear(2*encoder_hidden_size + decoder_hidden_size + embedding_size, trg_vocab_size)

    def forward(self, input: torch.Tensor, encoder_outputs: torch.Tensor, decoder_hidden: torch.Tensor):
        """Decoder processes one token (input) at a time
        """

        embedding = self.dropout(self.embedding(input))                             # (1, batch_size, embedding_size)
        context = self.get_context_with_attention(encoder_outputs, decoder_hidden)  # (1, batch_size, 2*encoder_hidden_size)

        gru_input = torch.cat((embedding, context), dim=2)                   # (1, batch_size, 2*encoder_hidden_size + embedding_size)
        decoder_outputs, decoder_hidden = self.gru(gru_input, decoder_hidden)
        
        fc_input = torch.cat((decoder_outputs, context, embedding), dim=2).squeeze(0)   
        # fc input (batch_size, 2*encoder_hidden_size + decoder_hidden_size + embedding_size)
        decoder_outputs = self.fc(fc_input).unsqueeze(0)                            # (1, batch_size, trg_vocab_size)

        return decoder_outputs, decoder_hidden


    def get_context_with_attention(self, encoder_outputs, decoder_hidden):

        attention_weights = self.attention(encoder_outputs, decoder_hidden) # (batch_size, src_len)
        attention_weights = attention_weights.unsqueeze(1)                  # (batch_size, 1, src_len)
        encoder_outputs = encoder_outputs.permute(1, 0, 2)                  # (batch_size, src_len, 2*encoder_hidden_size)
        context = attention_weights.bmm(encoder_outputs)                    # (batch_size, 1, 2*encoder_hidden_size)
        context = context.permute(1, 0, 2)                                  # (1, batch_size, 2*encoder_hidden_size)
        return context


class Seq2Seq(pl.LightningModule):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        
        TRG_PAD_IDX = TRG.vocab.stoi[TRG.pad_token]
        self.loss_fn = nn.CrossEntropyLoss(ignore_index = TRG_PAD_IDX)

    def forward(self, src, trg, tearcher_forcing_ratio=0.5):
        """Used for training, as it takes trg sequence as parameter
        """

        # src = [src len, batch size]
        # trg = [trg len, batch size]
        
        trg_len, batch_size = trg.shape
        trg_vocab_size = self.decoder.trg_vocab_size
        encoder_outputs, encoder_hidden = self.encoder(src)

        decoder_outputs = torch.zeros((trg_len, batch_size, trg_vocab_size))

        input = trg[0, :].unsqueeze(0)
        decoder_hidden = encoder_hidden
        for t in range(1, trg_len):
            decoder_output, decoder_hidden = self.decoder(input, encoder_outputs, decoder_hidden)

            use_teacher_forcing = self.use_teacher_forcing(tearcher_forcing_ratio)
            input = trg[t, :].unsqueeze(0) if use_teacher_forcing else decoder_output.argmax(dim=2)

            decoder_outputs[t] = decoder_output

        return decoder_outputs

    @staticmethod
    def use_teacher_forcing(tearcher_forcing_ratio):
        return torch.rand(1).item() < tearcher_forcing_ratio


    def configure_optimizers(self):
        return Adam(self.parameters())

    
    def training_step(self, batch, batch_idx):
        src = batch.src.to(self.device)
        trg = batch.trg.to(self.device)

        outputs = self(src, trg).to(self.device)

        # Ignore <sos> token
        loss = self.loss_fn(outputs[1:].view(-1, outputs.shape[2]), trg[1:].view(-1))
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        src = batch.src.to(self.device)
        trg = batch.trg.to(self.device)

        outputs = self(src, trg, tearcher_forcing_ratio=0.).to(self.device)

        # Ignore <sos> token
        loss = self.loss_fn(outputs[1:].view(-1, outputs.shape[2]), trg[1:].view(-1))
        self.log("val_loss", loss)
        return loss
        




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
                lower=True)

    TRG = Field(tokenize= lambda text: tokenize(text, spacy_lang=spacy_en),
                init_token='<sos>', 
                eos_token='<eos>', 
                lower=True)


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

    # Model definition
    #-------------------------------------------------------------------------------------
    SRC_VOCAB_SIZE = len(SRC.vocab)
    TRG_VOCAB_SIZE = len(TRG.vocab)
    ENC_EMB_DIM = 128
    DEC_EMB_DIM = 128
    ENC_HID_DIM = 256
    DEC_HID_DIM = 256
    ENC_DROPOUT = 0.5
    DEC_DROPOUT = 0.5

    encoder = Encoder(SRC_VOCAB_SIZE, ENC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, ENC_DROPOUT)
    decoder = Decoder(TRG_VOCAB_SIZE, DEC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, DEC_DROPOUT)
    model  = Seq2Seq(encoder, decoder)
    model.apply(init_weights)
    

    # Training
    #-------------------------------------------------------------------------------------
    checkpoint = ModelCheckpoint(monitor='val_loss')
    logger = TensorBoardLogger(save_dir=r".\logs")
    trainer = pl.Trainer(gpus=1,
                         gradient_clip_val=1.,
                         callbacks=[checkpoint],
                         logger=logger)

    
    trainer.fit(model, train_iterator, valid_iterator)
