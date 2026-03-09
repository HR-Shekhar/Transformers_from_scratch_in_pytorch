import torch
import torch.nn as nn
from torch.utils.data import Dataset

class BilingualDataset(Dataset):
    """
    Dataset class for bilingual translation data.

    Prepares source and target sequences with padding, special tokens, and masks.

    Attributes:
        seq_len (int): Maximum sequence length.
        ds: The raw dataset.
        tokenizer_src: Source language tokenizer.
        tokenizer_tgt: Target language tokenizer.
        src_lang (str): Source language code.
        tgt_lang (str): Target language code.
        sos_token (torch.Tensor): Start-of-sequence token ID.
        eos_token (torch.Tensor): End-of-sequence token ID.
        pad_token (torch.Tensor): Padding token ID.
    """

    def __init__(self, ds, tokenizer_src, tokenizer_tgt, src_lang, tgt_lang, seq_len):
        """
        Initialize the BilingualDataset.

        Args:
            ds: The raw dataset containing translation pairs.
            tokenizer_src: Tokenizer for the source language.
            tokenizer_tgt: Tokenizer for the target language.
            src_lang (str): Source language code (e.g., 'en').
            tgt_lang (str): Target language code (e.g., 'fr').
            seq_len (int): Maximum sequence length for padding.
        """
        super().__init__()
        self.seq_len = seq_len

        self.ds = ds
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang

        self.sos_token = torch.tensor([tokenizer_tgt.token_to_id("[SOS]")], dtype=torch.int64)
        self.eos_token = torch.tensor([tokenizer_tgt.token_to_id("[EOS]")], dtype=torch.int64)
        self.pad_token = torch.tensor([tokenizer_tgt.token_to_id("[PAD]")], dtype=torch.int64)

    def __len__(self):
        """
        Return the length of the dataset.

        Returns:
            int: Number of samples.
        """
        return len(self.ds)

    def __getitem__(self, idx):
        """
        Get a single data sample.

        Args:
            idx (int): Index of the sample.

        Returns:
            dict: Dictionary with encoder_input, decoder_input, masks, label, and texts.
        """
        src_target_pair = self.ds[idx]
        src_text = src_target_pair['translation'][self.src_lang]
        tgt_text = src_target_pair['translation'][self.tgt_lang]

        # Transform the text into tokens
        enc_input_tokens = self.tokenizer_src.encode(src_text).ids
        dec_input_tokens = self.tokenizer_tgt.encode(tgt_text).ids

        # Add padding to each sentence --> Padding tokens to add = length of sequence - length of sequence - 2(SOS and EOS)
        enc_num_padding_tokens = self.seq_len - len(enc_input_tokens) - 2
        dec_num_padding_tokens = self.seq_len - len(dec_input_tokens) - 1

        # encoder final input = sos + input sentence + eos + pad
        encoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(enc_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * enc_num_padding_tokens, dtype=torch.int64),
            ]
        )

        # Decoder final input = sos + input sentence + eos + pad
        decoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64),
            ]
        )

        # ADD EOS to the label(what we expect output from the decoder)
        label = torch.cat(
            [
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64),
            ]
        )

        # Double check the size of the tensors to make sure they are all seq_len long
        assert encoder_input.size(0) == self.seq_len
        assert decoder_input.size(0) == self.seq_len
        assert label.size(0) == self.seq_len

        return {
            "encoder_input": encoder_input,  # (seq_len)
            "decoder_input": decoder_input,  # (seq_len)

            # Creates a boolean mask which is false for padding tokens .int() to convert into 0s & 1s
            "encoder_mask": (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(), # (1, 1, seq_len)
            "decoder_mask": (decoder_input != self.pad_token).unsqueeze(0).int() & causal_mask(decoder_input.size(0)), # (1, seq_len) & (1, seq_len, seq_len),
            "label": label,  # (seq_len)
            "src_text": src_text,
            "tgt_text": tgt_text,
        }
    
def causal_mask(size):
    """
    Create a causal mask to hide future tokens in the decoder.

    Args:
        size (int): Size of the mask (sequence length).

    Returns:
        torch.Tensor: Boolean mask where True indicates allowed positions.
    """
    mask = torch.triu(torch.ones((1, size, size)), diagonal=1).type(torch.int)
    return mask == 0