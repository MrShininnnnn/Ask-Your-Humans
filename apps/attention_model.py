"""Attention model used in instructions generator decoder"""

import torch.nn as nn


class Attention(nn.Module):
  """
    Attention Network.
    https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning/
    """

  def __init__(self, encoder_dim, decoder_dim, attention_dim):
    """
        :param encoder_dim: feature size of encoded images
        :param decoder_dim: size of decoder's RNN
        :param attention_dim: size of the attention network
        """
    super(Attention, self).__init__()
    self.encoder_att = nn.Linear(
        encoder_dim, attention_dim)  # linear layer to transform encoded image
    self.decoder_att = nn.Linear(
        decoder_dim,
        attention_dim)  # linear layer to transform decoder's output
    self.full_att = nn.Linear(
        attention_dim, 1)  # linear layer to calculate values to be softmax-ed
    self.relu = nn.ReLU()
    self.softmax = nn.Softmax(dim=1)  # softmax layer to calculate weights

  def forward(self, encoder_out, decoder_hidden):
    """
        Forward propagation.
        :param encoder_out: encoded images, a tensor of dimension (batch_size,
        num_pixels, encoder_dim)
        :param decoder_hidden: previous decoder output, a tensor of dimension
        (batch_size, decoder_dim)
        :return: attention weighted encoding, weights
        """
    att1 = self.encoder_att(
        encoder_out)  # (batch_size, num_pixels, attention_dim)
    att2 = self.decoder_att(decoder_hidden)  # (batch_size, attention_dim)
    att = self.full_att(self.relu(att1 + att2.unsqueeze(1))).squeeze(
        2)  # (batch_size, num_pixels)
    alpha = self.softmax(att)  # (batch_size, num_pixels)
    attention_weighted_encoding = (encoder_out * alpha.unsqueeze(2)).sum(
        dim=1)  # (batch_size, encoder_dim)
    return attention_weighted_encoding, alpha


if __name__ == '__main__':
  attention = Attention(128, 32, 32)
