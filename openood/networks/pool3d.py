import torch
import torch.nn.functional as F


class DeterministicPool3d(torch.nn.Module):
    """A deterministic implementation of 3D pooling (average or max).

    This is useful for reproducibility of results when it's required to use a
    3d pooling layer, which the default PyTorch implementation is non-
    deterministic. The catch is that this implementation is much slower than
    the default one!
    """
    def __init__(self, kernel_size, stride=None, padding=0, pool_type='max'):
        super(DeterministicPool3d, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding
        self.pool_type = pool_type

    def forward(self, x):
        # Add padding to the input
        pad_value = float('-inf') if self.pool_type == 'max' else 0
        x = F.pad(x, (self.padding, self.padding, self.padding, self.padding,
                      self.padding, self.padding),
                  mode='constant',
                  value=pad_value)

        # Unfold the depth, height, and width dimensions
        x_unfolded = x.unfold(2, self.kernel_size, self.stride) \
                      .unfold(3, self.kernel_size, self.stride) \
                      .unfold(4, self.kernel_size, self.stride)

        # Reshape the unfolded tensor to apply pooling operation
        unfolded_shape = x_unfolded.size()
        x_unfolded = x_unfolded.contiguous().view(unfolded_shape[0],
                                                  unfolded_shape[1],
                                                  unfolded_shape[2],
                                                  unfolded_shape[3],
                                                  unfolded_shape[4], -1)

        if self.pool_type == 'max':
            # Perform max pooling by taking the max over the last dimension
            pooled, _ = torch.max(x_unfolded, dim=-1)
        elif self.pool_type == 'avg':
            # Perform avg pooling by taking the mean over the last dimension
            pooled = torch.mean(x_unfolded, dim=-1)
        else:
            raise ValueError(f'Unsupported pool_type: {self.pool_type}')

        return pooled
