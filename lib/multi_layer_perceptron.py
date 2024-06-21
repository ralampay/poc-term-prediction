import torch.nn as nn

class MultiLayerPerceptron(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config

        self.input_size     = self.config['input_size']
        self.output_size    = self.config['output_size']

        self.layers = []

        for index, item in enumerate(self.config['layers']):
            dim_in  = item.get('dim_in') or -1
            dim_out = item.get('dim_out') or -1

            if index == 0:
                dim_in  = self.input_size

            elif index == (len(self.config['layers']) - 1):
                dim_out = self.output_size

            self.layers.append(nn.Linear(dim_in, dim_out))
            
            if item['activation']['name'] == 'relu':
                self.layers.append(nn.ReLU())
                self.layers.append(nn.Dropout(0.2))
            elif item['activation']['name'] == 'leaky_relu':
                negative_slope = 0.01

                if item['activation'].get('negative_slope') is not None:
                    negative_slope = item['activation']['negative_slope']
                self.layers.append(nn.LeakyReLU(negative_slope=negative_slope))
            elif item['activation']['name'] == 'sigmoid':
                self.layers.append(nn.Sigmoid())
            elif item['activation']['name'] == 'tanh':
                self.layers.append(nn.Tanh())

        self.model = nn.Sequential(*self.layers)

    def forward(self, x):
        return self.model(x)
