import torch.nn as nn
from transformers import Dinov2Config, Dinov2Model

class DinoV2_6channels(nn.Module):
    def __init__(self, num_channels=6, num_classes=10):  # Adjust num_classes as needed
        super(DinoV2_6channels, self).__init__()

        # Define the Dinov2 model with a specified configuration
        configuration = Dinov2Config(num_channels=num_channels)
        self.dinov2 = Dinov2Model.from_pretrained('facebook/dinov2-base', config=configuration, ignore_mismatched_sizes=True)

        # Add a classifier layer ahead of the pooling layer output
        self.classifier = nn.Sequential(
            nn.Linear(in_features=self.dinov2.config.hidden_size, out_features=256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(in_features=256, out_features=num_classes)
        )

    def forward(self, input_data):
        # Get the output from the Dinov2 model
        dinov2_output = self.dinov2(input_data)['pooler_output']

        # Pass the pooled output through the classifier
        output = self.classifier(dinov2_output)

        return output
