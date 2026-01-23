import torch
import torch.nn as nn
import torch.nn.functional as F


class DeepBeatModel(nn.Module):
    """
    PyTorch implementation of the DeepBeat model for cardiac signal analysis.
    
    This model has two output heads:
    - qa_output: Quality assessment (3 classes)
    - rhythm_output: Rhythm classification (2 classes)
    
    Input: (batch_size, 800, 1) - 1D signal of length 800
    """
    
    def __init__(self):
        super(DeepBeatModel, self).__init__()
        
        # Initial conv layers (feature extraction)
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=10, 
                               stride=1, padding='same')
        self.maxpool1 = nn.MaxPool1d(kernel_size=3, stride=3)
        
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=45, kernel_size=8, 
                               stride=1, padding='same')
        self.maxpool2 = nn.MaxPool1d(kernel_size=3, stride=3)
        
        self.conv3 = nn.Conv1d(in_channels=45, out_channels=50, kernel_size=5, 
                               stride=1, padding='same')
        self.maxpool3 = nn.MaxPool1d(kernel_size=2, stride=2, padding=1)
        
        self.bn1 = nn.BatchNorm1d(50, momentum=0.01)  # Keras momentum=0.99 -> PyTorch momentum=0.01
        
        # Deeper conv layers with regularization
        self.conv4 = nn.Conv1d(in_channels=50, out_channels=64, kernel_size=4, 
                               stride=3, padding=1)
        self.leaky_relu1 = nn.LeakyReLU(negative_slope=0.1)
        self.bn2 = nn.BatchNorm1d(64, momentum=0.01)
        self.dropout1 = nn.Dropout(p=0.11824940188979882)
        
        self.conv5 = nn.Conv1d(in_channels=64, out_channels=35, kernel_size=4, 
                               stride=3, padding=1)
        self.leaky_relu2 = nn.LeakyReLU(negative_slope=0.1)
        self.bn3 = nn.BatchNorm1d(35, momentum=0.01)
        self.dropout2 = nn.Dropout(p=0.5449968090097298)
        
        self.conv6 = nn.Conv1d(in_channels=35, out_channels=64, kernel_size=4, 
                               stride=1, padding='same')
        self.leaky_relu3 = nn.LeakyReLU(negative_slope=0.1)
        self.bn4 = nn.BatchNorm1d(64, momentum=0.01)
        self.dropout3 = nn.Dropout(p=0.5678640009808424)
        
        # Branch 1 (for QA output)
        self.conv7_qa = nn.Conv1d(in_channels=64, out_channels=25, kernel_size=4, 
                                   stride=2, padding=1)
        self.bn5_qa = nn.BatchNorm1d(25, momentum=0.01)
        self.dropout4_qa = nn.Dropout(p=0.6717685126227927)
        
        # Branch 2 (for rhythm output)
        self.conv8_rhythm = nn.Conv1d(in_channels=64, out_channels=35, kernel_size=5, 
                                       stride=3, padding=2)
        self.bn6_rhythm = nn.BatchNorm1d(35, momentum=0.01)
        self.dropout5_rhythm = nn.Dropout(p=0.3737667081555568)
        
        self.conv9_rhythm = nn.Conv1d(in_channels=35, out_channels=25, kernel_size=4, 
                                       stride=3, padding=1)
        self.bn7_rhythm = nn.BatchNorm1d(25, momentum=0.01)
        self.dropout6_rhythm = nn.Dropout(p=0.4138296544325403)
        
        self.conv10_rhythm = nn.Conv1d(in_channels=25, out_channels=35, kernel_size=3, 
                                        stride=1, padding='same')
        self.bn8_rhythm = nn.BatchNorm1d(35, momentum=0.01)
        self.dropout7_rhythm = nn.Dropout(p=0.017043716246453067)
        
        # Fully connected layers (need to calculate input size based on conv output)
        # After all convolutions, we need to flatten and pass through FC layers
        self.fc1_qa = nn.Linear(375, 175)  # 25 * 15 = 375 (calculated from conv output)
        self.fc2_rhythm = nn.Linear(350, 175)  # 35 * 10 = 350 (calculated from conv output)
        
        # Output layers
        self.qa_output = nn.Linear(175, 3)
        self.rhythm_output = nn.Linear(175, 2)
        
        # L2 regularization weights (stored for manual regularization in training)
        self.l2_weights = {
            'conv4': 0.044189151376485825,
            'conv5': 0.04586048796772957,
            'conv6': 0.011329331435263157,
            'conv7_qa': 0.03589663654565811,
            'conv8_rhythm': 0.04263903945684433,
            'conv9_rhythm': 0.041482653468847275,
            'conv10_rhythm': 0.03098524734377861
        }
        
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, 1) or (batch_size, 1, seq_len)
        
        Returns:
            dict with 'qa_output' and 'rhythm_output' keys
        """
        # Handle input shape - convert (batch, seq_len, 1) to (batch, 1, seq_len)
        if x.shape[-1] == 1 and len(x.shape) == 3:
            x = x.permute(0, 2, 1)
        
        # Initial feature extraction
        x = F.relu(self.conv1(x))
        x = self.maxpool1(x)
        
        x = F.relu(self.conv2(x))
        x = self.maxpool2(x)
        
        x = F.relu(self.conv3(x))
        x = self.maxpool3(x)
        
        x = self.bn1(x)
        
        # Deeper layers with regularization
        x = self.conv4(x)
        x = self.leaky_relu1(x)
        x = self.bn2(x)
        x = self.dropout1(x)
        
        x = self.conv5(x)
        x = self.leaky_relu2(x)
        x = self.bn3(x)
        x = self.dropout2(x)
        
        x = self.conv6(x)
        x = self.leaky_relu3(x)
        x = self.bn4(x)
        x = self.dropout3(x)
        
        # Branch for QA output
        qa_branch = F.relu(self.conv7_qa(x))
        qa_branch = self.bn5_qa(qa_branch)
        qa_branch = self.dropout4_qa(qa_branch)
        qa_branch = qa_branch.flatten(1)  # Flatten
        qa_branch = F.relu(self.fc1_qa(qa_branch))
        qa_out = F.softmax(self.qa_output(qa_branch), dim=1)
        
        # Branch for rhythm output
        rhythm_branch = F.relu(self.conv8_rhythm(x))
        rhythm_branch = self.bn6_rhythm(rhythm_branch)
        rhythm_branch = self.dropout5_rhythm(rhythm_branch)
        
        rhythm_branch = F.relu(self.conv9_rhythm(rhythm_branch))
        rhythm_branch = self.bn7_rhythm(rhythm_branch)
        rhythm_branch = self.dropout6_rhythm(rhythm_branch)
        
        rhythm_branch = F.relu(self.conv10_rhythm(rhythm_branch))
        rhythm_branch = self.bn8_rhythm(rhythm_branch)
        rhythm_branch = self.dropout7_rhythm(rhythm_branch)
        rhythm_branch = rhythm_branch.flatten(1)  # Flatten
        rhythm_branch = F.relu(self.fc2_rhythm(rhythm_branch))
        rhythm_out = torch.sigmoid(self.rhythm_output(rhythm_branch))
        
        return {
            'qa_output': qa_out,
            'rhythm_output': rhythm_out
        }
    
    def get_l2_regularization(self):
        """
        Calculate L2 regularization loss for specified layers.
        
        Returns:
            L2 regularization loss
        """
        l2_reg = 0.0
        
        # Add L2 regularization for conv4
        l2_reg += self.l2_weights['conv4'] * torch.sum(self.conv4.weight ** 2)
        
        # Add L2 regularization for conv5
        l2_reg += self.l2_weights['conv5'] * torch.sum(self.conv5.weight ** 2)
        
        # Add L2 regularization for conv6
        l2_reg += self.l2_weights['conv6'] * torch.sum(self.conv6.weight ** 2)
        
        # Add L2 regularization for conv7_qa
        l2_reg += self.l2_weights['conv7_qa'] * torch.sum(self.conv7_qa.weight ** 2)
        
        # Add L2 regularization for conv8_rhythm
        l2_reg += self.l2_weights['conv8_rhythm'] * torch.sum(self.conv8_rhythm.weight ** 2)
        
        # Add L2 regularization for conv9_rhythm
        l2_reg += self.l2_weights['conv9_rhythm'] * torch.sum(self.conv9_rhythm.weight ** 2)
        
        # Add L2 regularization for conv10_rhythm
        l2_reg += self.l2_weights['conv10_rhythm'] * torch.sum(self.conv10_rhythm.weight ** 2)
        
        return l2_reg


def test_model():
    """Test the model with a sample input"""
    model = DeepBeatModel()
    
    # Test with sample input
    batch_size = 4
    seq_len = 800
    x = torch.randn(batch_size, seq_len, 1)
    
    print("Input shape:", x.shape)
    
    output = model(x)
    
    print("QA Output shape:", output['qa_output'].shape)
    print("Rhythm Output shape:", output['rhythm_output'].shape)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    return model


if __name__ == "__main__":
    test_model()
