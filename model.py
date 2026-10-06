import torch
import torch.nn as nn

class ContextAggregationModule(nn.Module):
    def __init__(self, in_channels=1, num_classes=24, C=64):
        super().__init__()
        
        # Exact 8 layer architecture from Yu & Koltun [MULTI-SCALE CONTEXT AGGREGATION BY DILATED CONVOLUTIONS] (see Readme)
        # padding must equal dilation for 3x3 kernels to preserve spatial dimensions (160x192)
        
        # Parallel dilated branches to capture different contexts simultaneously
        self.layer1 = nn.Conv2d(in_channels, C, kernel_size=3, padding=1, dilation=1)
        self.layer2 = nn.Conv2d(C,           C, kernel_size=3, padding=1, dilation=1)
        self.layer3 = nn.Conv2d(C,           C, kernel_size=3, padding=2, dilation=2)
        self.layer4 = nn.Conv2d(C,           C, kernel_size=3, padding=4, dilation=4)
        self.layer5 = nn.Conv2d(C,           C, kernel_size=3, padding=8, dilation=8)
        self.layer6 = nn.Conv2d(C,           C, kernel_size=3, padding=16, dilation=16)
        self.layer7 = nn.Conv2d(C,           C, kernel_size=3, padding=1, dilation=1)
        
        # Layer 8 is a 1x1 convolution projecting to the 24 segmentation classes.
        self.layer8 = nn.Conv2d(C, num_classes, kernel_size=1)
        
        self.relu = nn.ReLU(inplace=True)
        
        self._init_weights()
        
                      
    def forward(self, x):
        # Extract context at multiple scales
        out1 = self.relu(self.conv_d1(x))
        out2 = self.relu(self.conv_d2(x))
        out4 = self.relu(self.conv_d4(x))
        out8 = self.relu(self.conv_d8(x))
        
        # Concatenate features along the channel axis and compress
        combined = torch.cat([out1, out2, out4, out8], dim=1)
        return self.relu(self.fuse(combined))
    
    
if __name__ == "__main__":
    # Create a dummy batch resembling a real loader item: [Batch_Size, Channels, Height, Width]
    dummy_input = torch.randn(4, 1, 160, 192) 
    model = ContextAggregationModule(in_channels=1, num_classes=24)
    output = model(dummy_input)
    
    print("=== CAN Architecture Verification ===")
    print(f"Input Shape:  {dummy_input.shape}")   # Should be [4, 1, 160, 192]
    print(f"Output Shape: {output.shape}")        # Must be [4, 24, 160, 192]
    print("=====================================")