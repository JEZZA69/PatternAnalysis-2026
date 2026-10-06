import torch
import torch.nn as nn

class ContextAggregationModule(nn.Module):
    def __init__(self, in_channels=1, num_classes=24, C=64):
        super().__init__()
        
        # Exact 8 layer sequential architecture from Yu & Koltun (ICLR 2016)
        # Padding must equal dilation for 3x3 kernels to preserve spatial dimensions (160x192)
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
        
    def _init_weights(self):
        """ Yu & Koltun discovered that to prevent model hallucination and 'telephone game'-ness. 
        They set every weight in the network to 0 except for the dead-centre pixel of the 3x3 
        convolutional filter for the matching channels, which is set to 1.
        
        This acts as a perfect pass through.
        
        The model therefore does the following:
            The brain slice passes through smoothly with all 7 layers completely unaltered.
            The network makes a guess
            The loss function adjusts
            BACKPROPOGATION KICKS IN
            
        Because the signal wasn't randomly scrambled nonsense the signal can travel back through the network flawlessly.
        This allows the network to actually TRAIN
        """
        
        # grab layers 2 through 7 (the internal C to C layers)
        layers_to_init = [self.layer2, self.layer3, self.layer4, self.layer5, self.layer6, self.layer7]
        
        for layer in layers_to_init:
            
            # Whiteout. weights and biases are now 0.
            nn.init.constant_(layer.weight, 0.0)
            if layer.bias is not None:
                nn.init.constant_(layer.bias, 0.0)
                
            # Find the exact centre pixel of the 3x3 filter
            # A 3x3 matrix has indices: top=[0], middle=[1], bottom=[2]
            # So index [1,1] is the dead centre
            for i in range(layer.out_channels):
                # If input channel = output channel, (i,i), set the center pixel to 1.
                layer.weight.data[i,i,1,1] = 1.0
                      
    def forward(self, x):
        # Layers 1 to 7 use a point-wise truncation max(x, 0)
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.relu(self.layer3(x))
        x = self.relu(self.layer4(x))
        x = self.relu(self.layer5(x))
        x = self.relu(self.layer6(x))
        x = self.relu(self.layer7(x))
        
        # Layer 8 has NO activation function before returning the logits
        return self.layer8(x)
    

    # AI made this.
if __name__ == "__main__":
    # Create a dummy batch resembling a real loader item: [Batch_Size, Channels, Height, Width]
    dummy_input = torch.randn(4, 1, 160, 192) 
    
    # Initialize the model matching your exact task constraints
    model = ContextAggregationModule(in_channels=1, num_classes=24, C=64)
    output = model(dummy_input)
    
    print("=== CAN Architecture Verification ===")
    print(f"Input Shape:  {dummy_input.shape}")   # Expected: [4, 1, 160, 192]
    print(f"Output Shape: {output.shape}")        # Expected: [4, 24, 160, 192]
    print("=====================================")