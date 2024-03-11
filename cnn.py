from torch import nn 
from torchsummary import summary

class CNNetwork(nn.Module):
    ## costructor
    # architecture  ---> 4 cnn / flatten / linear / softmax
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=16, ## how many kernels we want
                kernel_size=3,
                stride=1,padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)                     
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=3,
                stride=1,padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)                     
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                stride=1,padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)                     
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                stride=1,padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)                     
        )
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(128 * 5 * 4, 10)
        self.softmax = nn.Softmax(dim=1)
    
    def forward (self,input_data): 
        x = self.conv1(input_data)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.flatten(x)
        logics = self.linear(x)
        predictions = self.softmax(logics)
        return predictions 
        

if __name__ == "__main__":
    cnn = CNNetwork()
    summary(cnn, (1, 64, 44))
