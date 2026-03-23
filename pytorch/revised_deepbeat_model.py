import torch
import torch.nn as nn

class revised_DeepBeatModel(nn.Module):
    def __init__(self, dropouts=None):
        super(revised_DeepBeatModel, self).__init__()
        
        if dropouts is None:
            dropouts = {
                'do57': 0.118, 'do58': 0.545, 'do59': 0.568,
                'do60': 0.672, 'do61': 0.374, 'do62': 0.414,
            }
        
        # revised to follow standard structure
        # Conv1D -> BatchNorm -> Activation -> Pooling  -> Dropout
        # --- Common Backbone ---
        # backbone block 1
        self.conv1d_81 = nn.Conv1d(1, 64, kernel_size=10, padding=5) # 'same'
        self.max_pool25 = nn.MaxPool1d(kernel_size=3, stride=3)
        self.bn_81 = nn.BatchNorm1d(64) # add this for stability
        self.backbone_1 = nn.Sequential(self.conv1d_81, self.bn_81, nn.ReLU(), self.max_pool25 )
        
        # backbone block 2
        self.conv1d_82 = nn.Conv1d(64, 45, kernel_size=8, padding=4)
        self.max_pool26 = nn.MaxPool1d(kernel_size=3, stride=3)
        self.bn_82 = nn.BatchNorm1d(45) # add this for stability
        self.backbone_2 = nn.Sequential(self.conv1d_82, self.bn_82, nn.ReLU(), self.max_pool26)
        
        # backbone block 3
        self.conv1d_83 = nn.Conv1d(45, 50, kernel_size=5, padding=2)
        self.max_pool27 = nn.MaxPool1d(kernel_size=2, stride=2, padding=1) # 'same' padding
        self.bn65 = nn.BatchNorm1d(50)
        self.backbone_3 = nn.Sequential(self.conv1d_83,self.bn65, nn.ReLU(),self.max_pool27 )
        

        # --- Shared Bottleneck ---
        
        # shared block 1
        self.conv1d_84 = nn.Conv1d(50, 64, kernel_size=4, stride=3, padding=1)
        self.lrelu = nn.LeakyReLU(0.1) 
        self.bn66 = nn.BatchNorm1d(64)
        self.do57 = nn.Dropout(dropouts['do57']) 
        self.sharedblock_1 = nn.Sequential(self.conv1d_84,self.bn66,self.lrelu,self.do57)
        

        self.conv1d_85 = nn.Conv1d(64, 35, kernel_size=4, stride=3, padding=1)
        self.bn67 = nn.BatchNorm1d(35)
        self.do58 = nn.Dropout(dropouts['do58'])
        self.sharedblock_2 = nn.Sequential(self.conv1d_85,self.bn67,self.lrelu,self.do58)

        self.conv1d_86 = nn.Conv1d(35, 64, kernel_size=4, padding=2) # Stride 1
        self.bn68 = nn.BatchNorm1d(64)
        self.do59 = nn.Dropout(dropouts['do59']) 
        self.sharedblock_3 = nn.Sequential(self.conv1d_86,self.bn68,self.lrelu,self.do59 )

        # --- QA Branch (Connected to Dropout 59) ---
        self.conv1d_87 = nn.Conv1d(64, 25, kernel_size=4, stride=2, padding=1)
        self.bn69 = nn.BatchNorm1d(25)
        self.do60 = nn.Dropout(dropouts['do60'])
        self.qa_conv_1 = nn.Sequential(self.conv1d_87, self.bn69, nn.ReLU(), self.do60)

        self.qa_out = nn.Linear(75, 3)   # flatten(25 * 3) -> 3 classes

        # --- Rhythm Branch (Connected to Dropout 59) ---
        self.conv1d_88 = nn.Conv1d(64, 35, kernel_size=5, stride=3, padding=2)
        self.bn70 = nn.BatchNorm1d(35)
        self.do61 = nn.Dropout(dropouts['do61'])
        self.rh_conv_1 = nn.Sequential(self.conv1d_88, self.bn70, nn.ReLU(), self.do61)

        self.conv1d_89 = nn.Conv1d(35, 25, kernel_size=4, stride=3, padding=2)
        self.bn71 = nn.BatchNorm1d(25)
        self.do62 = nn.Dropout(dropouts['do62'])
        self.rh_conv_2 = nn.Sequential(self.conv1d_89, self.bn71, nn.ReLU(), self.do62)

        self.rhythm_out = nn.Linear(25, 2)  # flatten(25 * 1) -> 2 classes

    def forward(self, x):
        # Backbone
        x = self.backbone_1(x)
        x = self.backbone_2(x)
        x = self.backbone_3(x)

        # Shared processing
        x = self.sharedblock_1(x)
        x = self.sharedblock_2(x)
        shared_feat = self.sharedblock_3(x)
        
        # QA Branch
        qa = self.qa_conv_1(shared_feat)
        qa = torch.flatten(qa, 1)   # (batch, 25*3=75)
        qa_out = self.qa_out(qa)

        # Rhythm Branch
        rh = self.rh_conv_1(shared_feat)
        rh = self.rh_conv_2(rh)
        rh = torch.flatten(rh, 1)   # (batch, 25*1=25)
        rhythm_out = self.rhythm_out(rh)
        return qa_out, rhythm_out
 
   