import torch
import torch.nn as nn
import torch.nn.functional as F

class revised_DeepBeatModel(nn.Module):
    def __init__(self, dropouts=None):
        super(revised_DeepBeatModel, self).__init__()
        
        if dropouts is None:
            dropouts = {
                'do57': 0.118, 'do58': 0.545, 'do59': 0.568,
                'do60': 0.672, 'do61': 0.374, 'do62': 0.414, 'do63': 0.017
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
        
        self.dense17 = nn.Linear(75, 175)
        self.qa_out = nn.Linear(175, 3)
        self.qa_MLP = nn.Sequential(self.dense17, nn.ReLU(), self.qa_out)
        

        # --- Rhythm Branch (Connected to Dropout 59) ---
        self.conv1d_88 = nn.Conv1d(64, 35, kernel_size=5, stride=3, padding=2)
        self.bn70 = nn.BatchNorm1d(35)
        self.do61 = nn.Dropout(dropouts['do61'])
        self.rh_conv_1 = nn.Sequential(self.conv1d_88,self.bn70, nn.ReLU(), self.do61 )
        
        self.conv1d_89 = nn.Conv1d(35, 25, kernel_size=4, stride=3, padding=2)
        self.bn71 = nn.BatchNorm1d(25)
        self.do62 = nn.Dropout(dropouts['do62'])
        self.rh_conv_2 = nn.Sequential(self.conv1d_89,self.bn71, nn.ReLU(),self.do62)

        self.conv1d_90 = nn.Conv1d(25, 35, kernel_size=3, padding=1)
        self.bn72 = nn.BatchNorm1d(35)
        self.do63 = nn.Dropout(dropouts['do63']) 
        self.rh_conv_3 = nn.Sequential(self.conv1d_90,self.bn72, nn.ReLU(), self.do63)
        
        self.dense18 = nn.Linear(35, 175)
        self.rhythm_out = nn.Linear(175, 2)
        self.rh_MLP = nn.Sequential(self.dense18, nn.ReLU(), self.rhythm_out)

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
        qa = torch.flatten(qa, 1)
        qa_out = self.qa_MLP(qa)

        # Rhythm Branch
        rh = self.rh_conv_1(shared_feat)
        rh = self.rh_conv_2(rh)
        rh = self.rh_conv_3(rh)
        rh = torch.flatten(rh, 1)
        rhythm_out = self.rh_MLP(rh)
        return qa_out, rhythm_out
 
   