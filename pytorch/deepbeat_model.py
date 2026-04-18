import torch
import torch.nn as nn
import torch.nn.functional as F

class DeepBeatModel(nn.Module):
    def __init__(self, dropouts=None):
        super(DeepBeatModel, self).__init__()
        
        if dropouts is None:
            dropouts = {
                'do57': 0.118, 'do58': 0.545, 'do59': 0.568,
                'do60': 0.672, 'do61': 0.374, 'do62': 0.414, 'do63': 0.017
            }
        
        # --- Common Backbone ---
        self.conv1d_81 = nn.Conv1d(1, 64, kernel_size=10, padding=5) # 'same'
        self.max_pool25 = nn.MaxPool1d(kernel_size=3, stride=3)
        self.batch_norm_81 = nn.BatchNorm1d(64)
        
        self.conv1d_82 = nn.Conv1d(64, 45, kernel_size=8, padding=4)
        self.max_pool26 = nn.MaxPool1d(kernel_size=3, stride=3)
        self.batch_norm_82 = nn.BatchNorm1d(45)
        
        self.conv1d_83 = nn.Conv1d(45, 50, kernel_size=5, padding=2)
        self.max_pool27 = nn.MaxPool1d(kernel_size=2, stride=2, padding=1) # 'same' padding
        
        self.bn65 = nn.BatchNorm1d(50)
        

        # --- Shared Bottleneck ---
        self.conv1d_84 = nn.Conv1d(50, 64, kernel_size=4, stride=3, padding=1)
        self.lrelu = nn.LeakyReLU(0.1) 
        self.bn66 = nn.BatchNorm1d(64)
        self.do57 = nn.Dropout(dropouts['do57']) 

        self.conv1d_85 = nn.Conv1d(64, 35, kernel_size=4, stride=3, padding=1)
        self.bn67 = nn.BatchNorm1d(35)
        self.do58 = nn.Dropout(dropouts['do58']) 

        self.conv1d_86 = nn.Conv1d(35, 64, kernel_size=4, padding=2) # Stride 1
        self.bn68 = nn.BatchNorm1d(64)
        self.do59 = nn.Dropout(dropouts['do59']) 

        # --- QA Branch (Connected to Dropout 59) ---
        self.conv1d_87 = nn.Conv1d(64, 25, kernel_size=4, stride=2, padding=1)
        self.bn69 = nn.BatchNorm1d(25)
        self.do60 = nn.Dropout(dropouts['do60']) 
        self.dense17 = nn.Linear(75, 175)
        self.qa_out = nn.Linear(175, 3)

        # --- Rhythm Branch (Connected to Dropout 59) ---
        self.conv1d_88 = nn.Conv1d(64, 35, kernel_size=5, stride=3, padding=2)
        self.bn70 = nn.BatchNorm1d(35)
        self.do61 = nn.Dropout(dropouts['do61']) 
        
        self.conv1d_89 = nn.Conv1d(35, 25, kernel_size=4, stride=3, padding=2)
        self.bn71 = nn.BatchNorm1d(25)
        self.do62 = nn.Dropout(dropouts['do62']) 

        self.conv1d_90 = nn.Conv1d(25, 35, kernel_size=3, padding=1)
        self.bn72 = nn.BatchNorm1d(35)
        self.do63 = nn.Dropout(dropouts['do63']) 
        
        self.dense18 = nn.Linear(35, 175)
        self.rhythm_out = nn.Linear(175, 2)

    def forward(self, x):
        # Backbone
        x = self.max_pool25(F.relu(self.conv1d_81(x)))
        x = self.batch_norm_81(x)
        x = self.max_pool26(F.relu(self.conv1d_82(x)))
        x = self.batch_norm_82(x)
        x = self.max_pool27(F.relu(self.conv1d_83(x)))
        x = self.bn65(x)

        # Shared processing
        x = self.do57(self.bn66(self.lrelu(self.conv1d_84(x))))
        x = self.do58(self.bn67(self.lrelu(self.conv1d_85(x))))
        shared_feat = self.do59(self.bn68(self.lrelu(self.conv1d_86(x))))

        # QA Branch
        qa = self.do60(self.bn69(F.relu(self.conv1d_87(shared_feat))))
        qa = torch.flatten(qa, 1)
        qa = F.relu(self.dense17(qa))
        qa_out = self.qa_out(qa) # raw logits

        # Rhythm Branch
        rh = self.do61(self.bn70(F.relu(self.conv1d_88(shared_feat))))
        rh = self.do62(self.bn71(F.relu(self.conv1d_89(rh))))
        rh = self.do63(self.bn72(F.relu(self.conv1d_90(rh))))
        rh = torch.flatten(rh, 1)
        rh = F.relu(self.dense18(rh))
        rhythm_out = self.rhythm_out(rh) 
        return qa_out, rhythm_out
 


# the following models are created for experimentation purposes       

class rhythm_only_deepbeat(nn.Module):
    def __init__(self, dropouts=None):
        super(rhythm_only_deepbeat, self).__init__()
        
        if dropouts is None:
            dropouts = {
                'do57': 0.25, 'do58': 0.25, 'do59': 0.25,
                'do60': 0.25, 'do61': 0.25, 'do62': 0.25, 'do63': 0.25
            }
        
        # --- Common Backbone ---
        self.conv1d_81 = nn.Conv1d(1, 64, kernel_size=10, padding=5) # 'same'
        self.max_pool25 = nn.MaxPool1d(kernel_size=3, stride=3)
        self.bn_81 = nn.BatchNorm1d(64)
        
        self.conv1d_82 = nn.Conv1d(64, 45, kernel_size=8, padding=4)
        self.max_pool26 = nn.MaxPool1d(kernel_size=3, stride=3)
        self.bn_82 = nn.BatchNorm1d(45)
        
        self.conv1d_83 = nn.Conv1d(45, 50, kernel_size=5, padding=2)
        self.max_pool27 = nn.MaxPool1d(kernel_size=2, stride=2, padding=1) # 'same' padding
        
        self.bn65 = nn.BatchNorm1d(50)
        

        # --- Shared Bottleneck ---
        self.conv1d_84 = nn.Conv1d(50, 64, kernel_size=4, stride=3, padding=1)
        self.lrelu = nn.LeakyReLU(0.1) 
        self.bn66 = nn.BatchNorm1d(64)
        self.do57 = nn.Dropout(dropouts['do57']) 

        self.conv1d_85 = nn.Conv1d(64, 35, kernel_size=4, stride=3, padding=1)
        self.bn67 = nn.BatchNorm1d(35)
        self.do58 = nn.Dropout(dropouts['do58']) 

        self.conv1d_86 = nn.Conv1d(35, 64, kernel_size=4, padding=2) # Stride 1
        self.bn68 = nn.BatchNorm1d(64)
        self.do59 = nn.Dropout(dropouts['do59']) 
        
        # --- Rhythm Branch (Connected to Dropout 59) ---
        self.conv1d_88 = nn.Conv1d(64, 35, kernel_size=5, stride=3, padding=2)
        self.bn70 = nn.BatchNorm1d(35)
        self.do61 = nn.Dropout(dropouts['do61']) 
        
        self.conv1d_89 = nn.Conv1d(35, 25, kernel_size=4, stride=3, padding=2)
        self.bn71 = nn.BatchNorm1d(25)
        self.do62 = nn.Dropout(dropouts['do62']) 

        self.conv1d_90 = nn.Conv1d(25, 35, kernel_size=3, padding=1)
        self.bn72 = nn.BatchNorm1d(35)
        self.do63 = nn.Dropout(dropouts['do63']) 
        
        self.dense18 = nn.Linear(35, 175)
        self.rhythm_out = nn.Linear(175, 2)
    
    def forward(self, x):
        # Backbone
        x = self.max_pool25(F.relu(self.conv1d_81(x)))
        x = self.bn_81(x)
        x = self.max_pool26(F.relu(self.conv1d_82(x)))
        x = self.bn_82(x)
        x = self.max_pool27(F.relu(self.conv1d_83(x)))
        x = self.bn65(x)

        # Shared processing
        x = self.do57(self.bn66(self.lrelu(self.conv1d_84(x))))
        x = self.do58(self.bn67(self.lrelu(self.conv1d_85(x))))
        shared_feat = self.do59(self.bn68(self.lrelu(self.conv1d_86(x))))

        # Rhythm Branch
        rh = self.do61(self.bn70(F.relu(self.conv1d_88(shared_feat))))
        rh = self.do62(self.bn71(F.relu(self.conv1d_89(rh))))
        rh = self.do63(self.bn72(F.relu(self.conv1d_90(rh))))
        rh = torch.flatten(rh, 1)
        rh = F.relu(self.dense18(rh))
        rhythm_out = self.rhythm_out(rh) 
        return rhythm_out
    

class balanced_DeepBeatModel(nn.Module):
    def __init__(self, dropouts=None):
        super(balanced_DeepBeatModel, self).__init__()
        
        if dropouts is None:
            dropouts = {
                'do57': 0.25, 'do58': 0.25, 'do59': 0.25,
                'do60': 0.25, 'do61': 0.25
            }
        
        # --- Common Backbone ---
        self.conv1d_81 = nn.Conv1d(1, 64, kernel_size=10, padding=5) # 'same'
        self.max_pool25 = nn.MaxPool1d(kernel_size=3, stride=3)
        self.bn_81 = nn.BatchNorm1d(64)
        
        self.conv1d_82 = nn.Conv1d(64, 45, kernel_size=8, padding=4)
        self.max_pool26 = nn.MaxPool1d(kernel_size=3, stride=3)
        self.bn_82 = nn.BatchNorm1d(45)
        
        self.conv1d_83 = nn.Conv1d(45, 50, kernel_size=5, padding=2)
        self.max_pool27 = nn.MaxPool1d(kernel_size=2, stride=2, padding=1) # 'same' padding
        
        self.bn65 = nn.BatchNorm1d(50)
        

        # --- Shared Bottleneck ---
        self.conv1d_84 = nn.Conv1d(50, 64, kernel_size=4, stride=3, padding=1)
        self.lrelu = nn.LeakyReLU(0.1) 
        self.bn66 = nn.BatchNorm1d(64)
        self.do57 = nn.Dropout(dropouts['do57']) 

        self.conv1d_85 = nn.Conv1d(64, 35, kernel_size=4, stride=3, padding=1)
        self.bn67 = nn.BatchNorm1d(35)
        self.do58 = nn.Dropout(dropouts['do58']) 

        self.conv1d_86 = nn.Conv1d(35, 64, kernel_size=4, padding=2) # Stride 1
        self.bn68 = nn.BatchNorm1d(64)
        self.do59 = nn.Dropout(dropouts['do59']) 

        # --- QA Branch (Connected to Dropout 59) ---
        self.conv1d_87 = nn.Conv1d(64, 25, kernel_size=4, stride=2, padding=1)
        self.bn69 = nn.BatchNorm1d(25)
        self.do60 = nn.Dropout(0.4) 
        self.dense17 = nn.Linear(75, 175)
        self.qa_out = nn.Linear(175, 2)

        # --- Rhythm Branch (Connected to Dropout 59) --- make it the same structure as QA branch
        self.conv1d_88 = nn.Conv1d(64, 25, kernel_size=4, stride=2, padding=1)
        self.bn70 = nn.BatchNorm1d(25)
        self.do61 = nn.Dropout(0.4) 
        self.dense18 = nn.Linear(75, 175)
        self.rhythm_out = nn.Linear(175, 2)

    def forward(self, x):
        # Backbone
        x = self.max_pool25(F.relu(self.conv1d_81(x)))
        x = self.bn_81(x)
        x = self.max_pool26(F.relu(self.conv1d_82(x)))
        x = self.bn_82(x)
        x = self.max_pool27(F.relu(self.conv1d_83(x)))
        x = self.bn65(x)

        # Shared processing
        x = self.do57(self.bn66(self.lrelu(self.conv1d_84(x))))
        x = self.do58(self.bn67(self.lrelu(self.conv1d_85(x))))
        shared_feat = self.do59(self.bn68(self.lrelu(self.conv1d_86(x))))

        # QA Branch
        qa = self.do60(self.bn69(F.relu(self.conv1d_87(shared_feat))))
        qa = torch.flatten(qa, 1)
        qa = F.relu(self.dense17(qa))
        qa_out = self.qa_out(qa) # raw logits

        # Rhythm Branch
        rh = self.do61(self.bn70(F.relu(self.conv1d_88(shared_feat))))
        rh = torch.flatten(rh, 1)
        rh = F.relu(self.dense18(rh))
        rhythm_out = self.rhythm_out(rh) 
        return qa_out, rhythm_out
    
    

    

    



        
 