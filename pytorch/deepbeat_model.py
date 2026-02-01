import torch
import torch.nn as nn
import torch.nn.functional as F

class DeepBeat(nn.Module):
    def __init__(self):
        super(DeepBeat, self).__init__()
        
        # --- Common Backbone ---
        self.conv1d_81 = nn.Conv1d(1, 64, kernel_size=10, padding=5) # 'same'
        self.max_pool25 = nn.MaxPool1d(kernel_size=3, stride=3)
        
        self.conv1d_82 = nn.Conv1d(64, 45, kernel_size=8, padding=4)
        self.max_pool26 = nn.MaxPool1d(kernel_size=3, stride=3)
        
        self.conv1d_83 = nn.Conv1d(45, 50, kernel_size=5, padding=2)
        self.max_pool27 = nn.MaxPool1d(kernel_size=2, stride=2, padding=1) # 'same' padding
        
        self.bn65 = nn.BatchNorm1d(50)
        

        # --- Shared Bottleneck ---
        self.conv1d_84 = nn.Conv1d(50, 64, kernel_size=4, stride=3, padding=1)
        self.lrelu = nn.LeakyReLU(0.1) # File uses Alpha 0.1 [cite: 91]
        self.bn66 = nn.BatchNorm1d(64)
        self.do57 = nn.Dropout(0.118) # Specific rate 

        self.conv1d_85 = nn.Conv1d(64, 35, kernel_size=4, stride=3, padding=1)
        self.bn67 = nn.BatchNorm1d(35)
        self.do58 = nn.Dropout(0.545) # Specific rate 

        self.conv1d_86 = nn.Conv1d(35, 64, kernel_size=4, padding=2) # Stride 1
        self.bn68 = nn.BatchNorm1d(64)
        self.do59 = nn.Dropout(0.568) # Specific rate 

        # --- QA Branch (Connected to Dropout 59) ---
        self.conv1d_87 = nn.Conv1d(64, 25, kernel_size=4, stride=2, padding=1)
        self.bn69 = nn.BatchNorm1d(25)
        self.do60 = nn.Dropout(0.672) # Specific rate [cite: 23549]
        self.dense17 = nn.Linear(75, 175)
        self.qa_out = nn.Linear(175, 3)

        # --- Rhythm Branch (Connected to Dropout 59) ---
        self.conv1d_88 = nn.Conv1d(64, 35, kernel_size=5, stride=3, padding=2)
        self.bn70 = nn.BatchNorm1d(35)
        self.do61 = nn.Dropout(0.374) # Specific rate [cite: 23547]
        
        self.conv1d_89 = nn.Conv1d(35, 25, kernel_size=4, stride=3, padding=2)
        self.bn71 = nn.BatchNorm1d(25)
        self.do62 = nn.Dropout(0.414) # Specific rate [cite: 23548]

        self.conv1d_90 = nn.Conv1d(25, 35, kernel_size=3, padding=1)
        self.bn72 = nn.BatchNorm1d(35)
        self.do63 = nn.Dropout(0.017) # Specific rate [cite: 23549]
        
        self.dense18 = nn.Linear(35, 175)
        self.rhythm_out = nn.Linear(175, 2)

    def forward(self, x):
        # Backbone
        x = self.max_pool25(F.relu(self.conv1d_81(x)))
        x = self.max_pool26(F.relu(self.conv1d_82(x)))
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
        rhythm_out = self.rhythm_out(rh) # uses nn.BCEWithLogitsLoss for loss, which uses sigmoid() for activation
        return qa_out, rhythm_out
 
    def check_forward_shape(self,x):
        """ print out output shape at each stage

        Args:
            x (array): size (Batch, 1, 800)
         
        """
        
        # Backbone
        x = self.max_pool25(F.relu(self.conv1d_81(x)))
        print(f"After Block 1 (Conv81 + Pool25): {x.shape}")
        x = self.max_pool26(F.relu(self.conv1d_82(x)))
        print(f"After Block 2 (Conv82 + Pool26): {x.shape}")
        x = self.max_pool27(F.relu(self.conv1d_83(x)))
        print(f"After Block 3 (Conv83 + Pool27): {x.shape}")
        x = self.bn65(x)
    
        # Shared processing
        x = self.do57(self.bn66(self.lrelu(self.conv1d_84(x))))
        print(f"After Bottleneck 1 (Conv84): {x.shape}")
        x = self.do58(self.bn67(self.lrelu(self.conv1d_85(x))))
        print(f"After Bottleneck 2 (Conv85): {x.shape}")
        shared_feat = self.do59(self.bn68(self.lrelu(self.conv1d_86(x))))
        print(f"Shared Features (Conv86): {shared_feat.shape}")

        # QA Branch
        qa = self.do60(self.bn69(F.relu(self.conv1d_87(shared_feat))))
        print(f"QA Branch - After Conv87: {qa.shape}")
        qa = torch.flatten(qa, 1)
        print(f"QA Branch - Flattened: {qa.shape}")
        qa = F.relu(self.dense17(qa))
        print(f"QA Branch - Dense: {qa.shape}")
        qa_out = self.qa_out(qa) # raw logits
        print(f"QA Final Output: {qa_out.shape}")
    
        # Rhythm Branch
        rh = self.do61(self.bn70(F.relu(self.conv1d_88(shared_feat))))
        print(f"Rhythm Branch - After Conv88: {rh.shape}")
        rh = self.do62(self.bn71(F.relu(self.conv1d_89(rh))))
        print(f"Rhythm Branch - After Conv89: {rh.shape}")
        rh = self.do63(self.bn72(F.relu(self.conv1d_90(rh))))
        print(f"Rhythm Branch - After Conv90: {rh.shape}")
        rh = torch.flatten(rh, 1)
        print(f"Rhythm Branch - Flattened: {rh.shape}")
        rh = F.relu(self.dense18(rh))
        print(f"Rhythm Branch - Dense: {rh.shape}")
        rhythm_out = self.rhythm_out(rh) 
        print(f"Rhythm Final Output: {rhythm_out.shape}")
        
        
 