import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from torch.nn.parallel import DataParallel

#---------------------------------------------------model------------------------------------------------------------------

class mutil_head_attention(nn.Module):
    def __init__(self,head = 8,conv = 32):
        super(mutil_head_attention,self).__init__()
        self.conv = conv
        self.head = head
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.drug_a = nn.Linear(self.conv * 3, self.conv * 3 * head)
        self.protein_a = nn.Linear(self.conv * 3, self.conv * 3 * head)
        self.scale = nn.Parameter(torch.sqrt(torch.FloatTensor([self.conv * 3])), requires_grad = False)

    def forward(self, drug, protein):
        batch_size, drug_c, drug_l = drug.shape
        batch_size, protein_c, protein_l = protein.shape
        drug_att = self.relu(self.drug_a(drug.permute(0, 2, 1))).view(batch_size,self.head,drug_l,drug_c)
        protein_att = self.relu(self.protein_a(protein.permute(0, 2, 1))).view(batch_size,self.head,protein_l,protein_c)
        interaction_map = torch.mean(self.tanh(torch.matmul(drug_att, protein_att.permute(0, 1, 3, 2)) / self.scale),1)
        Compound_atte = self.tanh(torch.sum(interaction_map, 2)).unsqueeze(1)
        Protein_atte = self.tanh(torch.sum(interaction_map, 1)).unsqueeze(1)
        drug = drug * Compound_atte
        protein = protein * Protein_atte
        return drug, protein

class DTI_LEKG(nn.Module):
    def __init__(self, protein_MAX_LENGH = 1200, protein_kernel = [4,8,12],
            drug_MAX_LENGH = 100, drug_kernel = [4,6,8], kg_dim = 400,
            conv = 64, char_dim = 128,head_num = 8,dropout_rate = 0.1):
        super(DTI_LEKG, self).__init__()
        self.protein_kernel = protein_kernel
        self.drug_kernel = drug_kernel

        self.dim = char_dim
        self.conv = conv
        self.dropout_rate = dropout_rate
        self.head_num = head_num
        self.drug_MAX_LENGH = drug_MAX_LENGH
        self.protein_MAX_LENGH = protein_MAX_LENGH
        self.kg_dim = kg_dim

        self.protein_embed = nn.Embedding(26, self.dim,padding_idx=0)
        self.drug_embed = nn.Embedding(65, self.dim,padding_idx=0)

        self.Drug_CNNs = nn.Sequential(
            nn.Conv1d(in_channels=self.dim, out_channels= self.conv,  kernel_size=self.drug_kernel[0]),
            nn.ReLU(),
            nn.Conv1d(in_channels=self.conv, out_channels= self.conv*2,  kernel_size=self.drug_kernel[1]),
            nn.ReLU(),
            nn.Conv1d(in_channels=self.conv*2, out_channels= self.conv*3,  kernel_size=self.drug_kernel[2]),
            nn.ReLU(),
        )

        self.Drug_max_pool_0 = nn.MaxPool1d(self.drug_MAX_LENGH-self.drug_kernel[0]-self.drug_kernel[1]-self.drug_kernel[2]+3)
        self.Drug_max_pool_1 = nn.AvgPool1d(192)

        self.Protein_CNNs = nn.Sequential(
            nn.Conv1d(in_channels=self.dim, out_channels=self.conv, kernel_size=self.protein_kernel[0]),
            nn.ReLU(),
            nn.Conv1d(in_channels=self.conv, out_channels=self.conv * 2, kernel_size=self.protein_kernel[1]),
            nn.ReLU(),
            nn.Conv1d(in_channels=self.conv * 2, out_channels=self.conv * 3, kernel_size=self.protein_kernel[2]),
            nn.ReLU(),
        )
        self.Protein_max_pool_0 = nn.MaxPool1d(self.protein_MAX_LENGH - self.protein_kernel[0] - self.protein_kernel[1] - self.protein_kernel[2] + 3)
        self.Protein_max_pool_1 = nn.AvgPool1d(192)

        self.attention = mutil_head_attention(head = self.head_num, conv=self.conv)
        
        self.dropout = nn.Dropout(self.dropout_rate)
        self.dropout_kg = nn.Dropout(self.dropout_rate)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.leaky_relu = nn.LeakyReLU()

        self.fc_drug_conv_0 = nn.Linear(self.conv * 3, self.conv * 3 )
        self.fc_target_conv_0 = nn.Linear(self.conv * 3 , self.conv * 3 )

        self.fc_drug_conv_1 = nn.Linear(85, self.conv * 3 )
        self.fc_target_conv_1 = nn.Linear(1179, self.conv * 3 )

        self.fc1 = nn.Linear(1568, 1024)
        self.dropout1 = nn.Dropout(self.dropout_rate)
        self.fc2 = nn.Linear(1024, 1024)
        self.dropout2 = nn.Dropout(self.dropout_rate)
        self.fc3 = nn.Linear(1024, 512)
        self.dropout3 = nn.Dropout(self.dropout_rate)

        self.out = nn.Linear(512, 1)
        torch.nn.init.constant_(self.out.bias, 5)

    def forward(self, drug_smile, drug_kg_embedding, target_sequence, target_kg_embedding):

        drugembed = self.drug_embed(drug_smile)
        proteinembed = self.protein_embed(target_sequence)
        drugembed = drugembed.permute(0, 2, 1)
        proteinembed = proteinembed.permute(0, 2, 1)

        drugConv = self.Drug_CNNs(drugembed)
        proteinConv = self.Protein_CNNs(proteinembed)

        drugConv_0 ,proteinConv_0 = self.attention(drugConv,proteinConv)

        drugConv_1 = drugConv_0.permute(0, 2, 1)
        proteinConv_1 = proteinConv_0.permute(0, 2, 1)

        drugConv_0 = self.Drug_max_pool_0(drugConv_0).squeeze(2)
        proteinConv_0 = self.Protein_max_pool_0(proteinConv_0).squeeze(2)

        drugConv_1 = self.Drug_max_pool_1(drugConv_1).squeeze(2)
        proteinConv_1 = self.Protein_max_pool_1(proteinConv_1).squeeze(2)

        drugConv_0 = self.fc_drug_conv_0(drugConv_0)
        drugConv_0 = self.dropout_kg(drugConv_0)
        proteinConv_0 = self.fc_target_conv_0(proteinConv_0)
        proteinConv_0 = self.dropout_kg(proteinConv_0)

        drugConv_1 = self.fc_drug_conv_1(drugConv_1)
        drugConv_1 = self.dropout_kg(drugConv_1)
        proteinConv_1 = self.fc_target_conv_1(proteinConv_1)
        proteinConv_1 = self.dropout_kg(proteinConv_1)

        all_info = torch.cat([drugConv_0, drugConv_1, drug_kg_embedding, proteinConv_0, proteinConv_1, target_kg_embedding], dim=1)

        fully1 = self.leaky_relu(self.fc1(all_info))
        fully1 = self.dropout1(fully1)
        fully2 = self.leaky_relu(self.fc2(fully1))
        fully2 = self.dropout2(fully2)
        fully3 = self.leaky_relu(self.fc3(fully2))
        fully3 = self.dropout3(fully3)
        predict = self.out(fully3)
        return predict
    
#-------------------------------------------------dataset------------------------------------------------------------

#----------------------------------------Test Custom Dataset-------------------------------------------------

class Custom_Test_Dataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        input1, input2, input3, input4 = self.data[idx]
        input1 = torch.tensor(input1)
        input2 = torch.tensor(input2, dtype=torch.float)
        input3 = torch.tensor(input3)
        input4 = torch.tensor(input4, dtype=torch.float)
        return input1, input2, input3, input4

#---------------------------------------Training Custom Dataset-----------------------------------------------

class Custom_Train_Dataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        input1, input2, input3, input4, label = self.data[idx]
        input1 = torch.tensor(input1)
        input2 = torch.tensor(input2, dtype=torch.float)
        input3 = torch.tensor(input3)
        input4 = torch.tensor(input4, dtype=torch.float)
        label = torch.tensor(label, dtype=torch.float)
        return input1, input2, input3, input4, label

#--------------------------------------Training prediction Model-----------------------------------------------

def calculate_average_loss(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for input1, input2, input3, input4, labels in data_loader:
            input1 = input1.to(device)
            input2 = input2.to(device)
            input3 = input3.to(device)
            input4 = input4.to(device)
            labels = labels.to(device)
            outputs = model(input1, input2, input3, input4)
            outputs = outputs.squeeze()
            loss = criterion(outputs, labels)
            total_loss += loss.item() * len(input1)
            total_samples += len(input1)

    return total_loss / total_samples


def Train_Prediction_Model(input_data, kg_dim = 96, num_epochs = 1000, batch_size = 512, device_ids = 'cuda:0', save_path = 'prediction_model/bilinear_model.pth', log_filename = 'log/training_log.txt', early_stop = True, early_stop_threshold=0.0001, lr = 0.001):
    gpu_ids = device_ids.split(',')
    final_device_ids = []
    for item in gpu_ids:
        final_device_ids.append(item)
    custom_dataset = Custom_Train_Dataset(input_data)
    train_loader = DataLoader(custom_dataset, batch_size = batch_size, shuffle=True)

    model = DTI_LEKG(kg_dim = kg_dim)

    """weight initialize"""
    weight_p, bias_p = [], []
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    for name, p in model.named_parameters():
        if 'bias' in name:
            bias_p += [p]
        else:
            weight_p += [p]
    """load trained model"""
    
    if len(final_device_ids) > 1:
        model = DataParallel(model, device_ids=final_device_ids)
        model = model.to(final_device_ids[0])
    else:
        model = model.to(final_device_ids[0]) 
    
    criterion = nn.BCEWithLogitsLoss()
    #criterion = nn.CrossEntropyLoss()   maybe try CrossEntropyLoss
    
    optimizer = optim.Adam(model.parameters(), lr=lr)

    clip_value = 1.0
    #best_loss = float('inf')

    if len(final_device_ids) == 1:#use a single gpu
        print("training with the following gpu:",final_device_ids)
        for epoch in tqdm(range(num_epochs)):
            #with tqdm(total=len(train_loader), desc=f'Epoch {epoch + 1}/{num_epochs}', unit='batch') as pbar:
            for i, (drug_smile, drug_kg_embedding, target_sequence, target_kg_embedding, labels) in enumerate(train_loader):
                #drug_smile, drug_embedding, target_sequence, target_embedding
                drug_smile = drug_smile.cuda(final_device_ids[0])
                drug_kg_embedding = drug_kg_embedding.cuda(final_device_ids[0])
                target_sequence = target_sequence.cuda(final_device_ids[0])
                target_kg_embedding = target_kg_embedding.cuda(final_device_ids[0])
                labels = labels.cuda(final_device_ids[0])

                optimizer.zero_grad()

                # Forward pass
                outputs = model(drug_smile, drug_kg_embedding, target_sequence, target_kg_embedding)
                loss = criterion(outputs.squeeze(), labels)
               
                # Backward pass and optimization
                loss.backward()
                
                # Clip gradients to prevent exploding gradients
                nn.utils.clip_grad_norm_(model.parameters(), clip_value)
                
                optimizer.step()
                #if (i + 1) % 1 == 0:
                #    print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(train_loader)}], Loss: {loss.item():.4f}')
                    #logging.info(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(train_loader)}], Loss: {loss.item():.4f}')
            
            # Calculate average loss for the epoch
            average_loss = calculate_average_loss(model, train_loader, criterion, device = final_device_ids[0])
    
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {average_loss}')
            #logging.info(f'Epoch [{epoch+1}/{num_epochs}], Loss: {average_loss}')

            # Check for early stopping
            if average_loss < early_stop_threshold and early_stop == True:
                print(f'Loss is below the early stopping threshold. Training stopped.')
                break
    else:#multiple gpu
        print("training with the following gpus:",final_device_ids)
        for epoch in tqdm(range(num_epochs)):
            #with tqdm(total=len(train_loader), desc=f'Epoch {epoch + 1}/{num_epochs}', unit='batch') as pbar:
            for i, (drug_smile, drug_kg_embedding, target_sequence, target_kg_embedding, labels) in enumerate(train_loader):
                #drug_smile, drug_embedding, target_sequence, target_embedding
                drug_smile = drug_smile.cuda(final_device_ids[0])
                drug_kg_embedding = drug_kg_embedding.cuda(final_device_ids[0])
                target_sequence = target_sequence.cuda(final_device_ids[0])
                target_kg_embedding = target_kg_embedding.cuda(final_device_ids[0])
                labels = labels.cuda(final_device_ids[0])

                optimizer.zero_grad()

                # Forward pass
                outputs = model(drug_smile, drug_kg_embedding, target_sequence, target_kg_embedding)
                loss = criterion(outputs.squeeze(), labels)
               
                # Backward pass and optimization
                loss.backward()
                
                # Clip gradients to prevent exploding gradients
                nn.utils.clip_grad_norm_(model.parameters(), clip_value)
                
                optimizer.step()
            
            # Calculate average loss for the epoch
            average_loss = calculate_average_loss(model, train_loader, criterion, device = final_device_ids[0])
    
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {average_loss}')
            #logging.info(f'Epoch [{epoch+1}/{num_epochs}], Loss: {average_loss}')

            # Check for early stopping
            if average_loss < early_stop_threshold and early_stop == True:
                print(f'Loss is below the early stopping threshold. Training stopped.')
                break

    print('Training finished!')
    torch.save(model.state_dict(), save_path)
    print('save model succesuffly')

def Attention_prediction(Test_data, kg_dim = 100, load_path = 'prediction_model/bilinear_MLP.pth', batch_size = 128, sigmoid_transform = True , device = 'cuda:0'):
    print('start loading data')

    model = DTI_LEKG(kg_dim = kg_dim)

    state_dict = torch.load(load_path, map_location=device)  # Load the state dictionary

    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

    model.load_state_dict(state_dict)

    model = model.to(device)

    model.eval()

    print('prediction load succesfully')

    Test_dataset = Custom_Test_Dataset(Test_data)

    Test_loader = DataLoader(Test_dataset, batch_size = batch_size, shuffle = False)

    all_outputs = []

    with torch.no_grad():

        for i, (drug_smile, drug_kg_embedding, target_sequence, target_kg_embedding) in enumerate(tqdm(Test_loader, desc='Inference')):
            drug_smile = drug_smile.cuda(device)
            drug_kg_embedding = drug_kg_embedding.cuda(device)
            target_sequence = target_sequence.cuda(device)
            target_kg_embedding = target_kg_embedding.cuda(device)
            logits = model(drug_smile, drug_kg_embedding, target_sequence, target_kg_embedding)
            if sigmoid_transform:
                logits = torch.sigmoid(logits)
            outputs_list = logits.tolist()

            all_outputs.extend(outputs_list)
    results = []

    for item in all_outputs:

        results.append(item[0])
        
    return results
    