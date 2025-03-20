import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataLoader.data_loader_class import SingleCellDataset, collate_csr_tensors
from DAE.dae import DeepAutoEncoder

def main():
    #Initialize dataset and DataLaoder
    data_dir = '../../july2024_census_data/full/'#TODO set up arguments to pass directly?, imagine from parent dir??


    print("creating human_dataset")
    human_dataset = SingleCellDataset(data_dir, species="human")

    print("creating human_loader")
    human_loader = DataLoader(human_dataset, batch_size=128, shuffle=True, collate_fn=collate_csr_tensors)
    print("Finished creating dataloader")

    #get inputsize from sample in dataset 
    sample_input, _ = human_dataset[0]
    input_size = sample_input.shape[0]


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DeepAutoEncoder(input_size).to(device)
    optimizer = optim.Adam(model.parameters(), lr = 0.001)
    criterion = nn.MSELoss() #TODO confirm this is desired criterion

    writer = SummaryWriter()

    epochs = 10

    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch, _ in human_loader:
            batch = batch.to(device)
            batch = batch.to_dense() #just dont bother with CSR
            optimizer.zero_grad()
            outputs = model(batch)
            loss = criterion(outputs,batch)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()


        avg_loss = epoch_loss / len(human_loader)
        writer.add_scalar("Loss/train", avg_loss, epoch)
        print(f"Epoch {epoch +1}/{epochs}, Loss : {avg_loss:.6f}")

    torch.save(model.state_dict(), "dae_model.pth")
    writer.close()

if __name__ == "__main__":
    main()