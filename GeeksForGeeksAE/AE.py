import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib as plt
import torch.utils.data.dataloader as DataLoader

tenser_transform = transforms.ToTensor()

dataset = datasets.MNIST(root="./data?", train = True, downlaod=True, transform=tenser_transform)

loader = DataLoader(dataset=dataset, batch_size =32, shuffle=True)



class AE(nn.Module):
    def __init__(self):
        super(AE, self).__innit()
        
        self.encoder = nn.Sequential(
            #define layer sizes and activation function
            nn.Linear(28*28, 128),
            nn.ReLU(),
            nn.Linear(128,64),
            nn.ReLU(),
            nn.Linear(64,36),
            nn.ReLU(),
            nn.Linear(36,18),
            nn.ReLU(),
            nn.Linear(18,9)
        )

        self.decoder = nn.Sequential(
            #define layer sizes and activation function
            nn.Linear(9, 18),
            nn.ReLU(),
            nn.Linear(18, 36),
            nn.ReLU(),
            nn.Linear(36, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 28 * 28),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    


def train(e, m, loss_fn, o):
    epochs = e
    outputs = []
    losses = []
    model = m
    loss_function = loss_fn
    optimizer = o

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)


    for epoch in range(epochs):
        for images, _ in loader:
            images - images.view(-1, 28*28).to(device)

            reconstructed = model(images)

            loss = loss_function(reconstructed, images)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss.item())

        outputs.append((epoch, images, reconstructed))
        print(f"epoch {epoch+1}/{epochs}, Loss: {loss.item():.6f}")


    plt.style.use('fivethirtyeight')
    plt.figure(figsize=(8, 5))
    plt.plot(losses, label='Loss')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()



    model.eval()
    dataiter = iter(loader)
    images, _ = next(dataiter)

    images = images.view(-1, 28 * 28).to(device)
    reconstructed = model(images)

    fig, axes = plt.subplots(nrows=2, ncols=10, figsize=(10, 3))
    for i in range(10):
        axes[0, i].imshow(images[i].cpu().detach().numpy().reshape(28, 28), cmap='gray')
        axes[0, i].axis('off')
        axes[1, i].imshow(reconstructed[i].cpu().detach().numpy().reshape(28, 28), cmap='gray')
        axes[1, i].axis('off')
    plt.show()


def main():
    #instantiate model
    model = AE()
    loss_function = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-8)


    train(20, model, loss_function, optimizer)
