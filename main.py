import torch as tc
import torchvision as tv
from classifier import SmallConvNetClassifier

training_data = tv.datasets.CIFAR10(
    root="data", train=True, download=True, transform=tv.transforms.Compose(
        [tv.transforms.RandomHorizontalFlip(p=0.5), tv.transforms.ToTensor()]
    )
)

# Download test data from open datasets.
test_data = tv.datasets.CIFAR10(
    root="data", train=False, download=True, transform=tv.transforms.ToTensor())

batch_size = 64

# Create data loaders.
train_dataloader = tc.utils.data.DataLoader(training_data, batch_size=batch_size, shuffle=True)
test_dataloader = tc.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True)


# Get cpu or gpu device for training.
device = "cuda" if tc.cuda.is_available() else "cpu"
print("Using {} device".format(device))

model = SmallConvNetClassifier(
    img_height=32, img_width=32, img_channels=3,
    num_filters=64, num_classes=10).to(device)
print(model)

try:
    model.load_state_dict(tc.load("model.pth"))
    print('successfully reloaded checkpoint. continuing training...')
except Exception:
    print('no checkpoint found. training from scratch...')

loss_fn = tc.nn.CrossEntropyLoss()
optimizer = tc.optim.Adam(model.parameters(), lr=1e-3)

def train(dataloader, model, loss_fn, optimizer):
    num_training_examples = len(dataloader.dataset)

    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss = loss.item()
            current_idx = batch_size * (batch-1) + len(X)
            print(f"loss: {loss:>7f}  [{current_idx:>5d}/{num_training_examples:>5d}]")

def test(dataloader, model):
    num_test_examples = len(dataloader.dataset)
    model.eval()
    test_loss, correct = 0, 0
    with tc.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += len(X) * loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(tc.float).sum().item()
    test_loss /= num_test_examples
    correct /= num_test_examples
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

epochs = 5
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model)
print("Done!")

tc.save(model.state_dict(), "model.pth")
print("Saved PyTorch Model State to model.pth")




