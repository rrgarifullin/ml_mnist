import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms


def train_model(model):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # загрузите данные
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.1307,), (0.3081,))])  # преобразование данных

    train_set = datasets.MNIST('data', train=True, download=True,
                               transform=transform)  # ссылка на данные для тренировки
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)

    test_set = datasets.MNIST('data', train=False, download=True,
                              transform=transform)  # ссылка на данные для тестирования
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=64, shuffle=True)

    n_epochs = 10

    for epoch in range(n_epochs):
        train_loss = 0.0
        model.train()
        for data, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * data.size(0)

        train_loss = train_loss / len(train_loader.dataset)
        print('Epoch: {} \tTraining Loss: {:.6f}'.format(epoch + 1, train_loss))

        # оценка модели на тестовых данных
        model.eval()
        test_loss = 0.0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                output = model(data)
                loss = criterion(output, target)
                test_loss += loss.item() * data.size(0)
                _, pred = torch.max(output, 1)
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss = test_loss / len(test_loader.dataset)
        test_accuracy = correct / len(test_loader.dataset)
        print('Epoch: {} \tTest Loss: {:.6f} \tTest Accuracy: {:.2f}%'.format(epoch + 1, test_loss, test_accuracy * 100))

    torch.save(model.state_dict(), 'weights/my_model_weights.pt')