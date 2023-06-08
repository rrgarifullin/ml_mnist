import torch
from model import Net

model = Net()
model.load_state_dict(torch.load("weights/my_model_weights.pt"))
model.eval()


def predict(image):
    image = image.unsqueeze(0)
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output.data, 1)
        return predicted.item()