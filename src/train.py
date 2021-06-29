import torch
from tqdm import tqdm


class Model(torch.nn.Module):
    def __init__(self, num_features, layer_1, layer_2, layer_3):
        super(Model, self).__init__()
        self.lin1 = torch.nn.Linear(num_features, layer_1)
        self.lin2 = torch.nn.Linear(layer_1, layer_2)
        self.lin3 = torch.nn.Linear(layer_2, layer_3)
        self.lin4 = torch.nn.Linear(layer_3, 1)
        self.selu = torch.nn.SELU()

    def forward(self, x):
        x = self.selu(self.lin1(x))
        x = self.selu(self.lin2(x))
        x = self.selu(self.lin3(x))
        x = self.lin4(x)
        return x


def train_model(num_features, device, x_train, y_train):
    model = Model(num_features, 197, 198, 112)
    criterion = torch.nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([14.80], device=device))
    optimizer = torch.optim.SGD(model.parameters(
    ), lr=0.03104, weight_decay=0, momentum=0.4204, nesterov=True)
    # weight_decay=0.01043
    model.train()
    model.to(device)
    no_epochs = 127
    for epoch in tqdm(range(no_epochs), desc="Training"):
        optimizer.zero_grad()
        outputs = model.forward(x_train)
        loss = criterion(outputs, y_train.view(-1, 1))
        loss.backward()
        optimizer.step()
    return model
