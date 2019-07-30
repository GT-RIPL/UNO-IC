import torch

N, D_in, H, D_out = 64, 1000, 100, 10


# Sum Loss
torch.manual_seed(0)
x = torch.randn(N, D_in)
y = torch.randn(N, D_out)

model = torch.nn.Sequential(
    torch.nn.Linear(D_in, H),
    torch.nn.ReLU(),
    torch.nn.Linear(H, D_out),
)
loss_fn = torch.nn.MSELoss(reduction='sum')

learning_rate = 1e-4
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

y_pred = model(x)

loss = loss_fn(y_pred, y)
loss.backward()
optimizer.step()
print(model[0].weight)

# Mean Loss
torch.manual_seed(0)
x = torch.randn(N, D_in)
y = torch.randn(N, D_out)

model = torch.nn.Sequential(
    torch.nn.Linear(D_in, H),
    torch.nn.ReLU(),
    torch.nn.Linear(H, D_out),
)
loss_fn = torch.nn.MSELoss(reduction='mean')

learning_rate = 1e-4
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

y_pred = model(x)

loss = loss_fn(y_pred, y)
loss.backward()
optimizer.step()
print(model[0].weight)