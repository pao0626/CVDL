VGG[V1]
optimizer = optim.Adam(model.parameters(), lr=0.0001)
scheduler = StepLR(optimizer, step_size=5, gamma=0.1)

VGG[V2]
optimizer = optim.Adam(model.parameters(), lr=0.00005)
scheduler = StepLR(optimizer, step_size=5, gamma=0.1)

ResNet[V1]
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = StepLR(optimizer, step_size=5, gamma=0.1)

ResNet[V2]