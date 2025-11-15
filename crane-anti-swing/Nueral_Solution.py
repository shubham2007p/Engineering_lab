# train_k_gains.py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
import joblib

# ---------- 1) dataset generation (analytic labels for critical damping) ----------
def generate_dataset(n_samples=20000, seed=0):
    rng = np.random.default_rng(seed)
    # sample ranges (reasonable ranges for crane-like systems)
    m = rng.uniform(50, 1000, n_samples)      # kg
    l = rng.uniform(1.0, 20.0, n_samples)     # m
    theta0 = rng.uniform(0.0, 30.0, n_samples) # deg initial swing
    wn_cl = rng.uniform(0.5, 3.0, n_samples)  # desired closed-loop natural freq (rad/s)
    g = 9.81

    # compute I = m l^2
    I = m * l**2

    # analytic Kp, Kd for critical damping:
    # Kp = I*(wn_cl^2 - g/l)
    # Kd = 2 * I * wn_cl
    Kp = I * (wn_cl**2 - g / l)
    Kd = 2.0 * I * wn_cl

    # ensure non-negative Kp (if wn_cl^2 < g/l then set Kp small positive)
    Kp = np.clip(Kp, a_min=0.0, a_max=None)
    Kd = np.clip(Kd, a_min=0.0, a_max=None)

    X = np.column_stack([m, l, theta0, wn_cl])
    Y = np.column_stack([Kp, Kd])
    return X.astype(np.float32), Y.astype(np.float32)

# ---------- 2) small regression network ----------
class GainsNet(nn.Module):
    def __init__(self, inp=4, out=2, hidden=[128, 64]):
        super().__init__()
        layers = []
        prev = inp
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        layers += [nn.Linear(prev, out)]
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x)

# ---------- 3) training pipeline ----------
def train_model(X, Y, epochs=60, batch=256, lr=1e-3, device='cpu'):
    # scale inputs and targets (log-scale targets help)
    xscaler = StandardScaler().fit(X)
    Xs = xscaler.transform(X)

    # transform Y: use log(1+Y) to stabilize training for big magnitudes
    Ylog = np.log1p(Y)
    yscaler = StandardScaler().fit(Ylog)
    Ys = yscaler.transform(Ylog)

    # to tensors
    Xt = torch.from_numpy(Xs).float()
    Yt = torch.from_numpy(Ys).float()

    dataset = TensorDataset(Xt, Yt)
    loader = DataLoader(dataset, batch_size=batch, shuffle=True, drop_last=True)

    model = GainsNet().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    for ep in range(1, epochs+1):
        model.train()
        total_loss = 0.0
        for xb, yb in loader:
            xb = xb.to(device); yb = yb.to(device)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            opt.zero_grad(); loss.backward(); opt.step()
            total_loss += loss.item()
        if ep % 10 == 0 or ep==1:
            print(f"Epoch {ep:03d} loss={total_loss/len(loader):.6f}")
    return model, xscaler, yscaler

# ---------- 4) inference helper ----------
def predict_gains(model, xscaler, yscaler, inputs, device='cpu'):
    # inputs: array shape (N,4) -> (m, l, theta0_deg, wn_cl)
    Xs = xscaler.transform(np.array(inputs, dtype=np.float32))
    Xt = torch.from_numpy(Xs).float().to(device)
    model.eval()
    with torch.no_grad():
        ypred_scaled = model(Xt).cpu().numpy()
    # invert scaling and log transform
    Ylog = yscaler.inverse_transform(ypred_scaled)
    Y = np.expm1(Ylog)  # invert log1p
    # enforce non-negativity
    Y = np.clip(Y, 0.0, None)
    return Y

# ---------- 5) run everything ----------
if __name__ == "__main__":
    X, Y = generate_dataset(25000, seed=42)
    model, xscaler, yscaler = train_model(X, Y, epochs=80, batch=512, lr=1e-3)
    # save model + scalers
    torch.save(model.state_dict(), "gains_net.pt")
    joblib.dump(xscaler, "xscaler.pkl")
    joblib.dump(yscaler, "yscaler.pkl")
    print("Saved model and scalers.")

    # quick test
    test_in = np.array([[500.0, 12.0, 17.0, 1.5]])  # your crane sample
    pred = predict_gains(model, xscaler, yscaler, test_in)
    print("Predicted Kp, Kd:", pred[0])


    

