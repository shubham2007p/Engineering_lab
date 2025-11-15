# 🏗️ **Anti-Swing Crane Control using PD Controller + Neural Network Auto-Tuning**

*A simulation + ML project to reduce load swing in cranes.*

---

## 📌 **1. Problem Overview**

When a crane lifts or moves a suspended load, the load behaves like a **pendulum** and starts swinging.
This swinging:

* is **unsafe**,
* reduces precision,
* stresses the crane structure,
* slows down operations.

Real industrial cranes use **Active Swing Control** to reduce oscillation.

This project recreates that system using:

1. **Physics-based pendulum model**
2. **PD Controller (Proportional + Derivative)**
3. **Neural Network that auto-selects Kp & Kd**
4. **Nonlinear simulation**
5. **Animation showing swing → stable**

---

## ⚙️ **2. Physical Model (Pendulum Dynamics)**

A 500 kg load hanging from a 12 m cable behaves like a simple pendulum.

Let:

* ( \theta ) = swing angle
* ( m ) = mass
* ( l ) = cable (arm) length
* ( u ) = motor torque at pivot

### **Full nonlinear equation:**

[
m l^2 \ddot{\theta} = -mgl\sin(\theta) + u
]

### **Linearized (small angle) form:**

[
\ddot{\theta} + \frac{g}{l}\theta = \frac{u}{ml^2}
]

---

## 🎮 **3. PD Controller for Anti-Swing**

We apply a torque:

[
u = -K_p \theta - K_d \dot{\theta}
]

This damps the swing and brings the load to rest.

### **Goal:**

Choose ( K_p ) and ( K_d ) so the load stops swinging **fast** and **without overshoot**.

---

## 📘 **4. Analytic Gain Design (Critical Damping)**

For a critically damped system:

[
K_p = m l^2 \left( \omega_{desired}^2 - \frac{g}{l} \right)
]
[
K_d = 2 m l^2 \omega_{desired}
]

Where:

* ( \omega_{desired} ) is the controller’s target speed (user-chosen).

These analytic values are used to generate training labels.

---

## 🤖 **5. Neural Network Auto-Tuning of Kp & Kd**

Instead of manually choosing gains, we train a small neural network that learns to map:

### **Inputs → Outputs**

| Inputs             | Meaning                  |
| ------------------ | ------------------------ |
| mass (m)           | load mass                |
| length (l)         | cable length             |
| initial angle (θ°) | starting swing           |
| desired ωn         | desired speed of damping |

→

| Outputs | Meaning           |
| ------- | ----------------- |
| (K_p)   | proportional gain |
| (K_d)   | derivative gain   |

### **Training Data**

We generate ~25k samples and compute **analytic critical damping gains** as labels.

The NN learns the pattern and can instantly output good gains for any crane geometry.

---

## 🧪 **6. Simulation (Nonlinear RK4)**

We simulate the full nonlinear pendulum with:

* analytic gains
* NN-predicted gains

and compare:

* angle vs time
* torque demand
* settling time
* max torque
* pendulum animation (swing → no swing)

We use **4th order Runge-Kutta (RK4)** for accuracy.

---

## 🎞️ **7. Animation Output**

The script generates a GIF:

### **`nn_vs_analytic.gif`**

This shows:

* Starting swing (e.g., 17°)
* Real-time pendulum motion
* How quickly analytic vs NN controllers remove swing

This GIF is perfect for your README or LinkedIn.

---

## 📁 **8. Repository Structure**

```
anti-swing-crane/
│
├── train_k_gains.py          # NN training script
├── animate_gains_inference.py # Sim + animation using trained model
├── gains_net.pt              # Trained model weights
├── xscaler.pkl
├── yscaler.pkl                # Normalization scalers
│
├── nn_vs_analytic.gif         # Animation output
├── analytic_vs_nn_angle.csv
├── analytic_vs_nn_torque.csv  # Results for plotting
│
└── README.md                  # Documentation (this file)
```

---

## 🧰 **9. How to Use**

### 🔹 **1. Train the model (optional — already trained)**

```
python train_k_gains.py
```

### 🔹 **2. Run the simulation and create animation**

```
python animate_gains_inference.py
```

### 🔹 **3. Change inputs** inside `animate_gains_inference.py`:

```python
mass = 500
length = 12
init_angle_deg = 17
desired_wn = 1.5
```

This will:

* Predict Kp, Kd
* Simulate swing
* Save GIF and CSV results

---

## 📈 **10. Example Result**

For a 500 kg load, 12 m cable:

```
Analytic Kp: ~102000
Analytic Kd: ~216000

NN Pred Kp: ~116000
NN Pred Kd: ~224000
```

**Settling time:**

* Analytic: ~3.0 s
* NN: ~2.9 s

**Max torque:**

* Analytic: 30k Nm
* NN: ~34k Nm

---

## 🔮 **11. Future Improvements**

* Add friction to pendulum model
* Train NN using optimization-based Kp,Kd
* Add noise → Kalman Filter
* Build Streamlit UI to predict gains
* Deploy as a small web-app
* Extend to 2D (sway + yaw) crane swing

---

## 🏁 **12. Summary**

This project replicates a **real industrial engineering problem**:

✔ Crane swing dynamics
✔ PD control
✔ Auto-tuning using neural networks
✔ Nonlinear simulation
✔ Visualization & animation
✔ Model comparison

It bridges **Machine Learning**, **Control Theory**, and **Physics Simulation**, making it a strong portfolio-ready project.

---


