import numpy as np, pandas as pd, tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split

DATA_PATH = "hamiltonian_dataset.csv"
SAVE_MODEL_PATH = "hamiltonian_tf_model.keras"

df = pd.read_csv(DATA_PATH)
X = df[["ux","uy","uth"]].values.astype("float32")
y = df["H"].values.astype("float32")
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=7)

inputs = keras.Input(shape=(3,), name="u")  # [ux, uy, uth]
ux = layers.Lambda(lambda z: z[:,0:1])(inputs)
uy = layers.Lambda(lambda z: z[:,1:2])(inputs)
uth= layers.Lambda(lambda z: z[:,2:3])(inputs)

ux2 = layers.Multiply()([ux, ux])
uy2 = layers.Multiply()([uy, uy])
v2  = layers.Add()([ux2, uy2])     # ux^2 + uy^2
w2  = layers.Multiply()([uth, uth])  # uth^2

features = layers.Concatenate(name="phys_features")([v2, w2])  # (N,2)

# H = a * (ux^2 + uy^2) + b * (uth^2)  con a,b >= 0
out = layers.Dense(
    1, use_bias=False, name="H",
    kernel_constraint=keras.constraints.NonNeg()
)(features)

model = keras.Model(inputs, out, name="hamiltonian_physlinear_nomass")
model.compile(
    optimizer=keras.optimizers.Adam(1e-3), loss="mse",
    metrics=[tf.keras.metrics.RootMeanSquaredError(name="rmse"),
             tf.keras.metrics.MeanAbsoluteError(name="mae")]
)

model.fit(
    Xtr, ytr, validation_data=(Xte, yte),
    epochs=200, batch_size=1024,
    callbacks=[
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=8, min_lr=1e-6),
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True),
        keras.callbacks.ModelCheckpoint(SAVE_MODEL_PATH, monitor="val_loss", save_best_only=True),
    ],
    verbose=2
)

W = model.get_layer("H").get_weights()[0].ravel()
print(f"Pesos aprendidos: a≈{W[0]:.6f}, b≈{W[1]:.6f}")
print(f"[OK] Modelo guardado en: {SAVE_MODEL_PATH}")
