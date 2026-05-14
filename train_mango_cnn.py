"""
train_mango_cnn.py
===================
Train a MobileNetV2 CNN to detect mango variety (Alphonso / Imam Pasand)
and health status (Healthy / Diseased) — 4 output classes.

Classes:
    Alphonso_Healthy
    Alphonso_Diseased
    ImamPasand_Healthy
    ImamPasand_Diseased

Dataset folder structure (create before running):
    D:/Agri_Assist/MangoDataset/
        Alphonso_Healthy/      ← 50–200 photos of healthy Alphonso leaves/fruits
        Alphonso_Diseased/     ← 50–200 photos of diseased Alphonso
        ImamPasand_Healthy/    ← 50–200 photos of healthy Imam Pasand
        ImamPasand_Diseased/   ← 50–200 photos of diseased Imam Pasand

Photo sources (free):
    • Your own orchard photos (best!)
    • Kaggle Mango Leaf Disease Dataset:
        https://www.kaggle.com/datasets/warcoder/mango-leaf-disease-dataset
    • Google Images (search each variety name + "leaf" or "disease")

Run:
    pip install tensorflow pillow numpy matplotlib scikit-learn
    python train_mango_cnn.py

Output files (saved in D:/Agri_Assist/):
    mango_variety_model.h5        ← main model (loaded by MangoVarietyDetector)
    mango_class_indices.json      ← class index mapping
    mango_training_history.png    ← accuracy/loss curves

Training time estimate:
    GPU (NVIDIA RTX):  ~5–8 min
    CPU only:          ~40–60 min
    Google Colab GPU:  ~4–5 min   (recommended if no GPU locally)
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless — no display required
import matplotlib.pyplot as plt
from pathlib import Path

# ── TensorFlow / Keras ────────────────────────
try:
    import tensorflow as tf
    from tensorflow.keras import layers, models, optimizers, callbacks
    from tensorflow.keras.applications import MobileNetV2
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    print(f"✅ TensorFlow {tf.__version__} detected")
except ImportError:
    raise SystemExit("❌ TensorFlow not found. Run: pip install tensorflow")

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
DATASET_DIR   = "MangoDataset"          # relative to script location
OUTPUT_MODEL  = "mango_variety_model.h5"
OUTPUT_IDX    = "mango_class_indices.json"
OUTPUT_PLOT   = "mango_training_history.png"

IMAGE_SIZE    = (224, 224)
BATCH_SIZE    = 16      # lower if RAM is limited; increase for GPU
EPOCHS        = 30      # early stopping will cut this short if needed
FINE_TUNE_AT  = 100     # unfreeze MobileNetV2 layers from this index onward
LEARNING_RATE = 1e-4
SEED          = 42

EXPECTED_CLASSES = [
    "Alphonso_Diseased",
    "Alphonso_Healthy",
    "ImamPasand_Diseased",
    "ImamPasand_Healthy",
]

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def check_dataset(dataset_dir: str):
    """Validate that the dataset folder exists and has the right subfolders."""
    p = Path(dataset_dir)
    if not p.exists():
        raise FileNotFoundError(
            f"\n❌ Dataset folder not found: {p.resolve()}\n"
            "Please create it with four subfolders:\n"
            "  MangoDataset/Alphonso_Healthy/\n"
            "  MangoDataset/Alphonso_Diseased/\n"
            "  MangoDataset/ImamPasand_Healthy/\n"
            "  MangoDataset/ImamPasand_Diseased/\n"
            "Then add at least 30 photos per folder and re-run."
        )
    found = [d.name for d in p.iterdir() if d.is_dir()]
    print(f"📁 Dataset folder: {p.resolve()}")
    for cls in EXPECTED_CLASSES:
        cls_path = p / cls
        if not cls_path.exists():
            print(f"  ⚠️  Missing folder: {cls} (create it and add photos)")
        else:
            n = len(list(cls_path.glob("*.jpg")) + list(cls_path.glob("*.jpeg"))
                    + list(cls_path.glob("*.png")) + list(cls_path.glob("*.webp")))
            print(f"  ✅ {cls}: {n} images")
    print()


def build_data_generators(dataset_dir: str):
    """Return (train_gen, val_gen, class_indices)."""
    train_aug = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=30,
        width_shift_range=0.15,
        height_shift_range=0.15,
        shear_range=0.12,
        zoom_range=0.20,
        horizontal_flip=True,
        brightness_range=[0.7, 1.3],
        fill_mode="nearest",
        validation_split=0.20,
    )
    val_gen_obj = ImageDataGenerator(
        rescale=1.0 / 255,
        validation_split=0.20,
    )

    train_gen = train_aug.flow_from_directory(
        dataset_dir,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        subset="training",
        seed=SEED,
        shuffle=True,
    )
    val_gen = val_gen_obj.flow_from_directory(
        dataset_dir,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        subset="validation",
        seed=SEED,
        shuffle=False,
    )
    return train_gen, val_gen


def build_model(num_classes: int) -> tf.keras.Model:
    """MobileNetV2 feature extractor + custom classification head."""
    base = MobileNetV2(
        input_shape=(*IMAGE_SIZE, 3),
        include_top=False,
        weights="imagenet",
    )
    base.trainable = False   # freeze in phase 1

    inputs = tf.keras.Input(shape=(*IMAGE_SIZE, 3))
    x = base(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.40)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.30)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs, outputs, name="MangoVarietyCNN")
    return model, base


def unfreeze_for_fine_tuning(model, base_model, fine_tune_at: int = 100):
    """Unfreeze top layers of the base for fine-tuning (phase 2)."""
    base_model.trainable = True
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False
    print(f"🔓 Fine-tuning: unfrozen {len(base_model.layers) - fine_tune_at} base layers (from index {fine_tune_at})")


def plot_history(history_phase1, history_phase2, save_path: str):
    """Plot and save training curves from both phases."""
    def merge(h1, h2, key):
        return h1.history.get(key, []) + h2.history.get(key, [])

    acc     = merge(history_phase1, history_phase2, "accuracy")
    val_acc = merge(history_phase1, history_phase2, "val_accuracy")
    loss    = merge(history_phase1, history_phase2, "loss")
    val_loss= merge(history_phase1, history_phase2, "val_loss")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(acc,     label="Train Acc",  color="#2e7d32")
    ax1.plot(val_acc, label="Val Acc",    color="#81c784", linestyle="--")
    ax1.axvline(len(history_phase1.history["accuracy"]) - 1,
                color="orange", linestyle=":", label="Fine-tune starts")
    ax1.set_title("🥭 Mango CNN — Accuracy", fontsize=13)
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Accuracy")
    ax1.legend(); ax1.grid(True, alpha=0.3)

    ax2.plot(loss,     label="Train Loss", color="#c62828")
    ax2.plot(val_loss, label="Val Loss",   color="#ef9a9a", linestyle="--")
    ax2.axvline(len(history_phase1.history["loss"]) - 1,
                color="orange", linestyle=":", label="Fine-tune starts")
    ax2.set_title("🥭 Mango CNN — Loss", fontsize=13)
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Loss")
    ax2.legend(); ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close()
    print(f"📊 Training curves saved → {save_path}")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  🥭  AgriAssist+ — Mango Variety CNN Trainer")
    print("=" * 60)

    # 1. Validate dataset
    check_dataset(DATASET_DIR)

    # 2. Data generators
    print("📂 Loading dataset...")
    train_gen, val_gen = build_data_generators(DATASET_DIR)
    num_classes = len(train_gen.class_indices)
    print(f"🏷️  Classes ({num_classes}): {train_gen.class_indices}")
    print(f"📊 Training samples : {train_gen.samples}")
    print(f"📊 Validation samples: {val_gen.samples}\n")

    if train_gen.samples < 40:
        print("⚠️  WARNING: Very few training images. Aim for ≥50 per class for good accuracy.\n")

    # Save class indices
    idx_map = {str(v): k for k, v in train_gen.class_indices.items()}
    with open(OUTPUT_IDX, "w") as f:
        json.dump(idx_map, f, indent=2)
    print(f"💾 Class indices saved → {OUTPUT_IDX}")

    # 3. Build model
    print("\n🧠 Building MobileNetV2 model...")
    model, base_model = build_model(num_classes)
    model.summary()

    # ── PHASE 1: Train head only ──────────────
    print("\n" + "─" * 50)
    print("⚡ PHASE 1: Training classification head (base frozen)")
    print("─" * 50)
    model.compile(
        optimizer=optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    cb_phase1 = [
        callbacks.EarlyStopping(monitor="val_accuracy", patience=6, restore_best_weights=True, verbose=1),
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6, verbose=1),
    ]
    steps_per_epoch = max(1, train_gen.samples // BATCH_SIZE)
    val_steps       = max(1, val_gen.samples // BATCH_SIZE)

    h1 = model.fit(
        train_gen,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_gen,
        validation_steps=val_steps,
        epochs=EPOCHS,
        callbacks=cb_phase1,
        verbose=1,
    )
    best_val_acc_p1 = max(h1.history.get("val_accuracy", [0]))
    print(f"\n✅ Phase 1 complete. Best val accuracy: {best_val_acc_p1*100:.1f}%")

    # ── PHASE 2: Fine-tune top layers ─────────
    print("\n" + "─" * 50)
    print("🔧 PHASE 2: Fine-tuning top MobileNetV2 layers")
    print("─" * 50)
    unfreeze_for_fine_tuning(model, base_model, FINE_TUNE_AT)
    model.compile(
        optimizer=optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    cb_phase2 = [
        callbacks.EarlyStopping(monitor="val_accuracy", patience=8, restore_best_weights=True, verbose=1),
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.4, patience=4, min_lr=1e-7, verbose=1),
        callbacks.ModelCheckpoint(
            OUTPUT_MODEL, monitor="val_accuracy", save_best_only=True, verbose=1
        ),
    ]

    # Reset generators for phase 2
    train_gen.reset()
    val_gen.reset()

    h2 = model.fit(
        train_gen,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_gen,
        validation_steps=val_steps,
        epochs=EPOCHS,
        callbacks=cb_phase2,
        verbose=1,
    )
    best_val_acc_p2 = max(h2.history.get("val_accuracy", [0]))
    print(f"\n✅ Phase 2 complete. Best val accuracy: {best_val_acc_p2*100:.1f}%")

    # 4. Save final model
    model.save(OUTPUT_MODEL)
    print(f"\n💾 Model saved → {OUTPUT_MODEL}")

    # 5. Plot training history
    plot_history(h1, h2, OUTPUT_PLOT)

    # 6. Quick evaluation
    print("\n📊 Evaluating on validation set...")
    val_gen.reset()
    loss_val, acc_val = model.evaluate(val_gen, steps=val_steps, verbose=0)
    print(f"   Validation Loss    : {loss_val:.4f}")
    print(f"   Validation Accuracy: {acc_val*100:.1f}%")

    # 7. Per-class accuracy
    print("\n🏷️  Per-class prediction check (first batch)...")
    val_gen.reset()
    x_batch, y_batch = next(val_gen)
    preds = model.predict(x_batch, verbose=0)
    pred_labels = np.argmax(preds, axis=1)
    true_labels = np.argmax(y_batch, axis=1)
    class_names = list(train_gen.class_indices.keys())
    correct = np.sum(pred_labels == true_labels)
    print(f"   Batch accuracy: {correct}/{len(true_labels)} correct")
    for i in range(min(5, len(pred_labels))):
        pred_cls = class_names[pred_labels[i]]
        true_cls = class_names[true_labels[i]]
        conf     = preds[i][pred_labels[i]] * 100
        icon     = "✅" if pred_cls == true_cls else "❌"
        print(f"   {icon} Pred: {pred_cls:<25} True: {true_cls:<25} Conf: {conf:.1f}%")

    print("\n" + "=" * 60)
    print("  🎉 Training complete!")
    print(f"  Model   : {OUTPUT_MODEL}")
    print(f"  Classes : {class_names}")
    print(f"  Val Acc : {acc_val*100:.1f}%")
    print("=" * 60)
    print("\n✅ Restart your Streamlit app (streamlit run app.py)")
    print("   The 🥭 Mango CNN detector will now be active in Tab 6.\n")


if __name__ == "__main__":
    main()
