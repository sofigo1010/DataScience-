from pathlib import Path
import random
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image


PROJECT_DIR = Path(__file__).resolve().parent
DATA_ROOT   = PROJECT_DIR / "PolyMNIST" / "MMNIST"   
SPLITS      = ["train", "test"]                      



def build_inventory(root: Path = DATA_ROOT) -> pd.DataFrame:

    records = []
    for split in SPLITS:
        for modality_dir in sorted((root / split).iterdir()):
            if not modality_dir.is_dir():
                continue
            modality = modality_dir.name           
            for img_path in modality_dir.glob("*.png"):
                tokens = img_path.stem.split("_")
                label_token = None
                if tokens[-1].isdigit() and len(tokens[-1]) == 1:
                    label_token = tokens[-1]
                else:
                    for tok in tokens:
                        if tok.isdigit() and len(tok) == 1:
                            label_token = tok
                            break
                if label_token is None:
                    if tokens[0][-1].isdigit():
                        label_token = tokens[0][-1]
                if label_token is None:
                    raise ValueError(f"Cannot parse label from filename: {img_path.name}")
                label = int(label_token)
                records.append(
                    dict(
                        split=split,
                        modality=modality,
                        label=label,
                        path=img_path,
                    )
                )

    df = pd.DataFrame(records)
    return df


def show_examples(df: pd.DataFrame, n: int = 5, split: str = "train") -> None:
    subset     = df[df["split"] == split]
    modalities = sorted(subset["modality"].unique())

    fig, axes = plt.subplots(
        nrows=len(modalities),
        ncols=n,
        figsize=(n * 1.6, len(modalities) * 1.6),
        sharex=True,
        sharey=True,
    )

    for i, modality in enumerate(modalities):
        rows = subset[subset["modality"] == modality].sample(n, random_state=42)
        for j, (_, row) in enumerate(rows.iterrows()):
            ax = axes[i, j] if len(modalities) > 1 else axes[j]
            ax.imshow(Image.open(row["path"]), cmap="gray")
            ax.set_title(str(row["label"]), fontsize=8)
            ax.axis("off")

        if len(modalities) > 1:
            axes[i, 0].set_ylabel(
                modality, rotation=0, labelpad=25, fontsize=10, va="center"
            )

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    df = build_inventory()
    print("Encabezado del inventario:\n", df.head(), "\n")
    print(
        "Distribución de clases por modalidad (train):\n",
        df[df["split"] == "train"]
        .groupby(["modality", "label"])
        .size()
        .unstack(fill_value=0)
        .head(),
        "\n",
    )
    show_examples(df, n=6, split="train")