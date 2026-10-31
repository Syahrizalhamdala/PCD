import imageio.v2 as imageio
import numpy as np
from scipy import ndimage
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

NAMA_FILE = "#Kabinet.png"


def ke_grayscale(img):
    if img.ndim == 2:
        return img.astype(np.uint8)
    if img.shape[2] == 4:
        img = img[:, :, :3]
    r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    return (0.299 * r + 0.587 * g + 0.114 * b).astype(np.uint8)


def citra_kontras_rendah(gray, bawah=90, atas=150):
    return (bawah + (gray.astype(np.float64) / 255.0) * (atas - bawah)).astype(np.uint8)


def histogram_equalization(gray):
    hist = np.bincount(gray.ravel(), minlength=256).astype(np.float64)
    cdf = np.cumsum(hist)
    cdf_min = cdf[hist > 0].min()
    total = gray.size
    cdf_norm = np.round((cdf - cdf_min) / (total - cdf_min) * 255)
    cdf_norm = np.clip(cdf_norm, 0, 255).astype(np.uint8)
    return cdf_norm[gray]


def histogram(gray):
    return np.bincount(gray.ravel(), minlength=256)


img = imageio.imread(NAMA_FILE)
gray = ke_grayscale(img)

kontras_rendah = citra_kontras_rendah(gray)
kontras_rendah = ndimage.median_filter(kontras_rendah, size=3)

hasil_eq = histogram_equalization(kontras_rendah)
hasil_eq_asli = histogram_equalization(gray)

print("Citra asli       :", img.shape)
print("Grayscale        :", gray.shape)
print("Rentang awal     :", gray.min(), "-", gray.max())
print("Rentang kontras rendah :", kontras_rendah.min(), "-", kontras_rendah.max())
print("Rentang hasil ekualisasi :", hasil_eq.min(), "-", hasil_eq.max())
print("Rata-rata sebelum :", round(gray.mean(), 2))
print("Rata-rata sesudah :", round(hasil_eq_asli.mean(), 2))
print("Standar deviasi sebelum :", round(gray.std(), 2))
print("Standar deviasi sesudah :", round(hasil_eq_asli.std(), 2))

fig, axes = plt.subplots(2, 3, figsize=(16, 9))

pasangan = [
    ("Grayscale Asli", gray),
    ("Kontras Rendah", kontras_rendah),
    ("Hasil Ekualisasi", hasil_eq),
]

for ax, (judul, citra) in zip(axes[0], pasangan):
    ax.imshow(citra, cmap="gray", vmin=0, vmax=255)
    ax.set_title(judul)
    ax.axis("off")

for ax, (judul, citra) in zip(axes[1], pasangan):
    ax.bar(range(256), histogram(citra), width=1, color="steelblue")
    ax.set_title(f"Histogram {judul}")
    ax.set_xlabel("Nilai Intensitas")
    ax.set_ylabel("Jumlah Piksel")
    ax.set_xlim(0, 255)

plt.tight_layout()
plt.savefig("hasil_ekualisasi.png", dpi=120, bbox_inches="tight")

fig2, axes2 = plt.subplots(1, 2, figsize=(12, 5))
axes2[0].imshow(gray, cmap="gray", vmin=0, vmax=255)
axes2[0].set_title("Sebelum Ekualisasi")
axes2[0].axis("off")
axes2[1].imshow(hasil_eq_asli, cmap="gray", vmin=0, vmax=255)
axes2[1].set_title("Sesudah Ekualisasi")
axes2[1].axis("off")
plt.tight_layout()
plt.savefig("hasil_ekualisasi_asli.png", dpi=120, bbox_inches="tight")
plt.show()
