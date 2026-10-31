import imageio.v2 as imageio
import numpy as np
from scipy import ndimage
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

NAMA_FILE = "#Kabinet.png"

KERNEL_MEAN_3 = np.ones((3, 3), dtype=np.float64) / 9.0
KERNEL_MEAN_5 = np.ones((5, 5), dtype=np.float64) / 25.0

KERNEL_GAUSS_3 = np.outer([1, 2, 1], [1, 2, 1]) / 16.0
KERNEL_GAUSS_5 = np.outer([1, 4, 6, 4, 1], [1, 4, 6, 4, 1]) / 256.0

KERNEL_SHARPEN = np.array(
    [
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0],
    ],
    dtype=np.float64,
)


def ke_grayscale(img):
    if img.ndim == 2:
        return img.astype(np.uint8)
    if img.shape[2] == 4:
        img = img[:, :, :3]
    r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    return (0.299 * r + 0.587 * g + 0.114 * b).astype(np.uint8)


def konvolusi(gray, kernel):
    hasil = ndimage.convolve(gray.astype(np.float64), kernel, mode="reflect")
    return np.clip(hasil, 0, 255).astype(np.uint8)


def mean_filter(gray, ukuran):
    kernel = KERNEL_MEAN_3 if ukuran == 3 else KERNEL_MEAN_5
    return konvolusi(gray, kernel)


def gaussian_filter(gray, ukuran):
    kernel = KERNEL_GAUSS_3 if ukuran == 3 else KERNEL_GAUSS_5
    return konvolusi(gray, kernel)


def median_filter(gray, ukuran):
    return ndimage.median_filter(gray, size=ukuran).astype(np.uint8)


def sharpening(gray):
    return konvolusi(gray, KERNEL_SHARPEN)


def statistik(gray):
    f = gray.astype(np.float64)
    gx = ndimage.sobel(f, axis=1)
    gy = ndimage.sobel(f, axis=0)
    kekuatan_tepi = np.hypot(gx, gy).mean()
    return {
        "min": int(gray.min()),
        "max": int(gray.max()),
        "mean": float(f.mean()),
        "std": float(f.std()),
        "tepi": float(kekuatan_tepi),
    }


def tambah_noise(gray, seed=42):
    rng = np.random.default_rng(seed)
    noisy = gray.astype(np.float64)

    gaussian_noise = rng.normal(0, 12, gray.shape)
    noisy = noisy + gaussian_noise

    jumlah_sp = int(0.02 * gray.size)
    baris = rng.integers(0, gray.shape[0], jumlah_sp)
    kolom = rng.integers(0, gray.shape[1], jumlah_sp)
    noisy[baris[: jumlah_sp // 2], kolom[: jumlah_sp // 2]] = 0
    noisy[baris[jumlah_sp // 2 :], kolom[jumlah_sp // 2 :]] = 255

    return np.clip(noisy, 0, 255).astype(np.uint8)


def terapkan_semua(gray):
    return [
        ("Citra Asli", gray),
        ("Mean 3x3", mean_filter(gray, 3)),
        ("Mean 5x5", mean_filter(gray, 5)),
        ("Gaussian 3x3", gaussian_filter(gray, 3)),
        ("Gaussian 5x5", gaussian_filter(gray, 5)),
        ("Median 3x3", median_filter(gray, 3)),
        ("Median 5x5", median_filter(gray, 5)),
        ("Sharpening (Laplacian)", sharpening(gray)),
    ]


def tampilkan(hasil, judul_gambar, nama_keluaran):
    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    for ax, (judul, citra) in zip(axes.ravel(), hasil):
        s = statistik(citra)
        ax.imshow(citra, cmap="gray", vmin=0, vmax=255)
        ax.set_title(
            f"{judul}\nmean={s['mean']:.2f}  std={s['std']:.2f}  "
            f"tepi={s['tepi']:.2f}",
            fontsize=11,
        )
        ax.axis("off")
    fig.suptitle(judul_gambar, fontsize=16)
    plt.tight_layout()
    plt.savefig(nama_keluaran, dpi=120, bbox_inches="tight")
    plt.close(fig)


img = imageio.imread(NAMA_FILE)
gray = ke_grayscale(img)

hasil_asli = terapkan_semua(gray)
tampilkan(
    hasil_asli,
    "Hasil Filter Spasial pada Citra Asli (#Kabinet.png)",
    "hasil_filter_spasial.png",
)

noisy = tambah_noise(gray)
hasil_noise = [("Citra Bernoise", noisy)] + [
    (f"Bernoise + {judul}", citra)
    for judul, citra in terapkan_semua(noisy)[1:]
]
tampilkan(
    hasil_noise,
    "Hasil Filter Spasial pada Citra dengan Noise (Gaussian + Salt & Pepper)",
    "hasil_filter_spasial_noise.png",
)

f = gray.astype(np.float64)
gx = ndimage.sobel(f, axis=1)
gy = ndimage.sobel(f, axis=0)
gradien = np.hypot(gx, gy)
baris, kolom = np.unravel_index(np.argmax(gradien), gradien.shape)
baris = int(np.clip(baris, 2, gray.shape[0] - 3))
kolom = int(np.clip(kolom, 2, gray.shape[1] - 3))
patch_asli = gray[baris - 2 : baris + 3, kolom - 2 : kolom + 3]

lines = []
lines.append(f"Citra sumber: {NAMA_FILE}")
lines.append(f"Ukuran citra: {img.shape}")
lines.append("")
lines.append("=" * 88)
lines.append("STATISTIK CITRA ASLI DAN HASIL FILTER")
lines.append("=" * 88)
lines.append(
    f"{'Citra':<26}{'Min':>6}{'Max':>6}{'Mean':>10}{'Std':>10}{'Kekuatan Tepi':>16}"
)
lines.append("-" * 88)
for judul, citra in hasil_asli:
    s = statistik(citra)
    lines.append(
        f"{judul:<26}{s['min']:>6}{s['max']:>6}{s['mean']:>10.2f}"
        f"{s['std']:>10.2f}{s['tepi']:>16.2f}"
    )

lines.append("")
lines.append("=" * 88)
lines.append("STATISTIK CITRA BERNOISE DAN HASIL FILTER")
lines.append("=" * 88)
lines.append(
    f"{'Citra':<26}{'Min':>6}{'Max':>6}{'Mean':>10}{'Std':>10}{'Kekuatan Tepi':>16}"
)
lines.append("-" * 88)
for judul, citra in hasil_noise:
    s = statistik(citra)
    lines.append(
        f"{judul:<26}{s['min']:>6}{s['max']:>6}{s['mean']:>10.2f}"
        f"{s['std']:>10.2f}{s['tepi']:>16.2f}"
    )

lines.append("")
lines.append("=" * 88)
lines.append(f"CONTOH NILAI INTENSITAS PATCH 5x5 (baris {baris}, kolom {kolom})")
lines.append("=" * 88)
for judul, citra in hasil_asli:
    patch = citra[baris - 2 : baris + 3, kolom - 2 : kolom + 3]
    lines.append(f"{judul}:")
    lines.append(np.array2string(patch, max_line_width=200))
    lines.append("")

teks = "\n".join(lines)
print(teks)
with open("nilai_intensitas_sesi5.txt", "w", encoding="utf-8") as file:
    file.write(teks)
