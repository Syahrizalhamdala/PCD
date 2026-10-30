import cv2
import numpy as np
import matplotlib.pyplot as plt

nama_file = "#Kabinet.png"

img_bgr = cv2.imread(nama_file)
if img_bgr is None:
    raise FileNotFoundError(f"Citra '{nama_file}' tidak ditemukan atau tidak dapat dibaca.")

img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def translation(image, tx, ty):
    matrix = np.float32([[1, 0, tx], [0, 1, ty]])
    ukuran = (image.shape[1], image.shape[0])
    return cv2.warpAffine(image, matrix, ukuran, borderValue=(255, 255, 255))


def scaling(image, faktor):
    return cv2.resize(
        image,
        None,
        fx=faktor,
        fy=faktor,
        interpolation=cv2.INTER_LINEAR,
    )


def rotation(image, sudut):
    tinggi, lebar = image.shape[:2]
    pusat = (lebar / 2, tinggi / 2)
    matrix = cv2.getRotationMatrix2D(pusat, sudut, 1.0)

    cos = abs(matrix[0, 0])
    sin = abs(matrix[0, 1])
    lebar_baru = int(tinggi * sin + lebar * cos)
    tinggi_baru = int(tinggi * cos + lebar * sin)

    matrix[0, 2] += lebar_baru / 2 - pusat[0]
    matrix[1, 2] += tinggi_baru / 2 - pusat[1]

    return cv2.warpAffine(
        image,
        matrix,
        (lebar_baru, tinggi_baru),
        borderValue=(255, 255, 255),
    )


def horizontal_flip(image):
    return cv2.flip(image, 1)


def vertical_flip(image):
    return cv2.flip(image, 0)


hasil = [
    ("Citra Asli", img),
    ("Translation (tx=100, ty=50)", translation(img, 100, 50)),
    ("Scaling 0.5x", scaling(img, 0.5)),
    ("Scaling 2x", scaling(img, 2.0)),
    ("Rotation 45°", rotation(img, 45)),
    ("Rotation 90°", rotation(img, 90)),
    ("Rotation 180°", rotation(img, 180)),
    ("Horizontal Flip", horizontal_flip(img)),
    ("Vertical Flip", vertical_flip(img)),
]

print(f"Citra Asli: {img.shape}")
for judul, citra in hasil[1:]:
    print(f"{judul}: {citra.shape}")

fig, axes = plt.subplots(3, 3, figsize=(15, 15))
for ax, (judul, citra) in zip(axes.ravel(), hasil):
    ax.imshow(citra)
    ax.set_title(judul)
    ax.axis("off")

plt.tight_layout()
plt.savefig("hasil_transformasi.png", dpi=120, bbox_inches="tight")
plt.show()
