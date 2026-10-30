import imageio.v2 as imageio
import numpy as np
import matplotlib.pyplot as plt

nama_file = "aril.jpg"
#1. membaca citra
nama_file = "#Kabinet.png"
img = imageio.imread(nama_file)
print("Citra RGB    :", img.shape)
#2. mengubah citra rgb ke gray
R = img[:,:,0]
G = img[:,:,1]
B = img[:,:,2]

img_gray = (
    0.299 * R +
    0.587 * G +
    0.114 * B
).astype(np.uint8)

print("Citra Grayscale  :",img_gray.shape)
#3. Hitung piksel
tinggi, lebar = img_gray.shape
jumlah_piksel = img_gray.size

print("Tinggi   :",tinggi, "Piksel")
print("Lebar   :",lebar, "Piksel")
print("Jumlah   :",jumlah_piksel, "Piksel")
#4. menghitung histogram
# Memastikan img_gray sudah terdefinisi secara otomatis jika belum dijalankan sebelumnya
if 'img_gray' not in globals() and 'img' in globals():
    R = img[:,:,0]
    G = img[:,:,1]
    B = img[:,:,2]
    img_gray = (0.299 * R + 0.587 * G + 0.114 * B).astype(np.uint8)

histogram = np.bincount(
    img_gray.flatten(),
    minlength=256
)
#5. menghitung intensitas
for intensitas in range(256):
    print(
        f"intensitas{intensitas: 3d} :"
        f"{histogram[intensitas]}"
    )
#6. mencari intensitas dominan
intensitas_dominan = np.argmax(histogram)
jumlah_intensitas = histogram[intensitas_dominan]

print("Intensitas Dominan   :",intensitas_dominan)
print("Jumlah Intensitas    :", jumlah_intensitas)

#7. tampilkan gambar
plt.figure(figsize=(14,5))

plt.subplot(1,2,1)
plt.imshow(img_gray, cmap="gray")
plt.title("Citra Grayscale")


plt.subplot(1,2,2)
plt.bar(
    range(256),
    histogram,
    width=1
)

plt.ylabel("Jumlah Piksel")
plt.xlabel("Nilai Intensitas")
plt.xlim(0,255)

plt.tight_layout()
plt.show()