import os

# Mevcut dizini kontrol et
current_dir = os.getcwd()
print(f"Mevcut dizin: {current_dir}")

# Code klasöründeki tüm içerikleri listele
code_dir = "C:\\Users\\ayogu\\Desktop\\Okul\\4.Year\\1.Semester\\NLP\\Project\\Code"
print(f"\nCode klasöründeki dosyalar:")
for item in os.listdir(code_dir):
    item_path = os.path.join(code_dir, item)
    if os.path.isdir(item_path):
        print(f"📁 {item}/")
    else:
        print(f"📄 {item}")

# Model klasöründe ara
print("\nModel dosyalarını arıyorum...")
for root, dirs, files in os.walk(code_dir):
    for dir_name in dirs:
        if "multiclass" in dir_name.lower():
            print(f"Bulundu: {os.path.join(root, dir_name)}")