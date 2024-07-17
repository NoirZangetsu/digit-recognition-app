import torch
from model import ImprovedDigitRecognitionCNN
from gui import DigitRecognitionApp
from train import train_model
import tkinter as tk
from tkinter import messagebox

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = ImprovedDigitRecognitionCNN().to(device)

    try:
        checkpoint = torch.load('best_digit_recognition_model.pth')
        model.load_state_dict(checkpoint['model_state_dict'])
        model.best_loss = checkpoint['best_loss']
        best_accuracy = checkpoint['best_accuracy']
        print(f"En iyi model yüklendi. En iyi doğruluk: {best_accuracy:.2f}%, En iyi kayıp: {model.best_loss:.4f}")
    except (FileNotFoundError, RuntimeError):
        print("Uyumlu bir eğitilmiş model bulunamadı. Yeni model oluşturuldu.")
        best_accuracy = 0
        model.best_loss = float('inf')

    model.to('cpu')  # GUI için CPU'ya taşı

    root = tk.Tk()
    root.withdraw()  # Ana pencereyi gizle

    if best_accuracy < 95:  # Eğer model performansı düşükse veya model yeni oluşturulduysa
        should_train = messagebox.askyesno("Model Eğitimi", "Model performansı düşük veya yeni bir model oluşturuldu. Şimdi eğitmek ister misiniz?")
        if should_train:
            print("Model eğitiliyor...")
            best_accuracy, best_loss = train_model(model)
            print(f"Model eğitildi. En iyi doğruluk: {best_accuracy:.2f}%, En iyi kayıp: {best_loss:.4f}")
            model.best_loss = best_loss
        else:
            print("Eğitim atlandı. Mevcut model kullanılacak.")

    app = DigitRecognitionApp(model)
    app.run()

if __name__ == "__main__":
    main()