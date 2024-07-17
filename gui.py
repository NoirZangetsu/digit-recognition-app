import tkinter as tk
from tkinter import messagebox, simpledialog
from PIL import Image, ImageDraw, ImageTk
import torch
import torchvision.transforms as transforms
from train import train_model
import threading

class DigitRecognitionApp:
    def __init__(self, model):
        self.model = model
        self.root = tk.Tk()
        self.root.title("Geliştirilmiş El Yazısı Rakam Tanıma")
        
        self.canvas = tk.Canvas(self.root, width=280, height=280, bg="black")
        self.canvas.pack(pady=20)
        
        self.image = Image.new("RGB", (280, 280), color="black")
        self.draw = ImageDraw.Draw(self.image)
        
        button_frame = tk.Frame(self.root)
        button_frame.pack(pady=10)
        
        self.button_clear = tk.Button(button_frame, text="Temizle", command=self.clear_canvas)
        self.button_clear.pack(side=tk.LEFT, padx=5)
        
        self.button_predict = tk.Button(button_frame, text="Tahmin Et", command=self.predict_digit)
        self.button_predict.pack(side=tk.LEFT, padx=5)
        
        self.button_feedback = tk.Button(button_frame, text="Geri Bildirim Ver", command=self.give_feedback)
        self.button_feedback.pack(side=tk.LEFT, padx=5)
        
        self.button_train = tk.Button(button_frame, text="Modeli Eğit", command=self.start_training)
        self.button_train.pack(side=tk.LEFT, padx=5)
        
        self.label_result = tk.Label(self.root, text="", font=("Helvetica", 18))
        self.label_result.pack(pady=10)
        
        self.label_accuracy = tk.Label(self.root, text="Model Doğruluğu: N/A", font=("Helvetica", 12))
        self.label_accuracy.pack(pady=5)
        
        self.label_loss = tk.Label(self.root, text="Model Kaybı: N/A", font=("Helvetica", 12))
        self.label_loss.pack(pady=5)
        
        self.canvas.bind("<B1-Motion>", self.paint)
        self.canvas.bind("<ButtonRelease-1>", self.reset_xy)
        self.x = self.y = None
        
        self.transform = transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
    def paint(self, event):
        if self.x and self.y:
            x1, y1 = (self.x, self.y)
            x2, y2 = (event.x, event.y)
            self.canvas.create_line((x1, y1, x2, y2), fill='white', width=20)
            self.draw.line([x1, y1, x2, y2], fill="white", width=20)
        self.x = event.x
        self.y = event.y

    def reset_xy(self, event):
        self.x = None
        self.y = None

    def clear_canvas(self):
        self.canvas.delete("all")
        self.image = Image.new("RGB", (280, 280), color="black")
        self.draw = ImageDraw.Draw(self.image)
        self.label_result.config(text="")

    def predict_digit(self):
        img = self.image.convert('L')
        img_tensor = self.transform(img).unsqueeze(0)
        
        self.model.eval()
        with torch.no_grad():
            output = self.model(img_tensor)
            pred = output.argmax(dim=1, keepdim=True)
        
        self.label_result.config(text=f"Tahmin: {pred.item()}")
        self.current_image = img_tensor

    def give_feedback(self):
        true_label = simpledialog.askinteger("Geri Bildirim", "Doğru rakamı girin:", minvalue=0, maxvalue=9)
        if true_label is not None:
            loss = self.model.update(self.current_image, torch.tensor(true_label))
            messagebox.showinfo("Güncelleme", f"Model güncellendi. Kayıp: {loss:.4f}")
            self.update_model_stats()

    def update_model_stats(self):
        # Bu fonksiyon normalde eğitim sonrası çağrılmalı
        # Şu an için sadece kaybı güncelliyoruz
        self.label_loss.config(text=f"Model Kaybı: {self.model.best_loss:.4f}")

    def start_training(self):
        threading.Thread(target=self.train_model_thread, daemon=True).start()

    def train_model_thread(self):
        self.button_train.config(state=tk.DISABLED)
        messagebox.showinfo("Eğitim Başladı", "Model eğitimi başladı. Bu işlem biraz zaman alabilir.")
        best_accuracy, best_loss = train_model(self.model)
        self.model.best_loss = best_loss
        self.label_accuracy.config(text=f"Model Doğruluğu: {best_accuracy:.2f}%")
        self.label_loss.config(text=f"Model Kaybı: {best_loss:.4f}")
        self.button_train.config(state=tk.NORMAL)
        messagebox.showinfo("Eğitim Tamamlandı", f"Model eğitimi tamamlandı. Doğruluk: {best_accuracy:.2f}%, Kayıp: {best_loss:.4f}")

    def run(self):
        self.root.mainloop()