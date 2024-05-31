import tkinter as tk
from tkinter import filedialog
from tkinter import *
import torch
from PIL import ImageTk, Image
from torchvision import transforms
from model import Net


BG_COLOR = "#0B181F"
TEXT_COLOR = "#EAECEE"


model = Net()
model.load_state_dict(torch.load('model.pth'))
model.eval()


preprocess = transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Resize((32, 32)),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

classes = ('airplane', 'bird', 'car', 'cat', 'deer', 'dog', 'horse', 'monkey', 'ship', 'truck')


def classify_image(image_path):
    image = preprocess(Image.open(image_path))
    # вказуємо що є всього один елемент
    image = image.unsqueeze(0)
    output = model(image)
    _, predicted = torch.max(output, 1)
    return predicted.item()


class Application:

    def __init__(self):
        self.window = Tk()
        self._setup_main_window()

    def run(self):
        self.window.mainloop()

    def _setup_main_window(self):
        self.window.title("Image guesser")
        self.window.resizable(width=False, height=False)
        self.window.configure(width=470, height=600, bg=BG_COLOR)

        self.text_label = Label(self.window, bg=BG_COLOR, fg=TEXT_COLOR, text='Hello! I\'m image guesser :)', pady=10, font=('Arial', 18))
        self.text_label.place(relwidth=1, rely=0.03)

        self.image_label = Label(self.window, width=300, height=300, bg=BG_COLOR)
        self.image_label.place(relwidth=1, rely=0.12)

        self.answer = tk.Label(self.window, bg=BG_COLOR, fg=TEXT_COLOR, pady=5, font=('Arial', 18))
        self.answer.place(relwidth=1, rely=0.68)

        # bottom label
        bottom_label = Label(self.window, bg=BG_COLOR, height=80)
        bottom_label.place(relwidth=1, rely=0.8)

        self.choose_button = Button(bottom_label, text='Select image', command=self.choose_image, bg='WHITE', fg=BG_COLOR, font=('Arial', 16))
        self.choose_button.place(relx=0.5, rely=0.038, anchor="center", relheight=0.06, relwidth=0.42)

    def choose_image(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            result = classify_image(file_path)
            self.display_image(file_path)
            self.answer.config(text='Hmmm... I think it\'s a ' + classes[result] + '!')

    def display_image(self, file_path):
        image = Image.open(file_path)
        image.thumbnail((300, 300))
        photo = ImageTk.PhotoImage(image)
        self.image_label.configure(image=photo)
        self.image_label.image = photo


if __name__ == "__main__":
    app = Application()
    app.run()
