import tkinter as tk
from tkinter import ttk
import sv_ttk

class Application(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("Title?")
        self.geometry("700x400")

        self.rowconfigure(0, weight=0)
        self.rowconfigure(1, weight=0)

        self.columnconfigure(0, weight=0)
        self.columnconfigure(1, weight=0)

        self.tab_bar()
        self.top_bar()
        
    def top_bar(self):
        self.open_file_button = ttk.Button(self, text="Open file...", command=lambda:print("open file pressed"))
        self.open_file_button.grid(row=1, column=0, sticky="nswe", padx=3, pady=5)

        self.open_db_button = ttk.Button(self, text="Open DB...", command=lambda:print("open db pressed"))
        self.open_db_button.grid(row=1, column=1, sticky="nswe", padx=3, pady=5)

    def tab_bar(self):
        tab_bar_obj = ttk.Notebook(self)
        tab1 = ttk.Frame(tab_bar_obj)
        tab2 = ttk.Frame(tab_bar_obj)

        tab_bar_obj.add(tab1, text ='Preprocess') 
        tab_bar_obj.add(tab2, text ='Classify')
        tab_bar_obj.grid(row=0, column=0, columnspan=2, sticky="w")


if __name__ == "__main__":
    app = Application()
    sv_ttk.set_theme("dark")
    app.mainloop()
