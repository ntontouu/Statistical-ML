import tkinter as tk
from tkinter import PhotoImage, filedialog
from tkinter import ttk
import sv_ttk

class Application(tk.Tk):
    def __init__(self):
        super().__init__()

        icon = PhotoImage(file="assets/dit_500.png")
        self.iconphoto(False, icon)

        self.title("Title?")
        self.geometry("700x400")

        self.tab_bar()
        self.top_bar()

    def onOpen(self):
        ftypes = [('ARF data files', '*.arff'), ('CSV data files', '*.csv'), ('All files', '*')]
        dlg = filedialog.Open(self, filetypes = ftypes)
        fl = dlg.show()

        if fl:
            text = self.readFile(fl)
            print(text)

    def readFile(self, filename):
        f = open(filename, "r")
        text = f.read()
        return text

    def top_bar(self):
        self.open_file_button = ttk.Button(self.preprocess_tab, text="Open file...", command=self.onOpen)
        self.open_file_button.grid(row=0, column=0, sticky="nswe", padx=3, pady=5)

        self.open_db_button = ttk.Button(self.preprocess_tab, text="Open DB...", command=lambda:print("open db pressed"))
        self.open_db_button.grid(row=0, column=1, sticky="nswe", padx=3, pady=5)

    def tab_bar(self):
        tab_bar_obj = ttk.Notebook(self)

        # Create tabs
        self.preprocess_tab = ttk.Frame(tab_bar_obj)
        self.classify_tab = ttk.Frame(tab_bar_obj)

        # Add tabs to the notebook
        tab_bar_obj.add(self.preprocess_tab, text='Preprocess')
        tab_bar_obj.add(self.classify_tab, text='Classify')

        # Add the notebook to the grid and make it expand
        tab_bar_obj.grid(row=0, column=0, sticky="nswe")

        # Configure the grid weights for the notebook
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

        self.preprocess_tab.grid_rowconfigure(0, weight=0)
        self.preprocess_tab.grid_rowconfigure(1, weight=1)
        self.preprocess_tab.grid_columnconfigure(0, weight=0)
        self.preprocess_tab.grid_columnconfigure(1, weight=0)
        self.preprocess_tab.grid_columnconfigure(2, weight=2)

        self.classify_tab.grid_rowconfigure(0, weight=1)
        self.classify_tab.grid_columnconfigure(0, weight=1)


if __name__ == "__main__":
    app = Application()
    sv_ttk.set_theme("dark")
    app.mainloop()
