import csv
import sys
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import (
    QApplication,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QGroupBox,
    QComboBox,
    QPushButton,
    QTextEdit,
    QLabel,
    QFormLayout,
    QFileDialog
)

from utils import csv_read, arff_read

class App(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Title ?")
        self.setGeometry(0, 0, 800, 600)

        app_icon = QIcon("assets/dit_grey_500.png") # not working?
        self.setWindowIcon(app_icon)

        self.initUI()

    def initUI(self):
        main_lyt = QVBoxLayout()
        classifier_group = QGroupBox("Classifier Configuration")
        form_layout = QFormLayout()
        
        self.classifier_combo = QComboBox()
        self.classifier_combo.addItems(["Naive Bayes"])
        
        self.test_mode_combo = QComboBox()
        self.test_mode_combo.addItems(["Training set", "Cross-validation", "Percentage split"])
        
        form_layout.addRow("Classifier:", self.classifier_combo)
        form_layout.addRow("Test mode:", self.test_mode_combo)
        classifier_group.setLayout(form_layout)

        results_group = QGroupBox("Results")
        results_lyt = QVBoxLayout()
        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        results_lyt.addWidget(self.results_text)
        results_group.setLayout(results_lyt)

        control_lyt = QHBoxLayout()
        start_btn = QPushButton("Start")
        stop_btn = QPushButton("Stop")
        clear_btn = QPushButton("Clear")
        control_lyt.addStretch()
        control_lyt.addWidget(start_btn)
        control_lyt.addWidget(stop_btn)
        control_lyt.addWidget(clear_btn)

        status_bar = QLabel("Ready")
        
        button_lyt = QHBoxLayout()
        open_file_btn = QPushButton("Open file...")
        open_db_btn = QPushButton("Open DB..")
        generate_btn = QPushButton("Generate..")
        button_lyt.addWidget(open_file_btn)
        button_lyt.addWidget(open_db_btn)
        button_lyt.addWidget(generate_btn)
    
        main_lyt.addLayout(button_lyt)
        main_lyt.addWidget(classifier_group)
        main_lyt.addWidget(results_group)
        main_lyt.addLayout(control_lyt)
        main_lyt.addWidget(status_bar)

        self.setLayout(main_lyt)

        start_btn.clicked.connect(self.on_start)
        stop_btn.clicked.connect(self.on_stop)
        open_file_btn.clicked.connect(self.open_file_dlg)
        clear_btn.clicked.connect(self.on_clear)

    def on_start(self):
        self.results_text.append("Start button pressed\n")

    def on_stop(self):
        self.results_text.append("Stop button pressed\n")

    def open_file_dlg(self):
        file_path, sfilter = QFileDialog.getOpenFileName(self, "Open File", "", "Arff files (*.arff);;CSV files (*.csv);;Any file (*.*)")
        if file_path:
            try:
                if(sfilter == "Arff files (*.arff)"):
                    with open(file_path, 'r', encoding='utf-8') as file:
                        txt = file.read()
                        self.results_text.append(arff_read.relation_val(txt))
                        self.results_text.append(str(arff_read.attribute_val(txt)))
                elif(sfilter == "CSV files (*.csv)"):
                    df = csv_read.read_csv(file_path)
                    self.results_text.append(str(df))
                else:
                    with open(file_path, 'r') as file:
                        txt = file.read()
                        self.results_text.append(f"Opened file {file_path}:\n{txt}\n")
            except Exception as e:
               print(f"Error reading file: {e}\n")
    
    def on_clear(self):
        self.results_text.clear()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = App()
    window.show()
    sys.exit(app.exec())