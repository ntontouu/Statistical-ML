import sys, gui, platform
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
    QFormLayout,
    QFileDialog,
    QTabWidget,
    QListWidget,
    QAbstractItemView,
    QListWidgetItem,
    QStyleFactory,
    QLineEdit,
    QLabel,
    QGridLayout,
    QRadioButton
)
from PySide6.QtCore import Qt
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from utils.data import data
from gaussian_NB import GaussianNaiveBayes

class App(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Title ?")
        self.setGeometry(500, 500, 800, 600)

        app_icon = QIcon("dit.png") # not working?
        self.setWindowIcon(app_icon)

        self.initUI()

    def initUI(self):
        main_lyt = QVBoxLayout()

        self.tab_widget = QTabWidget()
        self.tab1 = QWidget()
        self.tab2 = QWidget()
        self.tab_widget.addTab(self.tab1, "Preprocess")
        self.tab_widget.addTab(self.tab2, "Classify")
#        self.tab_widget.setTabEnabled(1, False)

        main_lyt.addWidget(self.tab_widget)

        self.preprocess_tab()
        self.classify_tab()

        self.setLayout(main_lyt)

    def preprocess_tab(self):
        layout = QVBoxLayout(self.tab1)

        button_lyt = QHBoxLayout()
        open_file_btn = QPushButton("Open file...")
        open_db_btn = QPushButton("Open DB...")
        generate_btn = QPushButton("Generate...")
        edit_btn = QPushButton("Edit...")
        save_btn = QPushButton("Save...")
        button_lyt.addWidget(open_file_btn)
        button_lyt.addWidget(open_db_btn)
        button_lyt.addWidget(generate_btn)
        button_lyt.addWidget(edit_btn)
        button_lyt.addWidget(save_btn)
        open_file_btn.clicked.connect(self.open_file_dlg)

        cur_rel_group = QGroupBox("Current Relation")
        rel_layout = QGridLayout()
        self.rel_label = QLabel(f"Relation: {None}")
        self.inst_label = QLabel(f"Instances: {None}")
        self.attr_label = QLabel(f"Attributes: {None}")
        self.sow_label = QLabel(f"Sum of weights: {None}")
        rel_layout.addWidget(self.rel_label, 0, 0)
        rel_layout.addWidget(self.inst_label, 1, 0)
        rel_layout.addWidget(self.attr_label, 0, 1)
        rel_layout.addWidget(self.sow_label, 1, 1)
        cur_rel_group.setLayout(rel_layout)

        attr_group = QGroupBox("Attributes")
        form_layout = QVBoxLayout()
        attr_btn_layout = QHBoxLayout()
        all_btn = QPushButton("All")
        none_btn = QPushButton("None")
        invert_btn = QPushButton("Invert")
        attr_btn_layout.addWidget(all_btn)
        attr_btn_layout.addWidget(none_btn)
        attr_btn_layout.addWidget(invert_btn)
        self.listwidget = QListWidget()
        self.listwidget.setSelectionMode(QAbstractItemView.SingleSelection)
        remove_btn = QPushButton("Remove")
        form_layout.addLayout(attr_btn_layout)
        form_layout.addWidget(self.listwidget)
        form_layout.addWidget(remove_btn)
        attr_group.setLayout(form_layout)

        selat_group = QGroupBox("Selected attribute")
        selat_lyt = QVBoxLayout()

        lbl_layout = QHBoxLayout()
        self.selat_name_label = QLabel(f"Name: {None}")
        self.selat_type_label = QLabel(f"Type: {None}")
        lbl_layout.addWidget(self.selat_name_label)
        lbl_layout.addWidget(self.selat_type_label)
        selat_lyt.addLayout(lbl_layout)

        self.selat_list = QListWidget()
        self.selat_list.setSelectionMode(QAbstractItemView.SingleSelection)
        selat_lyt.addWidget(self.selat_list)
        
        selat_group.setLayout(selat_lyt)

        left_lyt = QVBoxLayout()
        left_lyt.addWidget(cur_rel_group)
        left_lyt.addWidget(attr_group)
        
        right_lyt = QVBoxLayout()
        right_lyt.addWidget(selat_group)

        merged_lyt = QHBoxLayout()
        merged_lyt.addLayout(left_lyt)
        merged_lyt.addLayout(right_lyt)

        layout.addLayout(button_lyt)
        layout.addLayout(merged_lyt)

    def classify_tab(self):
        layout = QVBoxLayout(self.tab2)

        classifier_group = QGroupBox("Classifier Configuration")
        form_layout = QFormLayout()
        self.classifier_combo = QComboBox()
        self.classifier_combo.addItems(["Gaussian Naive Bayes"])
        form_layout.addRow("Classifier:", self.classifier_combo)
        classifier_group.setLayout(form_layout)
        layout.addWidget(classifier_group)

        testopt_group = QGroupBox("Test options")
        testopt_lyt = QVBoxLayout()
        self.training_set_radio = QRadioButton("Use training set")
        self.supplied_test_radio = QRadioButton("Supplied test set")
        self.percentage_split_radio = QRadioButton("Percentage split")
        self.percentage_split_radio.setChecked(True)
        self.set_button = QPushButton("Set...")
        self.set_button.setEnabled(False)

        testopt_lyt.addWidget(self.training_set_radio)
        slyt = QHBoxLayout()
        slyt.addWidget(self.supplied_test_radio)
        slyt.addWidget(self.set_button)
        
        plyt = QHBoxLayout()
        plyt.addWidget(self.percentage_split_radio)
        self.percentage_split_text = QLineEdit()
        self.percentage_split_text.setFixedHeight(28)
        self.percentage_split_text.setFixedWidth(28)
        self.percentage_split_text.setPlaceholderText("%")
        plyt.addWidget(self.percentage_split_text)
        
        testopt_lyt.addLayout(slyt)
        testopt_lyt.addLayout(plyt)
        testopt_group.setLayout(testopt_lyt)

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

        layout.addWidget(testopt_group)
        layout.addWidget(results_group)
        layout.addLayout(control_lyt)

        start_btn.clicked.connect(self.on_start)
        stop_btn.clicked.connect(self.on_stop)
        clear_btn.clicked.connect(self.on_clear)

    def on_start(self):

        train_features, test_features, train_labels, test_labels = train_test_split(
            self.features, self.labels, test_size=0.5, random_state=0
        )
        clf = GaussianNaiveBayes()
        clf.fit(train_features, train_labels)
        predictions = clf.predict(test_features)

        accuracy = accuracy_score(test_labels, predictions)
        precision, recall, fscore, _ = precision_recall_fscore_support(
            test_labels, predictions, average="macro"
        )

        self.results_text.append(f"Accuracy:  {accuracy:.3f}")
        self.results_text.append(f"Precision: {precision:.3f}")
        self.results_text.append(f"Recall:    {recall:.3f}")
        self.results_text.append(f"F-score:   {fscore:.3f}\n")
        self.results_text.append(f"Mislabeled points: {(predictions != test_labels).sum()}/{test_features.shape[0]}")

    def on_stop(self):
        self.results_text.append("Stop button pressed\n")

    def open_file_dlg(self):
        file_path, sfilter = QFileDialog.getOpenFileName(self, "Open File", "", "Arff files (*.arff);;CSV files (*.csv);;Any file (*.*)")
        if file_path:
            try:
                if(sfilter == "Arff files (*.arff)"):
                    pass
                elif(sfilter == "CSV files (*.csv)"):
                    data_obj = data()
                    data_obj.load_csv(file_path)
                    self.features = data_obj.features
                    self.labels = data_obj.labels
                    self.attributes = data_obj.attributes
                    self.rel_label.setText(f"Relation: {file_path.split('/')[-1].split('.')[0]}")
                    self.inst_label.setText(f"Instances: {len(data_obj.features)}")
                    self.attr_label.setText(f"Attributes: {len(data_obj.attributes)}")
                    for _, attribute in enumerate(self.attributes, start=1):
                        item = QListWidgetItem()
                        item.setText(f"{attribute}")
                        item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
                        item.setCheckState(Qt.Unchecked)
                        self.listwidget.addItem(item)

                    self.tab_widget.setTabEnabled(1, True)
                    
                else:
                    with open(file_path, 'r') as file:
                        txt = file.read()
                        self.results_text.append(f"Opened file {file_path}:\n{txt}\n")
                    file.close()
            except Exception as e:
               print(f"Error reading file: {e}\n")
    
    def on_clear(self):
        self.results_text.clear()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    pltf = platform.system()
    if pltf == "Windows":
        app.setStyle("windowsvista")
    window = gui.App()
    window.show()
    sys.exit(app.exec())
