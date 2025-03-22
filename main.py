import sys
import gui
from PySide6.QtWidgets import QApplication

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = gui.App()
    window.show()
    sys.exit(app.exec())