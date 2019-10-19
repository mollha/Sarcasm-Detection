from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.Qt import *
import sys
from Code.GUI.MenuWIdget import MenuWidget


class Main(QMainWindow):
    def __init__(self, parent=None):
        super(Main, self).__init__(parent)
        self.central_widget = QStackedWidget()
        self.setCentralWidget(self.central_widget)
        self.menu_widget = MenuWidget(self)
        self.central_widget.addWidget(self.menu_widget)
        self.central_widget.setCurrentWidget(self.menu_widget)

        self.configure_menu()
        self.setWindowTitle('Cyberfirst Quest')
        self.showMaximized()

        # --- Sets background to black ---
        co_palette = self.palette()
        co_palette.setColor(self.backgroundRole(), Qt.black)
        self.setPalette(co_palette)

        # --- Configure main window
    def configure_menu(self):
        menu = self.menuBar()
        options = menu.addMenu("Options")
        new_game = QAction("New", self)
        new_game.setShortcut("Ctrl+N")
        options.addAction(new_game)
        quit_game = QAction("Exit", self)
        options.addAction(quit_game)
        self.setWindowIcon(QIcon("resources/images/icon.png"))

    def quit(self):
        """
        Return central widget to main menu
        :return:
        """
        self.central_widget.setCurrentWidget(self.menu_widget)

if __name__ == '__main__':
    stylesheet = open('resources/style.txt', 'r').read()
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    QFontDatabase.addApplicationFont("resources/fonts/Westengland.ttf")
    app.setStyleSheet(stylesheet)
    ex = Main()
    ex.show()
    sys.exit(app.exec_())
