from PyQt5.QtWidgets import *
from PyQt5.Qt import *

class MenuWidget(QWidget):
    def __init__(self, parent):
        super(MenuWidget, self).__init__(parent)
        self.layout = QGridLayout(self)
        self.setMinimumSize(200, 200)

        # create something

        # -- create label
        self.title_label = QLabel(self)
        self.layout.addWidget(self.title_label, 0, 0, 3, 3, alignment=Qt.AlignCenter)
        self.title_label.setText('How sarcastic are you?')
        self.title_label.show()

        # -- Create the name box for the high score --
        self.name_box = QLineEdit(self)
        self.name_box.setPlaceholderText('Enter your name')
        self.name_box.setAlignment(Qt.AlignCenter)
        #self.name_box.hide()
        # self.name_box.setClearButtonEnabled(True)
        self.layout.addWidget(self.name_box, 2, 1, 1, 1, alignment=Qt.AlignCenter)
        self.name_box.returnPressed.connect(lambda: self.add_to_label(self.name_box.text()))

        # -- create blank label
        # -- Create the image for the title
        self.blank_label = QLabel(self)
        self.layout.addWidget(self.blank_label, 3, 0, 3, 3, alignment=Qt.AlignCenter)
        self.blank_label.show()

        # # -- Create the label for the main menu
        # win_width = self.window().frameGeometry().width()
        # self.question = QLabel("Main Menu", self)
        # my_movie = QMovie('resources/images/cfquest.gif')
        # self.question.setMovie(my_movie)
        # my_movie.start()
        # self.question.setAlignment(Qt.AlignHCenter | Qt.AlignBottom)
        # self.question.setScaledContents(True)
        # self.question.setStyleSheet('min-width: ' + str(win_width*0.8) + ' px;')
        # self.layout.addWidget(self.question, 2, 0, 1, 3)
        #
        # # -- Create the button to start the game
        # self.button1 = QPushButton("NEW GAME", self)
        # self.layout.addWidget(self.button1, 3, 1, alignment=Qt.AlignBottom)
        # self.button1.setStyleSheet("font-size: 30px;")
        #
        # # -- Create the button to start the game
        # self.button2 = QPushButton("HIGH SCORES", self)
        # self.layout.addWidget(self.button2, 4, 1, alignment=Qt.AlignTop)
        # self.button2.setStyleSheet("font-size: 30px;")
        #
        #
        # # -- Create the image for the title
        # self.title_label = QLabel(self)
        # self.title_movie = QMovie("resources/images/Bacchus2.gif")
        # self.title_label.setMovie(self.title_movie)
        # self.title_label.setStyleSheet('height: 500px; min-width: ' + str(win_width) + 'px')
        # self.title_label.setStyleSheet('min-width: 200px')
        # self.title_label.setScaledContents(True)
        # self.layout.addWidget(self.title_label, 2, 0, 3, 3, alignment=Qt.AlignCenter)
        # self.title_label.show()

    def add_to_label(self, text):
        print('his')
        print(text)
        self.name_box.clear()
        self.blank_label.setText(text)
        return