import sys
from PySide6.QtWidgets import QApplication
from PySide6.QtQml import QQmlApplicationEngine
from PySide6.QtCore import QUrl

if __name__ == "__main__":
    app = QApplication(sys.argv)

    # Create and configure QML engine
    engine = QQmlApplicationEngine()

    # Load the QML file
    engine.load(QUrl.fromLocalFile("mapping/main.qml"))

    # If no root objects are loaded, exit the application
    if not engine.rootObjects():
        sys.exit(-1)

    sys.exit(app.exec())
