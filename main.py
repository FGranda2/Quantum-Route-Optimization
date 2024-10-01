# main.py
import sys
from PySide6.QtCore import QObject, Slot, Signal
from PySide6.QtGui import QGuiApplication
from PySide6.QtQml import QQmlApplicationEngine

from optimization.optimize import process_map_items

class MapOptimizer(QObject):
    def __init__(self):
        super().__init__()
        self.map_items = []

    # Expose a signal to send data back to QML if needed
    optimization_result = Signal(list)

    @Slot(list)
    def set_map_items(self, items):
        """Sets the map_items array from QML"""
        self.map_items = items
        print("Map items received:", self.map_items)

    @Slot()
    def optimize(self):
        """Call the optimize function and return result"""
        result = process_map_items(self.map_items)
        print("Optimization result:", result)
        self.optimization_result.emit(result)  # Emit result to QML if needed


if __name__ == "__main__":
    app = QGuiApplication(sys.argv)

    engine = QQmlApplicationEngine()

    # Create an instance of MapOptimizer
    optimizer = MapOptimizer()

    # Expose the optimizer object to QML
    engine.rootContext().setContextProperty("optimizer", optimizer)

    # Load the QML file
    engine.load("mapping/main.qml")

    if not engine.rootObjects():
        sys.exit(-1)

    sys.exit(app.exec())
