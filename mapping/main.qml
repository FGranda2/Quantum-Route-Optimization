import QtQuick 6.3
import QtQuick.Controls 6.3
import QtPositioning 6.3
import QtLocation 6.3

ApplicationWindow {
    visible: true
    width: 1000
    height: 1000
    title: "OpenStreetMap Viewer"

    // Array to store clicked coordinates
    property var map_items: []

    Plugin {
        id: osmPlugin
        name: "osm"
    }

    Map {
        id: map
        anchors.fill: parent
        plugin: osmPlugin
        center: QtPositioning.coordinate(51.0443585623781, -114.06312895427341)
        zoomLevel: 16
        property geoCoordinate startCentroid

        // ... (same map handlers as before)

        MouseArea {
            id: mouseArea
            anchors.fill: parent
            visible: false  // Initially disabled
            onClicked: (mouse) => {
                var coordinate = map.toCoordinate(Qt.point(mouse.x, mouse.y))
                mapModel.append({coordinate: coordinate, label: clickCounter++})
                
                // Add the clicked coordinate to the map_items array
                map_items.push([coordinate.latitude, coordinate.longitude])
                console.log("Map Items: " + JSON.stringify(map_items))  // Log to the console
            }
        }

        ListModel {
            id: mapModel
        }

        MapItemView {
            model: mapModel
            delegate: MapQuickItem {
                coordinate: model.coordinate
                sourceItem: Rectangle {
                    width: 24
                    height: 24
                    color: "red"
                    radius: 12
                    border.color: "black"
                    border.width: 1

                    Text {
                        anchors.centerIn: parent
                        text: model.label
                        color: "white"
                    }
                }
                anchorPoint.x: sourceItem.width / 2
                anchorPoint.y: sourceItem.height
            }
        }
    }

    // Button to toggle map clicks
    Button {
        id: toggleButton
        text: "Enable Map Clicks"
        anchors.bottom: parent.bottom
        anchors.right: parent.right
        anchors.margins: 20
        checkable: true
        onToggled: {
            mouseArea.visible = toggleButton.checked
            toggleButton.text = toggleButton.checked ? "Disable Map Clicks" : "Enable Map Clicks"
        }
    }

    // Button to call the optimize function
    Button {
        text: "Optimize"
        anchors.bottom: parent.bottom
        anchors.right: toggleButton.left
        anchors.margins: 20

        onClicked: {
            // Send the map_items array to Python and call the optimize function
            optimizer.set_map_items(map_items)
            optimizer.optimize()
        }
    }

    // Property to keep track of the number of clicks
    property int clickCounter: 1
}
