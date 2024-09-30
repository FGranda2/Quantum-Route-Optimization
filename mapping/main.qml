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
        name: "osm"  // Specify OpenStreetMap as the plugin
    }

    Map {
        id: map
        anchors.fill: parent
        plugin: osmPlugin
        center: QtPositioning.coordinate(51.0443585623781, -114.06312895427341) // Calgary Tower
        zoomLevel: 16
        property geoCoordinate startCentroid

        PinchHandler {
            id: pinch
            target: null
            onActiveChanged: if (active) {
                map.startCentroid = map.toCoordinate(pinch.centroid.position, false)
            }
            onScaleChanged: (delta) => {
                map.zoomLevel += Math.log2(delta)
                map.alignCoordinateToPoint(map.startCentroid, pinch.centroid.position)
            }
            onRotationChanged: (delta) => {
                map.bearing -= delta
                map.alignCoordinateToPoint(map.startCentroid, pinch.centroid.position)
            }
            grabPermissions: PointerHandler.TakeOverForbidden
        }

        // Corrected WheelHandler for zooming
        WheelHandler {
            id: wheelHandler
            target: map
            onWheel: (wheel) => {
                if (wheel.angleDelta.y > 0) {
                    map.zoomLevel += 0.5  // Zoom in
                } else {
                    map.zoomLevel -= 0.5  // Zoom out
                }
            }
        }

        DragHandler {
            id: drag
            target: null
            onTranslationChanged: (delta) => map.pan(-delta.x, -delta.y)
        }

        Shortcut {
            enabled: map.zoomLevel < map.maximumZoomLevel
            sequence: StandardKey.ZoomIn
            onActivated: map.zoomLevel = Math.round(map.zoomLevel + 1)
        }

        Shortcut {
            enabled: map.zoomLevel > map.minimumZoomLevel
            sequence: StandardKey.ZoomOut
            onActivated: map.zoomLevel = Math.round(map.zoomLevel - 1)
        }

        // MouseArea to detect map clicks and place markers
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

        // Data model to store marker coordinates and their labels
        ListModel {
            id: mapModel
        }

        // Delegate for map markers
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

    // Toggle button to enable/disable map clicks
    Button {
        id: toggleButton
        text: "Enable Map Clicks"
        anchors.bottom: parent.bottom
        anchors.right: parent.right
        anchors.margins: 20  // Add margins for padding
        checkable: true
        onToggled: {
            if (toggleButton.checked) {
                mouseArea.visible = true
                toggleButton.text = "Disable Map Clicks"
            } else {
                mouseArea.visible = false
                toggleButton.text = "Enable Map Clicks"
            }
        }
    }

    // Property to keep track of the number of clicks
    property int clickCounter: 1
}
