import QtQuick 6.3
import QtQuick.Controls 6.3
import QtPositioning 6.3
import QtLocation 6.3

ApplicationWindow {
    visible: true
    width: 1200
    height: 600
    title: "QuantumFlow"

    // Array to store clicked coordinates
    property var map_items: []

    Plugin {
        id: osmPlugin
        name: "osm"
    }

    Connections {
        target: optimizer  // backend is the Python object exposed to QML

        onOptimization_result: {
            // When the listSignal is emitted, this will be triggered
            var receivedList = arguments[0]
            console.log("Received list:", receivedList)
            mapModel.clear()
            for (var i = 0; i < receivedList.length; i++) {
                var latLon = map_items[receivedList[i]]

                // Create a coordinate for the current lat/lon pair
                var coord = QtPositioning.coordinate(latLon[0], latLon[1])

                // Append the coordinate and label to the mapModel
                mapModel.append({coordinate: coord, label: i + 1})
            }

        }
    }

    Rectangle {
            id: bar
            width: 100
            height: parent.height
            color: "lightgray"  // Set background color
            radius: 5  // Optional: for rounded corners
            anchors.left: parent.left // Anchor to the left side of the window
            anchors.top: parent.top
            opacity: 0.8
            Column {
                anchors.fill: parent
                spacing: 0  // Space between buttons
                anchors.margins: 0  // Padding for the column

                // Button to toggle map clicks
                Button {
                    id: toggleButton
                    text: "Waypoints On"
                    anchors.left: parent.left
                    anchors.right: parent.right // Make button fill the width
                    height: parent.width
                    checkable: true
                    onToggled: {
                        mouseArea.visible = toggleButton.checked
                        toggleButton.text = toggleButton.checked ? "Waypoints Off" : "Waypoints"
                    }
                }

                // Button to call the optimize function
                Button {
                    id: quantumOptimizationButton
                    anchors.left: parent.left
                    anchors.right: parent.right // Make button fill the width
                    height: parent.width

                    contentItem: Text {
                        text: "Quantum\nOptimization"
                        horizontalAlignment: Text.AlignHCenter
                        verticalAlignment: Text.AlignVCenter
                        elide: Text.ElideNone
                        wrapMode: Text.WordWrap
                    }

                    background: Rectangle {
                        implicitWidth: 100
                        implicitHeight: 40
                        opacity: enabled ? 1 : 0.3
                        color: quantumOptimizationButton.down ? "#d6d6d6" : "#f6f6f6"
                        border.color: quantumOptimizationButton.down ? "#26282a" : "#8f8f8f"
                        border.width: 1
                        radius: 4
                    }

                    onClicked: {
                        // Send the map_items array to Python and call the optimize function
                        // optimizer.set_map_items(map_items)
                        // optimizer.optimize()
                        map.drawPolylines();
                    }
                }

                Button {

                    text: "Update"
                    anchors.left: parent.left
                    anchors.right: parent.right // Make button fill the width
                    height: parent.width
                    
                }

                Button {

                    text: "Explore"
                    anchors.left: parent.left
                    anchors.right: parent.right // Make button fill the width
                    height: parent.width
                    
                }

                Button {

                    text: "Center"
                    anchors.left: parent.left
                    anchors.right: parent.right // Make button fill the width
                    height: parent.width
                    
                }

                Button {

                    text: "Reset"
                    anchors.left: parent.left
                    anchors.right: parent.right // Make button fill the width
                    height: parent.width
                    
                }
                

            }
        }

    Map {
        id: map
        anchors.left: bar.right
        anchors.top: parent.top
        anchors.right: parent.right
        anchors.bottom: parent.bottom
        plugin: osmPlugin
        center: QtPositioning.coordinate(51.0443585623781, -114.06312895427341) // Calgary Tower
        zoomLevel: 16
        property geoCoordinate startCentroid

        Component {
            id: arrowComponent
            MapQuickItem {
                anchorPoint.x: sourceItem.width/2
                anchorPoint.y: sourceItem.height/2
                sourceItem: Canvas {
                    width: 20
                    height: 20
                    onPaint: {
                        var ctx = getContext("2d")
                        ctx.fillStyle = "red"
                        ctx.moveTo(0, 20)
                        ctx.lineTo(10, 0)
                        ctx.lineTo(20, 20)
                        ctx.closePath()
                        ctx.fill()
                    }
                }
            }
        }

        MapPolyline {
            id: routeLine
            line.width: 5
            line.color: 'red'
        }

        function drawPolylines() {
            var path = [];
            for (var i = 0; i < mapModel.count; i++) {
                path.push(mapModel.get(i).coordinate);
            }
            // Add the first point again to close the loop
            if (path.length > 0) {
                path.push(mapModel.get(0).coordinate);
            }
            routeLine.path = path;

            // Clear existing arrows
            for (var j = map.mapItems.length - 1; j >= 0; j--) {
                if (map.mapItems[j].objectName === "directionArrow") {
                    map.removeMapItem(map.mapItems[j]);
                }
            }

            // Add direction arrows
            for (var k = 0; k < path.length - 1; k++) {
                var start = path[k];
                var end = path[k+1];
                var midPoint = QtPositioning.coordinate(
                    (start.latitude + end.latitude) / 2,
                    (start.longitude + end.longitude) / 2
                );
                
                
                var arrow = arrowComponent.createObject(map, {
                    coordinate: midPoint,
                    objectName: "directionArrow"
                });
                
                map.addMapItem(arrow);
            }
        }

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
                    color: "purple"
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

    // Property to keep track of the number of clicks
    property int clickCounter: 1
}
