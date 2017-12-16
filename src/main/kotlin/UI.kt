
import javafx.application.Application
import javafx.beans.property.DoubleProperty
import javafx.beans.property.SimpleDoubleProperty
import javafx.beans.property.SimpleIntegerProperty
import javafx.geometry.Orientation
import javafx.geometry.Pos
import javafx.scene.canvas.GraphicsContext
import javafx.scene.paint.Color
import tornadofx.*

class AppStarter : App(MainView::class)

class MainView : View("Jacobi Splines") {
	val defaultSize = 5
	val canvasWidth = 900.0
	val canvasHeight = 500.0
	val numSliders = SimpleIntegerProperty()
	val pointDistance = canvasWidth.toProperty() / numSliders
	val pointList = mutableListOf<DoubleProperty>().observable()

	lateinit var gc: GraphicsContext

	override val root = borderpane {
		paddingAll = 10
		setPrefSize(1024.0, 768.0)

		top = hbox(20) {
			alignment = Pos.CENTER_LEFT

			vbox {
				alignment = Pos.CENTER
				label("Num Points")
				spinner(2, 32, defaultSize, 1, true, numSliders, true) {
					prefWidth = 60.0
				}
			}

			separator(Orientation.VERTICAL)

			hbox(10) {
				bindChildren(pointList) {
					slider(-canvasHeight / 2, canvasHeight / 2, null, Orientation.VERTICAL) {
						bind(it)
					}
				}
			}
		}

		center = canvas(canvasWidth, canvasHeight) {
			gc = graphicsContext2D
		}
	}

	init {
		numSliders.addListener { _, oldValue, newValue ->
			if(newValue.toInt() > oldValue.toInt()) {
				val count = newValue.toInt() - oldValue.toInt()
				for (i in 0 until count) {
					val point = SimpleDoubleProperty(0.0)
					pointList.add(point)
					point.onChange { draw()	}
				}
			} else {
				val count = oldValue.toInt() - newValue.toInt()
				for (i in 0 until count) {
					pointList.removeAt(pointList.size - 1)
				}
			}

			draw()
		}
		numSliders.value = defaultSize
	}

	private fun draw() {
		val interpolator = JacobiKernel.jacobiSpline(pointList.map { it.value }.toDoubleArray(), pointDistance.value)
		gc.clearRect(0.0, 0.0, canvasWidth, canvasHeight)

		var lastY = -interpolator(0.0) + canvasHeight / 2
		for (x in 1 until (canvasWidth - pointDistance.value).toInt()) {
			val currY = -interpolator(x.toDouble()) + canvasHeight / 2
			gc.strokeLine(x - 1.0, lastY, x.toDouble(), currY)
			lastY = currY
		}

		pointList.forEachIndexed { i, prop ->
			gc.fill = Color.RED
			gc.fillOval(pointDistance.value * i, -prop.value + canvasHeight / 2, 2.0, 2.0)
			gc.fill = Color.BLACK
		}
	}
}

fun main(args: Array<String>) = Application.launch(AppStarter::class.java)