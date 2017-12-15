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
	val canvasWidth = 700.0
	val canvasHeight = 200.0
	val numSliders = SimpleIntegerProperty()
	val pointDistance = canvasWidth.toProperty() / numSliders
	val pointList = mutableListOf<DoubleProperty>().observable()

	lateinit var gc: GraphicsContext

	override val root = borderpane {
		paddingAll = 10
		setPrefSize(800.0, 600.0)

		top = hbox(20) {
			alignment = Pos.CENTER_LEFT

			vbox {
				alignment = Pos.CENTER
				label("Num Points")
				spinner(2, 20, defaultSize, 1, true, numSliders, true) {
					prefWidth = 60.0
				}

				separator(Orientation.HORIZONTAL) { paddingBottom = 5; paddingTop = 10;}

				button("Calculate").action {
					val interpolator = JacobiKernel.jacobiSpline(pointList.map { it.value }.toDoubleArray(), pointDistance.value)

					gc.clearRect(0.0, 0.0, canvasWidth, canvasHeight)

					for (x in 0 until (canvasWidth - pointDistance.value).toInt()) {
						val y = -interpolator(x.toDouble())
						gc.fillOval(x.toDouble(), y / 1000 + canvasHeight / 2, 2.0, 2.0)
					}

					drawPoints()
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
		numSliders.onChange {
			pointList.clear()
			gc.clearRect(0.0, 0.0, canvasWidth, canvasHeight)

			for (i in 0 until it) {
				val point = SimpleDoubleProperty(1.0)
				pointList.add(point)

				point.addListener { _, oldValue, newValue ->
					gc.clearRect(pointDistance.value * i, -oldValue.toDouble() + canvasHeight / 2, 2.0, 2.0)
					gc.fill = Color.RED
					gc.fillOval(pointDistance.value * i, -newValue.toDouble() + canvasHeight / 2, 2.0, 2.0)
					gc.fill = Color.BLACK
				}

				point.value = 0.0
			}
		}
		numSliders.value = defaultSize
	}

	private fun drawPoints() {
		pointList.forEachIndexed { i, prop ->
			gc.fill = Color.RED
			gc.fillOval(pointDistance.value * i, -prop.value + canvasHeight / 2, 2.0, 2.0)
			gc.fill = Color.BLACK
		}
	}
}

fun main(args: Array<String>) = Application.launch(AppStarter::class.java)