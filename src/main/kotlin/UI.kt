import javafx.application.Application
import javafx.beans.binding.DoubleBinding
import javafx.beans.property.DoubleProperty
import javafx.beans.property.IntegerProperty
import javafx.beans.property.SimpleDoubleProperty
import javafx.beans.property.SimpleIntegerProperty
import javafx.collections.FXCollections
import javafx.collections.ObservableList
import javafx.geometry.Orientation
import javafx.geometry.Pos
import javafx.geometry.Rectangle2D
import javafx.scene.input.KeyCode
import javafx.stage.Stage
import tornadofx.*

class AppStarter : App(MainView::class) {
	override fun stop() {
		super.stop()
		System.exit(0)
	}
}

class MainView : View("Jacobi Splines") {
	private val defaultSize = 5

	/** List of values for the knots */
	private val knotList: ObservableList<DoubleProperty> = FXCollections.observableArrayList()

	private val glController by inject<OpenGLController>()
	private val clController by inject<JacobiSplineKernel>()

	/** number of knots used by the spinner */
	private val numKnots: IntegerProperty by lazy {
		val prop = SimpleIntegerProperty()
		prop.addListener { _, oldValue, newValue ->
			if (newValue.toInt() > oldValue.toInt()) {
				val count = newValue.toInt() - oldValue.toInt()
				for (i in 0 until count) knotList += SimpleDoubleProperty(0.0)
			} else {
				val count = oldValue.toInt() - newValue.toInt()
				for (i in 0 until count) knotList.removeAt(knotList.size - 1)
			}
		}
		prop
	}

	private val knotDistance: DoubleBinding = 2.0.toProperty() / numKnots
	private val knotDistanceHalf: DoubleBinding = knotDistance / 2

	init {
		glController.init()
		while (!glController.isInitialized) Thread.yield()
		updateGLWindowBounds()
		primaryStage.xProperty().onChange { updateGLWindowBounds() }
		primaryStage.yProperty().onChange { updateGLWindowBounds() }
		primaryStage.widthProperty().onChange { updateGLWindowBounds() }
		primaryStage.heightProperty().onChange { updateGLWindowBounds() }
		numKnots.value = defaultSize
	}

	override val root = borderpane {
		paddingAll = 10
		setPrefSize(1024.0, 160.0)

		top = hbox(20) {
			alignment = Pos.CENTER_LEFT

			vbox {
				alignment = Pos.CENTER

				label("Num Knots")
				spinner(2, 35, defaultSize, 1, true, numKnots, true) {
					prefWidth = 60.0
				}
			}

			separator(Orientation.VERTICAL)

			hbox(10) {
				bindChildren(knotList) {
					slider(-1.0, 1.0, null, Orientation.VERTICAL) { bind(it) }
				}
			}
		}
	}

	private fun updateGLWindowBounds() {
		with(primaryStage) {
			glController.windowDimensions.value = Rectangle2D(
					x, y + height,
					width, width * 9 / 16)
		}
	}
}

fun main(args: Array<String>) = Application.launch(AppStarter::class.java)
