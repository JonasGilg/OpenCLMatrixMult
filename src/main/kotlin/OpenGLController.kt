
import com.jogamp.newt.NewtFactory
import com.jogamp.newt.opengl.GLWindow
import com.jogamp.opengl.*
import com.jogamp.opengl.GL.*
import com.jogamp.opengl.GL2ES2.GL_FRAGMENT_SHADER
import com.jogamp.opengl.GL2ES2.GL_VERTEX_SHADER
import com.jogamp.opengl.GL2GL3.GL_LINE
import javafx.beans.property.SimpleObjectProperty
import javafx.geometry.Rectangle2D
import org.intellij.lang.annotations.Language
import tornadofx.*

class OpenGLController : Controller(), GLEventListener {


	val windowDimensions = SimpleObjectProperty<Rectangle2D>()

	private var program: Int = -1
	lateinit var context: GLContext
	lateinit var window: GLWindow
	var vbo: Int = -1

	fun init(windowBounds: Rectangle2D = Rectangle2D(0.0, 0.0, 800.0, 600.0)) {
		val dsp = NewtFactory.createDisplay(null)
		val screen = NewtFactory.createScreen(dsp, 0)
		val caps = GLCapabilities(GLProfile.get(GLProfile.GL3))
		window = GLWindow.create(screen, caps)

		window.setSize(800, 600)
		window.setPosition(0, 0)
		window.isUndecorated = true

		windowDimensions.onChange {
			window.setPosition(windowDimensions.value.minX.toInt(), windowDimensions.value.minY.toInt())
			window.setSize(windowDimensions.value.width.toInt(), windowDimensions.value.height.toInt())
		}

		window.addGLEventListener(this)

		window.isVisible = true
	}

	override fun init(drawable: GLAutoDrawable) {
		context = drawable.gl.context

		initShaderProgram()

		context.gl.glClearColor(0f, 0f, 0f, 0f)
	}

	private fun initShaderProgram() {
		program = context.gl.gL3.glCreateProgram()
		createShader(VERTEX_SHADER_SOURCE, GL_VERTEX_SHADER)
		createShader(FRAGMENT_SHADER_SOURCE, GL_FRAGMENT_SHADER)
		context.gl.gL3.glLinkProgram(program)
		context.gl.gL3.glUseProgram(program)
	}

	private fun createShader(code: String, type: Int): Int {
		val shader = context.gl.gL3.glCreateShader(type)
		context.gl.gL3.glShaderSource(shader, 1, arrayOf(code), null)
		context.gl.gL3.glCompileShader(shader)
		context.gl.gL3.glAttachShader(program, shader)
		return shader
	}

	fun initVBO(interpolations: Int) {

	}

	override fun reshape(drawable: GLAutoDrawable?, x: Int, y: Int, width: Int, height: Int) {

	}

	override fun display(drawable: GLAutoDrawable?) {
		context.gl.glClear(GL_COLOR_BUFFER_BIT or GL_DEPTH_BUFFER_BIT)

		context.gl.gL3.glUniform4f(0, 1f, 1f, 1f, 1f)
		context.gl.glDrawArrays(GL_LINE, 0, 3)

		context.gl.gL3.glUniform4f(0, 1f, 0f, 0f, 1f)
		context.gl.glDrawArrays(GL_POINTS, 0, 3)
	}

	override fun dispose(drawable: GLAutoDrawable?) {

	}

	fun close() {
		window.destroy()
	}

	companion object {

		@Language("GLSL")
		const val VERTEX_SHADER_SOURCE = """
			#version 150 core

			in vec2 inVertex;

			void main(void) {
				gl_Position = vec4(inVertex, 0.0f, 1.0f);
			}
			"""
		@Language("GLSL")
		const val FRAGMENT_SHADER_SOURCE = """
			#version 150 core

			out vec4 outColor;

			uniform vec4 color;

			void main(void) {
				outColor = color;
			}
			"""

	}
}