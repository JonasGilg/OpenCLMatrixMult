
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
import org.jocl.Sizeof
import tornadofx.*
import java.nio.IntBuffer

class OpenGLController : Controller(), GLEventListener {


	val windowDimensions = SimpleObjectProperty<Rectangle2D>()

	private var program: Int = -1
	lateinit var context: GLContext
	lateinit var window: GLWindow
	lateinit var gl: GL3
	var vbo: Int = -1
	var vao: Int = -1
	var interpolations: Int = -1

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
		gl = context.gl.gL3

		initShaderProgram()

		context.gl.glClearColor(0f, 0f, 0f, 0f)
	}

	private fun initShaderProgram() {
		program = gl.glCreateProgram()
		createShader(VERTEX_SHADER_SOURCE, GL_VERTEX_SHADER)
		createShader(FRAGMENT_SHADER_SOURCE, GL_FRAGMENT_SHADER)
		gl.glLinkProgram(program)
		gl.glUseProgram(program)
	}

	private fun createShader(code: String, type: Int): Int {
		val shader = gl.glCreateShader(type)
		gl.glShaderSource(shader, 1, arrayOf(code), null)
		gl.glCompileShader(shader)
		gl.glAttachShader(program, shader)
		return shader
	}

	fun initVBO(interpolations: Int) {
		this.interpolations = interpolations
		val tmpArray = IntArray(1)
		gl.glGenVertexArrays(1, IntBuffer.wrap(tmpArray))
		vao = tmpArray[0]
		gl.glBindVertexArray(vao)

		gl.glGenBuffers(1, IntBuffer.wrap(tmpArray))
		vbo = tmpArray[0]

		gl.glBindBuffer(GL_ARRAY_BUFFER, vbo)
		gl.glBufferData(GL_ARRAY_BUFFER, (interpolations * 2 * Sizeof.cl_float).toLong(), null, GL_DYNAMIC_DRAW)

		val location = gl.glGetAttribLocation(program, "inVertex")
		gl.glVertexAttribPointer(location, 2, GL_FLOAT, false, 0, 0)
		gl.glEnableVertexAttribArray(location)
	}

	override fun reshape(drawable: GLAutoDrawable?, x: Int, y: Int, width: Int, height: Int) {

	}

	override fun display(drawable: GLAutoDrawable?) {
		gl.glClear(GL_COLOR_BUFFER_BIT or GL_DEPTH_BUFFER_BIT)

		gl.glUniform4f(0, 1f, 1f, 1f, 1f)
		gl.glDrawArrays(GL_LINE, 0, interpolations)
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