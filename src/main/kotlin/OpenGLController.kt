
import javafx.beans.property.SimpleObjectProperty
import javafx.geometry.Rectangle2D
import org.intellij.lang.annotations.Language
import org.lwjgl.glfw.GLFW.*
import org.lwjgl.glfw.GLFWErrorCallback
import org.lwjgl.opengl.*
import org.lwjgl.opengl.GL.*
import org.lwjgl.opengl.GL11.*
import org.lwjgl.opengl.GL20.*
import org.lwjgl.system.MemoryUtil
import tornadofx.*
import java.nio.FloatBuffer

class OpenGLController : Controller() {
	val windowDimensions = SimpleObjectProperty<Rectangle2D>()
	private val glEventQueue by inject<GLEventLoop>()

	private var window: Long = -1
	private var program: Int = -1
	private lateinit var vb: FloatBuffer
	private var vao: Int = -1
	private var vbo: Int = -1

	var isInitialized = false

	fun init(windowBounds: Rectangle2D = Rectangle2D(0.0, 0.0, 800.0, 600.0)) {
		glEventQueue.enqueue {
			GLFWErrorCallback.createPrint(System.err).set()

			if (!glfwInit()) throw IllegalStateException("Unable to initialize GLFW")

			glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE)
			glfwWindowHint(GLFW_DECORATED, GLFW_FALSE)

			windowDimensions.value = windowBounds
			window = glfwCreateWindow(windowDimensions.value.width.toInt(), windowDimensions.value.height.toInt(), "Hello World", MemoryUtil.NULL, MemoryUtil.NULL)

			windowDimensions.onChange {
				glEventQueue.enqueue {
					glfwSetWindowPos(window, windowDimensions.value.minX.toInt(), windowDimensions.value.minY.toInt())
					glfwSetWindowSize(window, windowDimensions.value.width.toInt(), windowDimensions.value.height.toInt())
				}
			}

			glfwSetWindowSizeCallback(window) { _, width, height -> glViewport(0, 0, width, height) }

			glfwMakeContextCurrent(window)
			glfwShowWindow(window)

			createCapabilities()

			initShaderProgram()

			val vertices = floatArrayOf(
					0.0f, 0.5f, 0.0f,
					-0.5f, -0.5f, 0.0f,
					0.5f, -0.5f, 0.0f
			)

			vb = MemoryUtil.memAllocFloat(vertices.size)
			vb.put(vertices).flip()

			vao = GL30.glGenVertexArrays()
			GL30.glBindVertexArray(vao)

			vbo = GL15.glGenBuffers()
			GL15.glBindBuffer(GL15.GL_ARRAY_BUFFER, vbo)
			GL15.glBufferData(GL15.GL_ARRAY_BUFFER, vb, GL15.GL_STATIC_DRAW)
			glVertexAttribPointer(0, 3, GL_FLOAT, false, 0, 0)

			glEnableVertexAttribArray(0)

			MemoryUtil.memFree(vb)

			glClearColor(0f, 0f, 0f, 0f)
			glPointSize(10f)
			glEnable(GL_POINT_SMOOTH)

			glEventQueue.setLoopFunction {
				glClear(GL_COLOR_BUFFER_BIT or GL_DEPTH_BUFFER_BIT)

				glUniform4f(0, 0f, 1f, 0f, 1f)
				glDrawArrays(GL_TRIANGLES, 0, 3)

				glUniform4f(0, 1f, 1f, 1f, 1f)
				glDrawArrays(GL_LINE_LOOP, 0, 3)

				glUniform4f(0, 1f, 0f, 0f, 1f)
				glDrawArrays(GL_POINTS, 0, 3)

				glfwSwapBuffers(window)
				glfwPollEvents()
			}

			isInitialized = true
		}
	}

	private fun initShaderProgram() {
		program = glCreateProgram()
		createShader(VERTEX_SHADER_SOURCE, GL_VERTEX_SHADER)
		createShader(FRAGMENT_SHADER_SOURCE, GL_FRAGMENT_SHADER)
		glLinkProgram(program)
		glUseProgram(program)
	}

	private fun createShader(code: String, type: Int): Int {
		val shader = glCreateShader(type)
		glShaderSource(shader, code)
		glCompileShader(shader)
		glAttachShader(program, shader)
		return shader
	}

	companion object {
		@Language("GLSL")
		const val VERTEX_SHADER_SOURCE = """
			#version 150 core

			in  vec3 inVertex;

			void main(void) {
				gl_Position = vec4(inVertex, 1.0);
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