
import javafx.beans.property.SimpleObjectProperty
import javafx.geometry.Rectangle2D
import org.intellij.lang.annotations.Language
import org.lwjgl.glfw.GLFW
import org.lwjgl.glfw.GLFWErrorCallback
import org.lwjgl.opengl.*
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

			if (!GLFW.glfwInit()) throw IllegalStateException("Unable to initialize GLFW")

			GLFW.glfwWindowHint(GLFW.GLFW_RESIZABLE, GLFW.GLFW_TRUE)
			GLFW.glfwWindowHint(GLFW.GLFW_DECORATED, GLFW.GLFW_FALSE)

			windowDimensions.value = windowBounds
			window = GLFW.glfwCreateWindow(windowDimensions.value.width.toInt(), windowDimensions.value.height.toInt(), "Hello World", MemoryUtil.NULL, MemoryUtil.NULL)

			windowDimensions.onChange {
				glEventQueue.enqueue {
					GLFW.glfwSetWindowPos(window, windowDimensions.value.minX.toInt(), windowDimensions.value.minY.toInt())
					GLFW.glfwSetWindowSize(window, windowDimensions.value.width.toInt(), windowDimensions.value.height.toInt())
				}
			}

			GLFW.glfwSetWindowSizeCallback(window) { _, width, height -> GL11.glViewport(0, 0, width, height) }

			GLFW.glfwMakeContextCurrent(window)
			GLFW.glfwShowWindow(window)

			GL.createCapabilities()

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
			GL20.glVertexAttribPointer(0, 3, GL11.GL_FLOAT, false, 0, 0)

			GL20.glEnableVertexAttribArray(0)

			MemoryUtil.memFree(vb)

			GL11.glClearColor(0f, 0f, 0f, 0f)
			GL11.glPointSize(10f)
			GL11.glEnable(GL11.GL_POINT_SMOOTH)

			glEventQueue.setLoopFunction {
				GL11.glClear(GL11.GL_COLOR_BUFFER_BIT or GL11.GL_DEPTH_BUFFER_BIT)

				GL20.glUniform4f(0, 0f, 1f, 0f, 1f)
				GL11.glDrawArrays(GL11.GL_TRIANGLES, 0, 3)

				GL20.glUniform4f(0, 1f, 1f, 1f, 1f)
				GL11.glDrawArrays(GL11.GL_LINE_LOOP, 0, 3)

				GL20.glUniform4f(0, 1f, 0f, 0f, 1f)
				GL11.glDrawArrays(GL11.GL_POINTS, 0, 3)

				GLFW.glfwSwapBuffers(window)
				GLFW.glfwPollEvents()
			}

			isInitialized = true
		}
	}

	private fun initShaderProgram() {
		program = GL20.glCreateProgram()
		createShader(VERTEX_SHADER_SOURCE, GL20.GL_VERTEX_SHADER)
		createShader(FRAGMENT_SHADER_SOURCE, GL20.GL_FRAGMENT_SHADER)
		GL20.glLinkProgram(program)
		GL20.glUseProgram(program)
	}

	private fun createShader(code: String, type: Int): Int {
		val shader = GL20.glCreateShader(type)
		GL20.glShaderSource(shader, code)
		GL20.glCompileShader(shader)
		GL20.glAttachShader(program, shader)
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