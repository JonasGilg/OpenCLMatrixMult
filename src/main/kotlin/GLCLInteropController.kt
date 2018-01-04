import tornadofx.*

class GLCLInteropController : Controller() {
	val interpolations = 200

	val glController by inject<OpenGLController>()
	val clController by inject<JacobiSplineKernel>()

	fun init() {
		glController.init()
		clController.init(interpolations, glController.context)

		glController.initVBO(interpolations)
		clController.initVBO(glController.vbo, -1.0f, 1.0f)
	}
}