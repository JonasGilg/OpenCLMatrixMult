import tornadofx.*

class GLCLInteropController : Controller() {
	val interpolations = 200

	//val clController by inject<JacobiSplineKernel>()
	val glController by inject<OpenGLController>()

	fun init() {
		glController.init()
		//clController.init(interpolations, -1.0f, 1.0f)

		//glController.initVBO(interpolations)
		//val vbo = glController.vbo

	}
}