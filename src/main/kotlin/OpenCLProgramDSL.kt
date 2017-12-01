import com.jogamp.opencl.CLContext
import com.jogamp.opencl.CLProgram

fun CLContext.clProgram(op: OpenCLProgramBuilder.() -> Unit) : CLProgram =
		createProgram(OpenCLProgramBuilder().apply(op).build())

class OpenCLProgramBuilder internal constructor() {
	private val functionList = mutableListOf<String>()

	fun kernelFun(name: String, op: CLFunctionBuilder.() -> Unit) {
		functionList += CLFunctionBuilder(name, true).apply(op).build()
	}

	internal fun build() : String = functionList.joinToString("\n\n")
}

class CLFunctionBuilder internal constructor(private val name: String, private val isKernel: Boolean) {

	internal fun build() : String {
		val stringBuilder = StringBuilder()

		if(isKernel) stringBuilder.append("kernel")
		else stringBuilder.append("local")

		stringBuilder.append(" void $name(")

		stringBuilder.append(") {\n")

		stringBuilder.append("}\n")
		return stringBuilder.toString()
	}
}

interface CLExpression

fun main(args: Array<String>) {
	println(CLContext.create().clProgram {
		kernelFun("HelloWorld") {

		}
	}.source)
}