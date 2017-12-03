import com.jogamp.opencl.CLContext
import com.jogamp.opencl.CLProgram

fun CLContext.clProgram(op: OpenCLProgramBuilder.() -> Unit): CLProgram =
		createProgram(OpenCLProgramBuilder().apply(op).build())

class OpenCLProgramBuilder internal constructor() {
	private val functionList = mutableListOf<String>()

	fun OpenCLProgramBuilder.kernelFun(name: String, op: CLFunctionBuilder.() -> Unit) {
		functionList += CLFunctionBuilder(name, true).apply(op).build()
	}

	internal fun build(): String = functionList.joinToString("\n\n")
}

class CLFunctionBuilder internal constructor(private val name: String, private val isKernel: Boolean) {
	private var paramCounter = 0
	internal val parameters = mutableListOf<CLParameter>()

	fun CLFunctionBuilder.intParam(name: String? = null) {
		parameters += CLParameter(CLIntExpression(createName(name)), false, true)
	}

	private fun createName(name: String? = null): String {
		return name ?: "_${paramCounter++}"
	}

	internal fun build(): String = buildString {
		//######### FUNCTION #########\\
		if (isKernel)
			append("kernel")
		else
			append("local")

		append(" void $name(")

		//######### PARAMETER #########\\
		append(parameters.joinToString(", ") { it.build() })
		append(") {\n")

		//######### BODY #########\\

		append("}\n")
	}
}

internal data class CLParameter(val valueExpression: CLValueExpression, val readonly: Boolean, val isGlobal: Boolean) {
	fun build() = buildString {
		if (isGlobal)
			append("global ")
		else
			append("local ")

		if (readonly)
			append("const ")

		append(valueExpression.definition)
	}
}


interface CLExpression {
	fun build(): String
}

abstract class CLValueExpression(val name: String, val definition: String) : CLExpression {
	override fun build() = name
}

class CLIntExpression(name: String) : CLValueExpression(name, "int $name")
class CLFloatExpression(name: String) : CLValueExpression(name, "float $name")

class CLArrayExpression(name: String, definition: String) : CLValueExpression(name, definition) {
	operator fun get(index: CLIntExpression) = "$name[${index.name}]"
}

fun main(args: Array<String>) {
	println(CLContext.create().clProgram {
		kernelFun("HelloWorld") {
			intParam("numElements")
		}
	}.source)
}