
import com.jogamp.opencl.*
import java.io.FileNotFoundException
import java.io.InputStream
import java.lang.invoke.MethodHandles
import java.nio.FloatBuffer

fun clContext(platform: Int = 0, op: CLContext.() -> Unit): CLContext {
	val context = CLContext.create(CLPlatform.listCLPlatforms()[platform])
	context.op()
	return context
}

fun CLCommandQueue.enqueue(op: CLCommandQueue.() -> Unit) = op()

fun CLCommandQueue.writeBuffer(buffer: CLBuffer<*>, blockingRead: Boolean = false): CLCommandQueue
		= putWriteBuffer(buffer, blockingRead)

fun CLCommandQueue.readBuffer(buffer: CLBuffer<*>, blockingRead: Boolean = true): CLCommandQueue
		= putReadBuffer(buffer, blockingRead)

fun CLCommandQueue.kernel1DRange(kernel: CLKernel, globalWorkOffset: Long, globalWorkSize: Long, localWorkSize: Long): CLCommandQueue
		= put1DRangeKernel(kernel, globalWorkOffset, globalWorkSize, localWorkSize)

fun CLCommandQueue.kernel2DRange(kernel: CLKernel,
                                 globalWorkOffsetX: Long, globalWorkOffsetY: Long,
                                 globalWorkSizeX: Long, globalWorkSizeY: Long,
                                 localWorkSizeX: Long, localWorkSizeY: Long): CLCommandQueue
		= put2DRangeKernel(kernel, globalWorkOffsetX, globalWorkOffsetY, globalWorkSizeX, globalWorkSizeY, localWorkSizeX, localWorkSizeY)


fun CLProgram.createCLKernel(kernelName: String, op: KernelDSL.() -> Unit): CLKernel {
	val kernel = createCLKernel(kernelName)
	KernelDSL(kernel).op()
	return kernel
}

fun CLKernel.args(op: KernelDSL.() -> Unit) {
	KernelDSL(this).op()
}

class KernelDSL(private val kernel: CLKernel) {
	fun KernelDSL.arg(buffer: CLBuffer<*>) {
		kernel.putArg(buffer)
	}

	fun KernelDSL.arg(arg: Int) {
		kernel.putArg(arg)
	}

	fun KernelDSL.arg(arg: Float) {
		kernel.putArg(arg)
	}

	fun KernelDSL.local(size: Int) {
		kernel.putNullArg(size)
	}

	fun KernelDSL.rewind() {
		kernel.rewind()
	}
}

operator fun CLContext.invoke(op: CLContext.() -> Unit): CLContext {
	op()
	return this
}

fun FloatBuffer.toFloatArray(): FloatArray {
	val list = mutableListOf<Float>()

	while (hasRemaining())
		list.add(get())

	rewind()
	return list.toFloatArray()
}

fun FloatBuffer.toDoubleArray(): DoubleArray {
	val list = mutableListOf<Double>()

	while (hasRemaining())
		list.add(get().toDouble())

	rewind()
	return list.toDoubleArray()
}

fun String.asFileStream(): InputStream = loadFileStream(this)
fun loadFileStream(path:String): InputStream = MethodHandles.lookup().lookupClass().getResourceAsStream(path) ?: throw FileNotFoundException()

fun Float.format(padding: Int, digits: Int) = String.format("%$padding.${digits}f", this)

fun Int.getDigits(): Array<Int> {
	val digits = mutableListOf<Int>()
	getDigits(this, digits)
	return digits.toTypedArray()
}

private fun getDigits(i: Int, digits: MutableList<Int>) {
	if(i / 10 > 0) {
		getDigits(i / 10, digits)
	}
	digits.add(i % 10)
}

fun Int.toSubscriptString(): String {
	val digits = getDigits()
	return buildString {
		digits.forEach {
			append('\u2080' + it)
		}
	}
}