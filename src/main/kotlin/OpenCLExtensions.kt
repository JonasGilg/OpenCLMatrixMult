import com.jogamp.opencl.*
import java.io.InputStream

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

class KernelDSL(private val kernel: CLKernel) {
	operator fun CLBuffer<*>.not() {
		kernel.putArg(this)
	}

	operator fun Int.not() {
		kernel.putArg(this)
	}

	fun nullArg(size: Int) {
		kernel.putNullArg(size)
	}

	fun rewind() {
		kernel.rewind()
	}
}

operator fun CLContext.invoke(op: CLContext.() -> Unit): CLContext {
	op()
	return this
}

object Resource

fun String.asFileStream(): InputStream = Resource.javaClass.getResourceAsStream(this)

fun Float.format(padding: Int, digits: Int) = String.format("%$padding.${digits}f", this)