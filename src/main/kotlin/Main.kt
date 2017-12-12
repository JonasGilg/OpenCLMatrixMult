
import com.jogamp.opencl.*
import org.jocl.Sizeof
import java.nio.FloatBuffer
import java.util.*
import kotlin.math.ceil
import kotlin.math.log10
import kotlin.math.min
import kotlin.math.sqrt


object Main {
	private val context: CLContext
	private lateinit var queue: CLCommandQueue
	private lateinit var program: CLProgram
	private lateinit var initKernel: CLKernel
	private lateinit var jacobiStepKernelA: CLKernel
	private lateinit var jacobiStepKernelB: CLKernel
	private lateinit var differenceKernel: CLKernel

	private var maxLocalWorkSize1D = 256
	private var maxLocalWorkSize2D = 8

	init {
		context = clContext {
			println(platform)
			queue = maxFlopsDevice.createCommandQueue()
			program = createProgram("kernel.cl".asFileStream()).build()

			maxLocalWorkSize1D = maxFlopsDevice.maxWorkGroupSize
			maxLocalWorkSize2D = sqrt(maxFlopsDevice.maxWorkGroupSize.toDouble()).toInt()

			initKernel = program.createCLKernel("init")
			jacobiStepKernelA = program.createCLKernel("jacobiStep")
			jacobiStepKernelB = program.createCLKernel("jacobiStep")
			differenceKernel = program.createCLKernel("difference")
		}
	}

	fun jacobi(dimension: Int) {
		context {
			val localWorkSize2D = min(dimension, maxLocalWorkSize2D)
			val globalWorkSizeD = roundUp(localWorkSize2D, dimension)
			val globalWorkSizeDxD = globalWorkSizeD * globalWorkSizeD

			val xOld = createFloatBuffer(globalWorkSizeD, CLMemory.Mem.READ_WRITE)
			val xNew = createFloatBuffer(globalWorkSizeD, CLMemory.Mem.READ_WRITE)

			val A = createFloatBuffer(globalWorkSizeDxD, CLMemory.Mem.READ_ONLY)
			val b = createFloatBuffer(globalWorkSizeD, CLMemory.Mem.READ_ONLY)
			val diff = createFloatBuffer(globalWorkSizeD / localWorkSize2D, CLMemory.Mem.READ_WRITE)

			fillA(dimension, A.buffer)
			fillBuffer(b.buffer)

			println(getEquationAsString(A.buffer, b.buffer, dimension))

			initKernel.putArg(xOld)

			jacobiStepKernelA
					.putArg(A)
					.putArg(b)
					.putArg(xNew)
					.putArg(xOld)
					.putArg(dimension)

			jacobiStepKernelB
					.putArg(A)
					.putArg(b)
					.putArg(xOld)
					.putArg(xNew)
					.putArg(dimension)

			differenceKernel
					.putArg(xOld)
					.putArg(xNew)
					.putArg(diff)
					.putNullArg(localWorkSize2D * Sizeof.cl_float)

			queue.enqueue {
				kernel1DRange(initKernel, 0, dimension.toLong(), dimension.toLong())

				writeBuffer(A)
				writeBuffer(b)
				writeBuffer(xNew)
				writeBuffer(xOld)
				writeBuffer(diff)

				flush()
				finish()
			}

			var currIt = 0
			val maxIt = 100
			val eps = 1e-64
			do {
				queue.enqueue {
					kernel1DRange(jacobiStepKernelA, 0, globalWorkSizeDxD.toLong(), localWorkSize2D.toLong())
					flush()
					finish()

					kernel1DRange(jacobiStepKernelB, 0, globalWorkSizeDxD.toLong(), localWorkSize2D.toLong())
					flush()
					finish()

					kernel1DRange(differenceKernel, 0, globalWorkSizeD.toLong(), localWorkSize2D.toLong())
					readBuffer(diff)
					flush()
					finish()
				}
			} while (reduceDiffBuffer(diff.buffer) > eps && currIt++ < maxIt)

			queue.enqueue {
				readBuffer(xOld)
				flush()
				finish()
			}

			println("Result after $currIt steps:")
			for (i in 0 until dimension)
				print("x$i = ${String.format("%f", xOld.buffer[i])}${if(i < dimension - 1) ", " else ""}")
		}
	}

	fun reduceDiffBuffer(floatBuffer: FloatBuffer): Double {
		var sum = 0.0
		while (floatBuffer.hasRemaining()) {
			sum += floatBuffer.get()
		}

		floatBuffer.rewind()
		return sum
	}

	fun fillA(dimension: Int, buffer: FloatBuffer) {
		val random = Random()
		while (buffer.remaining() != 0)
			buffer.put(random.nextFloat() * 100)
		buffer.rewind()

		for (i in 0 until dimension)
			buffer.put(i * dimension + i, (1 + random.nextFloat() + dimension * dimension) * 100)
		buffer.rewind()
	}

	private fun getMatrixAsString(a: FloatBuffer, dimension: Int): String = buildString {
		for (row in 0 until dimension) {
			append("[")
			for (col in 0 until dimension)
				append(" ${a[row * dimension + col].format(6, 2)} ")
			append("]\n")
		}
	}

	private fun getVectorAsString(v: FloatBuffer): String = buildString {
		while (v.hasRemaining())
			append("[ ${v.get().format(6, 2)} ]\n")
		v.rewind()
	}

	private fun getEquationAsString(a: FloatBuffer, b: FloatBuffer, dimension: Int): String = buildString {
		for (row in 0 until dimension) {
			for (col in 0 until dimension) {
				if(col != 0) append(" ")
				append("${a[row * dimension + col].format(ceil(log10((dimension * dimension + 2) * 100.0)).toInt() + 5, 4)}x$col ")
				if (col < dimension - 1) append('+')
			}
			append("= ${b[row].format(7, 4)}\n")
		}
	}


	private fun roundUp(localWorkSize: Int, elements: Int): Int {
		val r = elements % localWorkSize
		return if (r == 0)
			elements
		else
			elements + localWorkSize - r
	}

	private fun fillBuffer(buffer: FloatBuffer) {
		val random = Random()
		while (buffer.remaining() != 0)
			buffer.put(random.nextFloat() * 100)
		buffer.rewind()
	}
}

fun main(args: Array<String>) {
	Main.jacobi(16)
}