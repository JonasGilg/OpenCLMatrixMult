import com.jogamp.opencl.*
import org.jocl.Sizeof
import java.nio.FloatBuffer
import java.util.*
import kotlin.math.ceil
import kotlin.math.log10
import kotlin.math.min
import kotlin.math.sqrt
import kotlin.system.measureTimeMillis


object Main {
	private val context: CLContext
	private lateinit var queue: CLCommandQueue
	private lateinit var program: CLProgram
	private lateinit var initKernel: CLKernel
	private lateinit var jacobiStepKernelA: CLKernel
	private lateinit var jacobiStepKernelB: CLKernel

	private lateinit var differenceKernel: CLKernel

	private lateinit var initRHSKernel: CLKernel
	private lateinit var jacobiSplineStepKernelA: CLKernel
	private lateinit var jacobiSplineStepKernelB: CLKernel
	private lateinit var computeABKernel: CLKernel

	private var maxLocalWorkSize1D = 256
	private var maxLocalWorkSize2D = 8

	private const val randomRange = 100.0

	init {
		context = clContext {
			println(platform)
			println(maxFlopsDevice)
			println()
			queue = maxFlopsDevice.createCommandQueue()
			program = createProgram("kernel.cl".asFileStream()).build()

			maxLocalWorkSize1D = maxFlopsDevice.maxWorkGroupSize
			maxLocalWorkSize2D = sqrt(maxFlopsDevice.maxWorkGroupSize.toDouble()).toInt()

			initKernel = program.createCLKernel("init")
			jacobiStepKernelA = program.createCLKernel("jacobiStep")
			jacobiStepKernelB = program.createCLKernel("jacobiStep")

			differenceKernel = program.createCLKernel("difference")

			initRHSKernel = program.createCLKernel("initRHS")
			jacobiSplineStepKernelA = program.createCLKernel("jacobiSplineStep")
			jacobiSplineStepKernelB = program.createCLKernel("jacobiSplineStep")
			computeABKernel = program.createCLKernel("computeAB")
		}
	}

	fun jacobi(dimension: Int, matA: FloatArray? = null, vecB: FloatArray? = null) {
		context {
			val localWorkSizeD = min(dimension, maxLocalWorkSize1D)
			val localWorkSizeDxD = localWorkSizeD * localWorkSizeD
			val globalWorkSizeD = roundUp(localWorkSizeD, dimension)
			val globalWorkSizeDxD = globalWorkSizeD * globalWorkSizeD

			val xOld = createFloatBuffer(globalWorkSizeD, CLMemory.Mem.READ_WRITE)
			val xNew = createFloatBuffer(globalWorkSizeD, CLMemory.Mem.READ_WRITE)

			val A = createFloatBuffer(globalWorkSizeDxD, CLMemory.Mem.READ_ONLY)
			val b = createFloatBuffer(globalWorkSizeD, CLMemory.Mem.READ_ONLY)
			val diff = createFloatBuffer(globalWorkSizeD / localWorkSizeD, CLMemory.Mem.READ_WRITE)

			if(matA != null) {
				for (value in matA)
					A.buffer.put(value)
				A.buffer.rewind()
			} else {
				fillA(dimension, A.buffer)
			}

			if(vecB != null) {
				for (value in vecB)
					b.buffer.put(value)
				b.buffer.rewind()
			} else {
				fillBuffer(b.buffer)
			}

			if (dimension < 64)
				println(getEquationAsString(A.buffer, b.buffer, dimension))

			initKernel.args {
				arg(xOld)
				rewind()
			}

			jacobiStepKernelA.args {
				arg(A)
				arg(b)
				arg(xNew)
				arg(xOld)
				arg(dimension)
				rewind()
			}

			jacobiStepKernelB.args {
				arg(A)
				arg(b)
				arg(xOld)
				arg(xNew)
				arg(dimension)
				rewind()
			}

			differenceKernel.args {
				arg(xOld)
				arg(xNew)
				arg(diff)
				local(localWorkSizeD * Sizeof.cl_float)
				rewind()
			}

			queue.enqueue {
				writeBuffer(xOld)
				kernel1DRange(initKernel, 0, dimension.toLong(), dimension.toLong())

				writeBuffer(A)
				writeBuffer(b)
				writeBuffer(xNew)
				writeBuffer(diff)

				flush()
				finish()
			}

			var currIt = 0
			val maxIt = 100
			val eps = 1e-20
			do {
				queue.enqueue {
					kernel1DRange(jacobiStepKernelA, 0, globalWorkSizeDxD.toLong(), localWorkSizeD.toLong())
					flush()
					finish()

					kernel1DRange(jacobiStepKernelB, 0, globalWorkSizeDxD.toLong(), localWorkSizeD.toLong())
					flush()
					finish()

					kernel1DRange(differenceKernel, 0, globalWorkSizeD.toLong(), localWorkSizeD.toLong())
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
			for (i in 0 until dimension) {
				val extension = if (dimension < 27) "${'a' + i}" else "x${i.toSubscriptString()}"
				print("$extension = ${String.format("%f", xOld.buffer[i])}${if (i < dimension - 1) ", " else ""}")
			}

			println()
			/*println()
			println("Difference for computed result vs given RHS")
			val results = computeRhsForComputedValues(A.buffer, xOld.buffer, dimension)
			for (i in 0 until dimension) {
				print("${String.format("expected: %.4f, actual: %.4f, difference: %.4f", b.buffer[i], results[i], results[i] - b.buffer[i])}\n")
			}
			println()*/
		}
	}

	fun jacobiSpline(dimension: Int, h: Float) {
		context {
			val localWorkSizeD = min(dimension, maxLocalWorkSize1D)
			val localWorkSizeDxD = localWorkSizeD * localWorkSizeD
			val globalWorkSizeD = roundUp(localWorkSizeD, dimension)
			val globalWorkSizeDxD = globalWorkSizeD * globalWorkSizeD

			val a = createFloatBuffer(globalWorkSizeD, CLMemory.Mem.READ_WRITE)
			val b = createFloatBuffer(globalWorkSizeD, CLMemory.Mem.READ_WRITE)
			val cOld = createFloatBuffer(globalWorkSizeD, CLMemory.Mem.READ_WRITE)
			val cNew = createFloatBuffer(globalWorkSizeD, CLMemory.Mem.READ_WRITE)
			val rhs = createFloatBuffer(globalWorkSizeD, CLMemory.Mem.READ_WRITE)

			val y = createFloatBuffer(globalWorkSizeD, CLMemory.Mem.READ_ONLY)

			val diff = createFloatBuffer(globalWorkSizeD / localWorkSizeD, CLMemory.Mem.READ_WRITE)

			fillBuffer(y.buffer)

			initKernel.args {
				arg(cOld)
				rewind()
			}
			queue.enqueue {
				writeBuffer(cOld)
				kernel1DRange(initKernel, 0, globalWorkSizeD.toLong(), localWorkSizeD.toLong())
				flush()
				finish()
			}

			initKernel.args {
				arg(cNew)
				rewind()
			}
			queue.enqueue {
				writeBuffer(cNew)
				kernel1DRange(initKernel, 0, globalWorkSizeD.toLong(), localWorkSizeD.toLong())
				flush()
				finish()
			}

			initRHSKernel.args {
				arg(y)
				arg(rhs)
				arg(h)
				rewind()
			}

			jacobiSplineStepKernelA.args {
				arg(rhs)
				arg(cOld)
				arg(cNew)
				rewind()
			}

			jacobiSplineStepKernelB.args {
				arg(rhs)
				arg(cNew)
				arg(cOld)
				rewind()
			}

			computeABKernel.args {
				arg(y)
				arg(cOld)
				arg(a)
				arg(b)
				arg(h)
				rewind()
			}

			differenceKernel.args {
				arg(cOld)
				arg(cNew)
				arg(diff)
				local(localWorkSizeD * Sizeof.cl_float)
				rewind()
			}

			queue.enqueue {
				writeBuffer(y)
				writeBuffer(rhs)
				kernel1DRange(initRHSKernel, 0, globalWorkSizeD.toLong(), localWorkSizeD.toLong())

				writeBuffer(a)
				writeBuffer(b)
				writeBuffer(diff)

				flush()
				finish()
			}

			var currIt = 0
			val maxIt = 100
			val eps = 1e-10
			do {
				queue.enqueue {
					kernel1DRange(jacobiSplineStepKernelA, 0, globalWorkSizeD.toLong(), localWorkSizeD.toLong())
					flush()
					finish()

					kernel1DRange(jacobiSplineStepKernelB, 0, globalWorkSizeD.toLong(), localWorkSizeD.toLong())
					flush()
					finish()

					kernel1DRange(differenceKernel, 0, globalWorkSizeD.toLong(), localWorkSizeD.toLong())
					readBuffer(diff)
					flush()
					finish()
				}
			} while (reduceDiffBuffer(diff.buffer) > eps && currIt++ < maxIt)

			queue.enqueue {
				kernel1DRange(computeABKernel, 0, globalWorkSizeD.toLong(), localWorkSizeD.toLong())
				readBuffer(a)
				readBuffer(b)
				readBuffer(cOld)
				flush()
				finish()
			}

			for (i in 0 until dimension) {
				println("${String.format("%4d", i)} -> a = ${a.buffer.get()}, b = ${b.buffer.get()}, c = ${cOld.buffer.get()}, y = ${y.buffer.get()}")
			}
		}
	}

	private fun computeRhsForComputedValues(A: FloatBuffer, x: FloatBuffer, size: Int) : FloatArray {
		val results = FloatArray(size)

		for (i in 0 until size) {
			var result = 0.0f
			for (j in 0 until size) {
				result += A[i * size + j] * x[j]
			}
			results[i] = result
		}

		return results
	}

	private fun reduceDiffBuffer(floatBuffer: FloatBuffer): Double {
		var sum = 0.0
		while (floatBuffer.hasRemaining()) {
			sum += floatBuffer.get()
		}

		floatBuffer.rewind()
		return sum
	}

	private fun fillA(dimension: Int, buffer: FloatBuffer) {
		val random = Random()
		while (buffer.remaining() != 0)
			buffer.put(random.nextFloat() * randomRange.toFloat())
		buffer.rewind()

		for (i in 0 until dimension)
			buffer.put(i * dimension + i, ((1 + random.nextFloat() + dimension * dimension) * randomRange).toFloat())
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
				val extension = if (dimension < 27) "${'a' + col}" else "x${col.toSubscriptString()}"
				if (col != 0) append(" ")
				append("${a[row * dimension + col].format(ceil(log10((dimension * dimension + 2) * randomRange)).toInt() + 5, 4)} $extension ")
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
			buffer.put(random.nextFloat() * randomRange.toFloat())
		buffer.rewind()
	}
}

fun main(args: Array<String>) {
	Main.jacobiSpline(16, 2f)
}

fun jacobiExamples() {
	run {
		val A = arrayOf(
				2.0f, 3.0f,
				4.0f, 9.0f
		).toFloatArray()

		val b = arrayOf(
				6.0f,
				15.0f
		).toFloatArray()
		Main.jacobi(b.size, A, b)
	}

	println("\n\n############\n")

	run {
		val A = arrayOf(
				37f,   2f,  -1f,  1f,
				2f, -32f,   4f,  2f,
				-1f,  .5f, -36f,  3f,
				1f,   3f,   1f, 37f
		).toFloatArray()
		val b = arrayOf(
				1f,
				-2f,
				0f,
				1f
		).toFloatArray()
		Main.jacobi(b.size, A, b)
	}

	println("\n\n############\n")
	Main.jacobi(32)

	println("\n\n############\n")
	println("\nEquation System of size 1024 took ${measureTimeMillis { Main.jacobi(1024) }}ms to solve")
}