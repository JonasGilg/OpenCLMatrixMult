
import com.jogamp.opencl.*
import org.jocl.Sizeof
import java.nio.FloatBuffer
import java.util.*
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

	private var maxLocalWorkSize1D = 256
	private var maxLocalWorkSize2D = 8

	init {
		CLPlatform.listCLPlatforms().forEach { println(it) }

		context = clContext {
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

	fun jakobi(dimension: Int) {
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

			initKernel.putArg(xOld)

			jacobiStepKernelA
					.putArg(A)
					.putArg(b)
					.putArg(xNew)
					.putArg(xOld)

			jacobiStepKernelB
					.putArg(A)
					.putArg(b)
					.putArg(xOld)
					.putArg(xNew)

			differenceKernel
					.putArg(xOld)
					.putArg(xNew)
					.putArg(diff)
					.putNullArg(localWorkSize2D * Sizeof.cl_float)

			queue.enqueue {
				//kernel1DRange(initKernel, 0, dimension.toLong(), dimension.toLong())

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
			val eps = 0.0001
			while (reduceDiffBuffer(diff.buffer) > eps && currIt++ < maxIt) {
				queue.enqueue {
					kernel1DRange(jacobiStepKernelA, 0, globalWorkSizeDxD.toLong(), localWorkSize2D.toLong())
					flush()
					finish()

					kernel1DRange(jacobiStepKernelB, 0, globalWorkSizeDxD.toLong(), localWorkSize2D.toLong())
					flush()
					finish()

					kernel1DRange(differenceKernel, 0, globalWorkSizeD.toLong(), dimension.toLong())
					readBuffer(diff)
					flush()
					finish()
				}
			}

			queue.enqueue {
				readBuffer(xOld)
				flush()
				finish()
			}

			while (xOld.buffer.hasRemaining())
				println(xOld.buffer.get())
		}
	}

	fun reduceDiffBuffer(floatBuffer: FloatBuffer) : Float {
		var sum = 0.0f
		while (floatBuffer.hasRemaining()) {
			sum += floatBuffer.get()
		}

		floatBuffer.rewind()
		return sum
	}

	fun fillA(dimension: Int, buffer: FloatBuffer) {
		val random = Random(12345)
		while (buffer.remaining() != 0)
			buffer.put(random.nextFloat() * 100)
		buffer.rewind()

		for (i in 0 until dimension)
			buffer.put(i * dimension + i, random.nextFloat() * 1000000)
		buffer.rewind()
	}

	fun addVectors(elementCount: Int) {
		context {
			val globalWorkSize = roundUp(maxLocalWorkSize1D, elementCount)

			val bufferA = createFloatBuffer(globalWorkSize, CLMemory.Mem.READ_ONLY)
			val bufferB = createFloatBuffer(globalWorkSize, CLMemory.Mem.READ_ONLY)
			val bufferC = createFloatBuffer(globalWorkSize, CLMemory.Mem.WRITE_ONLY)

			fillBuffer(bufferA.buffer)
			fillBuffer(bufferB.buffer)

			val kernel = program.createCLKernel("VectorAdd") {
				arg(bufferA)
				arg(bufferB)
				arg(bufferC)
				arg(elementCount)
			}

			val completeTime = measureTimeMillis {
				queue.enqueue {
					writeBuffer(bufferA)
					writeBuffer(bufferB)
					kernel1DRange(kernel, 0, globalWorkSize.toLong(), maxLocalWorkSize1D.toLong())
					readBuffer(bufferC)
				}
			}

			println("Vector addition of size $elementCount took $completeTime ms!")

			println("First ten Results of $elementCount:")
			for (i in 0..10)
				println("${bufferA.buffer[i].format(5, 2)} + ${bufferB.buffer[i].format(5, 2)} = ${bufferC.buffer[i].format(6, 2)}")
			println()
		}
	}

	fun multiplyMatrices(dimension: Int) {
		context {
			val localWorkSize2D = min(dimension, maxLocalWorkSize2D)
			val globalWorkSizeD = roundUp(localWorkSize2D, dimension)
			val globalWorkSizeDxD = globalWorkSizeD * globalWorkSizeD

			val bufferA = createFloatBuffer(globalWorkSizeDxD, CLMemory.Mem.READ_ONLY)
			val bufferB = createFloatBuffer(globalWorkSizeDxD, CLMemory.Mem.READ_ONLY)
			val bufferC = createFloatBuffer(globalWorkSizeDxD, CLMemory.Mem.WRITE_ONLY)

			fillBuffer(bufferA.buffer)
			fillBuffer(bufferB.buffer)

			val kernel = program.createCLKernel("MultiplyMatrices") {
				arg(bufferA)
				arg(bufferB)
				arg(bufferC)
			}

			val completeTime = measureTimeMillis {
				queue.enqueue {
					writeBuffer(bufferA)
					writeBuffer(bufferB)
					kernel2DRange(kernel, 0, 0, globalWorkSizeD.toLong(), globalWorkSizeD.toLong(), localWorkSize2D.toLong(), localWorkSize2D.toLong())
					readBuffer(bufferC)
				}
			}

			println("Matrix multiplication of two matrices with dimension $globalWorkSizeD took $completeTime ms!\n")
			if(dimension < 16)
				println(getMatrixResult(bufferA.buffer, bufferB.buffer, bufferC.buffer, globalWorkSizeD))
		}
	}

	fun multiplyMatricesShared(dimension: Int) {
		context {
			val localWorkSize2D = min(dimension, maxLocalWorkSize2D)
			val globalWorkSizeD = roundUp(localWorkSize2D, dimension)
			val globalWorkSizeDxD = globalWorkSizeD * globalWorkSizeD

			val bufferA = createFloatBuffer(globalWorkSizeDxD, CLMemory.Mem.READ_ONLY)
			val bufferB = createFloatBuffer(globalWorkSizeDxD, CLMemory.Mem.READ_ONLY)
			val bufferC = createFloatBuffer(globalWorkSizeDxD, CLMemory.Mem.WRITE_ONLY)

			fillBuffer(bufferA.buffer)
			fillBuffer(bufferB.buffer)

			val kernel = program.createCLKernel("MultiplyMatricesShared") {
				arg(bufferA)
				arg(bufferB)
				arg(bufferC)
				local(localWorkSize2D * localWorkSize2D * Sizeof.cl_float)
				local(localWorkSize2D * localWorkSize2D * Sizeof.cl_float)
			}

			val completeTime = measureTimeMillis {
				queue.enqueue {
					writeBuffer(bufferA)
					writeBuffer(bufferB)
					kernel2DRange(kernel, 0, 0, globalWorkSizeD.toLong(), globalWorkSizeD.toLong(), localWorkSize2D.toLong(), localWorkSize2D.toLong())
					readBuffer(bufferC)
				}
			}

			println("Shared matrix multiplication of two matrices with dimension $globalWorkSizeD took $completeTime ms!\n")
			if(dimension < 16)
				println(getMatrixResult(bufferA.buffer, bufferB.buffer, bufferC.buffer, globalWorkSizeD))
		}
	}

	private fun getMatrixResult(a: FloatBuffer, b: FloatBuffer, c: FloatBuffer, dimension: Int): String {
		val stringBuilder = StringBuilder()
		for (row in 0 until dimension) {
			stringBuilder.append("[")
			for (col in 0 until dimension)
				stringBuilder.append(" ${a[row * dimension + col].format(5, 2)} ")
			stringBuilder.append("] * [")
			for (col in 0 until dimension)
				stringBuilder.append(" ${b[row * dimension + col].format(5, 2)} ")
			stringBuilder.append("] = [")
			for (col in 0 until dimension)
				stringBuilder.append(" ${c[row * dimension + col].format(8, 2)} ")
			stringBuilder.append("]\n")
		}
		return stringBuilder.toString()
	}

	private fun roundUp(localWorkSize: Int, elements: Int): Int {
		val r = elements % localWorkSize
		return if (r == 0)
			elements
		else
			elements + localWorkSize - r
	}

	private fun fillBuffer(buffer: FloatBuffer) {
		val random = Random(12345)
		while (buffer.remaining() != 0)
			buffer.put(random.nextFloat() * 100)
		buffer.rewind()
	}
}

fun main(args: Array<String>) {
	Main.jakobi(16)
}