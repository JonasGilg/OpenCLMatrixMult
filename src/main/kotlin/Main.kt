
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

	private var maxLocalWorkSize1D = 256
	private var maxLocalWorkSize2D = 8

	init {
		context = CLContext.create(CLPlatform.listCLPlatforms()[0])({
			val device = maxFlopsDevice
			println("$device\n")
			queue = device.createCommandQueue()
			program = createProgram("kernel.cl".asFileStream()).build()

			maxLocalWorkSize1D = device.maxWorkGroupSize
			maxLocalWorkSize2D = sqrt(device.maxWorkGroupSize.toDouble()).toInt()
		})
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
				!bufferA
				!bufferB
				!bufferC
				!elementCount
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
				!bufferA
				!bufferB
				!bufferC
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
				!bufferA
				!bufferB
				!bufferC
				nullArg(localWorkSize2D * localWorkSize2D * Sizeof.cl_float)
				nullArg(localWorkSize2D * localWorkSize2D * Sizeof.cl_float)
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
	val numElements = 4096
	Main.addVectors(numElements)
	println("###############\n")
	Main.multiplyMatrices(numElements)
	println("###############\n")
	Main.multiplyMatricesShared(numElements)
}

