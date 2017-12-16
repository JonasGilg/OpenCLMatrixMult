
import com.jogamp.opencl.*
import org.jocl.Sizeof
import java.nio.FloatBuffer
import kotlin.math.min
import kotlin.math.sqrt


object JacobiSplineKernel {
	private val context: CLContext
	private lateinit var queue: CLCommandQueue
	private lateinit var program: CLProgram

	private lateinit var initKernel: CLKernel
	private lateinit var initRHSKernel: CLKernel

	private lateinit var jacobiSplineStepKernelA: CLKernel
	private lateinit var jacobiSplineStepKernelB: CLKernel
	private lateinit var computeABKernel: CLKernel

	private lateinit var differenceKernel: CLKernel

	private var maxLocalWorkSize1D = 256
	private var maxLocalWorkSize2D = 8

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
			initRHSKernel = program.createCLKernel("initRHS")
			jacobiSplineStepKernelA = program.createCLKernel("jacobiSplineStep")
			jacobiSplineStepKernelB = program.createCLKernel("jacobiSplineStep")
			computeABKernel = program.createCLKernel("computeAB")
			differenceKernel = program.createCLKernel("difference")
		}
	}

	fun jacobiSpline(yArray: DoubleArray, h: Float): JacobiInterpolator {
		var result: JacobiInterpolator? = null

		context {
			val localWorkSizeD = min(yArray.size, maxLocalWorkSize1D)
			val globalWorkSizeD = roundUp(localWorkSizeD, yArray.size)

			val a = createFloatBuffer(globalWorkSizeD, CLMemory.Mem.READ_WRITE)
			val b = createFloatBuffer(globalWorkSizeD, CLMemory.Mem.READ_WRITE)
			val c1 = createFloatBuffer(globalWorkSizeD, CLMemory.Mem.READ_WRITE)
			val c2 = createFloatBuffer(globalWorkSizeD, CLMemory.Mem.READ_WRITE)
			val rhs = createFloatBuffer(globalWorkSizeD, CLMemory.Mem.READ_WRITE)

			val y = createFloatBuffer(globalWorkSizeD, CLMemory.Mem.READ_ONLY)

			val diff = createFloatBuffer(globalWorkSizeD / localWorkSizeD, CLMemory.Mem.READ_WRITE)


			yArray.forEach {
				y.buffer.put(it.toFloat())
			}
			y.buffer.rewind()


			initKernel.args {
				arg(c1)
				rewind()
			}
			queue.enqueue {
				writeBuffer(c1)
				kernel1DRange(initKernel, 0, globalWorkSizeD.toLong(), localWorkSizeD.toLong())
				flush()
				finish()
			}

			c2.buffer.put(0, 0.0f)

			initRHSKernel.args {
				arg(y)
				arg(rhs)
				arg(h)
				rewind()
			}

			jacobiSplineStepKernelA.args {
				arg(rhs)
				arg(c1)
				arg(c2)
				rewind()
			}

			jacobiSplineStepKernelB.args {
				arg(rhs)
				arg(c2)
				arg(c1)
				rewind()
			}

			computeABKernel.args {
				arg(y)
				arg(c1)
				arg(a)
				arg(b)
				arg(h)
				rewind()
			}

			differenceKernel.args {
				arg(c1)
				arg(c2)
				arg(diff)
				local(localWorkSizeD * Sizeof.cl_float)
				rewind()
			}

			queue.enqueue {
				writeBuffer(y)
				writeBuffer(rhs)
				writeBuffer(a)
				writeBuffer(b)
				writeBuffer(diff)

				kernel1DRange(initRHSKernel, 0, globalWorkSizeD.toLong(), localWorkSizeD.toLong())

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
				readBuffer(c1)

				flush()
				finish()
			}

			result = JacobiInterpolator(
					a.buffer.toDoubleArray(),
					b.buffer.toDoubleArray(),
					c1.buffer.toDoubleArray(),
					h.toDouble())

			a.release()
			b.release()
			c1.release()
			c2.release()
			rhs.release()
			y.release()
			diff.release()
		}

		return result ?: throw IllegalStateException("Calculation failed")
	}

	private fun reduceDiffBuffer(floatBuffer: FloatBuffer): Double {
		var sum = 0.0
		while (floatBuffer.hasRemaining()) {
			sum += floatBuffer.get()
		}

		floatBuffer.rewind()
		return sum
	}

	private fun roundUp(localWorkSize: Int, elements: Int): Int {
		val r = elements % localWorkSize
		return if (r == 0)
			elements
		else
			elements + localWorkSize - r
	}
}

class JacobiInterpolator(val a: DoubleArray, val b: DoubleArray, val c: DoubleArray, val h: Double) {
	operator fun invoke(x: Double): Double {
		val i = (x / h).toInt() + 1
		val lowerBound = (i - 1) * h
		val upperBound = lowerBound + h

		return (1.0 / (6.0 * h)) * c[i] * ((x - lowerBound) * (x - lowerBound) * (x - lowerBound)) +
				(1.0 / (6.0 * h)) * c[i - 1] * ((upperBound - x) * (upperBound - x) * (upperBound - x)) +
				b[i] * (x - 0.5 * (lowerBound + upperBound)) + a[i]
	}
}