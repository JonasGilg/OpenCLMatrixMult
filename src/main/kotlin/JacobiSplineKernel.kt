import com.jogamp.opencl.*
import org.jocl.Sizeof
import tornadofx.*
import java.nio.FloatBuffer
import kotlin.math.min


class JacobiSplineKernel : Controller() {
	private val context: CLContext
	private lateinit var queue: CLCommandQueue

	private lateinit var initKernel: CLKernel
	private lateinit var initRHSKernel: CLKernel
	private lateinit var jacobiSplineStepKernelA: CLKernel
	private lateinit var jacobiSplineStepKernelB: CLKernel
	private lateinit var differenceKernel: CLKernel
	private lateinit var computeABKernel: CLKernel
	private lateinit var interpolationKernel: CLKernel

	private var maxLocalWorkSize1D = 256

	init {
		context = clContext {
			println(platform)
			println(maxFlopsDevice)
			println()
			queue = maxFlopsDevice.createCommandQueue()

			maxLocalWorkSize1D = maxFlopsDevice.maxWorkGroupSize

			createProgram("kernel.cl".asFileStream()).build().apply {
				initKernel = createCLKernel("init")
				initRHSKernel = createCLKernel("initRHS")
				jacobiSplineStepKernelA = createCLKernel("jacobiSplineStep")
				jacobiSplineStepKernelB = createCLKernel("jacobiSplineStep")
				computeABKernel = createCLKernel("computeAB")
				differenceKernel = createCLKernel("difference")
				interpolationKernel = createCLKernel("interpolate")
			}
		}
	}

	private lateinit var a: CLBuffer<FloatBuffer>
	private lateinit var b: CLBuffer<FloatBuffer>
	private lateinit var c1: CLBuffer<FloatBuffer>
	private lateinit var c2: CLBuffer<FloatBuffer>
	private lateinit var rhs: CLBuffer<FloatBuffer>
	private lateinit var y: CLBuffer<FloatBuffer>
	private lateinit var yInterpolated: CLBuffer<FloatBuffer>
	private lateinit var diff: CLBuffer<FloatBuffer>

	fun jacobiSpline(knots: DoubleArray, h: Float, interpolations: Int): DoubleArray {
		val localWorkSize = min(knots.size, maxLocalWorkSize1D)
		val globalWorkSize = roundUp(localWorkSize, knots.size)

		y = context.createFloatBuffer(globalWorkSize, CLMemory.Mem.READ_ONLY)
		y.buffer.fromArray(knots.toTypedArray())

		initC(globalWorkSize, localWorkSize)
		initRHS(globalWorkSize, localWorkSize, h)
		initStepsKernels()
		initDiff(globalWorkSize, localWorkSize)

		var currIt = 0
		val maxIt = 100
		val eps = 1e-10
		do {
			queue.enqueue {
				kernel1DRange(jacobiSplineStepKernelA, 0, globalWorkSize.toLong(), localWorkSize.toLong())
				kernel1DRange(jacobiSplineStepKernelB, 0, globalWorkSize.toLong(), localWorkSize.toLong())
				kernel1DRange(differenceKernel, 0, globalWorkSize.toLong(), localWorkSize.toLong())
				readBuffer(diff)
			}
		} while (reduceDiffBuffer(diff.buffer) > eps && currIt++ < maxIt)

		computeAB(globalWorkSize, localWorkSize, h)
		interpolate(interpolations, h)

		a.release()
		b.release()
		c1.release()
		c2.release()
		rhs.release()
		y.release()
		diff.release()

		return yInterpolated.buffer.toDoubleArray()
	}

	private fun interpolate(interpolations: Int, h: Float) {
		context {
			yInterpolated = createFloatBuffer(interpolations, CLMemory.Mem.READ_WRITE)
			interpolationKernel.args {
				arg(a)
				arg(b)
				arg(c1)
				arg(yInterpolated)
				arg(h)
				rewind()
			}

			queue.enqueue {
				writeBuffer(yInterpolated)
				kernel1DRange(interpolationKernel, 0, interpolations.toLong(), min(interpolations, maxLocalWorkSize1D).toLong())
				readBuffer(yInterpolated)
			}
		}
	}

	private fun computeAB(globalSize: Int, localSize: Int, h: Float) {
		context {
			a = createFloatBuffer(globalSize, CLMemory.Mem.READ_WRITE)
			b = createFloatBuffer(globalSize, CLMemory.Mem.READ_WRITE)

			computeABKernel.args {
				arg(y)
				arg(c1)
				arg(a)
				arg(b)
				arg(h)
				rewind()
			}

			queue.enqueue {
				writeBuffer(a)
				writeBuffer(b)
				kernel1DRange(computeABKernel, 0, globalSize.toLong(), localSize.toLong())

				readBuffer(a)
				readBuffer(b)
				readBuffer(c1)
			}
		}
	}

	private fun initDiff(globalSize: Int, localSize: Int) {
		context {
			diff = createFloatBuffer(globalSize / localSize, CLMemory.Mem.READ_WRITE)
			differenceKernel.args {
				arg(c1)
				arg(c2)
				arg(diff)
				local(localSize * Sizeof.cl_float)
				rewind()
			}

			queue.enqueue {
				writeBuffer(diff)
			}
		}
	}

	private fun initStepsKernels() {
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
	}

	private fun initC(globalSize: Int, localSize: Int) {
		context {
			c1 = createFloatBuffer(globalSize, CLMemory.Mem.READ_WRITE)
			c2 = createFloatBuffer(globalSize, CLMemory.Mem.READ_WRITE)

			initKernel.args {
				arg(c1)
				rewind()
			}
			c2.buffer.put(0, 0.0f)

			queue.enqueue {
				writeBuffer(c1)
				kernel1DRange(initKernel, 0, globalSize.toLong(), localSize.toLong())
			}
		}
	}

	private fun initRHS(globalSize: Int, localSize: Int, h: Float) {
		context {
			rhs = createFloatBuffer(globalSize, CLMemory.Mem.READ_WRITE)
			initRHSKernel.args {
				arg(y)
				arg(rhs)
				arg(h)
				rewind()
			}

			queue.enqueue {
				writeBuffer(y)
				writeBuffer(rhs)
				kernel1DRange(initRHSKernel, 0, globalSize.toLong(), localSize.toLong())
			}
		}
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