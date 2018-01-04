
import com.jogamp.opencl.CLBuffer
import com.jogamp.opencl.CLCommandQueue
import com.jogamp.opencl.CLKernel
import com.jogamp.opencl.CLMemory
import com.jogamp.opencl.gl.CLGLBuffer
import com.jogamp.opencl.gl.CLGLContext
import com.jogamp.opengl.GLContext
import org.jocl.Sizeof
import tornadofx.*
import java.nio.FloatBuffer
import kotlin.math.min


class JacobiSplineKernel : Controller() {
	private lateinit var context: CLGLContext
	private lateinit var queue: CLCommandQueue

	private lateinit var initKernel: CLKernel
	private lateinit var initRHSKernel: CLKernel
	private lateinit var jacobiSplineStepKernelA: CLKernel
	private lateinit var jacobiSplineStepKernelB: CLKernel
	private lateinit var differenceKernel: CLKernel
	private lateinit var computeABKernel: CLKernel
	private lateinit var interpolationKernel: CLKernel

	private var maxLocalWorkSize1D = 256

	private lateinit var a: CLBuffer<FloatBuffer>
	private lateinit var b: CLBuffer<FloatBuffer>
	private lateinit var c1: CLBuffer<FloatBuffer>
	private lateinit var c2: CLBuffer<FloatBuffer>
	private lateinit var rhs: CLBuffer<FloatBuffer>
	private lateinit var y: CLBuffer<FloatBuffer>
	private lateinit var diff: CLBuffer<FloatBuffer>

	private lateinit var vboMem: CLGLBuffer<FloatBuffer>
	private var interpolations: Long = -1L
	private var delta: Float = -1f
	private var min: Float = -1f

	fun init(interpolations: Int, glContext: GLContext) {
		context = clContext(glContext) {
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

		this.interpolations = interpolations.toLong()
	}

	fun initVBO(vbo: Int, min: Float, max: Float) {
		val vboMem = context.createFromGLBuffer(vbo, interpolations * 2 * Sizeof.cl_float, CLMemory.Mem.READ_WRITE)
		delta = (max - min) / interpolations.toFloat()
		this.min = min
	}

	fun jacobiSpline(knots: DoubleArray, h: Float): DoubleArray {
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
		interpolate(h)

		a.release()
		b.release()
		c1.release()
		c2.release()
		rhs.release()
		y.release()
		diff.release()

		return vboMem.buffer.toDoubleArray()
	}

	private fun interpolate(h: Float) {
		context {
			glContext.gl.glFinish()

			queue.enqueue { putAcquireGLObject(vboMem) }

			interpolationKernel.args {
				arg(a)
				arg(b)
				arg(c1)
				arg(vboMem)
				arg(h)
				arg(min)
				arg(delta)
				rewind()
			}

			queue.enqueue {
				writeBuffer(vboMem)
				kernel1DRange(interpolationKernel, 0, interpolations * 2, min(interpolations * 2, maxLocalWorkSize1D.toLong()))
				putReleaseGLObject(vboMem)
				finish()
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