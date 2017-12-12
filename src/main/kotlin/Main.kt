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
		}
	}

	fun jacobi(dimension: Int) {
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

			fillA(dimension, A.buffer)
			fillBuffer(b.buffer)

			if (dimension < 64)
				println(getEquationAsString(A.buffer, b.buffer, dimension))

			initKernel.putArg(xOld)

			jacobiStepKernelA.args {
				arg(A)
				arg(b)
				arg(xNew)
				arg(xOld)
				arg(dimension)
			}

			jacobiStepKernelB.args {
				arg(A)
				arg(b)
				arg(xOld)
				arg(xNew)
				arg(dimension)
			}

			differenceKernel.args {
				arg(xOld)
				arg(xNew)
				arg(diff)
				local(localWorkSizeD * Sizeof.cl_float)
			}

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
			val eps = 1e-10
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
	Main.jacobi(32)
}

/*
25725.5332a +    68.9865b +    51.9664c +    77.2394d +    24.1319e +    89.0070f +    82.6115g +    66.2115h +    21.2813i +    85.2429j +    88.9211k +    27.8932l +    29.0890m +    31.2333n +    53.6804o +    19.5824p = 72.8238
   25.8317a + 25799.9414b +    23.3571c +    80.3968d +    33.0682e +    35.4841f +    57.9939g +    75.9043h +    71.4098i +     0.0580j +    54.1121k +    19.2001l +    85.0673m +     9.0081n +    29.5185o +    35.9061p = 73.4438
   85.6768a +    57.3471b + 25743.5000c +    21.7553d +    84.2169e +    69.6364f +    42.1182g +    97.9979h +    75.7560i +    53.1349j +    40.0928k +    56.6330l +    36.8973m +    93.7875n +    19.2205o +    10.7584p = 51.6238
   32.0135a +     1.2869b +    93.5748c + 25784.8359d +    88.8866e +    87.8069f +     3.1156g +    12.1275h +    54.7160i +    69.7924j +    45.6119k +    17.7973l +    32.4587m +    22.0980n +    64.3001o +     6.0146p = 19.0783
   98.2618a +    11.5317b +    71.2697c +    99.3533d + 25763.2207e +    54.3159f +    44.2637g +    92.7576h +    41.4446i +     4.9954j +    92.9845k +    71.9747l +    19.2373m +     7.8384n +    28.7313o +    13.7123p =  3.8525
   21.6101a +    54.8100b +    99.6821c +    82.0671d +    89.3335e + 25724.7383f +    42.6113g +    92.2276h +    15.5792i +    59.0580j +     2.3554k +    28.3927l +    58.5256m +    79.1258n +    42.7639o +    53.3576p = 62.1883
   10.8609a +    10.6296b +     9.8827c +    83.9727d +    63.5789e +    14.4573f + 25706.3418g +    82.8849h +    86.9451i +    63.2257j +    67.1659k +    12.4288l +    43.9992m +    64.0265n +     9.3667o +    42.5771p = 81.7283
   18.3595a +    59.1335b +    96.5958c +     9.4081d +    90.6100e +    15.4474f +    39.1521g + 25782.9414h +    18.6965i +    85.5269j +    23.4013k +    60.7673l +    45.3724m +    81.8092n +    47.1276o +    88.3816p = 27.8386
   67.9592a +    98.7953b +    45.6162c +    98.3561d +     5.9793e +    32.4090f +    17.1323g +    96.7334h + 25725.2129i +    32.2855j +    30.1965k +    51.5219l +    22.8369m +    21.3636n +    68.4379o +    71.3071p = 18.1446
   15.9766a +    81.8455b +    78.2485c +    37.8692d +    75.7533e +    30.4137f +    24.2116g +    41.4070h +    38.1261i + 25749.7031j +     4.3782k +    10.3423l +    82.9857m +    13.4227n +    16.7885o +    19.6983p = 11.6973
   47.0881a +    80.3197b +    17.8473c +    79.3604d +    26.2560e +    54.3886f +    47.7611g +    24.3664h +    43.1692i +    22.7680j + 25774.5859k +    58.5178l +    71.9255m +    28.2321n +    12.0082o +    96.4008p =  7.6647
   32.1240a +    20.6181b +     8.5364c +    40.2746d +    30.7265e +    84.1855f +    76.5241g +    51.6793h +    46.9018i +     6.2844j +    61.1077k + 25751.6914l +    26.2815m +    55.3041n +    77.2691o +    45.1275p = 24.0729
   99.7096a +    49.4676b +    41.1852c +    93.6575d +    36.0962e +    34.4316f +    60.2552g +    39.9478h +    18.1648i +    22.7405j +    58.3302k +    22.0286l + 25788.3574m +    46.8997n +    46.6904o +    68.7641p = 24.2660
   98.7100a +    19.1779b +    66.7201c +    74.1358d +    38.0862e +    93.3019f +    46.2448g +    45.4964h +     5.4760i +    68.0783j +    30.1814k +    74.3681l +    32.2295m + 25709.8262n +    77.3662o +    36.8810p = 74.0136
   53.4288a +     5.0801b +     1.8493c +    53.8267d +    37.8308e +    19.3549f +    79.0577g +    30.7729h +    40.9023i +    66.3517j +    18.3429k +    86.5173l +    68.2525m +    31.5499n + 25711.7949o +    44.7146p = 20.1612
   38.5156a +    93.0413b +    57.2791c +    66.6893d +    38.2278e +    40.5713f +    74.8143g +     2.7283h +    98.9909i +    51.7995j +    96.4909k +    20.5616l +    27.0558m +    34.1905n +    84.5032o + 25706.1250p = 71.4896
 */