import tornadofx.*
import java.util.*
import java.util.concurrent.ConcurrentLinkedQueue
import java.util.concurrent.atomic.AtomicBoolean
import kotlin.concurrent.thread

class GLEventLoop : Controller() {
	private val queue: Queue<() -> Unit> = ConcurrentLinkedQueue()
	private var running = AtomicBoolean(false)
	private var loopFunction: () -> Unit = {}

	init { start() }

	fun enqueue(event: () -> Unit) = queue.offer(event)
	fun setLoopFunction(event: () -> Unit) { loopFunction = event }

	private fun loop() {
		thread {
			while (running.get()) {
				while (queue.isNotEmpty()) { queue.remove()() }
				loopFunction()
			}
		}
	}

	private fun start() {
		if(!running.get()) {
			running.set(true)
			loop()
		}
	}

	private fun stop() {
		if(running.get()) {
			running.set(false)
			queue.clear()
		}
	}
}