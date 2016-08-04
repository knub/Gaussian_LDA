package gaussian_lda

import scala.io.Source

object TranslationMain {

    def main(args: Array[String]): Unit = {
        // first arg embeddings
        val words = Source.fromFile(args(0)).getLines().map(_.split(' ')(0)).toBuffer
        Source.fromFile(args(1)).getLines.foreach { line =>
            val split = line.split(' ')
            println(split.map(_.toInt).map(words(_)).mkString(" "))
        }
    }

}
