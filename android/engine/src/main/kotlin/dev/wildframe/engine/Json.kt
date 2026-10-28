package dev.wildframe.engine

/** Minimal JSON reader (objects -> LinkedHashMap, arrays -> List, numbers -> Double), so the
 * engine has no dependency that clashes with Android's built-in org.json. Used for configs
 * shipped as assets and for the golden fixtures in tests. Accepts Python's NaN/Infinity. */
object Json {
    fun parse(text: String): Any? {
        val p = Parser(text)
        val v = p.value()
        p.skipWs()
        require(p.i == text.length) { "trailing data at offset ${p.i}" }
        return v
    }

    private class Parser(val s: String) {
        var i = 0

        fun skipWs() {
            while (i < s.length && s[i].isWhitespace()) i++
        }

        fun value(): Any? {
            skipWs()
            require(i < s.length) { "unexpected end of JSON" }
            return when (val c = s[i]) {
                '{' -> obj()
                '[' -> arr()
                '"' -> str()
                't' -> lit("true", true)
                'f' -> lit("false", false)
                'n' -> lit("null", null)
                'N' -> lit("NaN", Double.NaN)
                'I' -> lit("Infinity", Double.POSITIVE_INFINITY)
                else -> if (c == '-' && s.startsWith("-Infinity", i)) lit("-Infinity", Double.NEGATIVE_INFINITY) else num()
            }
        }

        fun lit(word: String, v: Any?): Any? {
            require(s.startsWith(word, i)) { "bad literal at offset $i" }
            i += word.length
            return v
        }

        fun obj(): Map<String, Any?> {
            val out = LinkedHashMap<String, Any?>()
            i++
            skipWs()
            if (s[i] == '}') { i++; return out }
            while (true) {
                skipWs()
                val k = str()
                skipWs()
                require(s[i] == ':') { "expected ':' at offset $i" }
                i++
                out[k] = value()
                skipWs()
                when (s[i]) {
                    ',' -> i++
                    '}' -> { i++; return out }
                    else -> error("expected ',' or '}' at offset $i")
                }
            }
        }

        fun arr(): List<Any?> {
            val out = ArrayList<Any?>()
            i++
            skipWs()
            if (s[i] == ']') { i++; return out }
            while (true) {
                out.add(value())
                skipWs()
                when (s[i]) {
                    ',' -> i++
                    ']' -> { i++; return out }
                    else -> error("expected ',' or ']' at offset $i")
                }
            }
        }

        fun str(): String {
            require(s[i] == '"') { "expected string at offset $i" }
            i++
            val sb = StringBuilder()
            while (true) {
                val c = s[i++]
                when (c) {
                    '"' -> return sb.toString()
                    '\\' -> {
                        when (val e = s[i++]) {
                            '"' -> sb.append('"')
                            '\\' -> sb.append('\\')
                            '/' -> sb.append('/')
                            'b' -> sb.append('\b')
                            'f' -> sb.append('\u000C')
                            'n' -> sb.append('\n')
                            'r' -> sb.append('\r')
                            't' -> sb.append('\t')
                            'u' -> { sb.append(s.substring(i, i + 4).toInt(16).toChar()); i += 4 }
                            else -> error("bad escape \\$e")
                        }
                    }
                    else -> sb.append(c)
                }
            }
        }

        fun num(): Double {
            val start = i
            while (i < s.length && (s[i].isDigit() || s[i] == '-' || s[i] == '+' || s[i] == '.' || s[i] == 'e' || s[i] == 'E')) i++
            return s.substring(start, i).toDouble()
        }
    }
}

/** Read-only view over a parsed rules.yaml / model.yaml dict (shipped as JSON). Mirrors the
 * Python code's `config.get(key, default)` access pattern. */
class Config(val map: Map<String, Any?>) {
    fun section(key: String): Config? = (map[key] as? Map<*, *>)?.let { m ->
        @Suppress("UNCHECKED_CAST")
        Config(m as Map<String, Any?>)
    }

    fun sectionOrEmpty(key: String): Config = section(key) ?: Config(emptyMap())
    fun double(key: String, default: Double): Double = (map[key] as? Number)?.toDouble() ?: default
    fun int(key: String, default: Int): Int = (map[key] as? Number)?.toInt() ?: default
    fun raw(key: String): Any? = map[key]
    fun has(key: String): Boolean = map.containsKey(key)

    companion object {
        @Suppress("UNCHECKED_CAST")
        fun parse(json: String): Config = Config(Json.parse(json) as Map<String, Any?>)
    }
}
